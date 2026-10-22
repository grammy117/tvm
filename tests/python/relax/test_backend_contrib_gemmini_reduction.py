# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Array sums: independent numerical, scheduling, fallback and queue checks."""

import ctypes
import unittest

import numpy as np
import tvm
from tvm import relax, te, tir
from tvm.relax.backend.contrib.gemmini import prepare_gemmini_graph
from tvm.relax.backend.contrib.gemmini_reduction import (
    make_reduction_schedule,
    reduction_blocks,
    schedule_reduction,
)
from tvm.tir.schedule.testing import verify_trace_roundtrip

import test_backend_contrib_gemmini_tensorir as reference


def workload(shape, axes, keepdims=False, source_dtype="int8", output_dtype="int32"):
    source = te.placeholder(shape, source_dtype, "source")
    reductions = {axis: te.reduce_axis((0, shape[axis]), "r" + str(axis)) for axis in axes}
    output_shape = tuple(
        1 if axis in axes else size
        for axis, size in enumerate(shape)
        if keepdims or axis not in axes
    )

    def value(*indices):
        spatial = iter(indices)
        access = []
        for axis in range(len(shape)):
            if axis in axes:
                access.append(reductions[axis])
                if keepdims:
                    next(spatial)
            else:
                access.append(next(spatial))
        return te.sum(source[tuple(access)].astype(output_dtype), axis=list(reductions.values()))

    output = te.compute(output_shape, value, name="unnamed_sum")
    return te.create_prim_func([source, output])


class GemminiReductionTests(unittest.TestCase):
    setUp = reference.GemminiTensorIRTests.setUp
    build = reference.GemminiTensorIRTests.build

    def test_int64_results_use_proven_int32_accumulation_and_cpu_widening(self):
        for shape, axes, keepdims in (
            ((3, 17), (1,), False),
            ((16, 32), (1,), False),
            ((17,), (0,), False),
            ((1,), (0,), False),
            ((2, 3, 4), (0, 2), False),
            ((2, 3, 4), (1, 2), True),
        ):
            with self.subTest(shape=shape, axes=axes, keepdims=keepdims):
                original = workload(shape, axes, keepdims, output_dtype="int64")
                saved = tvm.ir.save_json(original)
                result = make_reduction_schedule(original)
                self.assertTrue(result.metadata["exact_accumulator_narrowing"])
                self.assertEqual(result.metadata["output_dtype"], "int64")
                self.assertIn("UnsafeSetDType", {i.kind.name for i in result.trace.insts})
                replay = tir.Schedule(original, debug_mask="all")
                result.trace.apply_to_schedule(replay, remove_postproc=False)
                verify_trace_roundtrip(replay, original)
                self.assertEqual(tvm.ir.save_json(original), saved)
                runtime, control = self.build(result.scheduled_mod)
                values = (np.arange(np.prod(shape)).reshape(shape) % 256 - 128).astype("int8")
                for a in (values, np.full(shape, -128, "int8"), np.full(shape, 127, "int8")):
                    expected = a.astype("int64").sum(axis=axes, keepdims=keepdims)
                    av, output = tvm.nd.array(a), tvm.nd.empty(expected.shape, "int64")
                    control.emulator_reset()
                    runtime(av, output)
                    np.testing.assert_array_equal(output.numpy(), expected)
                    np.testing.assert_array_equal(av.numpy(), a)
                    self.assertGreater(control.emulator_commands(), 0)
                    self.assertEqual(control.emulator_faults(), 0)
                    self.assertEqual(control.emulator_pending(), 0)

    def test_sum_shapes_axes_extremes_and_full_trace(self):
        for shape, axes, keepdims in (
            ((3, 17), (1,), False),
            ((16, 32), (1,), False),
            ((17,), (0,), False),
            ((1,), (0,), False),
            ((2, 3, 4), (0, 2), False),
            ((2, 3, 4), (1, 2), True),
            ((2, 3, 4), (0, 1, 2), False),
        ):
            with self.subTest(shape=shape, axes=axes, keepdims=keepdims):
                original = workload(shape, axes, keepdims)
                result = make_reduction_schedule(original)
                runtime, control = self.build(result.scheduled_mod)
                values = (np.arange(np.prod(shape)).reshape(shape) % 256 - 128).astype("int8")
                for a in (values, np.full(shape, -128, "int8"), np.full(shape, 127, "int8")):
                    expected = a.astype("int64").sum(axis=axes, keepdims=keepdims).astype("int32")
                    av, output = tvm.nd.array(a), tvm.nd.empty(expected.shape, "int32")
                    control.emulator_reset()
                    runtime(av, output)
                    np.testing.assert_array_equal(output.numpy(), expected)
                    np.testing.assert_array_equal(av.numpy(), a)
                    self.assertEqual(control.emulator_faults(), 0)
                    self.assertEqual(control.emulator_pending(), 0)
                    self.assertGreater(control.emulator_commands(), 0)
                replay = tir.Schedule(original, debug_mask="all")
                result.trace.apply_to_schedule(replay, remove_postproc=True)
                scheduled, _, _ = schedule_reduction(original)
                tvm.ir.assert_structural_equal(replay.mod, scheduled.mod)

    def test_normal_relax_cast_sum_and_cpu_readout(self):
        a = relax.Var("a", relax.TensorStructInfo((3, 17), "int8"))
        builder = relax.BlockBuilder()
        with builder.function("main", [a]):
            with builder.dataflow():
                total = builder.emit(relax.op.sum(relax.op.astype(a, "int32"), axis=1))
                output = builder.emit_output(
                    relax.op.astype(total, "float32") / relax.const(np.float32(17))
                )
            builder.emit_func_output(output)
        report = []
        mod = prepare_gemmini_graph(builder.get(), report=report)
        self.assertEqual(sum(row["placement"] == "gemmini" for row in report), 1, report)
        path = self.root / "mean.so"
        executable = relax.build(mod, "llvm")
        executable.export_library(
            str(path), addons=[str(self.emulator)], cc="cc", options=["-std=c11"]
        )
        control = ctypes.CDLL(str(path))
        control.emulator_reset()
        vm = relax.VirtualMachine(tvm.runtime.load_module(str(path)), tvm.cpu())
        values = (np.arange(51).reshape(3, 17) - 40).astype("int8")
        values[:, 0] += 1  # Exercise non-integral means and the original FP32 division.
        output = vm["main"](tvm.nd.array(values))
        expected = values.astype("int64").sum(axis=1).astype("float32") / np.float32(17)
        np.testing.assert_array_equal(output.numpy(), expected)
        self.assertEqual(control.emulator_faults(), 0)

    def test_exported_sum_keeps_pytorch_dtype_shape_and_native_reduction(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is unavailable")
        from tvm.relax.frontend.torch import from_exported_program

        for shape, dims, keepdim, dtype in (
            ((3, 17), (1,), False, None),
            ((2, 3, 4), (0, 2), True, None),
            ((3, 17), (1,), False, "int32"),
            ((3, 17), (1,), True, "int8"),
        ):
            with self.subTest(shape=shape, dims=dims, keepdim=keepdim, dtype=dtype):

                class Sum(torch.nn.Module):
                    def forward(self, source):
                        return source.sum(
                            dim=dims,
                            keepdim=keepdim,
                            dtype=None if dtype is None else getattr(torch, dtype),
                        )

                values = (np.arange(np.prod(shape)).reshape(shape) * 13 - 128).astype("int8")
                model = Sum()
                example = torch.from_numpy(values)
                graph = from_exported_program(
                    torch.export.export(model, (example,)), unwrap_unit_return_tuple=True
                )
                report = []
                prepared = prepare_gemmini_graph(graph, report=report)
                self.assertEqual(sum(row["placement"] == "gemmini" for row in report), 1, report)
                path = self.root / (str(dtype) + "-" + str(len(shape)) + ".so")
                executable = relax.build(prepared, "llvm")
                executable.export_library(
                    str(path), addons=[str(self.emulator)], cc="cc", options=["-std=c11"]
                )
                control = ctypes.CDLL(str(path))
                control.emulator_reset()
                vm = relax.VirtualMachine(tvm.runtime.load_module(str(path)), tvm.cpu())
                actual = vm["main"](tvm.nd.array(values)).numpy()
                expected = model(example).numpy()
                self.assertEqual(actual.dtype, expected.dtype)
                self.assertEqual(actual.shape, expected.shape)
                np.testing.assert_array_equal(actual, expected)
                self.assertGreater(control.emulator_commands(), 0)
                self.assertEqual(control.emulator_faults(), 0)

    def test_standard_metaschedule_and_features(self):
        from tvm import meta_schedule as ms
        from tvm.relax.backend.contrib.gemmini_meta_schedule import gemmini_space_generator
        from tvm.relax.backend.contrib.gemmini_target import GemminiCapabilities
        from tvm.relax.backend.contrib.gemmini_tensorir import lower_tensorir_schedule

        a = relax.Var("a", relax.TensorStructInfo((3, 16), "int8"))
        builder = relax.BlockBuilder()
        with builder.function("main", [a]):
            with builder.dataflow():
                output = builder.emit_output(relax.op.sum(relax.op.astype(a, "int32"), axis=1))
            builder.emit_func_output(output)
        mathematical = tvm.transform.Sequential(
            [
                relax.transform.LegalizeOps(),
                relax.transform.AnnotateTIROpPattern(),
                relax.transform.FuseOps(),
                relax.transform.FuseTIR(),
            ]
        )(builder.get())
        original = next(f for f in mathematical.functions.values() if isinstance(f, tir.PrimFunc))
        context = ms.TuneContext(
            mod=original,
            target=GemminiCapabilities().target(),
            space_generator=gemmini_space_generator(),
            num_threads=1,
        )
        designs = context.generate_design_space()
        self.assertGreaterEqual(len(designs), 2)
        features = ms.feature_extractor.PerStoreFeature().extract_from(
            context, [ms.search_strategy.MeasureCandidate(design, []) for design in designs]
        )
        self.assertTrue(all(f.numpy().size and np.isfinite(f.numpy()).all() for f in features))
        for design in designs:
            design.enter_postproc()
            self.assertTrue(all(p.apply(design) for p in context.space_generator.postprocs))
            result = lower_tensorir_schedule(
                original, design, workload_metadata={"operation": "sum_i8_i32"}
            )
            runtime, control = self.build(result.scheduled_mod)
            values = np.full((3, 16), -128, "int8")
            output = tvm.nd.empty((3,), "int32")
            runtime(tvm.nd.array(values), output)
            np.testing.assert_array_equal(output.numpy(), np.full((3,), -2048, "int32"))
            self.assertEqual(control.emulator_faults(), 0)

    def test_numeric_and_dtype_boundaries(self):
        self.assertEqual(len(reduction_blocks(workload((1, 1 << 24), (1,)))), 1)
        self.assertFalse(reduction_blocks(workload((1, (1 << 24) + 1), (1,))))
        self.assertFalse(reduction_blocks(workload((65536, 65536, 1), (2,))))
        for dtype in ("int32", "float32"):
            self.assertFalse(
                reduction_blocks(workload((3, 17), (1,), source_dtype=dtype, output_dtype=dtype))
            )
        self.assertEqual(len(reduction_blocks(workload((3, 17), (1,), output_dtype="int64"))), 1)
        self.assertFalse(reduction_blocks(workload((1, (1 << 24) + 1), (1,), output_dtype="int64")))

    def test_metaschedule_keeps_wide_sum_contract_and_features(self):
        from tvm import meta_schedule as ms
        from tvm.relax.backend.contrib.gemmini_meta_schedule import gemmini_space_generator
        from tvm.relax.backend.contrib.gemmini_target import GemminiCapabilities
        from tvm.relax.backend.contrib.gemmini_tensorir import lower_tensorir_schedule
        from tvm.relax.backend.contrib.gemmini_reduction import reduction_metadata

        original = workload((3, 16), (1,), output_dtype="int64")
        context = ms.TuneContext(
            mod=original,
            target=GemminiCapabilities().target(),
            space_generator=gemmini_space_generator(),
            num_threads=1,
        )
        designs = context.generate_design_space()
        self.assertGreaterEqual(len(designs), 2)
        features = ms.feature_extractor.PerStoreFeature().extract_from(
            context, [ms.search_strategy.MeasureCandidate(design, []) for design in designs]
        )
        self.assertTrue(all(f.numpy().size and np.isfinite(f.numpy()).all() for f in features))
        metadata = reduction_metadata(reduction_blocks(original)[0])
        for design in designs:
            design.enter_postproc()
            self.assertTrue(all(p.apply(design) for p in context.space_generator.postprocs))
            result = lower_tensorir_schedule(original, design, workload_metadata=metadata)
            self.assertTrue(result.metadata["exact_accumulator_narrowing"])
            runtime, control = self.build(result.scheduled_mod)
            values = np.full((3, 16), -128, "int8")
            output = tvm.nd.empty((3,), "int64")
            runtime(tvm.nd.array(values), output)
            np.testing.assert_array_equal(output.numpy(), np.full((3,), -2048, "int64"))
            self.assertEqual(control.emulator_faults(), 0)


if __name__ == "__main__":
    unittest.main()
