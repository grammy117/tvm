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
"""Native DMA arithmetic/movement against independent bit-exact references."""

import unittest
import ctypes

import numpy as np
import tvm
from tvm import relax, te, tir
from tvm.relax.backend.contrib.gemmini import prepare_gemmini_graph
from tvm.relax.backend.contrib.gemmini_elementwise import (
    make_elementwise_schedule,
    schedule_elementwise,
)
from tvm.relax.backend.contrib.gemmini_tensorir import lower_tensorir_schedule

import test_backend_contrib_gemmini_tensorir as reference


def workload(shape, kind):
    dtype = (
        "float32"
        if kind == "copy_f32"
        else "int8"
        if kind in ("copy_i8", "widen_i8_i32", "relu_i8")
        else "int32"
    )
    a = te.placeholder(shape, dtype, "source")
    b = te.placeholder(shape, dtype, "other")
    value = te.compute(
        shape,
        lambda *i: (a[i].astype("int64") + b[i].astype("int64")).astype("int32")
        if kind == "add_i32_wrapping"
        else a[i] + b[i]
        if kind == "add_i32"
        else a[i].astype("int32")
        if kind == "widen_i8_i32"
        else te.max(a[i], tir.const(0, "int8"))
        if kind == "relu_i8"
        else a[i],
        name="value",
    )
    return te.create_prim_func([a, b, value] if kind.startswith("add_i32") else [a, value])


class GemminiElementwiseTests(unittest.TestCase):
    setUp = reference.GemminiTensorIRTests.setUp
    build = reference.GemminiTensorIRTests.build

    def test_reflection_and_edge_padding_use_indexed_dma_and_replay(self):
        for mode in ("reflect", "edge"):
            for shape in ((4,), (2, 3), (1, 2, 3), (1, 1, 2, 3), (1, 1, 2, 2, 3)):
                for dtype in ("int8", "int32", "float32"):
                    with self.subTest(mode=mode, shape=shape, dtype=dtype):
                        widths = [(0, 0) if dim == 1 else (1, 1) for dim in shape]
                        if dtype == "float32":
                            bits = np.array(
                                [0, 0x80000000, 0x7FC12345, 0x7F812345, 1, 0xFF800000],
                                "uint32",
                            )
                            values = np.resize(bits, shape).view("float32")
                        else:
                            values = (np.arange(np.prod(shape)).reshape(shape) * 17 - 128).astype(
                                dtype
                            )
                        expected = np.pad(values, widths, mode=mode)
                        source = relax.Var("source", relax.TensorStructInfo(shape, dtype))
                        builder = relax.BlockBuilder()
                        with builder.function("main", [source]):
                            with builder.dataflow():
                                output = builder.emit_output(
                                    relax.op.nn.pad(
                                        source,
                                        tuple(x for pair in widths for x in pair),
                                        pad_mode=mode,
                                    )
                                )
                            builder.emit_func_output(output)
                        graph = builder.get()
                        report = []
                        prepared = prepare_gemmini_graph(graph, report=report)
                        self.assertEqual(
                            sum(r["placement"] == "gemmini" for r in report), 0, report
                        )
                        device = next(
                            f for f in prepared.functions.values() if isinstance(f, tir.PrimFunc)
                        ).with_attr("global_symbol", "main")
                        runtime, control = self.build(tvm.IRModule({"main": device}))
                        input_value, actual = (
                            tvm.nd.array(values),
                            tvm.nd.empty(expected.shape, dtype),
                        )
                        runtime(input_value, actual)
                        np.testing.assert_array_equal(
                            actual.numpy().view("uint8"), expected.view("uint8")
                        )
                        np.testing.assert_array_equal(
                            input_value.numpy().view("uint8"), values.view("uint8")
                        )
                        self.assertEqual(control.emulator_commands(), 0)
                        self.assertEqual(control.emulator_faults(), 0)
                        self.assertEqual(control.emulator_pending(), 0)
                        legalized = relax.transform.LegalizeOps()(graph)
                        original = next(
                            f for f in legalized.functions.values() if isinstance(f, tir.PrimFunc)
                        )
                        saved = tvm.ir.save_json(original)
                        scheduled = make_elementwise_schedule(original, scalar=True)
                        replay = tir.Schedule(original, debug_mask="all")
                        scheduled.trace.apply_to_schedule(replay, remove_postproc=False)
                        replayed = lower_tensorir_schedule(
                            original, replay, workload_metadata={"operation": "copy_" + dtype}
                        )
                        tvm.ir.assert_structural_equal(
                            replayed.scheduled_mod, scheduled.scheduled_mod
                        )
                        self.assertEqual(tvm.ir.save_json(original), saved)
                        if mode == "reflect":
                            self.assertTrue(
                                any(
                                    instruction.kind.name == "AnnotateBufferAccess"
                                    for instruction in scheduled.trace.insts
                                )
                            )

    def test_reflection_pad_metaschedule_and_registered_vm_pipeline(self):
        from tvm import meta_schedule as ms
        from tvm.relax.backend.contrib.gemmini_meta_schedule import gemmini_space_generator
        from tvm.relax.backend.contrib.gemmini_target import GemminiCapabilities

        source = relax.Var("source", relax.TensorStructInfo((3, 4), "float32"))
        builder = relax.BlockBuilder()
        with builder.function("main", [source]):
            with builder.dataflow():
                output = builder.emit_output(
                    relax.op.nn.pad(source, (1, 1, 2, 1), pad_mode="reflect")
                )
            builder.emit_func_output(output)
        graph = builder.get()
        bits = np.resize(np.array([0, 0x80000000, 0x7F812345, 0x7FC12345], "uint32"), (3, 4))
        values = bits.view("float32")
        expected = np.pad(bits, ((1, 1), (2, 1)), mode="reflect")
        legalized = relax.transform.LegalizeOps()(graph)
        func = next(f for f in legalized.functions.values() if isinstance(f, tir.PrimFunc))
        context = ms.TuneContext(
            mod=func.with_attr("global_symbol", "main"),
            target=GemminiCapabilities().target(),
            space_generator=gemmini_space_generator(),
            num_threads=1,
        )
        legal = 0
        for design in context.generate_design_space():
            design.enter_postproc()
            if not all(p.apply(design) for p in context.space_generator.postprocs):
                continue
            lowered = lower_tensorir_schedule(
                context.mod["main"], design, workload_metadata={"operation": "copy_f32"}
            )
            runtime, control = self.build(lowered.scheduled_mod)
            actual = tvm.nd.empty(expected.shape, "float32")
            runtime(tvm.nd.array(values), actual)
            np.testing.assert_array_equal(actual.numpy().view("uint32"), expected)
            self.assertEqual(control.emulator_faults(), 0)
            legal += 1
        self.assertGreaterEqual(legal, 1)
        executable = relax.build(graph, "llvm", pipeline="gemmini_build")
        path = self.root / "reflected-vm.so"
        executable.export_library(
            str(path), addons=[str(self.emulator)], cc="cc", options=["-std=c11"]
        )
        control = ctypes.CDLL(str(path))
        control.emulator_reset()
        vm = relax.VirtualMachine(tvm.runtime.load_module(str(path)), tvm.cpu())
        actual = vm["main"](tvm.nd.array(values)).numpy()
        np.testing.assert_array_equal(actual.view("uint32"), expected)
        self.assertEqual(control.emulator_commands(), 0)
        self.assertEqual(control.emulator_faults(), 0)
        self.assertEqual(control.emulator_pending(), 0)

    def test_int8_relu_all_values_shapes_indexed_layouts_and_trace(self):
        for shape in ((1,), (31,), (16, 16), (3, 17), (2, 3, 4)):
            with self.subTest(shape=shape):
                original = workload(shape, "relu_i8")
                scheduled = make_elementwise_schedule(original)
                runtime, control = self.build(scheduled.scheduled_mod)
                values = np.resize(np.arange(-128, 128, dtype="int16"), shape).astype("int8")
                inputs, output = tvm.nd.array(values), tvm.nd.empty(shape, "int8")
                runtime(inputs, output)
                np.testing.assert_array_equal(output.numpy(), np.maximum(values, 0))
                np.testing.assert_array_equal(inputs.numpy(), values)
                self.assertEqual(control.emulator_faults(), 0)
                self.assertEqual(control.emulator_pending(), 0)
                calls = []
                tir.stmt_functor.post_order_visit(
                    scheduled.scheduled_mod["main"].body,
                    lambda node: calls.append(node.args[0].value)
                    if isinstance(node, tir.Call) and node.op == tvm.ir.Op.get("tir.call_extern")
                    else None,
                )
                self.assertIn("tvm_gemmini_dma_widen", calls)
                self.assertIn("tvm_gemmini_dma_relu_out", calls)
                replay = tir.Schedule(original, debug_mask="all")
                scheduled.trace.apply_to_schedule(replay, remove_postproc=False)
                tvm.ir.assert_structural_equal(
                    scheduled.scheduled_mod,
                    lower_tensorir_schedule(
                        original, replay, workload_metadata={"operation": "relu_i8"}
                    ).scheduled_mod,
                )
                if shape == (16, 16):
                    reference_output = tvm.nd.empty(shape, "int8")
                    tvm.build(original, "llvm")(inputs, reference_output)
                    np.testing.assert_array_equal(output.numpy(), reference_output.numpy())
                    control.emulator_reset()
                    control.emulator_wrong(1)
                    runtime(inputs, output)
                    self.assertFalse(np.array_equal(output.numpy(), np.maximum(values, 0)))

        a = te.placeholder((5, 3), "int8")
        b = te.compute((3, 5), lambda i, j: te.max(tir.const(0, "int8"), a[j, i]))
        result = make_elementwise_schedule(te.create_prim_func([a, b]))
        runtime, control = self.build(result.scheduled_mod)
        values = np.arange(-7, 8, dtype="int8").reshape(5, 3)
        output = tvm.nd.empty((3, 5), "int8")
        runtime(tvm.nd.array(values), output)
        np.testing.assert_array_equal(output.numpy(), np.maximum(values.T, 0))
        self.assertEqual(control.emulator_faults(), 0)

    def test_float32_movement_preserves_nan_payloads_and_signed_zero_bits(self):
        patterns = np.array(
            [
                0,
                0x80000000,
                0x7F800000,
                0xFF800000,
                0x7FC12345,
                0x7F812345,
                0xFFA00001,
                1,
                0x007FFFFF,
                0x00800000,
                0x3F800000,
                0xBF800000,
            ],
            "uint32",
        )
        for shape in ((1,), (31,), (3, 4), (16, 16), (2, 3, 4)):
            with self.subTest(shape=shape):
                bits = np.resize(patterns, shape)
                values = bits.view("float32")
                original = workload(shape, "copy_f32")
                result = make_elementwise_schedule(original)
                runtime, control = self.build(result.scheduled_mod)
                source, output = tvm.nd.array(values), tvm.nd.empty(shape, "float32")
                runtime(source, output)
                np.testing.assert_array_equal(output.numpy().view("uint32"), bits)
                np.testing.assert_array_equal(source.numpy().view("uint32"), bits)
                self.assertEqual(control.emulator_faults(), 0)
                self.assertEqual(control.emulator_pending(), 0)
        scalar = te.placeholder((), "float32", "scalar")
        filled = te.compute((3, 4), lambda i, j: scalar[()])
        result = make_elementwise_schedule(te.create_prim_func([scalar, filled]))
        runtime, control = self.build(result.scheduled_mod)
        bits = np.array(0x7F812345, "uint32")
        output = tvm.nd.empty((3, 4), "float32")
        runtime(tvm.nd.array(bits.view("float32")), output)
        np.testing.assert_array_equal(output.numpy().view("uint32"), np.full((3, 4), bits))
        self.assertEqual(control.emulator_faults(), 0)

    def test_relu_readout_completes_before_cpu_consumer(self):
        a = te.placeholder((3, 4), "int8", "source")
        b = te.compute((3, 4), lambda i, j: te.max(a[i, j], tir.const(0, "int8")), name="relu")
        c = te.compute((3, 4), lambda i, j: b[i, j].astype("int32") * 3 + 1, name="cpu")
        original = te.create_prim_func([a, b, c])
        result = make_elementwise_schedule(original)
        values = np.arange(-6, 6, dtype="int8").reshape(3, 4)
        expected = np.maximum(values.astype("int32"), 0) * 3 + 1
        for remove_wait in (False, True):
            module = result.scheduled_mod
            if remove_wait:

                def rewrite(node):
                    if (isinstance(node, tir.Call) and node.op == tvm.ir.Op.get("tir.call_extern")
                            and node.args[0].value == "tvm_gemmini_end"):
                        return tir.const(0, "int32")
                    if (
                        isinstance(node, tir.Call)
                        and node.op == tvm.ir.Op.get("tir.call_extern")
                        and node.args[0].value == "tvm_gemmini_dma_relu_out"
                    ):
                        return tir.call_extern(
                            "void", node.args[0].value, *node.args[1:-1], tir.const(0, "int32")
                        )
                    return None

                func = module["main"]
                module = tvm.IRModule(
                    {
                        "main": func.with_body(
                            tir.stmt_functor.ir_transform(func.body, None, rewrite)
                        )
                    }
                )
            runtime, control = self.build(module)
            intermediate = tvm.nd.array(np.full((3, 4), -77, "int8"))
            output = tvm.nd.empty((3, 4), "int32")
            runtime(tvm.nd.array(values), intermediate, output)
            if remove_wait:
                self.assertFalse(np.array_equal(output.numpy(), expected))
            else:
                np.testing.assert_array_equal(output.numpy(), expected)
                self.assertEqual(control.emulator_faults(), 0)

    def test_normal_relax_relu_and_float_movement_use_standard_metaschedule(self):
        from tvm import meta_schedule as ms
        from tvm.relax.backend.contrib.gemmini_meta_schedule import gemmini_space_generator
        from tvm.relax.backend.contrib.gemmini_target import GemminiCapabilities

        for dtype, operation in (("int8", "relu_i8"), ("float32", "copy_f32")):
            value = relax.Var("value", relax.TensorStructInfo((3, 4), dtype))
            builder = relax.BlockBuilder()
            with builder.function("main", [value]):
                with builder.dataflow():
                    output = builder.emit_output(
                        relax.op.nn.relu(value) if dtype == "int8" else relax.op.tile(value, (2, 1))
                    )
                builder.emit_func_output(output)
            report = []
            prepared = prepare_gemmini_graph(builder.get(), report=report)
            self.assertEqual([row["placement"] for row in report], ["gemmini" if dtype == "int8" else "cpu"], report)
            if dtype == "int8":
                self.assertEqual(report[0]["metadata"]["operation"], operation)
            path = self.root / f"relax-{dtype}.so"
            relax.build(prepared, "llvm").export_library(
                str(path), addons=[str(self.emulator)], cc="cc", options=["-std=c11"]
            )
            vm = relax.VirtualMachine(tvm.runtime.load_module(str(path)), tvm.cpu())
            values = np.arange(-6, 6, dtype=dtype).reshape(3, 4)
            np.testing.assert_array_equal(
                vm["main"](tvm.nd.array(values)).numpy(),
                np.maximum(values, 0) if dtype == "int8" else np.tile(values, (2, 1)),
            )
            func = workload((3, 4), operation)
            context = ms.TuneContext(
                mod=func,
                target=GemminiCapabilities().target(),
                space_generator=gemmini_space_generator(),
                num_threads=1,
            )
            designs = context.generate_design_space()
            self.assertGreaterEqual(len(designs), 2)
            for design in designs:
                design.enter_postproc()
                self.assertTrue(all(p.apply(design) for p in context.space_generator.postprocs))
                result = lower_tensorir_schedule(
                    func, design, workload_metadata={"operation": operation}
                )
                runtime, control = self.build(result.scheduled_mod)
                values = np.arange(-6, 6, dtype=dtype).reshape(3, 4)
                output = tvm.nd.empty((3, 4), dtype)
                runtime(tvm.nd.array(values), output)
                np.testing.assert_array_equal(
                    output.numpy(), np.maximum(values, 0) if dtype == "int8" else values
                )
                self.assertEqual(control.emulator_faults(), 0)

    def test_exact_bits_extremes_vectors_batches_and_prime_tails(self):
        bits = np.array([0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0x12345678], "uint32").view(
            "int32"
        )
        for shape in ((1,), (31,), (1, 7), (3, 4), (16, 16), (17, 19), (2, 3, 4)):
            for kind in ("copy_i8", "copy_i32", "widen_i8_i32"):
                with self.subTest(shape=shape, kind=kind):
                    original = workload(shape, kind)
                    result = make_elementwise_schedule(original)
                    runtime, control = self.build(result.scheduled_mod)
                    if kind in ("copy_i8", "widen_i8_i32"):
                        a = (np.arange(np.prod(shape)) % 256 - 128).astype("int8").reshape(shape)
                    else:
                        a = np.resize(bits, shape)
                    inputs = [tvm.nd.array(a)]
                    expected = a.astype("int32") if kind == "widen_i8_i32" else a
                    output = tvm.nd.empty(shape, str(expected.dtype))
                    runtime(*inputs, output)
                    np.testing.assert_array_equal(output.numpy(), expected)
                    np.testing.assert_array_equal(inputs[0].numpy(), a)
                    self.assertEqual(control.emulator_faults(), 0)
                    self.assertEqual(control.emulator_pending(), 0)
                    self.assertGreater(control.emulator_commands(), 0)
                    replay = tir.Schedule(original, debug_mask="all")
                    result.trace.apply_to_schedule(replay, remove_postproc=True)
                    scheduled, _, _ = schedule_elementwise(original)
                    tvm.ir.assert_structural_equal(replay.mod, scheduled.mod)

    def test_standard_metaschedule_postprocessing_and_mathematical_features(self):
        from tvm import meta_schedule as ms
        from tvm.relax.backend.contrib.gemmini_meta_schedule import gemmini_space_generator
        from tvm.relax.backend.contrib.gemmini_target import GemminiCapabilities

        func = workload((3, 4), "copy_i32")
        context = ms.TuneContext(
            mod=func,
            target=GemminiCapabilities().target(),
            space_generator=gemmini_space_generator(),
            num_threads=1,
        )
        designs = context.generate_design_space()
        self.assertGreaterEqual(len(designs), 2)
        features = ms.feature_extractor.PerStoreFeature().extract_from(
            context, [ms.search_strategy.MeasureCandidate(design, []) for design in designs]
        )
        self.assertTrue(
            all(feature.numpy().size and np.isfinite(feature.numpy()).all() for feature in features)
        )
        for design in designs:
            design.enter_postproc()
            self.assertTrue(all(p.apply(design) for p in context.space_generator.postprocs))
            result = lower_tensorir_schedule(
                func, design, workload_metadata={"operation": "copy_i32"}
            )
            runtime, control = self.build(result.scheduled_mod)
            a = np.full((3, 4), 2147483647, "int32")
            output = tvm.nd.empty((3, 4), "int32")
            runtime(tvm.nd.array(a), output)
            np.testing.assert_array_equal(output.numpy(), a)
            self.assertEqual(control.emulator_faults(), 0)

    def test_normal_relax_lowering_and_float_refusal(self):
        for dtype in ("int32", "float32"):
            a = relax.Var("a", relax.TensorStructInfo((3, 4), dtype))
            b = relax.Var("b", relax.TensorStructInfo((3, 4), dtype))
            builder = relax.BlockBuilder()
            with builder.function("main", [a, b]):
                with builder.dataflow():
                    result = builder.emit_output(relax.op.add(a, b))
                builder.emit_func_output(result)
            report = []
            prepare_gemmini_graph(builder.get(), report=report)
            self.assertEqual(
                sum(row["placement"] == "gemmini" for row in report), 0, report
            )

    def test_returned_device_buffer_is_ready_before_cpu_consumer(self):
        a = te.placeholder((3, 4), "int8", "a")
        b = te.placeholder((3, 4), "int8", "b")
        c = te.compute((3, 4), lambda i, j: a[i, j].astype("int32") + b[i, j].astype("int32"), name="device_result")
        d = te.compute((3, 4), lambda i, j: c[i, j] * 2, name="cpu_result")
        original = te.create_prim_func([a, b, c, d])
        scheduled = make_elementwise_schedule(original)
        av, bv = np.arange(12, dtype="int8").reshape(3, 4), np.full((3, 4), 7, "int8")
        runtime, control = self.build(scheduled.scheduled_mod)
        cv = tvm.nd.array(np.full((3, 4), -123, "int32"))
        dv = tvm.nd.empty((3, 4), "int32")
        runtime(tvm.nd.array(av), tvm.nd.array(bv), cv, dv)
        np.testing.assert_array_equal(cv.numpy(), av + bv)
        np.testing.assert_array_equal(dv.numpy(), (av + bv) * 2)
        self.assertEqual(control.emulator_faults(), 0)

        def remove_wait(node):
            if (isinstance(node, tir.Call) and node.op == tvm.ir.Op.get("tir.call_extern")
                    and node.args[0].value == "tvm_gemmini_end"):
                return tir.const(0, "int32")
            if (
                isinstance(node, tir.Call)
                and node.op == tvm.ir.Op.get("tir.call_extern")
                and node.args[0].value == "tvm_gemmini_dma_out"
            ):
                return tir.call_extern(
                    "void", node.args[0].value, *node.args[1:-1], tir.IntImm("int32", 0)
                )
            return None

        f = scheduled.scheduled_mod["main"]
        broken = f.with_body(tir.stmt_functor.ir_transform(f.body, None, remove_wait))
        wrong, _ = self.build(tvm.IRModule({"main": broken}))
        cv.copyfrom(np.full((3, 4), -123, "int32"))
        wrong(tvm.nd.array(av), tvm.nd.array(bv), cv, dv)
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(dv.numpy(), (av + bv) * 2)

    def test_indexed_movement_uses_framework_matching_and_preserves_inputs(self):
        for name in ("permute", "repeat", "diagonal", "slice"):
            a = te.placeholder((3, 4), "int8", "a")
            av = np.arange(12, dtype="int8").reshape(3, 4) - 7
            if name == "permute":
                c = te.compute((4, 3), lambda i, j: a[j, i], name="permutation")
                expected = av.T
            elif name == "repeat":
                c = te.compute((6, 8), lambda i, j: a[i % 3, j % 4], name="repeat")
                expected = np.tile(av, (2, 2))
            elif name == "diagonal":
                c = te.compute((3,), lambda i: a[i, i], name="diagonal")
                expected = np.diag(av)
            else:
                c = te.compute((3, 2), lambda i, j: a[i, j + 1], name="slice")
                expected = av[:, 1:3]
            original = te.create_prim_func([a, c])
            # Scalar DMA is an explicit capability experiment, never an
            # implicit production fallback after rectangular matching fails.
            result = make_elementwise_schedule(original, scalar=True)
            runtime, control = self.build(result.scheduled_mod)
            input_value = tvm.nd.array(av)
            output = tvm.nd.empty(expected.shape, "int8")
            runtime(input_value, output)
            np.testing.assert_array_equal(output.numpy(), expected)
            np.testing.assert_array_equal(input_value.numpy(), av)
            self.assertEqual(control.emulator_faults(), 0)
            self.assertEqual(control.emulator_pending(), 0)
            replay = tir.Schedule(original, debug_mask="all")
            result.trace.apply_to_schedule(replay, remove_postproc=True)
            if name == "permute":
                from tvm import meta_schedule as ms
                from tvm.relax.backend.contrib.gemmini_meta_schedule import gemmini_space_generator
                from tvm.relax.backend.contrib.gemmini_target import GemminiCapabilities

                context = ms.TuneContext(
                    mod=original,
                    target=GemminiCapabilities().target(),
                    space_generator=gemmini_space_generator(),
                    num_threads=1,
                )
                legal = 0
                for design in context.generate_design_space():
                    design.enter_postproc()
                    if all(p.apply(design) for p in context.space_generator.postprocs):
                        lowered = lower_tensorir_schedule(
                            original, design, workload_metadata={"operation": "copy_i8"}
                        )
                        selected, check = self.build(lowered.scheduled_mod)
                        selected(input_value, output)
                        np.testing.assert_array_equal(output.numpy(), expected)
                        self.assertEqual(check.emulator_faults(), 0)
                        legal += 1
                self.assertGreaterEqual(legal, 1)

    def test_scalar_fill_and_add_preserve_dtype_broadcast_and_wrap_contracts(self):
        for shape in ((1,), (7,), (3, 4), (17, 19), (2, 3, 4)):
            for source_dtype, output_dtype in (
                ("int8", "int8"),
                ("int32", "int32"),
                ("int8", "int32"),
            ):
                a = te.placeholder((), source_dtype, "scalar")
                c = te.compute(shape, lambda *i: a[()].astype(output_dtype), name="fill")
                func = te.create_prim_func([a, c])
                scheduled = make_elementwise_schedule(func)
                runtime, control = self.build(scheduled.scheduled_mod)
                scalar = np.array(-128 if source_dtype == "int8" else -2147483648, source_dtype)
                output = tvm.nd.empty(shape, output_dtype)
                runtime(tvm.nd.array(scalar), output)
                np.testing.assert_array_equal(
                    output.numpy(), np.full(shape, int(scalar), output_dtype)
                )
                self.assertEqual(control.emulator_faults(), 0)
                self.assertEqual(control.emulator_pending(), 0)
            for wrapping in (False, True):
                for scalar_left in (False, True):
                    x = te.placeholder(shape, "int32", "x")
                    a = te.placeholder((), "int32", "scalar")

                    def value(*indices):
                        left, right = (a[()], x[indices]) if scalar_left else (x[indices], a[()])
                        return (
                            (left.astype("int64") + right.astype("int64")).astype("int32")
                            if wrapping
                            else left + right
                        )

                    c = te.compute(shape, value, name="add_scalar")
                    # No byte-source bound: wrapping/full-domain i32 addition is
                    # outside the software spec's bounded-exact affine contract.
                    with self.assertRaises(ValueError):
                        make_elementwise_schedule(te.create_prim_func([x, a, c]))

    def test_registered_pipeline_preserves_graph_and_executes_dynamic_full(self):
        from tvm import meta_schedule as ms
        from tvm.relax.backend.contrib.gemmini_meta_schedule import gemmini_space_generator
        from tvm.relax.backend.contrib.gemmini_target import GemminiCapabilities

        scalar = relax.Var("scalar", relax.TensorStructInfo((), "int32"))
        builder = relax.BlockBuilder()
        with builder.function("main", [scalar]):
            with builder.dataflow():
                output = builder.emit_output(relax.op.full((3, 4), scalar, "int32"))
            builder.emit_func_output(output)
        graph = builder.get()
        report = []
        actual = relax.get_pipeline("gemmini", report=report)(graph)
        expected = prepare_gemmini_graph(graph)
        tvm.ir.assert_structural_equal(actual, expected)
        self.assertEqual(sum(row["placement"] == "gemmini" for row in report), 0, report)
        executable = relax.build(graph, "llvm", pipeline="gemmini_build")
        path = self.root / "registered-pipeline.so"
        executable.export_library(
            str(path), addons=[str(self.emulator)], cc="cc", options=["-std=c11"]
        )
        control = ctypes.CDLL(str(path))
        control.emulator_reset()
        vm = relax.VirtualMachine(tvm.runtime.load_module(str(path)), tvm.cpu())
        result = vm["main"](tvm.nd.array(np.array(-2147483648, "int32")))
        np.testing.assert_array_equal(result.numpy(), np.full((3, 4), -2147483648, "int32"))
        self.assertEqual(control.emulator_faults(), 0)
        legalized = relax.transform.LegalizeOps()(graph)
        func = next(
            f for f in legalized.functions.values() if isinstance(f, tir.PrimFunc)
        ).with_attr("global_symbol", "main")
        context = ms.TuneContext(
            mod=func,
            target=GemminiCapabilities().target(),
            space_generator=gemmini_space_generator(),
            num_threads=1,
        )
        for design in context.generate_design_space():
            design.enter_postproc()
            self.assertTrue(all(p.apply(design) for p in context.space_generator.postprocs))
            lowered = lower_tensorir_schedule(
                func, design, workload_metadata={"operation": "fill_i32"}
            )
            native, check = self.build(lowered.scheduled_mod)
            output = tvm.nd.empty((3, 4), "int32")
            native(tvm.nd.array(np.array(-2147483648, "int32")), output)
            np.testing.assert_array_equal(output.numpy(), np.full((3, 4), -2147483648, "int32"))
            self.assertEqual(check.emulator_faults(), 0)

    def test_hidden_primitive_scalar_abi_preserves_cpu_fallback(self):
        from tvm.relax.backend.contrib.gemmini_tensorir import ScheduleGemminiTensorIR

        original = workload((3, 4), "copy_i32")
        parameter = tir.Var("unused_scalar", "int32")
        function = tir.PrimFunc(
            [*original.params, parameter],
            original.body,
            buffer_map=original.buffer_map,
            attrs=original.attrs,
        )
        with self.assertRaisesRegex(ValueError, "buffer parameters"):
            make_elementwise_schedule(function)
        report = []
        result = ScheduleGemminiTensorIR(report=report, offload_copies=True)(tvm.IRModule({"main": function}))
        tvm.ir.assert_structural_equal(result["main"], function)
        self.assertEqual(report[0]["placement"], "cpu")
        self.assertIn("buffer parameters", report[0]["reason"])

    def test_default_vm_pipeline_accepts_prepared_single_input_functions(self):
        for kind in ("permute", "widen"):
            a = relax.Var("a", relax.TensorStructInfo((3, 4), "int8"))
            builder = relax.BlockBuilder()
            with builder.function("main", [a]):
                with builder.dataflow():
                    output = builder.emit_output(
                        relax.op.permute_dims(a, [1, 0])
                        if kind == "permute"
                        else relax.op.astype(a, "int32")
                    )
                builder.emit_func_output(output)
            report = []
            prepared = prepare_gemmini_graph(builder.get(), report=report)
            self.assertEqual(sum(row["placement"] == "gemmini" for row in report), 0, report)
            executable = relax.build(prepared, "llvm")
            path = self.root / (kind + "-default.so")
            executable.export_library(
                str(path), addons=[str(self.emulator)], cc="cc", options=["-std=c11"]
            )
            control = ctypes.CDLL(str(path))
            control.emulator_reset()
            vm = relax.VirtualMachine(tvm.runtime.load_module(str(path)), tvm.cpu())
            values = (np.arange(12).reshape(3, 4) - 8).astype("int8")
            actual = vm["main"](tvm.nd.array(values)).numpy()
            np.testing.assert_array_equal(
                actual, values.T if kind == "permute" else values.astype("int32")
            )
            self.assertEqual(control.emulator_commands(), 0)
            self.assertEqual(control.emulator_faults(), 0)


if __name__ == "__main__":
    unittest.main()
