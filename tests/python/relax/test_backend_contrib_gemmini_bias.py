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
"""Bias normalization proofs, standard scheduling and queued-DMA execution."""

import ctypes
from pathlib import Path
import tempfile
import unittest

import numpy as np
import tvm
from tvm import relax, te, tir
from tvm.relax.backend.contrib.gemmini import prepare_gemmini_graph
from tvm.relax.backend.contrib.gemmini_bias import fold_bias
from tvm.relax.backend.contrib.gemmini_tensorir import (
    contraction_block,
    lower_tensorir_schedule,
    make_tensorir_schedule,
)

from test_backend_contrib_gemmini_tensorir import EMULATOR


def biased_workload(m, n, k, bias_shape, *, batches=None, wrapping=False, shared=False):
    output_shape = (m, n) if batches is None else (batches, m, n)
    a_shape = (m, k) if batches is None else (batches, m, k)
    b_shape = (k, n) if batches is None else (batches, k, n)
    a = te.placeholder(a_shape, "int8", "left")
    b = te.placeholder(b_shape, "int8", "right")
    bias = te.placeholder(bias_shape, "int32", "bias")
    r = te.reduce_axis((0, k), "r")
    product = te.compute(
        output_shape,
        lambda *i: te.sum(
            a[(*i[:-2], i[-2], r)].astype("int32") * b[(*i[:-2], r, i[-1])].astype("int32"),
            axis=r,
        ),
        name="unbiased_product",
    )

    def add(*indices):
        bias_indices = tuple(
            0 if int(size) == 1 else index
            for size, index in zip(bias_shape, indices[len(indices) - len(bias_shape) :])
        )
        left, right = product[indices], bias[bias_indices]
        return (
            (left.astype("int64") + right.astype("int64")).astype("int32")
            if wrapping
            else left + right
        )

    result = te.compute(output_shape, add, name="biased_output")
    return te.create_prim_func([a, b, bias, *([product] if shared else []), result])


class GemminiBiasTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gemmini-bias-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.emulator = self.root / "emulator.c"
        self.emulator.write_text(EMULATOR)
        self.index = 0

    def build(self, mod):
        self.index += 1
        path = self.root / f"bias-{self.index}.so"
        tvm.build(mod, "llvm").export_library(
            str(path), addons=[str(self.emulator)], cc="cc", options=["-std=c11"]
        )
        control = ctypes.CDLL(str(path))
        control.emulator_reset()
        return tvm.runtime.load_module(str(path)), control

    def test_normalization_preserves_defined_results_despite_reordered_prefix_overflow(self):
        for wrapping in (False, True):
            original = biased_workload(3, 4, 32, (4,), wrapping=wrapping)
            normalized, proof = fold_bias(original)
            self.assertTrue(proof["applied"])
            self.assertEqual(
                proof["source_addition"],
                "explicit_modulo32" if wrapping else "defined_no_overflow",
            )
            self.assertIsInstance(contraction_block(normalized)[0][0].init.value, tir.BufferLoad)
            a = np.full((3, 32), 127, "int8")
            if not wrapping:
                a[:, 16:] = -127
                a[1] = -a[1]
            b = np.full((32, 4), 127, "int8")
            bias = np.array([2147483644, -2147483645, 17, -23], "int32")
            expected = (a.astype("int64") @ b.astype("int64") + bias).astype("int32")
            for func in (original, normalized):
                output = tvm.nd.empty((3, 4), "int32")
                tvm.build(func, "llvm")(
                    tvm.nd.array(a), tvm.nd.array(b), tvm.nd.array(bias), output
                )
                np.testing.assert_array_equal(output.numpy(), expected)

    def test_shared_or_returned_unbiased_product_is_not_rewritten(self):
        original = biased_workload(3, 4, 5, (4,), shared=True)
        unchanged, proof = fold_bias(original)
        self.assertTrue(unchanged.same_as(original))
        self.assertFalse(proof["applied"])
        self.assertIn("shared", proof["reason"])

    def test_small_bias_broadcast_cpu_alternative_exports_without_runtime_allocator(self):
        import sys
        sys.path.insert(0, str(Path(tvm.__file__).resolve().parents[2] / "apps/gemmini"))
        from baremetal import export_graph

        normalized, proof = fold_bias(biased_workload(4, 4, 8, (4,)))
        self.assertTrue(proof["applied"])
        buffers = [normalized.buffer_map[param] for param in normalized.params]
        builder = relax.BlockBuilder()
        gv = builder.add_func(normalized.with_attr("global_symbol", "product"), "product")
        inputs = [relax.Var(f"input_{i}", relax.TensorStructInfo(buf.shape, buf.dtype))
                  for i, buf in enumerate(buffers[:-1])]
        with builder.function("main", inputs):
            with builder.dataflow():
                result = builder.emit_output(relax.call_tir(gv, inputs,
                    out_sinfo=relax.TensorStructInfo(buffers[-1].shape, "int32")))
            builder.emit_func_output(result)
        folder = self.root / "cpu-static"
        report = export_graph(builder.get(), folder, target="llvm")
        self.assertFalse(report["runtime_workspace_allocator_required"])
        self.assertNotIn("TVMBackendAllocWorkspace", (folder / "operators.ll").read_text())

    def test_bias_schedules_replay_and_execute_with_delayed_dma(self):
        for m, n, k, bias_shape, batches in (
            (3, 4, 5, (4,), None),
            (1, 1, 1, (), None),
            (17, 19, 17, (17, 19), None),
            (3, 4, 32, (3, 1), None),
            (3, 4, 5, (2, 1, 4), 2),
        ):
            with self.subTest(shape=(m, n, k), bias=bias_shape, batches=batches):
                original = biased_workload(m, n, k, bias_shape, batches=batches)
                normalized, proof = fold_bias(original)
                self.assertTrue(proof["applied"])
                for order in ("ij", "ji"):
                    result = make_tensorir_schedule(normalized, order=order)
                    replay = tir.Schedule(normalized, debug_mask="all")
                    result.trace.apply_to_schedule(replay, remove_postproc=False)
                    tvm.ir.assert_structural_equal(
                        result.scheduled_mod,
                        lower_tensorir_schedule(normalized, replay).scheduled_mod,
                    )
                    self.assertTrue(result.metadata["bias_preloaded"])
                    calls = []
                    tir.stmt_functor.post_order_visit(
                        result.scheduled_mod["main"].body,
                        lambda node: calls.append(node.args[0].value)
                        if isinstance(node, tir.Call)
                        and node.op == tvm.ir.Op.get("tir.call_extern")
                        else None,
                    )
                    self.assertIn("tvm_gemmini_dma_acc_write", calls)
                    self.assertNotIn("tvm_gemmini_zero_acc", calls)
                    runtime, control = self.build(result.scheduled_mod)
                    leading = () if batches is None else (batches,)
                    a_shape, b_shape = (*leading, m, k), (*leading, k, n)
                    a = ((np.arange(np.prod(a_shape)).reshape(a_shape) * 13) % 256 - 128).astype(
                        "int8"
                    )
                    b = ((np.arange(np.prod(b_shape)).reshape(b_shape) * 7) % 256 - 128).astype(
                        "int8"
                    )
                    bias = (np.arange(np.prod(bias_shape)).reshape(bias_shape) * 17 - 23).astype(
                        "int32"
                    )
                    output = tvm.nd.empty((*leading, m, n), "int32")
                    runtime(tvm.nd.array(a), tvm.nd.array(b), tvm.nd.array(bias), output)
                    np.testing.assert_array_equal(
                        output.numpy(), a.astype("int64") @ b.astype("int64") + bias
                    )
                    self.assertEqual(control.emulator_faults(), 0)
                    self.assertEqual(control.emulator_pending(), 0)

    def test_bias_preload_handles_prefix_and_explicit_final_wrap(self):
        for wrapping in (False, True):
            normalized, proof = fold_bias(biased_workload(3, 4, 32, (4,), wrapping=wrapping))
            self.assertTrue(proof["applied"])
            result = make_tensorir_schedule(normalized)
            runtime, control = self.build(result.scheduled_mod)
            a = np.full((3, 32), 127, "int8")
            if not wrapping:
                a[:, 16:] = -127
                a[1] = -a[1]
            b = np.full((32, 4), 127, "int8")
            bias = np.array([2147483644, -2147483645, 17, -23], "int32")
            output = tvm.nd.empty((3, 4), "int32")
            runtime(tvm.nd.array(a), tvm.nd.array(b), tvm.nd.array(bias), output)
            np.testing.assert_array_equal(
                output.numpy(), (a.astype("int64") @ b.astype("int64") + bias).astype("int32")
            )
            self.assertEqual(control.emulator_faults(), 0)

    def test_relax_fusion_and_resource_refusal_restore_original_cpu_mathematics(self):
        from tvm.relax.backend.contrib.gemmini_target import GemminiCapabilities

        a = relax.Var("a", relax.TensorStructInfo((3, 5), "int8"))
        b = relax.Var("b", relax.TensorStructInfo((5, 4), "int8"))
        bias = relax.Var("bias", relax.TensorStructInfo((4,), "int32"))
        builder = relax.BlockBuilder()
        with builder.function("main", [a, b, bias]):
            with builder.dataflow():
                product = builder.emit(relax.op.matmul(a, b, out_dtype="int32"))
                output = builder.emit_output(relax.op.add(product, bias))
            builder.emit_func_output(output)
        for native in (True, False):
            capabilities = (
                GemminiCapabilities() if native else GemminiCapabilities(spad_rows=1, acc_rows=1)
            )
            report = []
            prepared = prepare_gemmini_graph(
                builder.get(), capabilities=capabilities, report=report
            )
            placement = "gemmini" if native else "cpu"
            self.assertEqual([row["placement"] for row in report], [placement], report)
            if native:
                self.assertTrue(report[0]["metadata"]["bias_preloaded"])
                self.assertTrue(report[0]["mathematical_normalization"]["applied"])
            else:
                self.assertIn("SRAM capacity", report[0]["reason"])
                self.assertTrue(
                    all(
                        not func.attrs or "gemmini.bias_preload" not in func.attrs
                        for func in prepared.functions.values()
                        if isinstance(func, tir.PrimFunc)
                    )
                )
            self.index += 1
            path = self.root / f"graph-{self.index}.so"
            relax.build(prepared, "llvm").export_library(
                str(path), addons=[str(self.emulator)], cc="cc", options=["-std=c11"]
            )
            vm = relax.VirtualMachine(tvm.runtime.load_module(str(path)), tvm.cpu())
            av = np.arange(15, dtype="int8").reshape(3, 5) - 7
            bv = np.arange(20, dtype="int8").reshape(5, 4) - 9
            dv = np.array([-19, 23, 0, 1], "int32")
            np.testing.assert_array_equal(
                vm["main"](tvm.nd.array(av), tvm.nd.array(bv), tvm.nd.array(dv)).numpy(),
                av.astype("int64") @ bv.astype("int64") + dv,
            )

    def test_bias_metaschedule_groups_and_standard_postprocessing(self):
        from tvm import meta_schedule as ms
        from tvm.relax.backend.contrib.gemmini_meta_schedule import gemmini_space_generator
        from tvm.relax.backend.contrib.gemmini_target import GemminiCapabilities

        normalized, proof = fold_bias(biased_workload(16, 16, 32, (16,)))
        self.assertTrue(proof["applied"])
        context = ms.TuneContext(
            mod=normalized,
            target=GemminiCapabilities().target(),
            space_generator=gemmini_space_generator(),
            num_threads=1,
        )
        designs = context.generate_design_space()
        self.assertGreaterEqual(len(designs), 3)  # OS, WS, pipelines and stock CPU sketches.
        # Check both traversal orders and the largest A/B reuse groups through
        # the normal tensorization and memory legality postprocessors.
        for design in designs:
            design.enter_postproc()
            self.assertTrue(
                all(postproc.apply(design) for postproc in context.space_generator.postprocs)
            )
            result = lower_tensorir_schedule(normalized, design)
            runtime, control = self.build(result.scheduled_mod)
            a = np.arange(512, dtype="int64").reshape(16, 32).astype("int8")
            b = (np.arange(512).reshape(32, 16) * 13).astype("int8")
            bias = np.arange(16, dtype="int32") - 9
            output = tvm.nd.empty((16, 16), "int32")
            runtime(tvm.nd.array(a), tvm.nd.array(b), tvm.nd.array(bias), output)
            np.testing.assert_array_equal(
                output.numpy(), a.astype("int64") @ b.astype("int64") + bias
            )
            self.assertEqual(control.emulator_faults(), 0)


if __name__ == "__main__":
    unittest.main()
