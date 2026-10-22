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

"""Independent CPU mathematics and primitive-contract tests, not device execution."""

import ctypes
import itertools
from pathlib import Path
import tempfile
import unittest

import numpy as np
import tvm
from tvm import relax, tir
from tvm.relax.backend.contrib.gemmini import DecomposeGemminiLinear, LowerGemminiMatmul, LowerGemminiScheduledMatmul, prepare_gemmini_graph_legacy as prepare_gemmini_graph
from tvm.relax.backend.contrib.gemmini_schedule import make_gemmini_matmul, make_semantic_matmul, tensorize_gemmini_matmul

# This emulator interprets row-addressed primitive calls independently of TVM
# loop construction. It checks DMA rectangles, loaded bytes, initialized acc
# bytes, row alignment/capacity, PE20 bounds and final int32 bounds. It does not
# model queues, instruction encoding, translation or hardware timing/coherence.
EMULATOR_C = r"""
#include "matmul.h"
#include <limits.h>
#include <stddef.h>
#include <string.h>
static int8_t sp[16384][16];
static int32_t acc[1024][16];
static unsigned char loaded[16384][16], initialized[1024][16];
static const int8_t *abase, *bbase;
static int32_t *cbase;
static int64_t gm, gn, gk, astride, bstride, cstride;
static int active, admitted, faults, forced_status, wrong_mode, counts[7];
void emulator_reset(void) { memset(counts, 0, sizeof(counts)); faults = active = admitted = 0; forced_status = wrong_mode = 0; }
void emulator_fail(int status) { forced_status = status; }
void emulator_wrong(int mode) { wrong_mode = mode; }
int emulator_count(int kind) { return counts[kind]; }
int emulator_faults(void) { return faults; }
int emulator_active(void) { return active; }
static int check(int condition) { if (!condition) faults++; return condition; }
static int tile(uint32_t row, uint32_t rows, uint32_t cols, uint32_t limit) {
    return check(active && !(row % 16) && row + 16 <= limit && rows >= 1 && rows <= 16 && cols >= 1 && cols <= 16);
}
static int rectangle(const void *base, const void *ptr, int64_t height, int64_t width, int64_t stride, uint32_t rows, uint32_t cols, size_t bytes) {
    uintptr_t start = (uintptr_t)base, addr = (uintptr_t)ptr;
    if (!check(addr >= start && !((addr - start) % bytes))) return 0;
    uint64_t offset = (addr - start) / bytes;
    return check(offset / stride + rows <= (uint64_t)height && offset % stride + cols <= (uint64_t)width);
}
int32_t tvm_gemmini_validate_matmul_i8_i32(const int8_t* a, const int8_t* b, int32_t* c, int64_t m, int64_t n, int64_t k, int64_t as, int64_t bs, int64_t cs) {
    counts[0]++;
    if (forced_status) return forced_status;
    if (!a || !b || !c || m <= 0 || n <= 0 || k <= 0 || k > 131071 || as < k || bs < n || cs < n) return -1;
    abase = a; bbase = b; cbase = c; gm = m; gn = n; gk = k; astride = as; bstride = bs; cstride = cs;
    admitted = 1;
    return 0;
}
void tvm_gemmini_begin(int64_t as, int64_t bs, int64_t cs) {
    counts[1]++;
    check(admitted && !active && as == astride && bs == bstride && cs == cstride);
    active = 1; admitted = 0;
    memset(loaded, 0, sizeof(loaded)); memset(initialized, 0, sizeof(initialized));
}
void tvm_gemmini_load_a(const int8_t* src, uint32_t row, uint32_t rows, uint32_t cols) {
    counts[2]++;
    if (!tile(row, rows, cols, 16384) || !rectangle(abase, src, gm, gk, astride, rows, cols, 1)) return;
    memset(loaded[row], 0, 16 * 16);
    for (uint32_t i = 0; i < rows; i++) for (uint32_t j = 0; j < cols; j++) {
        sp[row + i][j] = src[i * astride + j]; loaded[row + i][j] = 1;
    }
}
void tvm_gemmini_load_b(const int8_t* src, uint32_t row, uint32_t rows, uint32_t cols) {
    counts[3]++;
    if (!tile(row, rows, cols, 16384) || !rectangle(bbase, src, gk, gn, bstride, rows, cols, 1)) return;
    memset(loaded[row], 0, 16 * 16);
    for (uint32_t i = 0; i < rows; i++) for (uint32_t j = 0; j < cols; j++) {
        sp[row + i][j] = src[i * bstride + j]; loaded[row + i][j] = 1;
    }
}
void tvm_gemmini_compute(uint32_t ar, uint32_t br, uint32_t cr, uint32_t m, uint32_t n, uint32_t k, int32_t update) {
    counts[4]++;
    if (!tile(ar, m, k, 16384) || !tile(br, k, n, 16384) || !tile(cr, m, n, 1024) || !check(ar != br && (update == 0 || update == 1))) return;
    for (uint32_t i = 0; i < m; i++) for (uint32_t j = 0; j < n; j++) {
        int64_t partial = 0;
        for (uint32_t r = 0; r < k; r++) {
            check(loaded[ar + i][r] && loaded[br + r][j]);
            int8_t bv = sp[br + r][j];
            if (wrong_mode == 2) bv ^= 1;
            partial += (int64_t)sp[ar + i][r] * bv;
        }
        check(partial >= -524288 && partial <= 524287);
        if (update) check(initialized[cr + i][j]);
        int64_t sum = partial + ((update && wrong_mode != 1) ? acc[cr + i][j] : 0);
        check(sum >= INT32_MIN && sum <= INT32_MAX);
        acc[cr + i][j] = (int32_t)sum; initialized[cr + i][j] = 1;
    }
}
void tvm_gemmini_store(int32_t* dst, uint32_t row, uint32_t rows, uint32_t cols) {
    counts[5]++;
    if (!tile(row, rows, cols, 1024) || !rectangle(cbase, dst, gm, gn, cstride, rows, cols, 4)) return;
    for (uint32_t i = 0; i < rows; i++) for (uint32_t j = 0; j < cols; j++) {
        check(initialized[row + i][j]); dst[i * cstride + j] = acc[row + i][j];
    }
}
void tvm_gemmini_end(void) { counts[6]++; check(active); active = 0; }
"""


def graph(m=17, n=19, k=33, dtype="int8", out_dtype="int32", twice=False):
    a = relax.Var("a", relax.TensorStructInfo((m, k), dtype))
    b = relax.Var("b", relax.TensorStructInfo((k, n), dtype))
    builder = relax.BlockBuilder()
    with builder.function("main", [a, b]):
        with builder.dataflow():
            product = builder.emit(relax.op.matmul(a, b, out_dtype=out_dtype))
            duplicate = builder.emit(relax.op.matmul(a, b, out_dtype=out_dtype)) if twice else product
            result = builder.emit_output(relax.op.add(product, duplicate))
        builder.emit_func_output(result)
    return builder.get()


def external_calls(func):
    result = []
    tir.stmt_functor.post_order_visit(func.body, lambda node: result.append(node) if isinstance(node, tir.Call) and node.op == tvm.ir.Op.get("tir.call_extern") else None)
    return result


def quantized_graph(a_value=None):
    """Frozen weights, integer CPU transforms on both sides, three tuple outputs."""
    m, n, k = 17, 19, 33
    a = relax.Var("a", relax.TensorStructInfo((m, k), "int8"))
    w = relax.Var("w", relax.TensorStructInfo((k, n), "int8"))
    weight = ((np.arange(k * n).reshape(k, n) * 7 % 256) - 128).astype("int8")
    bias = np.arange(n, dtype="int32") - 5
    builder = relax.BlockBuilder()
    with builder.function("main", [a, w]):
        with builder.dataflow():
            wide = builder.emit(relax.op.astype(a, "int32"))
            offset = builder.emit(relax.op.add(wide, relax.const(np.int32(5))))
            bounded = builder.emit(relax.op.clip(offset, -128, 127))
            prepared = builder.emit(relax.op.astype(bounded, "int8"))
            constant_weight = builder.emit(relax.op.add(w, relax.const(np.zeros_like(weight))))
            product = builder.emit(relax.op.matmul(prepared, constant_weight, out_dtype="int32"))
            biased = builder.emit(relax.op.add(product, relax.const(bias)))
            activated = builder.emit(relax.op.nn.relu(biased))
            clipped = builder.emit(relax.op.clip(activated, 0, 300))
            shifted = builder.emit(relax.op.right_shift(clipped, relax.const(np.int32(3))))
            quantized = builder.emit(relax.op.astype(shifted, "int8"))
            output = builder.emit_output(relax.Tuple([quantized, product, prepared]))
        builder.emit_func_output(output)
    bindings = {"w": weight}
    if a_value is not None:
        bindings["a"] = a_value
    return relax.transform.BindParams("main", bindings)(builder.get()), weight, bias


def quantized_oracle(a, weight, bias):
    prepared = np.clip(a.astype("int64") + 5, -128, 127).astype("int8")
    product = prepared.astype("int64") @ weight.astype("int64")
    quantized = (np.clip(np.maximum(product + bias.astype("int64"), 0), 0, 300) >> 3).astype("int8")
    return quantized, product.astype("int32"), prepared


class GemminiScheduleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gemmini-schedule-", ignore_cleanup_errors=True)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.emulator = self.root / "primitive_emulator.c"
        self.emulator.write_text(EMULATOR_C)
        self.header = Path(tvm.__file__).resolve().parents[2] / "apps/gemmini"
        self.index = 0

    def build(self, mod, disable_assert=False):
        with tvm.transform.PassContext(config={"tir.disable_assert": disable_assert}):
            built = tvm.build(mod, "llvm")
        self.index += 1
        path = self.root / f"primitive_{self.index}.so"
        built.export_library(str(path), addons=[str(self.emulator)], cc="cc", options=["-std=c11", "-I" + str(self.header)])
        control = ctypes.CDLL(str(path))
        for name in ("emulator_count", "emulator_fail", "emulator_wrong"):
            getattr(control, name).argtypes = [ctypes.c_int]
        control.emulator_reset()
        return tvm.runtime.load_module(str(path)), control, built.get_source("ll")

    def build_graph(self, mod):
        executable = relax.build(mod, "llvm")
        self.index += 1
        path = self.root / f"graph_{self.index}.so"
        executable.export_library(str(path), addons=[str(self.emulator)], cc="cc", options=["-std=c11", "-I" + str(self.header)])
        control = ctypes.CDLL(str(path))
        for name in ("emulator_count", "emulator_fail", "emulator_wrong"):
            getattr(control, name).argtypes = [ctypes.c_int]
        control.emulator_reset()
        return relax.VirtualMachine(tvm.runtime.load_module(str(path)), tvm.cpu()), control

    def check_run(self, scheduled, runtime, control, a, b):
        m, k = a.shape
        n = b.shape[1]
        a_device, b_device = tvm.nd.array(a), tvm.nd.array(b)
        c_device = tvm.nd.array(np.full((m, n), 0x123456, dtype="int32"))
        control.emulator_reset()
        runtime(a_device, b_device, c_device)
        np.testing.assert_array_equal(c_device.numpy(), a.astype("int64") @ b.astype("int64"))
        np.testing.assert_array_equal(a_device.numpy(), a)
        np.testing.assert_array_equal(b_device.numpy(), b)
        self.assertEqual(control.emulator_faults(), 0)
        self.assertEqual(control.emulator_active(), 0)
        expected = [1, 1] + [scheduled.metadata[key] for key in ("load_a", "load_b", "compute", "store")] + [1]
        self.assertEqual([control.emulator_count(i) for i in range(7)], expected)
        return c_device

    def test_all_macro_tiles_full_and_tail_math_and_primitives(self):
        for tile_i, tile_j in itertools.product((1, 2, 4), repeat=2):
            for m, n, k in ((32, 32, 32), (33, 19, 49)):
                with self.subTest(tiles=(tile_i, tile_j), shape=(m, n, k)):
                    semantic = make_semantic_matmul(m, n, k, tile_i, tile_j)
                    self.assertTrue(tir.analysis.verify_well_formed(semantic["main"]))
                    a = ((np.arange(m * k).reshape(m, k) % 256) - 128).astype("int8")
                    b = ((np.arange(k * n).reshape(k, n) * 7 % 256) - 128).astype("int8")
                    semantic_output = tvm.nd.array(np.full((m, n), -111, dtype="int32"))
                    tvm.build(semantic, "llvm")(tvm.nd.array(a), tvm.nd.array(b), semantic_output)
                    np.testing.assert_array_equal(semantic_output.numpy(), a.astype("int64") @ b.astype("int64"))
                    scheduled = tensorize_gemmini_matmul(semantic)
                    self.assertTrue(tir.analysis.verify_well_formed(scheduled.scheduled_mod["main"]))
                    runtime, control, llvm = self.build(scheduled.scheduled_mod)
                    retained = self.check_run(scheduled, runtime, control, a, b)
                    saved = retained.numpy()
                    self.check_run(scheduled, runtime, control, np.zeros_like(a), b)
                    self.check_run(scheduled, runtime, control, a, b)
                    np.testing.assert_array_equal(retained.numpy(), saved)
                    self.assertNotIn("tvm_gemmini_matmul_i8_i32", llvm)
                    self.assertFalse(any("%" + name in llvm for name in ("AL", "BL", "CL")))
                    self.assertNotIn("alloca", llvm)
                    calls = external_calls(scheduled.scheduled_mod["main"])
                    for call in calls:
                        symbol = call.args[0].value
                        if symbol == "tvm_gemmini_compute":
                            self.assertEqual([str(arg.dtype) for arg in call.args[1:]], ["uint32"] * 6 + ["int32"])
                        if symbol in ("tvm_gemmini_load_a", "tvm_gemmini_load_b", "tvm_gemmini_store"):
                            self.assertEqual([str(arg.dtype) for arg in call.args[2:]], ["uint32"] * 3)
                    self.assertTrue(any(inst.kind.name == "Tensorize" for inst in scheduled.trace.insts))
                    self.assertLess(scheduled.lowering_steps.index("LowerMatchBuffer"), scheduled.lowering_steps.index("default TVM lowering"))

    def test_small_shapes_k_tails_and_maximum_exact_reduction(self):
        for m, n, k in ((1, 1, 1), (3, 7, 5), (1, 17, 17), (17, 1, 16), (1, 1, 131071)):
            with self.subTest(shape=(m, n, k)):
                scheduled = make_gemmini_matmul(m, n, k, 2, 2)
                a, b = np.full((m, k), -128, "int8"), np.full((k, n), -128, "int8")
                semantic_output = tvm.nd.array(np.zeros((m, n), "int32"))
                tvm.build(scheduled.semantic_mod, "llvm")(tvm.nd.array(a), tvm.nd.array(b), semantic_output)
                np.testing.assert_array_equal(semantic_output.numpy(), a.astype("int64") @ b.astype("int64"))
                runtime, control, _ = self.build(scheduled.scheduled_mod)
                self.check_run(scheduled, runtime, control, a, b)

    def test_real_reuse_and_distinct_traces(self):
        counts, traces = [], []
        for tile in (1, 2, 4):
            scheduled = make_gemmini_matmul(64, 64, 33, tile, tile)
            runtime, control, _ = self.build(scheduled.scheduled_mod)
            self.check_run(scheduled, runtime, control, np.ones((64, 33), "int8"), np.ones((33, 64), "int8"))
            counts.append((control.emulator_count(2), control.emulator_count(3), control.emulator_count(4), control.emulator_count(5)))
            traces.append(str(scheduled.trace))
        self.assertEqual(counts, [(48, 48, 48, 16), (24, 24, 48, 16), (12, 12, 48, 16)])
        self.assertEqual(len(set(traces)), 3)

    def test_trace_replay_and_mismatched_mathematics_rejection(self):
        semantic = make_semantic_matmul(17, 19, 33, 2, 2)
        result = tensorize_gemmini_matmul(semantic)
        replay = tir.Schedule(semantic, debug_mask=1)
        result.trace.apply_to_schedule(replay, remove_postproc=True)
        replayed = tir.transform.LowerMatchBuffer()(replay.mod)["main"]
        inner = result.scheduled_mod["main"].body.body.else_case.seq[1]
        tvm.ir.assert_structural_equal(replayed.body, inner, map_free_vars=True)

        def change_product(node):
            if isinstance(node, tir.Mul) and isinstance(node.a, tir.Cast) and isinstance(node.b, tir.Cast) and node.dtype == "int32":
                return node + tir.IntImm("int32", 1)
            return None

        changed = semantic["main"].with_body(tir.stmt_functor.ir_transform(semantic["main"].body, None, change_product, ["tir.Mul"]))
        with self.assertRaises(tir.schedule.ScheduleError):
            tensorize_gemmini_matmul(tvm.IRModule({"main": changed}))

    def test_wrong_update_and_operand_controls_fail_oracle(self):
        scheduled = make_gemmini_matmul(17, 19, 33, 2, 2)
        runtime, control, _ = self.build(scheduled.scheduled_mod)
        a, b = np.ones((17, 33), "int8"), np.ones((33, 19), "int8")
        for mode in (1, 2):
            control.emulator_reset()
            control.emulator_wrong(mode)
            output = tvm.nd.array(np.zeros((17, 19), "int32"))
            runtime(tvm.nd.array(a), tvm.nd.array(b), output)
            self.assertFalse(np.array_equal(output.numpy(), a.astype("int64") @ b.astype("int64")))
            self.assertEqual(control.emulator_faults(), 0)
        self.check_run(scheduled, runtime, control, a, b)

    def test_admission_precedes_device_calls_with_assertions_disabled(self):
        scheduled = make_gemmini_matmul(3, 7, 5)
        runtime, control, _ = self.build(scheduled.scheduled_mod, disable_assert=True)
        a, b = tvm.nd.array(np.ones((3, 5), "int8")), tvm.nd.array(np.ones((5, 7), "int8"))
        output = tvm.nd.array(np.full((3, 7), -123, "int32"))
        control.emulator_fail(-5)
        with self.assertRaisesRegex(tvm.error.TVMError, "scheduled matmul rejected"):
            runtime(a, b, output)
        self.assertEqual([control.emulator_count(i) for i in range(7)], [1, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(output.numpy(), -123)
        self.check_run(scheduled, runtime, control, a.numpy(), b.numpy())

    def test_shape_stride_capacity_and_tile_rejections(self):
        invalid = [(0, 1, 1, 1, 1), (1, -1, 1, 1, 1), (1, 1, 131072, 1, 1), (1, 1, True, 1, 1),
                   (1, 1, 1, 8, 1), (1, 1, 1, 1, 64), (1, 1, 1, True, 1), (1, 1 << 30, 1, 1, 1),
                   (1 << 63, 1, 1, 1, 1)]
        for args in invalid:
            with self.subTest(args=args), self.assertRaises(ValueError):
                make_semantic_matmul(*args)
        scheduled = make_gemmini_matmul(1, 1, 1, 4, 4)
        self.assertEqual((scheduled.metadata["spad_rows"], scheduled.metadata["acc_rows"]), (128, 256))

    def test_relax_pass_reuse_fallback_attributes_and_reference_route(self):
        original = graph(twice=True).with_attr("test_attribute", "preserved")
        lowered = LowerGemminiScheduledMatmul(2, 2)(original)
        self.assertTrue(relax.analysis.well_formed(lowered))
        self.assertEqual(str(lowered.attrs["test_attribute"]), "preserved")
        primitives = [func for func in lowered.functions.values() if isinstance(func, tir.PrimFunc)]
        self.assertEqual(len(primitives), 1)
        self.assertIn("tvm_gemmini_compute", primitives[0].script())
        self.assertNotIn("tvm_gemmini_matmul_i8_i32", primitives[0].script())
        tvm.ir.assert_structural_equal(LowerGemminiScheduledMatmul(2, 2)(lowered), lowered)
        host_adds = []
        relax.analysis.post_order_visit(lowered["main"].body, lambda node: host_adds.append(node) if isinstance(node, relax.Call) and node.op == tvm.ir.Op.get("relax.add") else None)
        self.assertEqual(len(host_adds), 1)
        for mod in (graph(dtype="float32", out_dtype="float32"), graph(k=131072), graph(n=1 << 30)):
            tvm.ir.assert_structural_equal(LowerGemminiScheduledMatmul()(mod), mod)
        reference = LowerGemminiMatmul()(original)
        reference_primitives = [func for func in reference.functions.values() if isinstance(func, tir.PrimFunc)]
        self.assertEqual(len(reference_primitives), 1)
        self.assertIn("tvm_gemmini_matmul_i8_i32", reference_primitives[0].script())
        self.assertNotIn("tvm_gemmini_compute", reference_primitives[0].script())

    def test_linear_vectors_and_batches_preserve_shapes_and_exact_primitives(self):
        for shape in ((33,), (2, 17, 33), (2, 1, 17, 33)):
            a = ((np.arange(np.prod(shape)).reshape(shape) * 11 % 256) - 128).astype("int8")
            b = ((np.arange(33 * 19).reshape(33, 19) * 7 % 256) - 128).astype("int8")
            lhs = relax.Var("a", relax.TensorStructInfo(shape, "int8"))
            builder = relax.BlockBuilder()
            with builder.function("main", [lhs]):
                builder.emit_func_output(relax.op.matmul(lhs, relax.const(b), out_dtype="int32"))
            original = builder.get()
            for optimize in (False, True):
                with self.subTest(shape=shape, optimize=optimize):
                    choices = {(int(np.prod(shape[:-1])), 19, 33): (1, 4)} if optimize else None
                    lowered = prepare_gemmini_graph(original, 2, 2, optimize=optimize, tile_choices=choices)
                    self.assertTrue(relax.analysis.well_formed(lowered))
                    vm, control = self.build_graph(lowered)
                    device_a = tvm.nd.array(a)
                    first = vm["main"](device_a)
                    expected = a.astype("int64") @ b.astype("int64")
                    self.assertEqual(first.dtype, "int32")
                    np.testing.assert_array_equal(first.numpy(), expected)
                    saved = first.numpy()
                    np.testing.assert_array_equal(vm["main"](tvm.nd.array(np.zeros_like(a))).numpy(), np.zeros_like(expected))
                    np.testing.assert_array_equal(first.numpy(), saved)
                    np.testing.assert_array_equal(device_a.numpy(), a)
                    self.assertEqual(control.emulator_faults(), 0)
                    self.assertEqual(control.emulator_count(0), 2)

    def test_linear_admission_preserves_unsupported_calls(self):
        for left, right, dtype, output in (((2, 3, 33), (2, 33, 19), "int8", "int32"), ((2, 3, 33), (33, 19), "float32", "float32"), ((2, 3, 131072), (131072, 19), "int8", "int32"), ((2, 0, 33), (33, 19), "int8", "int32")):
            a = relax.Var("a", relax.TensorStructInfo(left, dtype))
            b = relax.Var("b", relax.TensorStructInfo(right, dtype))
            builder = relax.BlockBuilder()
            with builder.function("main", [a, b]):
                builder.emit_func_output(relax.op.matmul(a, b, out_dtype=output))
            mod = builder.get()
            tvm.ir.assert_structural_equal(DecomposeGemminiLinear()(mod), mod)

    def test_onnx_expand_known_shape_values_follow_broadcast_contract(self):
        import onnx
        from onnx.reference import ReferenceEvaluator
        from tvm.relax.frontend.onnx.onnx_frontend import Expand
        cases = (((3, 1), (2, 1, 6)), ((2, 3), (1,)), ((1,), ()),
                 ((0, 3), (1, 3)), ((1,), (0,)))
        for input_shape, target in cases:
            for dtype, kind in (("int8", onnx.TensorProto.INT8), ("float32", onnx.TensorProto.FLOAT)):
                with self.subTest(input_shape=input_shape, target=target, dtype=dtype):
                    value = np.arange(np.prod(input_shape), dtype=dtype).reshape(input_shape)
                    graph = onnx.helper.make_graph(
                        [onnx.helper.make_node("Expand", ["x", "target"], ["out"])], "expand",
                        [onnx.helper.make_tensor_value_info("x", kind, input_shape)],
                        [onnx.helper.make_tensor_value_info("out", kind, None)],
                        [onnx.numpy_helper.from_array(np.array(target, dtype="int64"), "target")])
                    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
                    expected = ReferenceEvaluator(model).run(None, {"x": value})[0]
                    x = relax.Var("x", relax.TensorStructInfo(input_shape, dtype))
                    builder = relax.BlockBuilder()
                    with builder.function("main", [x]):
                        with builder.dataflow():
                            # A shape-valued Var carries known values, not a
                            # TensorStructInfo.shape. No runtime tensor cast.
                            shape = builder.emit(relax.ShapeExpr(target))
                            output = builder.emit_output(Expand._impl_v13(builder, [x, shape], {}, [{}, {}]))
                        builder.emit_func_output(output)
                    vm = relax.VirtualMachine(relax.build(builder.get(), "llvm"), tvm.cpu())
                    actual = vm["main"](tvm.nd.array(value)).numpy()
                    self.assertEqual(actual.dtype, value.dtype)
                    np.testing.assert_array_equal(actual, expected)
        x = relax.Var("x", relax.TensorStructInfo((2, 3), "float32"))
        for target in ((2, 2), (-1, 3)):
            with self.assertRaises(ValueError):
                Expand._impl_v13(relax.BlockBuilder(), [x, relax.ShapeExpr(target)], {}, [{}, {}])
        with self.assertRaisesRegex(NotImplementedError, "dynamic shape values"):
            Expand._impl_v13(relax.BlockBuilder(), [x, relax.Var("s", relax.ShapeStructInfo(ndim=2))], {}, [{}, {}])

    def test_onnx_expand_full_shape_identity_unknown_extents_and_partial_refusal(self):
        import onnx
        from onnx.reference import ReferenceEvaluator
        from tvm.relax.frontend.onnx import from_onnx
        from tvm.relax.frontend.onnx.onnx_frontend import Expand, Shape
        x = relax.Var("x", relax.TensorStructInfo(dtype="float32", ndim=2))
        builder = relax.BlockBuilder()
        with builder.function("main", [x]):
            shape = builder.emit(relax.op.shape_of(x))
            output = Expand._impl_v13(builder, [x, shape], {}, [{}, {}])
            self.assertTrue(output.same_as(x))
            builder.emit_func_output(output)
        unknown_vm = relax.VirtualMachine(relax.build(builder.get(), "llvm"), tvm.cpu())
        with self.assertRaisesRegex(NotImplementedError, "Shape slicing requires known shape values"):
            Shape._impl_v15(relax.BlockBuilder(), [x], {"start": 0, "end": 1}, [{}, {}])
        nodes = [onnx.helper.make_node("Max", ["x"], ["same"]),
                 onnx.helper.make_node("Shape", ["same"], ["shape"]),
                 onnx.helper.make_node("Expand", ["x", "shape"], ["out"])]
        graph = onnx.helper.make_graph(nodes, "identity",
            [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [None, 3])],
            [onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, [None, 3])])
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 22)])
        original = model.SerializeToString()
        mod = from_onnx(model, keep_params_in_input=False)
        self.assertEqual(model.SerializeToString(), original)
        vm = relax.VirtualMachine(relax.build(mod, "llvm"), tvm.cpu())
        for dims in ((2, 3), (0, 3)):
            value = np.arange(np.prod(dims), dtype="float32").reshape(dims)
            expected = ReferenceEvaluator(model).run(None, {"x": value})[0]
            np.testing.assert_array_equal(vm["main"](tvm.nd.array(value)).numpy(), expected)
            np.testing.assert_array_equal(unknown_vm["main"](tvm.nd.array(value)).numpy(), expected)
        # Partial Shape must not be mistaken for the full shape_of identity.
        graph.node[1].attribute.extend([onnx.helper.make_attribute("start", 0),
                                      onnx.helper.make_attribute("end", 1)])
        partial = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 22)])
        with self.assertRaisesRegex(NotImplementedError, "provably broadcast-compatible"):
            from_onnx(partial, keep_params_in_input=False)
        known = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 22)])
        known.graph.input[0].CopyFrom(onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2, 1]))
        known.graph.output[0].CopyFrom(onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, [2, 2]))
        mod = from_onnx(known, keep_params_in_input=False)
        vm = relax.VirtualMachine(relax.build(mod, "llvm"), tvm.cpu())
        value = np.array([[3], [5]], dtype="float32")
        expected = ReferenceEvaluator(known).run(None, {"x": value})[0]
        self.assertEqual(expected.shape, (2, 2))
        np.testing.assert_array_equal(vm["main"](tvm.nd.array(value)).numpy(), expected)

    def test_onnx_gather_nd_rank_from_unknown_length_index_provenance(self):
        import onnx
        from onnx.reference import ReferenceEvaluator
        from tvm.relax.frontend.onnx.onnx_frontend import GatherND

        # Both index lengths are unknown: only Unsqueeze/Concat prove K=2.
        data = relax.Var("data", relax.TensorStructInfo((2, 3), "float32"))
        rows = relax.Var("rows", relax.TensorStructInfo(dtype="int64", ndim=1))
        cols = relax.Var("cols", relax.TensorStructInfo(dtype="int64", ndim=1))
        builder = relax.BlockBuilder()
        with builder.function("main", [data, rows, cols]):
            with builder.dataflow():
                left = builder.emit(relax.op.expand_dims(rows, -1))
                right = builder.emit(relax.op.expand_dims(cols, 1))
                indices = builder.emit(relax.op.concat([left, right], axis=-1))
                self.assertIsNone(indices.struct_info.shape)
                output = GatherND._impl_v13(builder, [data, indices], {}, [{}, {}])
                self.assertEqual(output.struct_info.ndim, 1)
                self.assertEqual(str(output.struct_info.dtype), "float32")
                output = builder.emit_output(relax.op.permute_dims(output, [0]))
            builder.emit_func_output(output)
        script = builder.get().script()
        self.assertIn("R.match_cast", script)
        self.assertIn("R.gather_nd", script)
        self.assertNotIn("R.astype", script)
        graph = onnx.helper.make_graph([
            onnx.helper.make_node("Unsqueeze", ["rows", "axes"], ["r"]),
            onnx.helper.make_node("Unsqueeze", ["cols", "axes"], ["c"]),
            onnx.helper.make_node("Concat", ["r", "c"], ["indices"], axis=-1),
            onnx.helper.make_node("GatherND", ["data", "indices"], ["gathered"]),
            onnx.helper.make_node("Transpose", ["gathered"], ["out"], perm=[0]),
        ], "rank", [
            onnx.helper.make_tensor_value_info("data", onnx.TensorProto.FLOAT, [2, 3]),
            onnx.helper.make_tensor_value_info("rows", onnx.TensorProto.INT64, [None]),
            onnx.helper.make_tensor_value_info("cols", onnx.TensorProto.INT64, [None]),
        ], [onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, [None])],
            [onnx.numpy_helper.from_array(np.array([-1], dtype="int64"), "axes")])
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
        values = np.arange(6, dtype="float32").reshape(2, 3)
        for row_values, col_values in (([0, 1, -1], [2, 0, -1]), ([], [])):
            row_values, col_values = np.array(row_values, dtype="int64"), np.array(col_values, dtype="int64")
            actual = ReferenceEvaluator(model).run(None, {"data": values, "rows": row_values, "cols": col_values})[0]
            np.testing.assert_array_equal(actual, values[row_values, col_values])
            self.assertEqual(actual.ndim, output.struct_info.ndim)

    def test_onnx_gather_nd_static_values_batch_domain_and_dynamic_refusal(self):
        import onnx
        from onnx.reference import ReferenceEvaluator
        from tvm.relax.frontend.onnx import from_onnx
        from tvm.relax.frontend.onnx.onnx_frontend import GatherND
        for shape, index_values, batch in (
            ((2, 3), [[0, 2], [1, 0]], 0),
            ((2, 3), [[0], [1]], 0),
            ((2, 3, 4), [[[0], [2]], [[1], [0]]], 1),
        ):
            with self.subTest(shape=shape, batch=batch):
                value = np.arange(np.prod(shape), dtype="float32").reshape(shape)
                indices = np.array(index_values, dtype="int64")
                graph = onnx.helper.make_graph(
                    [onnx.helper.make_node("GatherND", ["x", "indices"], ["out"], batch_dims=batch)],
                    "gather", [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, shape)],
                    [onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, None)],
                    [onnx.numpy_helper.from_array(indices, "indices")])
                model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
                source = model.SerializeToString()
                expected = ReferenceEvaluator(model).run(None, {"x": value})[0]
                mod = from_onnx(model, keep_params_in_input=False)
                self.assertEqual(source, model.SerializeToString())
                self.assertEqual(mod["main"].ret_struct_info.ndim, expected.ndim)
                vm = relax.VirtualMachine(relax.build(mod, "llvm"), tvm.cpu())
                np.testing.assert_array_equal(vm["main"](tvm.nd.array(value)).numpy(), expected)
        data = relax.Var("x", relax.TensorStructInfo((2, 3), "float32"))
        for index_shape, batch in (((2, 0), 0), ((2, 3), 0), ((2, 1), -1), ((2, 1), 2)):
            with self.assertRaises(ValueError):
                indices = relax.Var("i", relax.TensorStructInfo(index_shape, "int64"))
                GatherND._impl_v13(relax.BlockBuilder(), [data, indices], {"batch_dims": batch}, [{}, {}])
        with self.assertRaisesRegex(ValueError, "batch dimensions must match"):
            indices = relax.Var("i", relax.TensorStructInfo((3, 1, 1), "int64"))
            GatherND._impl_v13(relax.BlockBuilder(), [data, indices], {"batch_dims": 1}, [{}, {}])
        with self.assertRaisesRegex(NotImplementedError, "provable last indices dimension"):
            indices = relax.Var("i", relax.TensorStructInfo(dtype="int64", ndim=2))
            GatherND._impl_v13(relax.BlockBuilder(), [data, indices], {}, [{}, {}])
        with self.assertRaisesRegex(NotImplementedError, "known input ranks"):
            indices = relax.Var("i", relax.TensorStructInfo(dtype="int64"))
            GatherND._impl_v13(relax.BlockBuilder(), [data, indices], {}, [{}, {}])

    def test_onnx_reshape_closed_shape_calculation_matches_reference(self):
        import onnx
        from onnx.reference import ReferenceEvaluator
        from tvm.relax.frontend.onnx import from_onnx
        initializers = [
            onnx.numpy_helper.from_array(np.float32(1), "one"),
            onnx.numpy_helper.from_array(np.array([1, 6, 5], dtype="int64"), "grid_shape"),
            onnx.numpy_helper.from_array(np.array([0], dtype="int64"), "axis"),
            onnx.numpy_helper.from_array(np.array([1], dtype="int64"), "end"),
            onnx.numpy_helper.from_array(np.int64(0), "index"),
            onnx.numpy_helper.from_array(np.array([-1], dtype="int64"), "vector_shape"),
        ]
        nodes = [
            onnx.helper.make_node("Expand", ["one", "grid_shape"], ["ones"]),
            onnx.helper.make_node("Cast", ["ones"], ["mask"], to=onnx.TensorProto.BOOL),
            onnx.helper.make_node("Slice", ["mask", "axis", "end", "axis"], ["slice"]),
            onnx.helper.make_node("Squeeze", ["slice", "axis"], ["grid"]),
            onnx.helper.make_node("Gather", ["grid", "index"], ["rows"], axis=1),
            onnx.helper.make_node("Gather", ["grid", "index"], ["cols"], axis=0),
            onnx.helper.make_node("Cast", ["rows"], ["rows_i"], to=onnx.TensorProto.INT64),
            onnx.helper.make_node("Cast", ["cols"], ["cols_i"], to=onnx.TensorProto.INT64),
            onnx.helper.make_node("ReduceSum", ["rows_i"], ["height"], keepdims=0),
            onnx.helper.make_node("ReduceSum", ["cols_i"], ["width"], keepdims=0),
            onnx.helper.make_node("Cast", ["height"], ["height_i"], to=onnx.TensorProto.INT64),
            onnx.helper.make_node("Cast", ["width"], ["width_i"], to=onnx.TensorProto.INT64),
            onnx.helper.make_node("Mul", ["height_i", "width_i"], ["count"]),
            onnx.helper.make_node("Reshape", ["count", "vector_shape"], ["vector"]),
            onnx.helper.make_node("Concat", ["vector"], ["target"], axis=0),
            onnx.helper.make_node("Reshape", ["x", "target"], ["out"]),
        ]
        graph = onnx.helper.make_graph(nodes, "closed-shape",
            [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2, 15])],
            [onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, [30])], initializers)
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 22)])
        source = model.SerializeToString()
        value = np.arange(-15, 15, dtype="float32").reshape(2, 15)
        expected = ReferenceEvaluator(model).run(None, {"x": value})[0]
        mod = from_onnx(model, keep_params_in_input=False)
        self.assertEqual(model.SerializeToString(), source)
        self.assertEqual(tuple(int(dim) for dim in mod["main"].ret_struct_info.shape.values), (30,))
        vm = relax.VirtualMachine(relax.build(mod, "llvm"), tvm.cpu())
        actual = vm["main"](tvm.nd.array(value)).numpy()
        self.assertEqual(actual.dtype, value.dtype)
        np.testing.assert_array_equal(actual, expected)

    def test_onnx_reshape_shape_folding_refuses_unknown_inputs_and_large_closures(self):
        from unittest.mock import patch
        from tvm.relax.frontend.onnx.onnx_frontend import Reshape
        data = relax.Var("x", relax.TensorStructInfo((2, 3), "float32"))
        unknown = relax.Var("shape", relax.TensorStructInfo((1,), "int64"))
        with patch.object(relax.transform, "FoldConstant") as fold:
            with self.assertRaisesRegex(NotImplementedError, "bounded closed shape calculation"):
                Reshape._impl_v13(relax.BlockBuilder(), [data, unknown], {}, [{}, {}])
            fold.assert_not_called()
        builder = relax.BlockBuilder()
        with builder.function("main", [data]):
            with builder.dataflow():
                # The result is one int64, but producing its sum would first
                # materialize a 256 KiB tensor. Budget before invoking TVM.
                large = builder.emit(relax.op.broadcast_to(relax.const(np.int64(1)), [256, 128]))
                scalar = builder.emit(relax.op.sum(large))
                shape = builder.emit(relax.op.reshape(scalar, [1]))
                with patch.object(relax.transform, "FoldConstant") as fold:
                    with self.assertRaisesRegex(NotImplementedError, "bounded closed shape calculation"):
                        Reshape._impl_v13(builder, [data, shape], {}, [{}, {}])
                    fold.assert_not_called()
                output = builder.emit_output(data)
            builder.emit_func_output(output)

    def test_onnx_closed_nonzero_and_expand_prove_scatter_lengths(self):
        import onnx
        from onnx.reference import ReferenceEvaluator
        from tvm.relax.frontend.onnx import from_onnx
        initializers = [
            onnx.numpy_helper.from_array(np.float32(1), "one"),
            onnx.numpy_helper.from_array(np.float32(0), "zero"),
            onnx.numpy_helper.from_array(np.array([1, 2, 2], dtype="int64"), "grid_shape"),
            onnx.numpy_helper.from_array(np.array([4], dtype="int64"), "data_shape"),
            onnx.numpy_helper.from_array(np.array([0], dtype="int64"), "axis"),
            onnx.numpy_helper.from_array(np.array([-1], dtype="int64"), "vector_shape"),
        ]
        nodes = [
            onnx.helper.make_node("Expand", ["one", "grid_shape"], ["ones"]),
            onnx.helper.make_node("Cast", ["ones"], ["mask"], to=onnx.TensorProto.BOOL),
            onnx.helper.make_node("Squeeze", ["mask", "axis"], ["grid"]),
            onnx.helper.make_node("Reshape", ["grid", "vector_shape"], ["flat"]),
            onnx.helper.make_node("NonZero", ["flat"], ["coordinates"]),
            onnx.helper.make_node("Transpose", ["coordinates"], ["indices"], perm=[1, 0]),
            onnx.helper.make_node("Cast", ["flat"], ["mask_i"], to=onnx.TensorProto.INT64),
            onnx.helper.make_node("ReduceSum", ["mask_i"], ["target"], keepdims=1),
            onnx.helper.make_node("Expand", ["x", "target"], ["updates"]),
            onnx.helper.make_node("Expand", ["zero", "data_shape"], ["data"]),
            onnx.helper.make_node("ScatterND", ["data", "indices", "updates"], ["out"]),
        ]
        graph = onnx.helper.make_graph(nodes, "closed-indexing",
            [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [4])],
            [onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, [4])], initializers)
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 22)])
        source = model.SerializeToString()
        value = np.array([-2.5, 0, 1, 8.25], dtype="float32")
        expected = ReferenceEvaluator(model).run(None, {"x": value})[0]
        mod = from_onnx(model, keep_params_in_input=False)
        self.assertEqual(model.SerializeToString(), source)
        self.assertNotIn("R.nonzero", mod.script())
        self.assertNotIn("R.match_cast", mod.script())
        vm = relax.VirtualMachine(relax.build(mod, "llvm"), tvm.cpu())
        np.testing.assert_array_equal(vm["main"](tvm.nd.array(value)).numpy(), expected)

    def test_onnx_constant_nonzero_order_empty_nonfinite_and_scalar_semantics(self):
        import onnx
        from onnx.reference import ReferenceEvaluator
        from tvm.relax.frontend.onnx import from_onnx
        from tvm.relax.frontend.onnx.onnx_frontend import NonZero, constant_numpy
        cases = (
            np.array([[0, 2, -3], [4, 0, -1]], dtype="int32"),
            np.zeros((2, 3), dtype="float32"),
            np.empty((2, 0, 3), dtype="bool"),
            np.array([-0., 0., np.nan, np.inf, -np.inf, np.finfo("float32").tiny], dtype="float32"),
            np.array(0, dtype="float32"), np.array(3, dtype="float32"),
            np.array(np.nan, dtype="float32"),
        )
        for value in cases:
            with self.subTest(shape=value.shape, dtype=value.dtype):
                graph = onnx.helper.make_graph(
                    [onnx.helper.make_node("NonZero", ["x"], ["out"])], "constant-nonzero", [],
                    [onnx.helper.make_tensor_value_info("out", onnx.TensorProto.INT64, None)],
                    [onnx.numpy_helper.from_array(value, "x")])
                model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
                source = model.SerializeToString()
                if value.ndim:
                    expected = ReferenceEvaluator(model).run(None, {})[0]
                else:
                    # ONNX specifies (0, N) for scalars. The installed reference
                    # evaluator instead uses NumPy at-least-1D, returning (1, N).
                    expected = np.empty((0, 0 if value.item() == 0 else 1), dtype="int64")
                mod = from_onnx(model, keep_params_in_input=False)
                self.assertEqual(model.SerializeToString(), source)
                folded = NonZero._impl_v9(relax.BlockBuilder(), [relax.const(value)], {}, [{}, {}])
                self.assertIsInstance(folded, relax.Constant)
                self.assertEqual(str(folded.struct_info.dtype), "int64")
                actual = constant_numpy(folded)
                self.assertEqual(actual.shape, expected.shape)
                np.testing.assert_array_equal(actual, expected)
                if expected.size:
                    vm = relax.VirtualMachine(relax.build(mod, "llvm"), tvm.cpu())
                    np.testing.assert_array_equal(vm["main"]().numpy(), expected)
                # Empty coordinate payloads have correct importer semantics.
                # This TVM's VM JIT constant loader rejects some zero-size
                # constants; empty VM execution is separately unsupported.

    def test_onnx_nonzero_unknown_and_budget_exhausted_masks_keep_dynamic_path(self):
        from unittest.mock import patch
        from tvm.relax.frontend.onnx.onnx_frontend import NonZero
        unknown = relax.Var("mask", relax.TensorStructInfo((4,), "bool"))
        builder = relax.BlockBuilder()
        with builder.function("main", [unknown]):
            with builder.dataflow():
                with patch.object(np, "argwhere") as coordinates:
                    output = NonZero._impl_v9(builder, [unknown], {}, [{}, {}])
                    coordinates.assert_not_called()
                output = builder.emit_output(output)
            builder.emit_func_output(output)
        self.assertIn("R.nonzero", builder.get().script())
        # Even a small boolean input can produce more than 64 KiB of indices.
        builder = relax.BlockBuilder()
        with builder.function("main", []):
            with builder.dataflow():
                mask = relax.const(np.ones((8193,), dtype="bool"))
                with patch.object(np, "argwhere") as coordinates, patch.object(relax.transform, "FoldConstant") as fold:
                    output = NonZero._impl_v9(builder, [mask], {}, [{}, {}])
                    coordinates.assert_not_called()
                    fold.assert_not_called()
                output = builder.emit_output(output)
            builder.emit_func_output(output)
        self.assertIn("R.nonzero", builder.get().script())
        builder = relax.BlockBuilder()
        with builder.function("main", []):
            with builder.dataflow():
                large = builder.emit(relax.op.broadcast_to(relax.const(np.int64(1)), [256, 128]))
                scalar = builder.emit(relax.op.sum(large))
                boolean = builder.emit(relax.op.astype(scalar, "bool"))
                mask = builder.emit(relax.op.reshape(boolean, [1]))
                with patch.object(np, "argwhere") as coordinates, patch.object(relax.transform, "FoldConstant") as fold:
                    output = NonZero._impl_v9(builder, [mask], {}, [{}, {}])
                    coordinates.assert_not_called()
                    fold.assert_not_called()
                output = builder.emit_output(output)
            builder.emit_func_output(output)
        self.assertIn("R.nonzero", builder.get().script())

    def test_onnx_variadic_broadcast_constants_and_llvm_match_reference(self):
        import onnx
        from onnx.reference import ReferenceEvaluator
        from tvm.relax.frontend.onnx import from_onnx
        cases = [
            ("float32", [np.array(2.), np.array([[1., 4., -2.], [3., -1., 5.]]), np.array([[[0., 1., 2.]], [[4., -3., 1.]]])]),
            ("float16", [np.array([[2048., -2048.]]), np.array(1.), np.array([-2048., 2048.])]),
            ("int32", [np.array([[2**30, -2**30]]), np.array(2), np.array([[[1, -1]], [[3, -3]]])]),
        ]
        for dtype, values in cases:
            values = [value.astype(dtype) for value in values]
            kind = onnx.helper.np_dtype_to_tensor_dtype(np.dtype(dtype))
            shape = np.broadcast_shapes(*(value.shape for value in values))
            for operation in (("Min", "Max") if dtype == "int32" else ("Min", "Max", "Mean", "Sum")):
                # The installed Mean oracle uses += and cannot enlarge its
                # first operand. Independent ONNX Expand nodes implement the
                # specified multidirectional broadcast before that oracle.
                expanded = ["ea", "eb", "ec"]
                oracle_nodes = [onnx.helper.make_node("Expand", [name, "shape"], [out]) for name, out in zip(("a", "b", "c"), expanded)]
                oracle_nodes.append(onnx.helper.make_node(operation, expanded, ["out"]))
                infos = [onnx.helper.make_tensor_value_info(name, kind, value.shape) for name, value in zip(("a", "b", "c"), values)]
                outputs = [onnx.helper.make_tensor_value_info("out", kind, shape)]
                oracle = onnx.helper.make_model(onnx.helper.make_graph(oracle_nodes, "oracle", infos, outputs, [onnx.numpy_helper.from_array(np.array(shape, "int64"), "shape")]), opset_imports=[onnx.helper.make_opsetid("", 13)])
                feed = dict(zip(("a", "b", "c"), values))
                expected = ReferenceEvaluator(oracle).run(None, feed)[0]
                for constants in (False, True):
                    with self.subTest(dtype=dtype, operation=operation, constants=constants):
                        graph = onnx.helper.make_graph([onnx.helper.make_node(operation, list(feed), ["out"])], "variadic", [] if constants else infos, outputs, [onnx.numpy_helper.from_array(value, name) for name, value in feed.items()] if constants else [])
                        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
                        mod = from_onnx(model, keep_params_in_input=False)
                        vm = relax.VirtualMachine(relax.build(mod, "llvm"), tvm.cpu())
                        actual = vm["main"](*([] if constants else [tvm.nd.array(value) for value in values])).numpy()
                        self.assertEqual(actual.dtype, np.dtype(dtype))
                        if operation == "Mean" and not constants:
                            # Existing TIR constant division may become a
                            # reciprocal multiply. This local arithmetic check
                            # allows its observed final rounding, not a policy
                            # quality or compiler-session threshold change.
                            np.testing.assert_allclose(actual, expected, rtol=np.finfo(dtype).eps, atol=0)
                        else:
                            np.testing.assert_array_equal(actual, expected)

    def test_onnx_variadic_min_max_nonfinite_and_zero_bits(self):
        import onnx
        from onnx.reference import ReferenceEvaluator
        from tvm.relax.frontend.onnx import from_onnx
        values = [np.array([-0., 0., np.nan, 1., np.inf, -np.inf], "float32"), np.array([0., -0., 2., np.nan, -np.inf, np.inf], "float32"), np.array([-0., 0., 3., 4., 5., -5.], "float32")]
        for operation in ("Min", "Max"):
            infos = [onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [6]) for name in ("a", "b", "c")]
            outputs = [onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, [6])]
            graph = onnx.helper.make_graph([onnx.helper.make_node(operation, ["a", "b", "c"], ["out"])], "nonfinite", infos, outputs)
            model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
            expected = ReferenceEvaluator(model).run(None, dict(zip(("a", "b", "c"), values)))[0]
            for constants in (False, True):
                with self.subTest(operation=operation, constants=constants):
                    if constants:
                        model = onnx.helper.make_model(onnx.helper.make_graph(list(graph.node), "constants", [], outputs, [onnx.numpy_helper.from_array(value, name) for name, value in zip(("a", "b", "c"), values)]), opset_imports=[onnx.helper.make_opsetid("", 13)])
                    mod = from_onnx(model, keep_params_in_input=False)
                    vm = relax.VirtualMachine(relax.build(mod, "llvm"), tvm.cpu())
                    actual = vm["main"](*([] if constants else [tvm.nd.array(value) for value in values])).numpy()
                    np.testing.assert_array_equal(actual, expected)
                    np.testing.assert_array_equal(actual.view("uint32"), expected.view("uint32"))

    def test_onnx_variadic_version_type_shape_refusals_and_unary_identity(self):
        from tvm.relax.frontend.onnx.onnx_frontend import Min, Max, Mean, Sum
        builder = relax.BlockBuilder()
        x = relax.Var("x", relax.TensorStructInfo(dtype="float32", ndim=3))
        for converter in (Min, Max, Mean, Sum):
            for version in (1, 8, 13):
                self.assertIs(converter.get_converter(version)(builder, [x], {}, [{}, {}]), x)
            for version in (1, 6):
                with self.assertRaisesRegex(ValueError, "equal shapes"):
                    converter.get_converter(version)(builder, [relax.const(np.ones((2, 3), "float32")), relax.const(np.float32(1))], {}, [{}, {}])
            with self.assertRaises(NotImplementedError):
                converter.get_converter(6)(builder, [x, x], {}, [{}, {}])
            for constants in (False, True):
                values = [relax.const(np.ones(shape, "float32")) if constants else relax.Var(name, relax.TensorStructInfo(shape, "float32")) for name, shape in (("a", (2, 3)), ("b", (2, 2)))]
                with self.assertRaises((ValueError, tvm.error.TVMError)):
                    builder.normalize(converter.get_converter(13)(builder, values, {}, [{}, {}]))
            with self.assertRaisesRegex(ValueError, "identical dtype"):
                converter.get_converter(13)(builder, [relax.const(np.float32(1)), relax.const(np.float16(2))], {}, [{}, {}])
            integer = [relax.const(np.array([1, 3], "int32")), relax.const(np.int32(2))]
            if converter in (Min, Max):
                with self.assertRaises(ValueError):
                    converter.get_converter(8)(builder, integer, {}, [{}, {}])
                with self.assertRaises(ValueError):
                    converter.get_converter(8)(builder, integer[:1], {}, [{}, {}])
                result = converter.get_converter(12)(builder, integer, {}, [{}, {}])
                self.assertEqual(result.struct_info.dtype, "int32")
                np.testing.assert_array_equal(result.data.numpy(), [1, 2] if converter is Min else [2, 3])
            else:
                with self.assertRaises(ValueError):
                    converter.get_converter(13)(builder, integer, {}, [{}, {}])
                with self.assertRaises(ValueError):
                    converter.get_converter(13)(builder, integer[:1], {}, [{}, {}])

    def test_onnx_variadic_bf16_arithmetic_boundaries_and_symbolic_shapes(self):
        import ml_dtypes
        import onnx
        from onnx.reference import ReferenceEvaluator
        from tvm.relax.frontend.onnx.onnx_frontend import Min, Max, Mean, Sum, constant_numpy
        # 256 + 1 rounds to 256 in BF16; adding -256 next must give zero,
        # whereas a promoted whole reduction would give one.
        values = [np.array([256., -256.], dtype=ml_dtypes.bfloat16), np.array(1., dtype=ml_dtypes.bfloat16), np.array([-256., 256.], dtype=ml_dtypes.bfloat16)]
        constants = [relax.const(value, "bfloat16") for value in values]
        for converter in (Mean, Sum):
            with self.assertRaises(ValueError):
                converter.get_converter(8)(relax.BlockBuilder(), constants, {}, [{}, {}])
            names = ["a", "b", "c"]
            infos = [onnx.helper.make_tensor_value_info(name, onnx.TensorProto.BFLOAT16, [2]) for name in names]
            oracle = onnx.helper.make_model(onnx.helper.make_graph([onnx.helper.make_node(converter.__name__, names, ["out"])], "bf16", infos, [onnx.helper.make_tensor_value_info("out", onnx.TensorProto.BFLOAT16, [2])]), opset_imports=[onnx.helper.make_opsetid("", 13)])
            expected = ReferenceEvaluator(oracle).run(None, {name: np.broadcast_to(value, [2]).copy() for name, value in zip(names, values)})[0]
            result = converter.get_converter(13)(relax.BlockBuilder(), constants, {}, [{}, {}])
            self.assertEqual(result.struct_info.dtype, "bfloat16")
            np.testing.assert_array_equal(constant_numpy(result).view("uint16"), expected.view("uint16"))
            builder = relax.BlockBuilder()
            inputs = [relax.Var(name, relax.TensorStructInfo(value.shape, "bfloat16")) for name, value in zip(("a", "b", "c"), values)]
            with builder.function("main", inputs):
                builder.emit_func_output(converter.get_converter(13)(builder, inputs, {}, [{}, {}]))
            vm = relax.VirtualMachine(relax.build(builder.get(), "llvm"), tvm.cpu())
            actual = vm["main"](*(value.data for value in constants)).numpy()
            np.testing.assert_array_equal(actual, expected.view("uint16"))
        n = tir.Var("n", "int64")
        for converter in (Min, Max, Mean, Sum):
            builder = relax.BlockBuilder()
            x = relax.Var("x", relax.TensorStructInfo([n, 3], "float32"))
            with builder.function("main", [x]):
                builder.emit_func_output(converter.get_converter(13)(builder, [x, relax.const(np.float32(1))], {}, [{}, {}]))
            self.assertTrue(relax.analysis.well_formed(builder.get()))
            tvm.ir.assert_structural_equal(builder.get()["main"].ret_struct_info, x.struct_info)

    def test_onnx_closed_scalar_cast_range_shapes_match_reference_and_llvm(self):
        import onnx
        from onnx.reference import ReferenceEvaluator
        from tvm.relax.frontend.onnx import from_onnx
        from tvm.relax.frontend.onnx.onnx_frontend import _fold_bounded_closed_tensor
        initializers = [onnx.numpy_helper.from_array(value, name) for name, value in (
            ("shape", np.array([32], "int64")), ("one", np.array(True)),
            ("axis", np.array([0], "int64")), ("start", np.float32(0)),
            ("step", np.float32(1)))]
        nodes = [onnx.helper.make_node("Expand", ["one", "shape"], ["ones"]),
                 onnx.helper.make_node("Cast", ["ones"], ["integers"], to=onnx.TensorProto.INT64),
                 onnx.helper.make_node("ReduceSum", ["integers", "axis"], ["sum"], keepdims=0),
                 onnx.helper.make_node("Cast", ["sum"], ["end"], to=onnx.TensorProto.FLOAT),
                 onnx.helper.make_node("Range", ["start", "end", "step"], ["range"]),
                 onnx.helper.make_node("Greater", ["range", "compare"], ["greater"]),
                 onnx.helper.make_node("Shape", ["range"], ["range_shape"]),
                 onnx.helper.make_node("Expand", ["one", "range_shape"], ["mask"]),
                 onnx.helper.make_node("And", ["greater", "mask"], ["out"])]
        outputs = [onnx.helper.make_tensor_value_info("range", onnx.TensorProto.FLOAT, [32]),
                   onnx.helper.make_tensor_value_info("out", onnx.TensorProto.BOOL, [32])]
        compare = np.arange(32, dtype="float32") - 1
        model = onnx.helper.make_model(onnx.helper.make_graph(nodes, "closed_range", [onnx.helper.make_tensor_value_info("compare", onnx.TensorProto.FLOAT, [32])], outputs, initializers), opset_imports=[onnx.helper.make_opsetid("", 22)])
        expected = ReferenceEvaluator(model).run(None, {"compare": compare})
        mod = from_onnx(model, keep_params_in_input=False)
        vm = relax.VirtualMachine(relax.build(mod, "llvm"), tvm.cpu())
        actual = vm["main"](tvm.nd.array(compare))
        for value, reference in zip(actual, expected):
            self.assertEqual(value.numpy().dtype, reference.dtype)
            np.testing.assert_array_equal(value.numpy(), reference)
        # A real, lossy scalar conversion must execute before using a literal.
        builder = relax.BlockBuilder()
        with builder.function("main", []):
            with builder.dataflow():
                converted = builder.emit(relax.op.astype(relax.const(2**24 + 1, "int64"), "float32"))
                binding = builder.lookup_binding(converted)
                folded = _fold_bounded_closed_tensor(builder, converted)
                self.assertEqual(folded.struct_info.dtype, "float32")
                self.assertEqual(folded.data.numpy().item(), np.float32(2**24 + 1))
                self.assertTrue(builder.lookup_binding(converted).same_as(binding))
                output = builder.emit_output(converted)
            builder.emit_func_output(output)

    def test_onnx_range_exact_integral_progressions_and_empty_import_semantics(self):
        import onnx
        from onnx.reference import ReferenceEvaluator
        from tvm.relax.frontend.onnx.onnx_frontend import Range
        cases = (("float32", (0, 32, 1)), ("float64", (10, 4, -2)),
                 ("int16", (-3, 3, 2)), ("int32", (10, 4, -2)),
                 ("int64", (0, 8, 2)), ("float32", (0, -1, 1)))
        for dtype, scalars in cases:
            with self.subTest(dtype=dtype, scalars=scalars):
                values = [np.array(value, dtype=dtype) for value in scalars]
                kind = onnx.helper.np_dtype_to_tensor_dtype(np.dtype(dtype))
                names = ["start", "end", "step"]
                graph = onnx.helper.make_graph([onnx.helper.make_node("Range", names, ["out"])], "range", [onnx.helper.make_tensor_value_info(name, kind, []) for name in names], [onnx.helper.make_tensor_value_info("out", kind, None)])
                model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 11)])
                expected = ReferenceEvaluator(model).run(None, dict(zip(names, values)))[0]
                result = Range.get_converter(11)(relax.BlockBuilder(), [relax.const(value, dtype) for value in values], {}, [{}, {}])
                self.assertIsInstance(result, relax.Constant)
                self.assertEqual(result.struct_info.dtype, dtype)
                np.testing.assert_array_equal(result.data.numpy(), expected)
                # Existing empty-constant VM serialization is unsupported;
                # check its importer payload/shape independently above.
                if expected.size:
                    builder = relax.BlockBuilder()
                    with builder.function("main", []):
                        builder.emit_func_output(result)
                    vm = relax.VirtualMachine(relax.build(builder.get(), "llvm"), tvm.cpu())
                    actual = vm["main"]().numpy()
                    self.assertEqual(actual.dtype, expected.dtype)
                    np.testing.assert_array_equal(actual, expected)

    def test_onnx_range_unproved_endpoints_keep_runtime_and_invalid_inputs_refuse(self):
        from unittest.mock import patch
        from tvm.relax.frontend.onnx.onnx_frontend import Range
        for scalars in ((0, 4, 0), (0, np.inf, 1), (np.nan, 4, 1)):
            with self.assertRaises((ValueError, NotImplementedError)):
                Range.get_converter(11)(relax.BlockBuilder(), [relax.const(np.float32(value)) for value in scalars], {}, [{}, {}])
        for values in ([relax.const(np.array([0], "float32")), relax.const(np.float32(4)), relax.const(np.float32(1))],
                       [relax.const(np.float16(value)) for value in (0, 4, 1)],
                       [relax.const(np.float32(0), "float32"), relax.const(np.float64(4), "float64"), relax.const(np.float32(1), "float32")]):
            with self.assertRaises(ValueError):
                Range.get_converter(11)(relax.BlockBuilder(), values, {}, [{}, {}])
        unproved = (("float32", (0., 20000., 1.)), ("float32", (0., 4., 0.5)),
                    ("float32", (2**24 + 2., 2**24 + 4., 2.)), ("float32", (-0., 4., 1.)),
                    ("float64", (0, 2**24 + 1, 2**24 + 1)), ("int16", (-32768, 32767, 1)))
        for dtype, scalars in unproved:
            builder = relax.BlockBuilder()
            with builder.function("main", []):
                with patch.object(np, "arange") as materialize:
                    output = Range.get_converter(11)(builder, [relax.const(np.array(value, dtype=dtype), dtype) for value in scalars], {}, [{}, {}])
                    materialize.assert_not_called()
                self.assertIsInstance(output, relax.Var)
                self.assertEqual(builder.lookup_binding(output).op, tvm.ir.Op.get("relax.call_tir"))
                builder.emit_func_output(output)
        # Scalar endpoint is observation-dependent: no value specialization.
        builder = relax.BlockBuilder()
        end = relax.Var("end", relax.TensorStructInfo([], "float32"))
        with builder.function("main", [end]):
            output = Range.get_converter(11)(builder, [relax.const(np.float32(0)), end, relax.const(np.float32(1))], {}, [{}, {}])
            self.assertIsInstance(output, relax.Var)
            builder.emit_func_output(output)
        # Closed scalar still refuses if its producer exceeds the old budget.
        builder = relax.BlockBuilder()
        with builder.function("main", []):
            with builder.dataflow():
                large = builder.emit(relax.op.broadcast_to(relax.const(1, "int64"), [8193]))
                total = builder.emit(relax.op.sum(large))
                end = builder.emit(relax.op.astype(total, "float32"))
                binding = builder.lookup_binding(end)
                with patch.object(np, "arange") as materialize:
                    output = Range.get_converter(11)(builder, [relax.const(np.float32(0)), end, relax.const(np.float32(1))], {}, [{}, {}])
                    materialize.assert_not_called()
                self.assertIsInstance(output, relax.Var)
                self.assertTrue(builder.lookup_binding(end).same_as(binding))
                output = builder.emit_output(output)
            builder.emit_func_output(output)

    def test_onnx_range_literal_and_symbolic_primitive_scalar_compatibility(self):
        from tvm.relax.frontend.onnx.onnx_frontend import Range
        for dtype, scalar in (("int64", tir.IntImm), ("float32", tir.FloatImm)):
            result = Range.get_converter(11)(relax.BlockBuilder(), [relax.PrimValue(scalar(dtype, value)) for value in (0, 8, 2)], {}, [{}, {}])
            self.assertIsInstance(result, relax.Constant)
            self.assertEqual(result.struct_info.dtype, dtype)
            np.testing.assert_array_equal(result.data.numpy(), np.array([0, 2, 4, 6], dtype=dtype))
        with self.assertRaises(ValueError):
            Range.get_converter(11)(relax.BlockBuilder(), [relax.const(0, "int64"), relax.PrimValue(tir.IntImm("int32", 8)), relax.const(1, "int64")], {}, [{}, {}])
        n = tir.Var("n", "int64")
        for endpoint in (n, n + 1):
            builder = relax.BlockBuilder()
            with builder.function("main", []):
                output = Range.get_converter(11)(builder, [relax.const(0, "int64"), relax.PrimValue(endpoint), relax.const(1, "int64")], {}, [{}, {}])
                self.assertIsInstance(output, relax.Var)
                self.assertEqual(builder.lookup_binding(output).op, tvm.ir.Op.get("relax.call_tir"))
                self.assertEqual(output.struct_info.dtype, "int64")
                builder.emit_func_output(output)

    def test_onnx_round_ties_even_and_signed_zero(self):
        import onnx
        from tvm.relax.frontend.onnx import from_onnx
        for dtype, kind in (("float16", onnx.TensorProto.FLOAT16), ("float32", onnx.TensorProto.FLOAT), ("float64", onnx.TensorProto.DOUBLE)):
            ties = np.array([-3.5, -2.5, -1.5, -0.5, -0.25, -0., 0., 0.25, 0.5, 1.5, 2.5, 3.5], dtype=dtype)
            values = np.concatenate((ties, np.nextafter(ties, np.array(np.inf, dtype=dtype)), np.nextafter(ties, np.array(-np.inf, dtype=dtype)), np.array([np.inf, -np.inf, np.nan, np.finfo(dtype).max], dtype=dtype)))
            graph = onnx.helper.make_graph([onnx.helper.make_node("Round", ["input"], ["out"])], "round", [onnx.helper.make_tensor_value_info("input", kind, values.shape)], [onnx.helper.make_tensor_value_info("out", kind, values.shape)])
            model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)])
            mod = from_onnx(model, keep_params_in_input=False)
            vm = relax.VirtualMachine(relax.build(mod, "llvm"), tvm.cpu())
            actual = vm["main"](tvm.nd.array(values)).numpy()
            expected = np.rint(values)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(np.signbit(actual[expected == 0]), np.signbit(expected[expected == 0]))

    def test_onnx_integer_matmul_signed_zero_points_and_refusals(self):
        import onnx
        from tvm.relax.frontend.onnx import from_onnx
        for zero_points in (None, (0, 0), (1, 0)):
            for dtype in ("int8", "uint8"):
                shape, k, n = (2, 3, 17), 17, 19
                a = ((np.arange(np.prod(shape)).reshape(shape) * 11 % 256) - 128).astype(dtype)
                b = ((np.arange(k * n).reshape(k, n) * 7 % 256) - 128).astype(dtype)
                kind = onnx.TensorProto.INT8 if dtype == "int8" else onnx.TensorProto.UINT8
                initializers = [onnx.numpy_helper.from_array(b, "b")]
                names = ["a", "b"]
                if zero_points is not None:
                    for name, value in zip(("az", "bz"), zero_points):
                        initializers.append(onnx.numpy_helper.from_array(np.array(value, dtype=dtype), name))
                        names.append(name)
                node = onnx.helper.make_node("MatMulInteger", names, ["out"])
                graph = onnx.helper.make_graph([node], "linear", [onnx.helper.make_tensor_value_info("a", kind, shape)], [onnx.helper.make_tensor_value_info("out", onnx.TensorProto.INT32, (2, 3, n))], initializers)
                model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)])
                onnx.checker.check_model(model, full_check=True)
                if dtype == "uint8" or zero_points == (1, 0):
                    with self.assertRaises(NotImplementedError):
                        from_onnx(model, keep_params_in_input=False)
                    continue
                mod = from_onnx(model, keep_params_in_input=False)
                vm, control = self.build_graph(prepare_gemmini_graph(mod, 2, 2))
                np.testing.assert_array_equal(vm["main"](tvm.nd.array(a)).numpy(), a.astype("int64") @ b.astype("int64"))
                self.assertEqual(control.emulator_faults(), 0)
                self.assertEqual(control.emulator_count(0), 1)

    def test_graph_pipeline_frozen_weight_cpu_fusion_and_tuple_semantics(self):
        original, weight, bias = quantized_graph()
        original = original.with_attr("test_attribute", "preserved")
        plans = [prepare_gemmini_graph(original, 2, 2, optimize=value) for value in (False, True)]
        cpu_functions = []
        boundaries = []
        for mod in plans:
            self.assertTrue(relax.analysis.well_formed(mod))
            self.assertEqual(str(mod.attrs["test_attribute"]), "preserved")
            primitives = [func for func in mod.functions.values() if isinstance(func, tir.PrimFunc)]
            boundary = [func for func in primitives if any(call.args[0].value == "tvm_gemmini_compute" for call in external_calls(func))]
            self.assertEqual(len(boundary), 1)
            self.assertEqual(int(boundary[0].attrs["op_pattern"]), 8)
            boundaries.append(boundary[0])
            cpu_functions.append([func for func in primitives if func != boundary[0]])
        self.assertEqual(len(cpu_functions[1]), 2)
        self.assertGreater(len(cpu_functions[0]), len(cpu_functions[1]))
        cpu_module = tvm.IRModule({gv: func.with_attr("global_symbol", gv.name_hint) for gv, func in plans[1].functions_items() if isinstance(func, tir.PrimFunc) and func != boundaries[1]})
        cpu_llvm = tvm.build(cpu_module, "llvm").get_source("ll")
        for gv in cpu_module.get_global_vars():
            self.assertIn(str(gv.name_hint), cpu_llvm)
        self.assertNotIn("TVMBackendAllocWorkspace", cpu_llvm)
        self.assertNotIn("TVMBackendFreeWorkspace", cpu_llvm)
        tvm.ir.assert_structural_equal(boundaries[0], boundaries[1], map_free_vars=True)
        expected_body = make_gemmini_matmul(17, 19, 33, 2, 2).scheduled_mod["main"].body
        tvm.ir.assert_structural_equal(boundaries[1].body, expected_body, map_free_vars=True)
        calls = []
        relax.analysis.post_order_visit(plans[1]["main"].body, lambda node: calls.append(node) if isinstance(node, relax.Call) and node.op == tvm.ir.Op.get("relax.call_tir") else None)
        boundary_call = next(call for call in calls if plans[1][call.args[0]] == boundaries[1])
        self.assertIsInstance(boundary_call.args[1].fields[1], relax.Constant)
        np.testing.assert_array_equal(boundary_call.args[1].fields[1].data.numpy(), weight)

        for mod in plans:
            vm, control = self.build_graph(mod)
            retained = None
            for a in (((np.arange(17 * 33).reshape(17, 33) % 256) - 128).astype("int8"), np.full((17, 33), -128, "int8"), np.full((17, 33), 127, "int8")):
                device_a = tvm.nd.array(a)
                control.emulator_reset()
                outputs = vm["main"](device_a)
                for actual, expected in zip(outputs, quantized_oracle(a, weight, bias)):
                    self.assertEqual(actual.dtype, str(expected.dtype))
                    np.testing.assert_array_equal(actual.numpy(), expected)
                np.testing.assert_array_equal(device_a.numpy(), a)
                self.assertEqual([control.emulator_count(i) for i in range(7)], [1, 1, 6, 6, 12, 4, 1])
                self.assertEqual((control.emulator_faults(), control.emulator_active()), (0, 0))
                if retained is None:
                    retained = [(value, value.numpy()) for value in outputs]
                else:
                    for value, saved in retained:
                        np.testing.assert_array_equal(value.numpy(), saved)
            control.emulator_reset()
            control.emulator_fail(-5)
            with self.assertRaisesRegex(tvm.error.TVMError, "scheduled matmul rejected"):
                vm["main"](tvm.nd.array(np.zeros((17, 33), "int8")))
            self.assertEqual([control.emulator_count(i) for i in range(7)], [1, 0, 0, 0, 0, 0, 0])
            control.emulator_reset()
            recovery = np.zeros((17, 33), "int8")
            for actual, expected in zip(vm["main"](tvm.nd.array(recovery)), quantized_oracle(recovery, weight, bias)):
                np.testing.assert_array_equal(actual.numpy(), expected)

    def test_graph_pipeline_constant_only_folds_before_device_lowering(self):
        a = ((np.arange(17 * 33).reshape(17, 33) % 256) - 128).astype("int8")
        original, weight, bias = quantized_graph(a)
        optimized = prepare_gemmini_graph(original, 2, 2)
        self.assertTrue(relax.analysis.well_formed(optimized))
        self.assertEqual(len(optimized["main"].params), 0)
        self.assertFalse(any(isinstance(func, tir.PrimFunc) for func in optimized.functions.values()))
        vm, control = self.build_graph(optimized)
        for actual, expected in zip(vm["main"](), quantized_oracle(a, weight, bias)):
            np.testing.assert_array_equal(actual.numpy(), expected)
        self.assertEqual([control.emulator_count(i) for i in range(7)], [0] * 7)

    def test_graph_pipeline_rejects_device_input_before_constant_evaluation(self):
        semantic = graph()
        for device in (LowerGemminiScheduledMatmul()(semantic), LowerGemminiMatmul()(semantic)):
            with self.assertRaisesRegex(ValueError, "semantic graph before device lowering"):
                prepare_gemmini_graph(device)
        for kwargs in ({"tile_i": 8}, {"optimize": 1}):
            with self.assertRaises(ValueError):
                prepare_gemmini_graph(semantic, **kwargs)


if __name__ == "__main__":
    unittest.main()
