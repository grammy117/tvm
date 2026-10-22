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

"""Host reference tests of the external-call boundary; no Gemmini execution."""

import ctypes
from pathlib import Path
import tempfile
import unittest

import numpy as np
import tvm
from tvm import relax, tir
from tvm.relax.backend.contrib.gemmini import LowerGemminiMatmul
from tvm.script import relax as R

REFERENCE_C = r"""
#include "matmul.h"
static int calls = 0;
static int failure = 0;
int host_reference_calls(void) { return calls; }
void host_reference_fail(int value) { failure = value; }
int32_t tvm_gemmini_matmul_i8_i32(const int8_t* a, const int8_t* b, int32_t* c, int64_t m, int64_t n, int64_t k, int64_t as, int64_t bs, int64_t cs) {
    calls++;
    if (failure) return failure;
    if (m <= 0 || n <= 0 || k <= 0 || k > 131071 || as < k || bs < n || cs < n) return -1;
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            int64_t sum = 0;
            for (int64_t x = 0; x < k; x++) sum += (int64_t)a[i * as + x] * b[x * bs + j];
            c[i * cs + j] = (int32_t)sum;
        }
    }
    return 0;
}
"""


def graph(left_shape=(3, 5), right_shape=(5, 7), dtype="int8", out_dtype="int32", twice=False, weight=None):
    a = relax.Var("a", relax.TensorStructInfo(left_shape, dtype))
    b = relax.Var("b", relax.TensorStructInfo(right_shape, dtype)) if weight is None else relax.const(weight)
    builder = relax.BlockBuilder()
    with builder.function("main", [a, b] if weight is None else [a]):
        with builder.dataflow():
            product = builder.emit(relax.op.matmul(a, b, out_dtype=out_dtype))
            value = builder.emit(relax.op.matmul(a, b, out_dtype=out_dtype)) if twice else product
            result = builder.emit_output(relax.op.add(product, value) if twice else product)
        builder.emit_func_output(result)
    return builder.get()


def calls(mod, op):
    found = []
    for func in mod.functions.values():
        if isinstance(func, relax.Function):
            relax.analysis.post_order_visit(func.body, lambda node: found.append(node) if isinstance(node, relax.Call) and isinstance(node.op, tvm.ir.Op) and node.op.name == op else None)
    return found


class ExternalMatmulTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gemmini-external-matmul-", ignore_cleanup_errors=True)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reference = self.root / "host_reference.c"
        self.reference.write_text(REFERENCE_C)
        self.header = Path(tvm.__file__).resolve().parents[2] / "apps/gemmini"
        self.build_index = 0

    def build(self, mod, mode="bytecode"):
        lowered = LowerGemminiMatmul()(mod)
        self.assertTrue(relax.analysis.well_formed(lowered))
        executable = relax.build(lowered, "llvm", exec_mode=mode)
        self.build_index += 1
        path = self.root / (mode + str(self.build_index) + ".so")
        executable.export_library(str(path), addons=[str(self.reference)], cc="cc", options=["-std=c11", "-I" + str(self.header)])
        counter = ctypes.CDLL(str(path))
        counter.host_reference_calls.restype = ctypes.c_int
        counter.host_reference_fail.argtypes = [ctypes.c_int]
        loaded = tvm.runtime.load_module(str(path))
        return relax.VirtualMachine(loaded, tvm.cpu()), counter, lowered

    def test_rectangles_boundaries_repeated_execution_and_constants(self):
        for mode in ("bytecode", "compiled"):
            for m, k, n in ((3, 5, 7), (17, 33, 19), (1, 131071, 1)):
                with self.subTest(mode=mode, shape=(m, k, n)):
                    a = np.full((m, k), -128, dtype="int8") if k == 131071 else ((np.arange(m * k).reshape(m, k) % 256) - 128).astype("int8")
                    b = np.full((k, n), -128, dtype="int8") if k == 131071 else ((np.arange(k * n).reshape(k, n) * 7 % 256) - 128).astype("int8")
                    vm, counter, lowered = self.build(graph((m, k), (k, n), weight=b), mode)
                    self.assertEqual(len(calls(lowered, "relax.call_tir")), 1)
                    self.assertEqual(len(calls(lowered, "relax.matmul")), 0)
                    before = counter.host_reference_calls()
                    device_a = tvm.nd.array(a)
                    for value in (a, np.zeros_like(a), a):
                        device_a.copyfrom(value)
                        actual = vm["main"](device_a).numpy()
                        self.assertEqual(actual.dtype, np.dtype("int32"))
                        np.testing.assert_array_equal(actual, value.astype("int64") @ b.astype("int64"))
                        np.testing.assert_array_equal(device_a.numpy(), value)
                    self.assertEqual(counter.host_reference_calls() - before, 3)

    def test_duplicate_calls_share_wrapper_and_preserve_host_add(self):
        original = graph(twice=True).with_attr("test_attribute", "preserved")
        vm, counter, lowered = self.build(original)
        self.assertEqual(str(lowered.attrs["test_attribute"]), "preserved")
        self.assertEqual(sum(isinstance(func, tir.PrimFunc) for func in lowered.functions.values()), 1)
        self.assertEqual(len(calls(lowered, "relax.call_tir")), 2)
        self.assertEqual(len(calls(lowered, "relax.add")), 1)
        a = np.arange(15, dtype="int8").reshape(3, 5)
        b = np.arange(35, dtype="int8").reshape(5, 7)
        np.testing.assert_array_equal(vm["main"](tvm.nd.array(a), tvm.nd.array(b)).numpy(), 2 * (a.astype("int32") @ b.astype("int32")))
        self.assertEqual(counter.host_reference_calls(), 2)
        tvm.ir.assert_structural_equal(LowerGemminiMatmul()(lowered), lowered)
        self.assertEqual(len(calls(original, "relax.matmul")), 2)

    def test_unsupported_contracts_remain_relax(self):
        symbolic = tir.Var("m", "int64")
        cases = [
            graph(dtype="float32", out_dtype="float32"),
            graph(out_dtype="int8"),
            graph(out_dtype=None),
            graph((2, 3, 5), (2, 5, 7)),
            graph((symbolic, 5), (5, 7)),
            graph((0, 5), (5, 7)),
            graph((1, 131072), (131072, 1)),
            graph((1 << 62, 1), (1, 1)),
        ]
        for original in cases:
            with self.subTest(ir=str(original)):
                lowered = LowerGemminiMatmul()(original)
                tvm.ir.assert_structural_equal(lowered, original)
                self.assertEqual(len(calls(lowered, "relax.call_tir")), 0)
        vm, counter, _ = self.build(cases[0])
        a = np.arange(15, dtype="float32").reshape(3, 5)
        b = np.arange(35, dtype="float32").reshape(5, 7)
        np.testing.assert_allclose(vm["main"](tvm.nd.array(a), tvm.nd.array(b)).numpy(), a @ b)
        self.assertEqual(counter.host_reference_calls(), 0)

    def test_if_dataflow_scopes_and_function_attributes(self):
        @R.function
        def main(condition: R.Tensor((), "bool"), a: R.Tensor((3, 5), "int8"), b: R.Tensor((5, 7), "int8")) -> R.Tensor((3, 7), "int32"):
            R.func_attr({"test_function_attribute": "preserved"})
            if condition:
                with R.dataflow():
                    product = R.matmul(a, b, out_dtype="int32")
                    result = R.add(product, product)
                    R.output(result)
            else:
                with R.dataflow():
                    result = R.matmul(a, b, out_dtype="int32")
                    R.output(result)
            return result

        original = tvm.IRModule({"main": main})
        vm, counter, lowered = self.build(original)
        self.assertEqual(str(lowered["main"].attrs["test_function_attribute"]), "preserved")
        self.assertEqual(lowered["main"].is_pure, original["main"].is_pure)
        tvm.ir.assert_structural_equal(lowered["main"].ret_struct_info, original["main"].ret_struct_info)
        self.assertEqual(len(calls(lowered, "relax.call_tir")), 2)
        self.assertEqual(len(calls(lowered, "relax.add")), 1)
        a = np.arange(15, dtype="int8").reshape(3, 5)
        b = np.arange(35, dtype="int8").reshape(5, 7)
        expected = a.astype("int32") @ b.astype("int32")
        for condition in (True, False):
            actual = vm["main"](tvm.nd.array(np.array(condition)), tvm.nd.array(a), tvm.nd.array(b)).numpy()
            np.testing.assert_array_equal(actual, expected * (2 if condition else 1))
        self.assertEqual(counter.host_reference_calls(), 2)

    def test_existing_wrapper_name_is_preserved(self):
        original = graph()
        existing = graph(dtype="float32", out_dtype="float32")["main"].without_attr("global_symbol")
        original["gemmini_matmul"] = existing
        lowered = LowerGemminiMatmul()(original)
        self.assertTrue(relax.analysis.well_formed(lowered))
        tvm.ir.assert_structural_equal(lowered["gemmini_matmul"], existing)
        rewritten = calls(lowered, "relax.call_tir")
        self.assertEqual(len(rewritten), 1)
        self.assertNotEqual(rewritten[0].args[0].name_hint, "gemmini_matmul")
        self.assertIsInstance(lowered[rewritten[0].args[0]], tir.PrimFunc)

    def test_non_cpu_placement_remains_relax(self):
        device = tvm.ir.VDevice(tvm.target.Target("cuda -arch=sm_50"))
        a = relax.Var("a", relax.TensorStructInfo((3, 5), "int8", vdevice=device))
        b = relax.Var("b", relax.TensorStructInfo((5, 7), "int8", vdevice=device))
        builder = relax.BlockBuilder()
        with builder.function("main", [a, b]):
            builder.emit_func_output(relax.op.matmul(a, b, out_dtype="int32"))
        original = builder.get()
        self.assertIsNotNone(original["main"].ret_struct_info.vdevice)
        tvm.ir.assert_structural_equal(LowerGemminiMatmul()(original), original)

    def test_existing_external_annotations_remain_unchanged(self):
        for attribute, value in (("Codegen", "existing_backend"), ("Composite", "existing_backend.matmul")):
            with self.subTest(attribute=attribute):
                original = graph()
                original["main"] = original["main"].with_attr(attribute, value)
                lowered = LowerGemminiMatmul()(original)
                tvm.ir.assert_structural_equal(lowered, original)
                self.assertEqual(len(calls(lowered, "relax.call_tir")), 0)

    def test_mixed_external_and_cpu_matmuls(self):
        a = relax.Var("a", relax.TensorStructInfo((3, 5), "int8"))
        b = relax.Var("b", relax.TensorStructInfo((5, 7), "int8"))
        x = relax.Var("x", relax.TensorStructInfo((3, 5), "float32"))
        y = relax.Var("y", relax.TensorStructInfo((5, 7), "float32"))
        builder = relax.BlockBuilder()
        with builder.function("main", [a, b, x, y]):
            with builder.dataflow():
                external = builder.emit(relax.op.matmul(a, b, out_dtype="int32"))
                fallback = builder.emit(relax.op.matmul(x, y))
                result = builder.emit_output(relax.Tuple([external, fallback]))
            builder.emit_func_output(result)
        vm, counter, lowered = self.build(builder.get())
        self.assertEqual(len(calls(lowered, "relax.call_tir")), 1)
        self.assertEqual(len(calls(lowered, "relax.matmul")), 1)
        a_values = np.arange(15, dtype="int8").reshape(3, 5)
        b_values = np.arange(35, dtype="int8").reshape(5, 7)
        x_values = np.arange(15, dtype="float32").reshape(3, 5) / 4
        y_values = np.arange(35, dtype="float32").reshape(5, 7) / 8
        external, fallback = vm["main"](*(tvm.nd.array(value) for value in (a_values, b_values, x_values, y_values)))
        np.testing.assert_array_equal(external.numpy(), a_values.astype("int32") @ b_values.astype("int32"))
        np.testing.assert_allclose(fallback.numpy(), x_values @ y_values)
        self.assertEqual(counter.host_reference_calls(), 1)

    def test_distinct_shapes_use_distinct_wrappers(self):
        a = relax.Var("a", relax.TensorStructInfo((3, 5), "int8"))
        b = relax.Var("b", relax.TensorStructInfo((5, 7), "int8"))
        x = relax.Var("x", relax.TensorStructInfo((2, 4), "int8"))
        y = relax.Var("y", relax.TensorStructInfo((4, 6), "int8"))
        builder = relax.BlockBuilder()
        with builder.function("main", [a, b, x, y]):
            with builder.dataflow():
                first = builder.emit(relax.op.matmul(a, b, out_dtype="int32"))
                second = builder.emit(relax.op.matmul(x, y, out_dtype="int32"))
                result = builder.emit_output(relax.Tuple([first, second]))
            builder.emit_func_output(result)
        vm, counter, lowered = self.build(builder.get())
        rewritten = calls(lowered, "relax.call_tir")
        self.assertEqual(len(rewritten), 2)
        self.assertFalse(rewritten[0].args[0].same_as(rewritten[1].args[0]))
        self.assertEqual(sum(isinstance(func, tir.PrimFunc) for func in lowered.functions.values()), 2)
        values = [np.arange(m * n, dtype="int8").reshape(m, n) for m, n in ((3, 5), (5, 7), (2, 4), (4, 6))]
        first, second = vm["main"](*(tvm.nd.array(value) for value in values))
        np.testing.assert_array_equal(first.numpy(), values[0].astype("int32") @ values[1].astype("int32"))
        np.testing.assert_array_equal(second.numpy(), values[2].astype("int32") @ values[3].astype("int32"))
        self.assertEqual(counter.host_reference_calls(), 2)

    def test_disabled_tir_assertions_preserve_external_call_and_failure(self):
        with tvm.transform.PassContext(config={"tir.disable_assert": True}):
            vm, counter, _ = self.build(graph())
        a = tvm.nd.array(np.zeros((3, 5), dtype="int8"))
        b = tvm.nd.array(np.zeros((5, 7), dtype="int8"))
        counter.host_reference_fail(7)
        with self.assertRaisesRegex(tvm.TVMError, "external matmul returned an error"):
            vm["main"](a, b)
        counter.host_reference_fail(0)
        np.testing.assert_array_equal(vm["main"](a, b).numpy(), np.zeros((3, 7), dtype="int32"))
        self.assertEqual(counter.host_reference_calls(), 2)

    def test_external_failure_is_propagated(self):
        vm, counter, _ = self.build(graph())
        counter.host_reference_fail(7)
        with self.assertRaisesRegex(tvm.TVMError, "external matmul returned an error"):
            vm["main"](tvm.nd.array(np.zeros((3, 5), dtype="int8")), tvm.nd.array(np.zeros((5, 7), dtype="int8")))
        counter.host_reference_fail(0)
        np.testing.assert_array_equal(vm["main"](tvm.nd.array(np.zeros((3, 5), dtype="int8")), tvm.nd.array(np.zeros((5, 7), dtype="int8"))).numpy(), np.zeros((3, 7), dtype="int32"))
        self.assertEqual(counter.host_reference_calls(), 2)


if __name__ == "__main__":
    unittest.main()
