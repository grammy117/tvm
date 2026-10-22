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
"""Semantic convolution checks using an independent direct int64 oracle."""

import unittest

import numpy as np
import tvm
from tvm import relax, tir
from tvm.relax.backend.contrib.gemmini_conv import DecomposeGemminiConv2D, get_conv2d_contract


def graph(data_shape=(1, 3, 5, 7), weight_shape=(19, 3, 3, 3), dtype="int8", out_dtype="int32", vdevice=None, **attrs):
    data = relax.Var("data", relax.TensorStructInfo(data_shape, dtype, vdevice=vdevice))
    weight = relax.Var("weight", relax.TensorStructInfo(weight_shape, dtype, vdevice=vdevice))
    builder = relax.BlockBuilder()
    with builder.function("main", [data, weight]):
        with builder.dataflow():
            output = builder.emit_output(relax.op.nn.conv2d(data, weight, out_dtype=out_dtype, **attrs))
        builder.emit_func_output(output)
    return builder.get()


def calls(mod, op):
    result = []
    for func in mod.functions.values():
        if isinstance(func, relax.Function):
            relax.analysis.post_order_visit(func.body, lambda node: result.append(node) if isinstance(node, relax.Call) and isinstance(node.op, tvm.ir.Op) and node.op.name == op else None)
    return result


def direct_conv(data, weight, strides=(1, 1), padding=(0, 0, 0, 0), dilation=(1, 1)):
    """Traverse output pixels and original kernel coordinates; no matrix packing."""
    batch, channels, height, width = data.shape
    out_channels, _, kh, kw = weight.shape
    oh = (height + padding[0] + padding[2] - dilation[0] * (kh - 1) - 1) // strides[0] + 1
    ow = (width + padding[1] + padding[3] - dilation[1] * (kw - 1) - 1) // strides[1] + 1
    output = np.zeros((batch, out_channels, oh, ow), dtype="int64")
    for n in range(batch):
        for o in range(out_channels):
            for y in range(oh):
                for x in range(ow):
                    for c in range(channels):
                        for r in range(kh):
                            iy = y * strides[0] - padding[0] + r * dilation[0]
                            for s in range(kw):
                                ix = x * strides[1] - padding[1] + s * dilation[1]
                                if 0 <= iy < height and 0 <= ix < width:
                                    output[n, o, y, x] += np.int64(data[n, c, iy, ix]) * np.int64(weight[o, c, r, s])
    return output


def run_cpu(mod, *inputs):
    executable = relax.build(relax.transform.LegalizeOps()(mod), "llvm")
    vm = relax.VirtualMachine(executable, tvm.cpu())
    return vm["main"](*(tvm.nd.array(value) for value in inputs)).numpy()


class GemminiConvTests(unittest.TestCase):
    def test_direct_semantic_cases(self):
        cases = [
            ((1, 16, 4, 4), (16, 16, 1, 1), (1, 1), (0, 0, 0, 0), (1, 1)),
            ((2, 3, 6, 8), (19, 3, 2, 3), (2, 2), (1, 2, 0, 1), (2, 2)),
            ((1, 2, 8, 9), (3, 2, 7, 7), (1, 1), (3, 3, 3, 3), (1, 1)),
            ((1, 17, 3, 4), (17, 17, 3, 3), (1, 1), (1, 1, 1, 1), (1, 1)),
        ]
        rng = np.random.default_rng(10)
        for data_shape, weight_shape, strides, padding, dilation in cases:
            with self.subTest(data_shape=data_shape, weight_shape=weight_shape):
                data = rng.integers(-128, 128, size=data_shape, dtype="int8")
                weight = rng.integers(-128, 128, size=weight_shape, dtype="int8")
                mod = graph(data_shape, weight_shape, strides=strides, padding=padding, dilation=dilation)
                lowered = DecomposeGemminiConv2D()(mod)
                self.assertEqual(len(calls(lowered, "relax.nn.conv2d")), 0)
                self.assertEqual(len(calls(lowered, "relax.matmul")), 1)
                self.assertEqual(len(calls(lowered, "relax.call_tir")), 3)
                tvm.ir.assert_structural_equal(lowered, DecomposeGemminiConv2D()(lowered))
                expected = direct_conv(data, weight, strides, padding, dilation)
                np.testing.assert_array_equal(run_cpu(lowered, data, weight), expected.astype("int32"))

    def test_extreme_inputs_and_frozen_packing(self):
        data = np.full((2, 17, 4, 5), -128, dtype="int8")
        weight = np.full((19, 17, 3, 3), -128, dtype="int8")
        weight[::2] = 127
        mod = graph(data.shape, weight.shape, padding=(1, 2, 0, 1))
        frozen = relax.transform.BindParams("main", {"weight": weight})(mod)
        lowered = DecomposeGemminiConv2D()(frozen)
        folded = relax.transform.FoldConstant()(lowered)
        self.assertEqual(len(calls(folded, "relax.matmul")), 1)
        self.assertEqual(len(calls(folded, "relax.call_tir")), 2)
        expected = direct_conv(data, weight, padding=(1, 2, 0, 1))
        np.testing.assert_array_equal(run_cpu(folded, data), expected.astype("int32"))
        # Freeze all inputs: mathematical contraction may fold before scheduling.
        constant = relax.transform.BindParams("main", {"data": data})(frozen)
        constant = relax.transform.FoldConstant()(DecomposeGemminiConv2D()(constant))
        self.assertEqual(len(calls(constant, "relax.matmul")), 0)
        np.testing.assert_array_equal(run_cpu(constant), expected.astype("int32"))

    def test_contract_bytes_and_limits(self):
        call = calls(graph((2, 3, 6, 8), (19, 3, 2, 3), strides=(2, 2), padding=(1, 2, 0, 1), dilation=(2, 2)), "relax.nn.conv2d")[0]
        contract = get_conv2d_contract(call)
        self.assertEqual(contract.output_shape, (2, 19, 3, 4))
        self.assertEqual(contract.matmul_shape, (24, 19, 18))
        self.assertEqual(contract.tensor_bytes["im2col"], 24 * 18)
        self.assertEqual(contract.tensor_bytes["matmul_output"], 24 * 19 * 4)
        for k, admitted in ((131071, True), (131072, False)):
            mod = graph((1, k, 1, 1), (1, k, 1, 1))
            self.assertEqual(get_conv2d_contract(calls(mod, "relax.nn.conv2d")[0]) is not None, admitted)
        huge = graph((1, 1, 1 << 40, 1 << 24), (1, 1, 1, 1))
        self.assertIsNone(get_conv2d_contract(calls(huge, "relax.nn.conv2d")[0]))
        wide = graph((1, 1, 1, 1), (1 << 30, 1, 1, 1))
        self.assertIsNone(get_conv2d_contract(calls(wide, "relax.nn.conv2d")[0]))

    def test_unsupported_preserved(self):
        symbolic = tir.Var("batch", "int64")
        cases = [
            graph(dtype="float32", out_dtype="float32"),
            graph(out_dtype="int8"),
            graph((1, 4, 5, 7), (6, 2, 3, 3), groups=2),
            graph((1, 5, 7, 3), (19, 3, 3, 3), data_layout="NHWC"),
            graph((1, 3, 5, 7), (3, 3, 3, 19), kernel_layout="HWIO"),
            graph(out_layout="NHWC"),
            graph((symbolic, 3, 5, 7)),
            graph((1, 131072, 1, 1), (1, 131072, 1, 1)),
            graph((1, 3, 1, 1), (19, 3, 3, 3)),
            graph(vdevice=tvm.ir.VDevice(tvm.target.Target("cuda -arch=sm_50"))),
        ]
        for mod in cases:
            with self.subTest(ir=mod.script()):
                tvm.ir.assert_structural_equal(mod, DecomposeGemminiConv2D()(mod))
        for attr in ("Codegen", "Composite"):
            mod = graph()
            mod["main"] = mod["main"].with_attr(attr, "external")
            tvm.ir.assert_structural_equal(mod, DecomposeGemminiConv2D()(mod))

    def test_inferred_output_shape_required(self):
        original = calls(graph(), "relax.nn.conv2d")[0]
        for output in ((1, 19, 4, 5), (1, 19, 3, 4), (1, 19, 0, 5)):
            call = relax.Call(original.op, original.args, original.attrs)
            relax.expr._update_struct_info(call, relax.TensorStructInfo(output, "int32"))
            self.assertIsNone(get_conv2d_contract(call))

    def test_cpu_fallback_execution(self):
        data = np.arange(1 * 3 * 5 * 7, dtype="int8").reshape(1, 3, 5, 7)
        weight = np.ones((19, 3, 3, 3), dtype="int8")
        mod = graph(data.shape, weight.shape, out_dtype="int8")
        np.testing.assert_array_equal(run_cpu(DecomposeGemminiConv2D()(mod), data, weight), direct_conv(data, weight).astype("int8"))


if __name__ == "__main__":
    unittest.main()
