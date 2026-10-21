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
"""Compile-time constant packing preserves independent convolution references."""

import unittest
import numpy as np
import tvm
from tvm import relax, tir
from tvm.relax.backend.contrib.gemmini import prepare_gemmini_graph
from tvm.relax.backend.contrib.gemmini_tensorir import prepare_gemmini_tensorir_workloads

import test_backend_contrib_gemmini_tensorir as reference


class GemminiConstantPackingTests(unittest.TestCase):
    setUp = reference.GemminiTensorIRTests.setUp

    def test_integral_carrier_bias_and_packed_conv_match_source_quantization(self):
        import torch
        import torch.ao.quantization.fx._decomposed
        from tvm.relax.frontend.torch import from_exported_program

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "weight", torch.arange(144, dtype=torch.int8).reshape(4, 4, 3, 3) - 70
                )
                self.register_buffer("bias", torch.tensor([-7617.0, 5424.0, 7731.0, 32775.0]))
                self.register_buffer("scale", torch.full((1, 4, 1, 1), 0.0031129))

            def forward(self, x):
                a = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                    x, 1.0, 0, -128, 127, torch.int8
                )
                b = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                    self.weight, 1.0, 0, -128, 127, torch.int8
                )
                result = (
                    torch.relu(torch.nn.functional.conv2d(a, b, self.bias, padding=1)) * self.scale
                )
                return torch.ops.quantized_decomposed.quantize_per_tensor.default(
                    result, 1.0, 0, -128, 127, torch.int8
                )

        model = Model()
        values = torch.arange(64, dtype=torch.int8).reshape(1, 4, 4, 4) - 32
        source = from_exported_program(
            torch.export.export(model, (values,)), unwrap_unit_return_tuple=True
        )
        report = []
        prepared = prepare_gemmini_graph(source, report=report)
        native = [row for row in report if row["placement"] == "gemmini"]
        self.assertEqual(len(native), 1, report)
        self.assertTrue(native[0]["metadata"]["bias_preloaded"])
        self.assertTrue(native[0]["metadata"]["readout_fused"])
        function = next(f for f in prepared.functions.values() if isinstance(f, tir.PrimFunc))
        self.assertEqual(tuple(map(int, function.buffer_map[function.params[1]].shape)), (48, 4))
        path = self.root / "packed-bias.so"
        relax.build(prepared, "llvm").export_library(
            str(path), addons=[str(self.emulator)], cc="cc", options=["-std=c11"]
        )
        vm = relax.VirtualMachine(tvm.runtime.load_module(str(path)), tvm.cpu())
        actual = vm["main"](tvm.nd.array(values.numpy())).numpy()
        np.testing.assert_array_equal(actual, model(values).numpy())

    def test_constant_weights_are_folded_once_and_native_results_match_cpu(self):
        for channels, output_channels, kernel in ((3, 17, 3), (2, 7, 1)):
            shape = (1, channels, 5, 6)
            x = relax.Var("x", relax.TensorStructInfo(shape, "int8"))
            weights = np.resize(
                np.array([-128, 127, 0, -1, 3], "int8"), (output_channels, channels, kernel, kernel)
            )
            builder = relax.BlockBuilder()
            with builder.function("main", [x]):
                with builder.dataflow():
                    y = builder.emit_output(
                        relax.op.nn.conv2d(
                            x, relax.const(weights), padding=kernel // 2, out_dtype="int32"
                        )
                    )
                builder.emit_func_output(y)
            source = builder.get()
            mathematical, _, _ = prepare_gemmini_tensorir_workloads(source)
            functions = [f for f in mathematical.functions.values() if isinstance(f, tir.PrimFunc)]
            self.assertEqual(len(functions), 1)  # The preprocessor was folded out.
            weight_buffer = functions[0].buffer_map[functions[0].params[1]]
            self.assertEqual(
                tuple(map(int, weight_buffer.shape)),
                (
                    ((channels * kernel * kernel + 15) // 16) * 16,
                    ((output_channels + 15) // 16) * 16
                    if output_channels > 16
                    else output_channels,
                ),
            )
            report = []
            prepared = prepare_gemmini_graph(source, report=report)
            self.assertEqual(sum(row["placement"] == "gemmini" for row in report), 1, report)
            # No remaining runtime block writes the packed weight parameter.
            function = next(f for f in prepared.functions.values() if isinstance(f, tir.PrimFunc))
            weight_data = function.buffer_map[function.params[1]].data
            stores = []
            tir.stmt_functor.post_order_visit(
                function.body,
                lambda node: stores.append(node)
                if isinstance(node, tir.BufferStore) and node.buffer.data.same_as(weight_data)
                else None,
            )
            self.assertFalse(stores)
            values = np.resize(np.array([-128, 127, -3, 0, 5], "int8"), shape)
            cpu = relax.VirtualMachine(relax.build(source, "llvm"), tvm.cpu())
            expected = cpu["main"](tvm.nd.array(values)).numpy()
            self.index += 1
            path = self.root / f"packed-{self.index}.so"
            relax.build(prepared, "llvm").export_library(
                str(path), addons=[str(self.emulator)], cc="cc", options=["-std=c11"]
            )
            vm = relax.VirtualMachine(tvm.runtime.load_module(str(path)), tvm.cpu())
            actual = vm["main"](tvm.nd.array(values)).numpy()
            np.testing.assert_array_equal(actual, expected)
