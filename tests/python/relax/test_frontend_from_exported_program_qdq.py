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
"""Source quantization semantics and proof-limited integer carrier canonicalization."""

import numpy as np
import pytest
import tvm
from tvm import relax
import importlib

torch = pytest.importorskip("torch")
importlib.import_module("torch.ao.quantization.fx._decomposed")
from_exported_program = importlib.import_module("tvm.relax.frontend.torch").from_exported_program


def check(model, inputs):
    expected = model(*inputs).detach().numpy()
    mod = from_exported_program(torch.export.export(model, inputs), unwrap_unit_return_tuple=True)
    vm = relax.VirtualMachine(relax.build(mod, "llvm"), tvm.cpu())
    actual = vm["main"](*[tvm.nd.array(x.numpy()) for x in inputs]).numpy()
    np.testing.assert_array_equal(actual, expected)
    return mod


@pytest.mark.parametrize("axis", [0, 1, 2, -1, -3])
def test_stack_all_axis_positions(axis):
    class Model(torch.nn.Module):
        def forward(self, a, b):
            return torch.stack((a, b), dim=axis)

    check(Model(), (torch.arange(6).reshape(2, 3), torch.arange(6, 12).reshape(2, 3)))


@pytest.mark.parametrize("size,chunks", [(6, 4), (7, 3), (3, 8), (8, 2)])
def test_chunk_uses_ceil_sized_sections(size, chunks):
    class Model(torch.nn.Module):
        def forward(self, values):
            return torch.cat(
                [value * (i + 1) for i, value in enumerate(values.chunk(chunks, -1))], -1
            )

    check(Model(), (torch.arange(2 * size, dtype=torch.float32).reshape(2, size),))


@pytest.mark.parametrize("scale,zero", [(0.25, 0), (0.13, -3), (1.0, 0)])
def test_tensor_qdq_rne_saturation_and_zero_points(scale, zero):
    class Model(torch.nn.Module):
        def forward(self, values):
            q = torch.ops.quantized_decomposed.quantize_per_tensor.default(
                values, scale, zero, -128, 127, torch.int8
            )
            return torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                q, scale, zero, -128, 127, torch.int8
            )

    values = torch.tensor([-1000.0, -2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5, 1000.0]) * scale
    check(Model(), (values,))


def test_channel_dequantization_preserves_scale_dtype_and_axis():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("scales", torch.tensor([0.13, 0.37, 1.0], dtype=torch.float64))
            self.register_buffer("zeros", torch.tensor([-3, 0, 7]))

        def forward(self, values):
            return torch.ops.quantized_decomposed.dequantize_per_channel.default(
                values, self.scales, self.zeros, 1, -128, 127, torch.int8
            )

    check(Model(), (torch.tensor([[-128, -17, 127], [3, 0, -7]], dtype=torch.int8),))


@pytest.mark.parametrize("depth,scale", [(17, 1.0), (1024, 1.0), (1025, 1.0), (17, 0.5)])
def test_integer_carrier_contraction_requires_prefix_proof(depth, scale):
    class Model(torch.nn.Module):
        def forward(self, a, b):
            a = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                a, scale, 0, -128, 127, torch.int8
            )
            b = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                b, scale, 0, -128, 127, torch.int8
            )
            return torch.nn.functional.linear(a, b)

    a = torch.full((2, depth), -128, dtype=torch.int8)
    b = torch.full((3, depth), -128, dtype=torch.int8)
    mod = check(Model(), (a, b))
    products = []
    relax.analysis.post_order_visit(
        mod["main"],
        lambda node: products.append(node)
        if isinstance(node, relax.Call) and node.op == tvm.ir.Op.get("relax.matmul")
        else None,
    )
    assert len(products) == 1
    assert (products[0].attrs.out_dtype == "int32") == (depth <= 1024 and scale == 1)


def test_integer_carrier_convolution_preserves_floating_bias():
    class Model(torch.nn.Module):
        def forward(self, a, b, bias):
            a = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                a, 1.0, 0, -128, 127, torch.int8
            )
            b = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                b, 1.0, 0, -128, 127, torch.int8
            )
            return torch.nn.functional.conv2d(a, b, bias, padding=1)

    generator = torch.Generator().manual_seed(17)
    a = torch.randint(-128, 128, (1, 3, 4, 5), dtype=torch.int8, generator=generator)
    b = torch.randint(-128, 128, (4, 3, 3, 3), dtype=torch.int8, generator=generator)
    check(Model(), (a, b, torch.tensor([-0.25, 1.75, 0.0, -1.5])))


@pytest.mark.parametrize("weight_code,integer", [(1, True), (-128, False)])
def test_frozen_weight_bounds_extend_exact_carrier_reductions(weight_code, integer):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weight", torch.full((3, 2048), weight_code, dtype=torch.int8))

        def forward(self, x):
            a = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                x, 1.0, 0, -128, 127, torch.int8
            )
            b = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                self.weight, 1.0, 0, -128, 127, torch.int8
            )
            return torch.nn.functional.linear(a, b)

    x = torch.full((2, 2048), -128, dtype=torch.int8)
    mod = check(Model(), (x,))
    products = []
    relax.analysis.post_order_visit(
        mod["main"],
        lambda node: products.append(node)
        if isinstance(node, relax.Call) and node.op == tvm.ir.Op.get("relax.matmul")
        else None,
    )
    assert len(products) == 1
    assert (products[0].attrs.out_dtype == "int32") == integer


def test_serialized_export_owns_aligned_constants_and_restores_input_metadata(tmp_path):
    import os
    import subprocess
    import sys

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("offset_constant", torch.arange(18, dtype=torch.float32)[1:])

        def forward(self, values):
            return values + self.offset_constant

    values = torch.arange(17, dtype=torch.float32)
    path = tmp_path / "offset.pt2"
    torch.export.save(torch.export.export(Model(), (values,)), path)
    script = """
import sys, torch, tvm, numpy as np
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program
loaded = torch.export.load(sys.argv[1])
for node in loaded.graph.nodes:
    node.meta.pop('tensor_meta', None)
mod = from_exported_program(loaded, unwrap_unit_return_tuple=True)
for value in loaded.state_dict.values():
    value.zero_()
vm = relax.VirtualMachine(relax.build(mod, 'llvm'), tvm.cpu())
actual = vm['main'](tvm.nd.array(np.arange(17, dtype='float32'))).numpy()
np.testing.assert_array_equal(actual, np.arange(17, dtype='float32') * 2 + 1)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        env=dict(os.environ),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout


@pytest.mark.parametrize(
    "bias,integer",
    [([3.0, -5.0, 32775.0], True), ([0.25, -1.75, 0.0], False), ([16777216.0, 0.0, 0.0], False)],
)
def test_frozen_bias_normalization_requires_integrality_and_fp32_prefix_bound(bias, integer):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weight", torch.ones((3, 17), dtype=torch.int8))
            self.register_buffer("bias", torch.tensor(bias, dtype=torch.float32))

        def forward(self, x):
            a = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                x, 1.0, 0, -128, 127, torch.int8
            )
            b = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                self.weight, 1.0, 0, -128, 127, torch.int8
            )
            return torch.nn.functional.linear(a, b, self.bias)

    mod = check(Model(), (torch.full((2, 17), -128, dtype=torch.int8),))
    additions = []
    relax.analysis.post_order_visit(
        mod["main"],
        lambda node: additions.append(node)
        if isinstance(node, relax.Call) and node.op == tvm.ir.Op.get("relax.add")
        else None,
    )
    assert len(additions) == 1
    assert (additions[0].struct_info.dtype == "int32") == integer


@pytest.mark.parametrize("weighted,epsilon", [(True, None), (False, 1e-5)])
def test_fp32_rms_norm_and_flatten(weighted, epsilon):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weight", torch.tensor([0.5, 1.0, 1.5, -0.25]))

        def forward(self, values):
            result = torch.nn.functional.rms_norm(
                values, (4,), self.weight if weighted else None, eps=epsilon
            )
            return result.flatten(1, -1)

    values = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 8 - 1
    model = Model()
    mod = from_exported_program(
        torch.export.export(model, (values,)), unwrap_unit_return_tuple=True
    )
    vm = relax.VirtualMachine(relax.build(mod, "llvm"), tvm.cpu())
    np.testing.assert_allclose(
        vm["main"](tvm.nd.array(values.numpy())).numpy(),
        model(values).numpy(),
        rtol=1e-6,
        atol=1e-7,
    )
