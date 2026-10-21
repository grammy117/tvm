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
"""Verify mathematical PyTorch view imports independently of hardware scheduling."""

import numpy as np
import pytest
import tvm
from tvm import relax


@pytest.mark.parametrize("dtype", ["int8", "int32", "float32"])
@pytest.mark.parametrize(
    "kind",
    [
        "squeeze_dim",
        "squeeze_nonunit",
        "squeeze_dims",
        "unsqueeze",
        "view",
        "permute",
        "select",
        "slice",
        "split_sizes",
        "split_integer",
        "diagonal",
        "diagonal_negative",
        "as_strided",
        "as_strided_repeat",
        "as_strided_rebased",
        "as_strided_empty",
    ],
)
def test_view_import_preserves_values_shapes_and_bits(kind, dtype):
    torch = pytest.importorskip("torch")
    from tvm.relax.frontend.torch import from_exported_program

    class Views(torch.nn.Module):
        def forward(self, x):
            if kind == "squeeze_dim":
                return torch.squeeze(x, 0)
            if kind == "squeeze_nonunit":
                return torch.squeeze(x, 1)
            if kind == "squeeze_dims":
                return torch.squeeze(x, (0, 1))
            if kind == "unsqueeze":
                return torch.unsqueeze(x, -1)
            if kind == "view":
                return x.view(-1, 6)
            if kind == "permute":
                return x.permute(2, 0, 1)
            if kind == "select":
                return torch.select(x, 1, -1)
            if kind == "slice":
                return torch.ops.aten.slice.Tensor(x, 1, 0, 3, 2)
            if kind == "split_sizes":
                return torch.split(x, [1, 2], dim=1)
            if kind == "split_integer":
                return torch.split(x, 5, dim=-1)
            if kind == "diagonal":
                return torch.diagonal(x, offset=1, dim1=1, dim2=2)
            if kind == "diagonal_negative":
                return torch.diagonal(x, offset=-1, dim1=-1, dim2=-2)
            if kind == "as_strided_repeat":
                return torch.as_strided(x, (3, 4), (0, 1))
            if kind == "as_strided_rebased":
                return torch.as_strided(x[:, 1:, :], (2, 3), (4, 1))
            if kind == "as_strided_empty":
                return torch.as_strided(x, (0, 2), (4, 1), 99)
            return torch.as_strided(x, (2, 2), (4, 1), 1)

    shape = (1, 3, 12) if kind == "split_integer" else (1, 3, 4)
    if dtype == "float32":
        values = np.resize(
            np.array([0x80000000, 0x7FC12345, 0x7F812345, 1, 0xFF800000], "uint32"), shape
        ).view("float32")
    else:
        values = (np.arange(np.prod(shape)).reshape(shape) * 37 - 128).astype(dtype)
    model = Views()
    example = torch.from_numpy(values)
    graph = from_exported_program(
        torch.export.export(model, (example,)), unwrap_unit_return_tuple=True
    )
    vm = relax.VirtualMachine(relax.build(graph, "llvm"), tvm.cpu())
    source = tvm.nd.array(values)
    actual, expected = vm["main"](source), model(example)
    actual = list(actual) if isinstance(expected, tuple) else [actual]
    expected = list(expected) if isinstance(expected, tuple) else [expected]
    assert len(actual) == len(expected)
    for output, reference in zip(actual, expected):
        output, reference = output.numpy(), reference.numpy()
        assert output.shape == reference.shape
        assert output.dtype == reference.dtype
        np.testing.assert_array_equal(
            output.view("uint8"), np.ascontiguousarray(reference).view("uint8")
        )
    np.testing.assert_array_equal(source.numpy().view("uint8"), values.view("uint8"))


@pytest.mark.parametrize(
    "dtype,value", [("float32", -0.0), ("int8", -128), ("int32", -2147483648), ("int64", 7)]
)
def test_scalar_tensor_uses_captured_dtype(dtype, value):
    torch = pytest.importorskip("torch")
    from tvm.relax.frontend.torch import from_exported_program

    class Scalar(torch.nn.Module):
        def forward(self, x):
            return torch.scalar_tensor(value, dtype=getattr(torch, dtype))

    model = Scalar()
    example = torch.ones(1)
    graph = from_exported_program(
        torch.export.export(model, (example,)), unwrap_unit_return_tuple=True
    )
    vm = relax.VirtualMachine(relax.build(graph, "llvm"), tvm.cpu())
    actual = vm["main"](tvm.nd.array(example.numpy())).numpy()
    expected = model(example).numpy()
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    np.testing.assert_array_equal(
        actual.reshape(1).view("uint8"), expected.reshape(1).view("uint8")
    )


@pytest.mark.parametrize("kind", ["noncontiguous", "outside_logical_input"])
def test_as_strided_refuses_unrepresented_storage(kind):
    torch = pytest.importorskip("torch")
    from tvm.relax.frontend.torch import from_exported_program

    class Storage(torch.nn.Module):
        def forward(self, x):
            if kind == "noncontiguous":
                return torch.as_strided(x.transpose(0, 1), (2, 2), (4, 1))
            return torch.as_strided(x[1:, :], (2, 2), (4, 1), 0)

    exported = torch.export.export(Storage(), (torch.arange(12.0).reshape(3, 4),))
    with pytest.raises(NotImplementedError, match="contiguous|logical Relax input"):
        from_exported_program(exported)
