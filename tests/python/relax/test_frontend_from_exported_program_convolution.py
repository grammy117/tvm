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
"""Convolution dtype, geometry and gradient contracts without device scheduling."""

import itertools

import numpy as np
import pytest
import tvm
from tvm import relax


def execute(model, *values):
    import torch
    from tvm.relax.frontend.torch import from_exported_program

    def flatten(value):
        if hasattr(value, "numpy"):
            return [value.numpy()]
        return [array for item in value for array in flatten(item)]

    args = tuple(torch.from_numpy(value) for value in values)
    expected = flatten(model(*args))
    graph = from_exported_program(torch.export.export(model, args), unwrap_unit_return_tuple=True)
    vm = relax.VirtualMachine(relax.build(graph, "llvm"), tvm.cpu())
    sources = [tvm.nd.array(value) for value in values]
    outputs = flatten(vm["main"](*sources))
    assert len(outputs) == len(expected)
    for actual, reference in zip(outputs, expected):
        assert actual.dtype == reference.dtype and actual.shape == reference.shape
        if actual.dtype.kind in "iu":
            np.testing.assert_array_equal(actual, reference)
        else:
            np.testing.assert_allclose(
                actual, reference, rtol=1e-3 if actual.dtype == np.float16 else 1e-6, atol=1e-6
            )
    for source, value in zip(sources, values):
        np.testing.assert_array_equal(
            source.numpy().reshape(-1).view("uint8"), value.reshape(-1).view("uint8")
        )
    return graph, outputs


def model_forward(dim, transpose, direct, strides=1, padding=0, dilation=1, groups=1, extra=0):
    import torch

    def axes(value):
        return [value] * dim if isinstance(value, int) else list(value)

    class Conv(torch.nn.Module):
        def forward(self, x, w, bias):
            if direct:
                return torch.ops.aten.convolution.default(
                    x,
                    w,
                    bias,
                    axes(strides),
                    axes(padding),
                    axes(dilation),
                    transpose,
                    axes(extra),
                    groups,
                )
            op = getattr(torch.nn.functional, f"conv{'_transpose' if transpose else ''}{dim}d")
            kwargs = {"output_padding": extra} if transpose else {}
            return op(
                x,
                w,
                bias,
                stride=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
                **kwargs,
            )

    return Conv()


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize(
    "transpose,dtype",
    [(False, dtype) for dtype in ("int8", "int32", "int64", "float16", "float32", "float64")]
    + [(True, dtype) for dtype in ("int64", "float16", "float32", "float64")],
)
def test_direct_and_named_convolution_preserve_captured_dtype(dim, direct, transpose, dtype):
    pytest.importorskip("torch")
    shape = (1, 2) + tuple(4 + i % 2 for i in range(dim))
    weight_shape = ((2, 3) if transpose else (3, 2)) + (2,) * dim
    values = (np.arange(np.prod(shape)).reshape(shape) % 13 - 7).astype(dtype)
    weights = (np.arange(np.prod(weight_shape)).reshape(weight_shape) % 7 - 4).astype(dtype)
    bias = np.array([-17, 0, 23], dtype)
    graph, _ = execute(model_forward(dim, transpose, direct), values, weights, bias)
    if not transpose and dtype in ("int8", "float16", "float32", "float64"):
        found = []
        relax.analysis.post_order_visit(
            graph["main"].body,
            lambda node: found.append(node)
            if isinstance(node, relax.Call)
            and isinstance(node.op, tvm.ir.Op)
            and node.op.name == f"relax.nn.conv{dim}d"
            else None,
        )
        assert len(found) == 1


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize(
    "transpose,dtype", [(False, "int8"), (False, "float32"), (True, "float32")]
)
@pytest.mark.parametrize("unbatched", [False, True])
def test_grouped_strided_dilated_convolution_and_named_unbatched(dim, transpose, dtype, unbatched):
    pytest.importorskip("torch")
    shape = (() if unbatched else (1,)) + (4,) + (5,) * dim
    weight_shape = (4, 2) + (2,) * dim
    values = (np.arange(np.prod(shape)).reshape(shape) % 13 - 7).astype(dtype)
    weights = (np.arange(np.prod(weight_shape)).reshape(weight_shape) % 7 - 4).astype(dtype)
    bias = np.array([-17, 23, 0, -3], dtype)
    execute(
        model_forward(
            dim,
            transpose,
            False,
            strides=2,
            padding=1,
            dilation=1 if dtype == "int8" else 2,
            groups=2,
            extra=int(transpose),
        ),
        values,
        weights,
        bias,
    )


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_transpose_output_padding_can_be_less_than_dilation_instead_of_stride(dim):
    pytest.importorskip("torch")
    shape = (1, 2) + (3,) * dim
    weight_shape = (2, 3) + (2,) * dim
    values = (np.arange(np.prod(shape)).reshape(shape) % 13 - 7).astype("float32")
    weights = (np.arange(np.prod(weight_shape)).reshape(weight_shape) % 7 - 4).astype("float32")
    execute(
        model_forward(dim, True, True, strides=1, dilation=2, extra=1),
        values,
        weights,
        np.array([-17, 0, 23], "float32"),
    )


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("transpose,dtype", [(False, "int32"), (False, "int64"), (True, "int64")])
def test_full_width_convolution_matches_unbounded_integer_oracle(dim, transpose, dtype):
    pytest.importorskip("torch")
    bits = np.dtype(dtype).itemsize * 8
    seed = np.array([-(1 << (bits - 1)), (1 << (bits - 1)) - 1, -17, 33], dtype)
    shape = (1, 2) + (3,) * dim
    w_shape = (2, 2) + (2,) * dim
    values, weights = np.resize(seed, shape), np.resize(seed[::-1], w_shape)
    bias = np.array([(1 << (bits - 1)) - 1, -(1 << (bits - 1))], dtype)
    _, outputs = execute(model_forward(dim, transpose, True), values, weights, bias)
    expected = np.empty(outputs[0].shape, dtype)
    for n, oc, *position in np.ndindex(expected.shape):
        total = int(bias[oc])
        for ic, tap in itertools.product(range(2), np.ndindex((2,) * dim)):
            source = tuple(p - k if transpose else p + k for p, k in zip(position, tap))
            if all(0 <= p < 3 for p in source):
                total += int(values[(n, ic) + source]) * int(
                    weights[((ic, oc) if transpose else (oc, ic)) + tap]
                )
        expected[(n, oc) + tuple(position)] = (total + (1 << (bits - 1))) % (1 << bits) - (
            1 << (bits - 1)
        )
    np.testing.assert_array_equal(outputs[0], expected)


def model_backward(
    dim, transpose, groups, stride, dilation, mask=(True, True, True), selected=None
):
    import torch

    class Gradient(torch.nn.Module):
        def forward(self, g, x, w):
            result = torch.ops.aten.convolution_backward.default(
                g,
                x,
                w,
                [g.shape[1]],
                [stride] * dim,
                [1] * dim,
                [dilation] * dim,
                transpose,
                [0] * dim,
                groups,
                list(mask),
            )
            return result if selected is None else result[selected]

    return Gradient()


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("groups", [1, 2])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("stride,dilation", [(1, 1), (2, 2)])
def test_convolution_backward_composes_stock_adjoint_and_mathematical_contractions(
    dim, transpose, groups, dtype, stride, dilation
):
    torch = pytest.importorskip("torch")
    shape = (1, 4) + (5,) * dim
    w_shape = ((4, 2) if transpose else (4, 4 // groups)) + (2,) * dim
    values = ((np.arange(np.prod(shape)).reshape(shape) % 13 - 7) / 4).astype(dtype)
    weights = ((np.arange(np.prod(w_shape)).reshape(w_shape) % 7 - 4) / 4).astype(dtype)
    op = getattr(torch.nn.functional, f"conv{'_transpose' if transpose else ''}{dim}d")
    output = op(
        torch.from_numpy(values),
        torch.from_numpy(weights),
        stride=stride,
        padding=1,
        dilation=dilation,
        groups=groups,
    )
    gradient = ((np.arange(output.numel()).reshape(output.shape) % 11 - 6) / 4).astype(dtype)
    execute(model_backward(dim, transpose, groups, stride, dilation), gradient, values, weights)


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("selected", [0, 1, 2])
def test_convolution_backward_output_masks_only_build_requested_tensors(transpose, selected):
    torch = pytest.importorskip("torch")
    values = (np.arange(40).reshape(1, 2, 4, 5) % 13 - 7).astype("float32")
    weights = (np.arange(16).reshape(2, 2, 2, 2) % 7 - 4).astype("float32")
    op = torch.nn.functional.conv_transpose2d if transpose else torch.nn.functional.conv2d
    output = op(torch.from_numpy(values), torch.from_numpy(weights), padding=1)
    gradient = (np.arange(output.numel()).reshape(output.shape) % 11 - 6).astype("float32")
    mask = tuple(index == selected for index in range(3))
    execute(model_backward(2, transpose, 1, 1, 1, mask, selected), gradient, values, weights)


@pytest.mark.parametrize("transpose", [False, True])
def test_empty_batch_convolution_backward_has_zero_weight_and_bias_gradients(transpose):
    torch = pytest.importorskip("torch")
    values = np.empty((0, 2, 4, 5), "float32")
    weights = np.arange(16, dtype="float32").reshape(2, 2, 2, 2)
    op = torch.nn.functional.conv_transpose2d if transpose else torch.nn.functional.conv2d
    output = op(torch.from_numpy(values), torch.from_numpy(weights), padding=1)
    gradient = np.empty(output.shape, "float32")
    execute(model_backward(2, transpose, 1, 1, 1), gradient, values, weights)
