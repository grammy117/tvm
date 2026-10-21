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
"""Source gradient and overlap contracts, independent of device scheduling."""

import numpy as np
import pytest
import tvm
from tvm import relax


def execute(model, *values):
    import torch
    from tvm.relax.frontend.torch import from_exported_program

    args = [torch.from_numpy(value) for value in values]
    expected = model(*args).numpy()
    graph = from_exported_program(
        torch.export.export(model, tuple(args)), unwrap_unit_return_tuple=True
    )
    vm = relax.VirtualMachine(relax.build(graph, "llvm"), tvm.cpu())
    inputs = [tvm.nd.array(value) for value in values]
    actual = vm["main"](*inputs).numpy()
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    if expected.dtype.kind in "biu":
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(
            actual, expected, rtol=1e-3 if expected.dtype == np.float16 else 1e-6, atol=1e-7
        )
    for source, original in zip(inputs, values):
        np.testing.assert_array_equal(
            source.numpy().reshape(-1).view("uint8"), original.reshape(-1).view("uint8")
        )
    return actual, graph


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64", "int64"])
@pytest.mark.parametrize("unbatched", [False, True])
@pytest.mark.parametrize(
    "case", ["default", "exclude", "ceil", "ceil_exclude", "override", "negative"]
)
def test_average_pool_backward_windows_divisors_and_dtype(dtype, unbatched, case):
    torch = pytest.importorskip("torch")
    ceil = case in ("ceil", "ceil_exclude", "negative")
    include = case not in ("exclude", "ceil_exclude")
    divisor = 5 if case == "override" else -7 if case == "negative" else None

    class Gradient(torch.nn.Module):
        def forward(self, g, x):
            return torch.ops.aten.avg_pool2d_backward.default(
                g, x, [3, 3], [2, 2], [1, 1], ceil, include, divisor
            )

    shape = ((2,) if unbatched else (1, 2)) + (5, 6)
    values = (np.arange(np.prod(shape)).reshape(shape) % 23 - 12).astype(dtype)
    output = torch.nn.functional.avg_pool2d(
        torch.from_numpy(values), 3, 2, 1, ceil_mode=ceil, count_include_pad=include
    )
    gradient = (np.arange(output.numel()).reshape(output.shape) % 19 - 11).astype(dtype)
    _, graph = execute(Gradient(), gradient, values)
    if dtype in ("float32", "float64") and divisor is None:
        found = []
        relax.analysis.post_order_visit(
            graph["main"].body,
            lambda node: found.append(node)
            if isinstance(node, relax.Call)
            and isinstance(node.op, tvm.ir.Op)
            and node.op.name == "relax.grad.avg_pool2d_backward"
            else None,
        )
        assert len(found) == 1


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
@pytest.mark.parametrize("unbatched", [False, True])
@pytest.mark.parametrize("output", [(2, 3), (7, 8), (1, 1)])
def test_adaptive_pool_backward_uneven_and_expanding_windows(dtype, unbatched, output):
    torch = pytest.importorskip("torch")

    class Gradient(torch.nn.Module):
        def forward(self, g, x):
            return torch.ops.aten._adaptive_avg_pool2d_backward.default(g, x)

    shape = ((2,) if unbatched else (1, 2)) + (5, 6)
    values = (np.arange(np.prod(shape)).reshape(shape) % 23 - 12).astype(dtype)
    g_shape = shape[:-2] + output
    gradient = ((np.arange(np.prod(g_shape)).reshape(g_shape) % 19 - 11) / 4).astype(dtype)
    execute(Gradient(), gradient, values)


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64", "bool"])
@pytest.mark.parametrize("unbatched", [False, True])
@pytest.mark.parametrize(
    "image,kernel,dilation,padding,strides",
    [
        ((4, 5), (2, 3), (1, 1), (0, 0), (1, 1)),
        ((5, 6), (3, 2), (1, 2), (1, 1), (2, 1)),
        ((6, 7), (2, 2), (2, 1), (0, 1), (3, 2)),
    ],
)
def test_col2im_overlap_padding_dilation_and_holes(
    dtype, unbatched, image, kernel, dilation, padding, strides
):
    torch = pytest.importorskip("torch")

    class Fold(torch.nn.Module):
        def forward(self, x):
            return torch.ops.aten.col2im.default(x, image, kernel, dilation, padding, strides)

    column_shape = tuple(
        (size + 2 * pad - dil * (k - 1) - 1) // stride + 1
        for size, pad, dil, k, stride in zip(image, padding, dilation, kernel, strides)
    )
    shape = (() if unbatched else (1,)) + (2 * kernel[0] * kernel[1], np.prod(column_shape))
    values = ((np.arange(np.prod(shape)).reshape(shape) % 19 - 11) / 4).astype(dtype)
    actual, _ = execute(Fold(), values)
    batched = values[None] if unbatched else values
    expected = np.zeros((1, 2) + image, dtype)
    for c, ky, kx, y, x in np.ndindex((2,) + kernel + column_shape):
        iy = y * strides[0] - padding[0] + ky * dilation[0]
        ix = x * strides[1] - padding[1] + kx * dilation[1]
        if 0 <= iy < image[0] and 0 <= ix < image[1]:
            row = (c * kernel[0] + ky) * kernel[1] + kx
            expected[0, c, iy, ix] += batched[0, row, y * column_shape[1] + x]
    np.testing.assert_array_equal(actual, expected[0] if unbatched else expected)


def test_integer_pool_backward_wraps_overlapping_deltas():
    torch = pytest.importorskip("torch")

    class Gradient(torch.nn.Module):
        def forward(self, g, x):
            return torch.ops.aten.avg_pool2d_backward.default(
                g, x, [3, 3], [1, 1], [1, 1], False, True, 1
            )

    seed = np.array([-(1 << 63), (1 << 63) - 1, -17, 33], "int64")
    gradient = np.resize(seed, (1, 1, 4, 5))
    values = np.zeros_like(gradient)
    actual, _ = execute(Gradient(), gradient, values)
    expected = np.empty_like(values)
    for y, x in np.ndindex(values.shape[-2:]):
        total = sum(
            int(gradient[0, 0, oy, ox])
            for oy in range(max(y - 1, 0), min(y + 2, 4))
            for ox in range(max(x - 1, 0), min(x + 2, 5))
        )
        expected[0, 0, y, x] = (total + (1 << 63)) % (1 << 64) - (1 << 63)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64", "int64"])
@pytest.mark.parametrize("holes", [False, True])
def test_average_pool_backward_broadcast_parameters_empty_stride_and_uncovered_inputs(dtype, holes):
    torch = pytest.importorskip("torch")
    kernel, strides, padding = ([1], [3], [0]) if holes else ([3], [], [1])

    class Gradient(torch.nn.Module):
        def forward(self, g, x):
            return torch.ops.aten.avg_pool2d_backward.default(
                g, x, kernel, strides, padding, False, True, 1
            )

    values = (np.arange(30).reshape(1, 1, 5, 6) - 17).astype(dtype)
    shape = torch.nn.functional.avg_pool2d(torch.from_numpy(values), kernel, strides, padding).shape
    gradient = (np.arange(np.prod(shape)).reshape(shape) + 1).astype(dtype)
    execute(Gradient(), gradient, values)
