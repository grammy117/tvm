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
"""Independent ATen dtype, promotion, overflow and rounding checks."""

import numpy as np
import pytest
import tvm
from tvm import relax


def execute(model, values):
    import torch
    from tvm.relax.frontend.torch import from_exported_program

    examples = [torch.from_numpy(value) for value in values]
    expected = model(*examples).numpy()
    graph = from_exported_program(
        torch.export.export(model, tuple(examples)), unwrap_unit_return_tuple=True
    )
    vm = relax.VirtualMachine(relax.build(graph, "llvm"), tvm.cpu())
    inputs = [tvm.nd.array(value) for value in values]
    actual = vm["main"](*inputs).numpy()
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    for source, value in zip(inputs, values):
        np.testing.assert_array_equal(
            source.numpy().reshape(-1).view("uint8"), value.reshape(-1).view("uint8")
        )
    return actual, expected


@pytest.mark.parametrize("dtype", ["int8", "int32", "int64", "float16", "float32", "float64"])
@pytest.mark.parametrize(
    "kind",
    [
        "mm",
        "bmm",
        "addmm",
        "add.Tensor",
        "sub.Tensor",
        "add.Scalar",
        "sub.Scalar",
        "mul.Scalar",
        "div.Scalar",
        "neg",
    ],
)
def test_arithmetic_preserves_dtype_and_modular_results(kind, dtype):
    torch = pytest.importorskip("torch")

    class Arithmetic(torch.nn.Module):
        def forward(self, x, y, bias):
            if kind == "mm":
                return torch.mm(x, y)
            if kind == "bmm":
                return torch.bmm(x, y)
            if kind == "addmm":
                return torch.addmm(bias, x, y, alpha=-3, beta=2)
            if kind == "add.Tensor":
                return torch.add(x, y, alpha=3)
            if kind == "sub.Tensor":
                return torch.sub(x, y, alpha=-3)
            if kind == "add.Scalar":
                return torch.ops.aten.add.Scalar(x, 17)
            if kind == "sub.Scalar":
                return torch.ops.aten.sub.Scalar(x, 17)
            if kind == "mul.Scalar":
                return torch.ops.aten.mul.Scalar(x, -3)
            if kind == "div.Scalar":
                return torch.ops.aten.div.Scalar(x, 3)
            return torch.neg(x)

    shapes = (
        ((2, 3), (3, 4), (4,))
        if kind in ("mm", "addmm")
        else (((2, 2, 3), (2, 3, 4), (4,)) if kind == "bmm" else ((2, 3), (2, 3), (3,)))
    )
    if dtype.startswith("int"):
        limits = np.iinfo(dtype)
        seed = np.array([limits.min, limits.max, -1, 0, 1, 17], dtype)
    else:
        seed = np.array([-3.75, 1.125, 0, -0.0, 1, 0.25], dtype)
    values = [
        np.resize(seed if i % 2 == 0 else seed[::-1], shape) for i, shape in enumerate(shapes)
    ]
    if dtype == "float16" and kind in ("add.Tensor", "sub.Tensor"):
        with pytest.raises(NotImplementedError, match="scalar-tail and vector"):
            execute(Arithmetic(), values)
        return
    actual, expected = execute(Arithmetic(), values)
    np.testing.assert_array_equal(
        actual.reshape(-1).view("uint8"), expected.reshape(-1).view("uint8")
    )


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
@pytest.mark.parametrize(
    "kind", ["add", "subtract", "scalar_add", "scalar_multiply", "scalar_divide"]
)
def test_floating_alpha_and_weak_scalar_rounding(kind, dtype):
    torch = pytest.importorskip("torch")

    class Arithmetic(torch.nn.Module):
        def forward(self, x, y):
            if kind == "add":
                return torch.add(x, y, alpha=1.1)
            if kind == "subtract":
                return torch.sub(x, y, alpha=1.1)
            if kind == "scalar_add":
                return torch.ops.aten.add.Scalar(x, 1.1)
            if kind == "scalar_multiply":
                return torch.ops.aten.mul.Scalar(x, 1.1)
            return torch.ops.aten.div.Scalar(x, 1.1)

    generator = np.random.default_rng(73019)
    values = [generator.normal(size=(3, 43)).astype(dtype) for _ in range(2)]
    if dtype == "float16" and kind in ("add", "subtract"):
        with pytest.raises(NotImplementedError, match="scalar-tail and vector"):
            execute(Arithmetic(), values)
        return
    actual, expected = execute(Arithmetic(), values)
    np.testing.assert_array_equal(actual.view("uint8"), expected.view("uint8"))


@pytest.mark.parametrize("alpha,beta", [(0, 0), (0, 2), (1, 0)])
@pytest.mark.parametrize("dtype", ["int8", "int32", "int64", "float32"])
def test_addmm_zero_coefficients_keep_output_shape_and_ignore_unused_bias(alpha, beta, dtype):
    torch = pytest.importorskip("torch")

    class Addmm(torch.nn.Module):
        def forward(self, bias, x, y):
            return torch.addmm(bias, x, y, alpha=alpha, beta=beta)

    bias = np.array([1, -3, 7, 11], dtype)
    if dtype == "float32" and beta == 0:
        bias[:] = np.nan
    values = [
        bias,
        np.arange(6, dtype=dtype).reshape(2, 3),
        np.arange(12, dtype=dtype).reshape(3, 4),
    ]
    actual, expected = execute(Addmm(), values)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("kind", ["add", "multiply", "divide"])
def test_integer_weak_float_scalar_promotes_to_captured_dtype(kind):
    torch = pytest.importorskip("torch")

    class Promoted(torch.nn.Module):
        def forward(self, x):
            if kind == "add":
                return torch.ops.aten.add.Scalar(x, 0.5)
            if kind == "multiply":
                return torch.ops.aten.mul.Scalar(x, 0.5)
            return torch.ops.aten.div.Scalar(x, 3)

    actual, expected = execute(Promoted(), [np.array([-2147483648, -1, 0, 2147483647], "int32")])
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("kind", ["matmul", "add", "multiply"])
def test_full_range_int64_arithmetic_matches_independent_unbounded_integers(kind):
    torch = pytest.importorskip("torch")

    class Arithmetic(torch.nn.Module):
        def forward(self, x, y):
            if kind == "matmul":
                return torch.matmul(x, y)
            if kind == "add":
                return torch.add(x, y, alpha=-7)
            return torch.mul(x, y)

    generator = np.random.default_rng(9317)
    shapes = ((2, 3, 11), (2, 11, 4)) if kind == "matmul" else ((3, 17), (17,))
    values = [
        generator.integers(0, 1 << 64, size=shape, dtype="uint64").view("int64") for shape in shapes
    ]
    actual, expected = execute(Arithmetic(), values)
    lhs, rhs = [value.astype(object) for value in values]
    exact = lhs @ rhs if kind == "matmul" else lhs + -7 * rhs if kind == "add" else lhs * rhs
    independent = np.array(
        [(int(value) + (1 << 63)) % (1 << 64) - (1 << 63) for value in exact.flat], "int64"
    ).reshape(exact.shape)
    np.testing.assert_array_equal(expected, independent)
    np.testing.assert_array_equal(actual, independent)


@pytest.mark.parametrize("dtype", ["int8", "int32", "int64", "float32"])
@pytest.mark.parametrize("kind", ["mm", "bmm", "addmm"])
def test_empty_contraction_has_initialized_zero_and_bias_results(dtype, kind):
    torch = pytest.importorskip("torch")

    class Empty(torch.nn.Module):
        def forward(self, x, y, bias):
            if kind == "addmm":
                return torch.addmm(bias, x, y)
            return torch.bmm(x, y) if kind == "bmm" else torch.mm(x, y)

    shapes = ((2, 3, 0), (2, 0, 4), (4,)) if kind == "bmm" else ((3, 0), (0, 4), (4,))
    values = [np.zeros(shape, dtype) for shape in shapes]
    values[-1][:] = [1, -3, 7, 11]
    actual, expected = execute(Empty(), values)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", ["int8", "int32", "int64"])
@pytest.mark.parametrize(
    "shapes", [((5,), (5,)), ((5,), (2, 5, 3)), ((2, 3, 5), (5,)), ((2, 1, 3, 5), (4, 5, 2))]
)
def test_existing_matmul_vector_and_batch_broadcast_contracts(dtype, shapes):
    torch = pytest.importorskip("torch")

    class Matmul(torch.nn.Module):
        def forward(self, x, y):
            return torch.matmul(x, y)

    values = [(np.arange(np.prod(shape)).reshape(shape) % 13 - 7).astype(dtype) for shape in shapes]
    actual, expected = execute(Matmul(), values)
    np.testing.assert_array_equal(actual, expected)


def test_zero_floating_alpha_preserves_nonfinite_propagation():
    torch = pytest.importorskip("torch")

    class Alpha(torch.nn.Module):
        def forward(self, x, y):
            return torch.add(x, y, alpha=0)

    values = [
        np.array([-0.0, 0.0, 1, -1, np.inf, np.nan], "float32"),
        np.array([0.0, -0.0, np.nan, np.inf, 0, 1], "float32"),
    ]
    actual, expected = execute(Alpha(), values)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    finite = ~np.isnan(expected)
    np.testing.assert_array_equal(actual[finite].view("uint32"), expected[finite].view("uint32"))


@pytest.mark.parametrize("dtype", ["int8", "int32", "int64"])
@pytest.mark.parametrize("scalar_kind", ["minimum", "maximum", "ordinary"])
@pytest.mark.parametrize("alpha", [-1, 1, 2])
def test_constant_integer_subtraction_folds_offset_in_the_result_ring(dtype, scalar_kind, alpha):
    torch = pytest.importorskip("torch")
    bits = np.dtype(dtype).itemsize * 8
    minimum, maximum = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    scalar = minimum if scalar_kind == "minimum" else maximum if scalar_kind == "maximum" else 17

    class Subtract(torch.nn.Module):
        def forward(self, x):
            return torch.ops.aten.sub.Scalar(x, scalar, alpha)

    values = np.array([minimum, maximum, -1, 0, 1, 17], dtype)
    actual, source = execute(Subtract(), [values])
    expected = np.array(
        [(int(value) - scalar * alpha - minimum) % (1 << bits) + minimum for value in values], dtype
    )
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(source, expected)
