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
"""ATen pooling window, divisor, shape and dtype contracts without device scheduling."""

import numpy as np
import pytest
import tvm
from tvm import relax


def execute(model, values):
    import torch
    from tvm.relax.frontend.torch import from_exported_program

    example = torch.from_numpy(values)
    expected = model(example).numpy()
    graph = from_exported_program(
        torch.export.export(model, (example,)), unwrap_unit_return_tuple=True
    )
    vm = relax.VirtualMachine(relax.build(graph, "llvm"), tvm.cpu())
    source = tvm.nd.array(values)
    actual = vm["main"](source).numpy()
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    if values.dtype.kind in "iu":
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(
            actual, expected, rtol=1e-3 if values.dtype == np.float16 else 1e-6, atol=1e-7
        )
    np.testing.assert_array_equal(
        source.numpy().reshape(-1).view("uint8"), values.reshape(-1).view("uint8")
    )
    return graph


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("unbatched", [False, True])
@pytest.mark.parametrize("case", ["default", "exclude", "ceil", "trailing"])
def test_average_pool_windows_padding_and_last_window_rule(dim, unbatched, case):
    torch = pytest.importorskip("torch")
    trailing = case == "trailing"
    kernel, stride, pad = (2 if trailing else 3,) * dim, (2,) * dim, (1,) * dim

    class Pool(torch.nn.Module):
        def forward(self, x):
            return getattr(torch.nn.functional, f"avg_pool{dim}d")(
                x,
                kernel,
                stride,
                pad,
                ceil_mode=case in ("ceil", "trailing"),
                count_include_pad=case != "exclude",
            )

    shape = ((2,) if unbatched else (1, 2)) + (
        (3,) * dim if trailing else tuple(6 + i for i in range(dim))
    )
    values = ((np.arange(np.prod(shape)).reshape(shape) % 23 - 11) / 4).astype("float32")
    execute(Pool(), values)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("divisor", [5, -7, "volume"])
@pytest.mark.parametrize("unbatched", [False, True])
def test_average_pool_explicit_divisor_is_applied_once(dim, divisor, unbatched):
    torch = pytest.importorskip("torch")
    value = 3**dim if divisor == "volume" else divisor

    class Pool(torch.nn.Module):
        def forward(self, x):
            return getattr(torch.nn.functional, f"avg_pool{dim}d")(
                x,
                (3,) * dim,
                (2,) * dim,
                (1,) * dim,
                ceil_mode=divisor == -7,
                count_include_pad=False,
                divisor_override=value,
            )

    shape = ((2,) if unbatched else (1, 2)) + tuple(4 + i for i in range(dim))
    values = ((np.arange(np.prod(shape)).reshape(shape) % 19 - 9) / 4).astype("float32")
    execute(Pool(), values)


@pytest.mark.parametrize(
    "dim,dtype",
    [(d, t) for d in (1, 2, 3) for t in ("int64", "float64")] + [(1, "float16"), (2, "float16")],
)
def test_average_pool_integer_truncation_and_floating_accumulation_dtype(dim, dtype):
    torch = pytest.importorskip("torch")

    class Pool(torch.nn.Module):
        def forward(self, x):
            return getattr(torch.nn.functional, f"avg_pool{dim}d")(
                x, (3,) * dim, (2,) * dim, (1,) * dim, ceil_mode=True, count_include_pad=False
            )

    shape = (1, 2) + tuple(4 + i for i in range(dim))
    values = (np.arange(np.prod(shape)).reshape(shape) % 19 - 10).astype(dtype)
    execute(Pool(), values)


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("unbatched", [False, True])
@pytest.mark.parametrize("case", ["uneven", "expand", "identity_axis"])
def test_adaptive_pool_variable_windows_and_shape_metadata(dim, unbatched, case):
    torch = pytest.importorskip("torch")
    output = (7,) * dim if case == "expand" else (2,) * dim
    if case == "identity_axis":
        output = (5,) if dim == 1 else (None,) + (2,) * (dim - 1)

    class Pool(torch.nn.Module):
        def forward(self, x):
            return getattr(torch.nn.functional, f"adaptive_avg_pool{dim}d")(x, output)

    shape = ((2,) if unbatched else (1, 2)) + tuple(5 + i for i in range(dim))
    values = ((np.arange(np.prod(shape)).reshape(shape) % 19 - 10) / 4).astype("float32")
    execute(Pool(), values)


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("dtype", ["float16", "float64"])
def test_adaptive_pool_floating_dtype_preservation(dim, dtype):
    torch = pytest.importorskip("torch")

    class Pool(torch.nn.Module):
        def forward(self, x):
            return getattr(torch.nn.functional, f"adaptive_avg_pool{dim}d")(x, (2,) * dim)

    shape = (1, 2) + (5,) * dim
    values = ((np.arange(np.prod(shape)).reshape(shape) % 19 - 10) / 4).astype(dtype)
    execute(Pool(), values)


@pytest.mark.parametrize("dim", [2, 3])
def test_internal_adaptive_overload_uses_same_mathematics(dim):
    torch = pytest.importorskip("torch")

    class Pool(torch.nn.Module):
        def forward(self, x):
            return getattr(torch.ops.aten, f"_adaptive_avg_pool{dim}d").default(x, [2] * dim)

    shape = (1, 1) + (5,) * dim
    values = (np.arange(np.prod(shape)).reshape(shape) % 13 - 7).astype("float32")
    execute(Pool(), values)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_int64_window_overflow_matches_unbounded_integer_oracle(dim):
    import itertools

    torch = pytest.importorskip("torch")
    divisor = 3 if dim == 1 else 5

    class Pool(torch.nn.Module):
        def forward(self, x):
            kwargs = {} if dim == 1 else {"divisor_override": divisor}
            return getattr(torch.nn.functional, f"avg_pool{dim}d")(
                x, (3,) * dim, (2,) * dim, (1,) * dim, **kwargs
            )

    seed = np.array([-(1 << 63), (1 << 63) - 1, -1, 0, 17, -33], "int64")
    values = np.resize(seed, (1, 1) + (4,) * dim)
    graph = execute(Pool(), values)
    vm = relax.VirtualMachine(relax.build(graph, "llvm"), tvm.cpu())
    expected = np.empty((1, 1) + (2,) * dim, "int64")
    for output in np.ndindex((2,) * dim):
        total = 0
        for offset in itertools.product(range(3), repeat=dim):
            point = tuple(index * 2 - 1 + delta for index, delta in zip(output, offset))
            if all(0 <= index < 4 for index in point):
                total += int(values[(0, 0) + point])
        wrapped = (total + (1 << 63)) % (1 << 64) - (1 << 63)
        expected[(0, 0) + output] = (
            -(abs(wrapped) // divisor) if wrapped < 0 else wrapped // divisor
        )
    np.testing.assert_array_equal(vm["main"](tvm.nd.array(values)).numpy(), expected)
