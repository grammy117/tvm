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
"""Signed-byte _int_mm intake keeps the wider result and modular boundary."""

import numpy as np
import pytest
import tvm
from tvm import relax


@pytest.mark.parametrize("shape", [(2, 3, 4), (17, 33, 19), (1, 131072, 1)])
def test_int_mm_preserves_int32_values_and_overflow(shape):
    torch = pytest.importorskip("torch")
    from tvm.relax.frontend.torch import from_exported_program

    class Product(torch.nn.Module):
        def forward(self, left, right):
            return torch.ops.aten._int_mm.default(left, right)

    m, k, n = shape
    a = np.full((m, k), -128, "int8")
    b = np.full((k, n), -128, "int8")
    source = (torch.from_numpy(a), torch.from_numpy(b))
    exported = torch.export.export(Product(), source)
    assert any(str(node.target) == "aten._int_mm.default" for node in exported.graph.nodes)
    graph = from_exported_program(exported, unwrap_unit_return_tuple=True)
    vm = relax.VirtualMachine(relax.build(graph, "llvm"), tvm.cpu())
    actual = vm["main"](tvm.nd.array(a), tvm.nd.array(b)).numpy()
    expected = (a.astype("int64") @ b.astype("int64")).astype("int32")
    assert actual.dtype == expected.dtype == np.dtype("int32")
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, Product()(*source).numpy())


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_torch_round_uses_half_even_including_negative_zero(dtype):
    torch = pytest.importorskip("torch")
    from tvm.relax.frontend.torch import from_exported_program

    class Round(torch.nn.Module):
        def forward(self, values):
            return torch.round(values)

    values = np.array(
        [-np.inf, -4.5, -3.5, -2.5, -1.5, -0.5, -0.0, 0.0, 0.5, 1.5, 2.5, 3.5, np.inf],
        dtype=dtype,
    )
    source = torch.from_numpy(values)
    graph = from_exported_program(
        torch.export.export(Round(), (source,)), unwrap_unit_return_tuple=True
    )
    vm = relax.VirtualMachine(relax.build(graph, "llvm"), tvm.cpu())
    actual = vm["main"](tvm.nd.array(values)).numpy()
    expected = Round()(source).numpy()
    np.testing.assert_array_equal(actual.view("uint8"), expected.view("uint8"))
