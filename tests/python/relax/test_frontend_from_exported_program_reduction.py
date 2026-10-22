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
"""Verify PyTorch reduction shape and dtype semantics before target scheduling."""

import numpy as np
import pytest
import tvm
from tvm import relax


@pytest.mark.parametrize(
    "kind,input_dtype,dims,keepdim,output_dtype,shape",
    [
        ("sum", dtype, dims, keep, None, (2, 3, 4))
        for dtype in ("int8", "int32", "bool", "float32")
        for dims, keep in (((1, 2), False), ((-1, 1), True), ((), True), (None, False))
    ]
    + [
        ("mean", "float32", dims, keep, None, (2, 3, 4))
        for dims, keep in (((1, 2), False), ((-1, 1), True), ((), True), (None, False))
    ]
    + [
        ("sum", "int8", (1,), True, "int8", (2, 7)),
        ("sum", "int32", (1,), False, "int32", (2, 7)),
        ("sum", "int8", (1,), False, "int32", (2, 7)),
        ("sum", "float32", (1,), True, "int64", (2, 7)),
        ("sum", "int32", (1,), True, "bool", (2, 7)),
        ("mean", "int8", (1,), True, "float32", (2, 7)),
        ("sum", "int8", (0,), True, None, ()),
        ("mean", "float32", (-1,), True, None, ()),
        ("sum", "int8", (0,), False, None, (0, 3)),
        ("mean", "float32", (0,), False, None, (0, 3)),
        ("sum", "bool", (0,), False, "bool", (0, 3)),
    ],
)
def test_reduction_executes_with_pytorch_contract(
    kind, input_dtype, dims, keepdim, output_dtype, shape
):
    torch = pytest.importorskip("torch")
    from tvm.relax.frontend.torch import from_exported_program

    class Reduction(torch.nn.Module):
        def forward(self, source):
            kwargs = {} if output_dtype is None else {"dtype": getattr(torch, output_dtype)}
            if dims is None and not keepdim:
                return getattr(torch, kind)(source, **kwargs)
            return getattr(torch, kind)(source, dim=dims, keepdim=keepdim, **kwargs)

    size = int(np.prod(shape))
    values = np.arange(size).reshape(shape) * 17 - 128
    if input_dtype == "int32":
        values = np.resize(np.array([2147483647, -2147483648, 2147483647, -1], "int32"), shape)
    elif input_dtype == "float32":
        values = values * 0.25
    elif input_dtype == "bool":
        values = np.arange(size).reshape(shape) % 3 == 0
    values = np.asarray(values, dtype=input_dtype)
    model = Reduction()
    example = torch.from_numpy(values)
    graph = from_exported_program(
        torch.export.export(model, (example,)), unwrap_unit_return_tuple=True
    )
    vm = relax.VirtualMachine(relax.build(graph, "llvm"), tvm.cpu())
    source = tvm.nd.array(values)
    actual = vm["main"](source).numpy()
    expected = model(example).numpy()
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    if actual.dtype.kind == "f":
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6, equal_nan=True)
    else:
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(source.numpy(), values)
