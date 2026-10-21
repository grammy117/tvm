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
"""Creation dtype, shape and payload contracts independent of accelerator scheduling."""

import numpy as np
import pytest
import tvm
from tvm import relax


@pytest.mark.parametrize("kind", ["full", "full_like"])
@pytest.mark.parametrize(
    "dtype,value,shape,default_dtype",
    [
        (None, 7, (2, 3), "float32"),
        (None, True, (2, 3), "float32"),
        (None, -0.0, (2, 3), "float64"),
        ("int8", -128, (3, 17), "float32"),
        ("int32", -2147483648, (2, 3), "float32"),
        ("int64", -9223372036854775807, (2, 3), "float32"),
        ("int32", -3.75, (2, 3), "float32"),
        ("float32", -0.0, (2, 3), "float32"),
        ("float32", float("nan"), (2, 3), "float32"),
        ("float32", float("-inf"), (2, 3), "float32"),
        ("float32", -3.75, (), "float32"),
        ("float32", 3.0, (0, 3), "float32"),
        ("bool", False, (2, 3), "float32"),
    ],
)
def test_creation_preserves_captured_dtype_shape_and_bits(kind, dtype, value, shape, default_dtype):
    torch = pytest.importorskip("torch")
    from tvm.relax.frontend.torch import from_exported_program

    class Creation(torch.nn.Module):
        def forward(self, x):
            kwargs = {} if dtype is None else {"dtype": getattr(torch, dtype)}
            return (
                torch.full(shape, value, **kwargs)
                if kind == "full"
                else torch.full_like(x, value, **kwargs)
            )

    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(getattr(torch, default_dtype))
        model = Creation()
        values = np.arange(np.prod(shape), dtype="int32").reshape(shape)
        example = torch.from_numpy(values)
        exported = torch.export.export(model, (example,))
        expected = model(example).numpy()
        graph = from_exported_program(exported, unwrap_unit_return_tuple=True)
    finally:
        torch.set_default_dtype(previous)
    vm = relax.VirtualMachine(relax.build(graph, "llvm"), tvm.cpu())
    source = tvm.nd.array(values)
    actual = vm["main"](source).numpy()
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    np.testing.assert_array_equal(
        actual.reshape(-1).view("uint8"), expected.reshape(-1).view("uint8")
    )
    np.testing.assert_array_equal(source.numpy(), values)
