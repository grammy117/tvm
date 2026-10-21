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
"""Execute functional PyTorch scatter imports independently of target scheduling."""

import numpy as np
import pytest
import tvm
from tvm import relax


@pytest.mark.parametrize("dtype", ["int8", "int32", "float32"])
@pytest.mark.parametrize(
    "kind,axis,start,stop,step",
    [
        ("slice", 1, None, None, 2),
        ("slice", -1, -6, -1, 2),
        ("slice", 0, -99, 99, 1),
        ("slice", 1, 5, 2, 1),
        ("select", 1, -1, None, 1),
        ("select", -2, 0, None, 1),
    ],
)
def test_exported_program_scatter_matches_torch(dtype, kind, axis, start, stop, step):
    torch = pytest.importorskip("torch")
    from tvm.relax.frontend.torch import from_exported_program

    class Scatter(torch.nn.Module):
        def forward(self, base, updates):
            if kind == "select":
                return torch.select_scatter(base, updates, dim=axis, index=start)
            return torch.slice_scatter(base, updates, dim=axis, start=start, end=stop, step=step)

    shape = (3, 7)
    if kind == "select":
        update_shape = tuple(size for i, size in enumerate(shape) if i != axis % len(shape))
    else:
        update_shape = list(shape)
        bounds = slice(start, stop, step).indices(shape[axis])
        update_shape[axis] = len(range(*bounds))
    values = np.arange(np.prod(shape)).reshape(shape).astype(dtype)
    updates = (np.arange(np.prod(update_shape)).reshape(update_shape) * 17 - 128).astype(dtype)
    if dtype == "float32":
        updates = np.resize(
            np.array([0x80000000, 0x7F812345, 1, 0xFF800000], "uint32"), update_shape
        ).view("float32")
    model = Scatter()
    examples = (torch.from_numpy(values), torch.from_numpy(updates))
    graph = from_exported_program(
        torch.export.export(model, examples), unwrap_unit_return_tuple=True
    )
    vm = relax.VirtualMachine(relax.build(graph, "llvm"), tvm.cpu())
    a, b = tvm.nd.array(values), tvm.nd.array(updates)
    actual = vm["main"](a, b).numpy()
    expected = model(*examples).numpy()
    np.testing.assert_array_equal(actual.view("uint8"), expected.view("uint8"))
    np.testing.assert_array_equal(a.numpy().view("uint8"), values.view("uint8"))
    np.testing.assert_array_equal(b.numpy().view("uint8"), updates.view("uint8"))
