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
"""Padding modes must retain their semantics through ordinary legalization."""

import numpy as np
import pytest

import tvm
from tvm import relax, tir


def pad_graph(shape, dtype, widths, mode):
    source = relax.Var("source", relax.TensorStructInfo(shape, dtype))
    builder = relax.BlockBuilder()
    with builder.function("main", [source]):
        with builder.dataflow():
            output = builder.emit_output(
                relax.op.nn.pad(source, tuple(x for pair in widths for x in pair), pad_mode=mode)
            )
        builder.emit_func_output(output)
    return builder.get()


@pytest.mark.parametrize("mode", ["reflect", "edge"])
@pytest.mark.parametrize("dtype", ["int8", "int32", "int64", "float32"])
@pytest.mark.parametrize(
    "shape,widths",
    [((4,), ((2, 1),)), ((3, 4), ((1, 2), (2, 1))), ((2, 3, 4), ((0, 0), (2, 1), (1, 3)))],
)
def test_pad_modes_execute_without_target_rules(shape, widths, dtype, mode):
    graph = pad_graph(shape, dtype, widths, mode)
    legalized = relax.transform.LegalizeOps()(graph)
    func = next(f for f in legalized.functions.values() if isinstance(f, tir.PrimFunc))
    if dtype == "float32":
        bits = np.array([0, 0x80000000, 0x7FC12345, 0x7F812345, 1, 0xFF800000], "uint32")
        values = np.resize(bits, shape).view("float32")
    else:
        values = (np.arange(np.prod(shape)).reshape(shape) * 7 - 101).astype(dtype)
    expected = np.pad(values, widths, mode=mode)
    output = tvm.nd.empty(expected.shape, dtype)
    runtime = tvm.build(func.with_attr("global_symbol", "main"), "llvm")
    source = tvm.nd.array(values)
    runtime(source, output)
    np.testing.assert_array_equal(output.numpy().view("uint8"), expected.view("uint8"))
    np.testing.assert_array_equal(source.numpy().view("uint8"), values.view("uint8"))


@pytest.mark.parametrize("mode", ["reflect", "edge"])
def test_padding_with_cropping(mode):
    values = np.arange(20, dtype="int32").reshape(4, 5)
    widths = ((-1, 1), (1, -1))
    expected = np.pad(values, ((0, 1), (1, 0)), mode=mode)[1:, :-1].copy()
    graph = relax.transform.LegalizeOps()(pad_graph(values.shape, "int32", widths, mode))
    func = next(f for f in graph.functions.values() if isinstance(f, tir.PrimFunc))
    output = tvm.nd.empty(expected.shape, "int32")
    tvm.build(func.with_attr("global_symbol", "main"), "llvm")(tvm.nd.array(values), output)
    np.testing.assert_array_equal(output.numpy(), expected)


def test_edge_padding_can_exceed_input_extent():
    values = np.arange(3, dtype="int32").reshape(1, 3)
    widths = ((3, 2), (4, 5))
    graph = relax.transform.LegalizeOps()(pad_graph(values.shape, "int32", widths, "edge"))
    func = next(f for f in graph.functions.values() if isinstance(f, tir.PrimFunc))
    expected = np.pad(values, widths, mode="edge")
    output = tvm.nd.empty(expected.shape, "int32")
    tvm.build(func.with_attr("global_symbol", "main"), "llvm")(tvm.nd.array(values), output)
    np.testing.assert_array_equal(output.numpy(), expected)


def test_unsafe_reflection_and_empty_edge_are_not_mislegalized():
    for shape, widths, mode in [((2,), ((2, 0),), "reflect"), ((0,), ((1, 1),), "edge")]:
        graph = pad_graph(shape, "float32", widths, mode)
        tvm.ir.assert_structural_equal(relax.transform.LegalizeOps()(graph), graph)
    n = tir.Var("n", "int64")
    graph = pad_graph((n,), "float32", ((1, 1),), "reflect")
    tvm.ir.assert_structural_equal(relax.transform.LegalizeOps()(graph), graph)


def test_unknown_padding_mode_is_rejected():
    with pytest.raises(ValueError, match="Unsupported relax.nn.pad mode"):
        relax.transform.LegalizeOps()(pad_graph((3,), "float32", ((1, 1),), "unknown"))


@pytest.mark.parametrize(
    "operator,spatial",
    [
        ("reflection_pad1d", 1),
        ("reflection_pad2d", 2),
        ("reflection_pad3d", 3),
        ("replication_pad1d", 1),
        ("replication_pad2d", 2),
        ("replication_pad3d", 3),
        ("constant_pad_nd", 2),
        ("pad", 2),
    ],
)
def test_exported_program_padding_uses_ordinary_relax(operator, spatial):
    torch = pytest.importorskip("torch")
    from tvm.relax.frontend.torch import from_exported_program

    class Padding(torch.nn.Module):
        def forward(self, x):
            op = getattr(torch.ops.aten, operator).default
            padding = tuple(
                width for axis in range(spatial) for width in ((1, 0), (0, 2), (2, 1))[axis]
            )
            if operator == "pad":
                return op(x, padding, "reflect")
            if operator == "constant_pad_nd":
                return op(x, padding, 3.25)
            return op(x, padding)

    shape = (1, 1) + (3,) * spatial
    values = np.arange(np.prod(shape), dtype="float32").reshape(shape) - 13
    model = Padding()
    example = torch.from_numpy(values)
    graph = from_exported_program(
        torch.export.export(model, (example,)), unwrap_unit_return_tuple=True
    )
    vm = relax.VirtualMachine(relax.build(graph, "llvm"), tvm.cpu())
    actual = vm["main"](tvm.nd.array(values)).numpy()
    np.testing.assert_array_equal(actual, model(example).numpy())
