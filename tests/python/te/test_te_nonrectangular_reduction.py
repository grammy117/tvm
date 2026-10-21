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
"""Independent execution checks for initialization across dependent TE domains."""

import numpy as np
import pytest
import tvm
from tvm import te, tir
from tvm.tir.schedule.testing import verify_trace_roundtrip


def execute(func, inputs, expected):
    assert tir.analysis.verify_well_formed(func)
    assert tir.analysis.verify_ssa(func)
    module = tvm.build(func, target="llvm")
    actual = [tvm.nd.array(np.full(value.shape, 77, dtype=value.dtype)) for value in expected]
    module(*[tvm.nd.array(value) for value in inputs], *actual)
    for value, reference in zip(actual, expected):
        np.testing.assert_array_equal(value.numpy(), reference)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("dtype", ["int32", "float32"])
@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
def test_mixed_constant_and_spatial_dependent_reductions(reverse, dtype, index_dtype):
    data = te.placeholder((7, 8, 3), dtype, "data")
    seed = te.placeholder((7,), dtype, "seed")

    def compute(i):
        variable = te.reduce_axis(tvm.ir.Range.from_min_extent(i % 2, i % 3 + 1), "variable")
        constant = te.reduce_axis((0, 3), "constant")
        reducer = te.comm_reducer(lambda a, b: a + b, lambda _: seed[i], name="seeded_sum")
        axes = [constant, variable] if reverse else [variable, constant]
        return reducer(data[i, variable, constant], axes)

    output = te.compute((7,), compute, name="dependent_sum")
    func = te.create_prim_func([data, seed, output], index_dtype_override=index_dtype)
    values = (np.arange(168).reshape(7, 8, 3) % 31 - 17).astype(dtype)
    seeds = (np.arange(7) * 13 - 29).astype(dtype)
    expected = np.array(
        [seeds[i] + values[i, i % 2 : i % 2 + i % 3 + 1, :].sum() for i in range(7)], dtype
    )
    saved = tvm.ir.save_json(func)
    execute(func, [values, seeds], [expected])
    schedule = tir.Schedule(func, debug_mask="all")
    init = schedule.get_block("dependent_sum_init")
    schedule.split(schedule.get_loops(init)[0], factors=[None, 2])
    verify_trace_roundtrip(schedule, tvm.IRModule({"main": func}))
    assert tvm.ir.save_json(func) == saved
    execute(schedule.mod["main"], [values, seeds], [expected])


@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
def test_nested_dependent_reductions_include_empty_domains(index_dtype):
    data = te.placeholder((7, 7, 7), "int32", "data")

    def compute(i):
        outer = te.reduce_axis((0, i), "outer")
        inner = te.reduce_axis((0, outer), "inner")
        return te.sum(data[i, outer, inner], [outer, inner])

    output = te.compute((7,), compute, name="triangular")
    func = te.create_prim_func([data, output], index_dtype_override=index_dtype)
    values = (np.arange(343).reshape(7, 7, 7) % 29 - 17).astype("int32")
    expected = np.array(
        [sum(int(values[i, j, k]) for j in range(i) for k in range(j)) for i in range(7)], "int32"
    )
    execute(func, [values], [expected])


@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
def test_scalar_nonzero_origin_nested_reduction(index_dtype):
    data = te.placeholder((6, 6), "int32", "data")
    outer = te.reduce_axis((2, 6), "outer")
    inner = te.reduce_axis((0, outer), "inner")
    output = te.compute((), lambda: te.sum(data[outer, inner], [outer, inner]), name="scalar")
    values = (np.arange(36).reshape(6, 6) - 17).astype("int32")
    expected = np.array(sum(int(values[j, k]) for j in range(2, 6) for k in range(j)), "int32")
    execute(
        te.create_prim_func([data, output], index_dtype_override=index_dtype), [values], [expected]
    )


@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
def test_tuple_reduction_initializes_each_output_once(index_dtype):
    data = te.placeholder((5, 5, 3), "int32", "data")

    def compute(i):
        variable = te.reduce_axis((0, i), "variable")
        constant = te.reduce_axis((0, 3), "constant")
        reducer = te.comm_reducer(
            lambda a, b: (a[0] + b[0], tir.max(a[1], b[1])),
            lambda dtype0, dtype1: (tir.const(0, dtype0), tir.min_value(dtype1)),
            name="sum_max",
        )
        return reducer(
            (data[i, variable, constant], data[i, variable, constant]), [variable, constant]
        )

    outputs = te.compute((5,), compute, name="tuple_sum_max")
    values = (np.arange(75).reshape(5, 5, 3) % 29 - 17).astype("int32")
    sums = np.array([values[i, :i, :].sum() for i in range(5)], "int32")
    maxima = np.array(
        [values[i, :i, :].max() if i else np.iinfo("int32").min for i in range(5)], "int32"
    )
    execute(
        te.create_prim_func([data, *outputs], index_dtype_override=index_dtype),
        [values],
        [sums, maxima],
    )


@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
def test_index_normalization_preserves_nonzero_point_domain_extents(index_dtype):
    data = te.placeholder((7, 8), "int32", "data")

    def compute(i):
        axis = te.reduce_axis((0, i % 3 + 1), "k")
        return te.sum(data[i, axis], axis)

    output = te.compute((7,), compute, name="variable_extent")
    func = te.create_prim_func([data, output], index_dtype_override=index_dtype)
    found = []

    def check(node):
        if isinstance(node, tir.Block):
            for axis in node.iter_vars:
                if isinstance(axis.dom.min, tir.Var):
                    found.append(axis)
                    assert tvm.arith.Analyzer().can_prove(axis.dom.extent == 1)

    tir.stmt_functor.post_order_visit(func.body, check)
    assert found
    values = (np.arange(56).reshape(7, 8) - 29).astype("int32")
    expected = np.array([values[i, : i % 3 + 1].sum() for i in range(7)], "int32")
    execute(func, [values], [expected])
