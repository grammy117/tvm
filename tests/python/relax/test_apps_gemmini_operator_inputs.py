# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Retain actual constant domains and propagate declared mathematical inputs."""

from pathlib import Path
import sys

import numpy as np
import pytest
import tvm
from tvm import relax, te
from tvm.relax.backend.contrib.gemmini_target import GemminiCapabilities
from tvm.relax.backend.contrib.gemmini_search import WORKLOAD_ATTR, workload_key

sys.path.insert(0, str(Path(tvm.__file__).resolve().parents[2] / "apps/gemmini"))
from operator_inputs import _checked, array_identity, check_integer_divisors, resolve_operator_inputs
from tune_models import extract_inventory


def test_scalar_constant_keeps_primfunc_abi_rank():
    scalar = _checked(np.array(127., dtype="float32"), (), "float32")
    assert scalar.shape == ()
    assert tvm.nd.array(scalar).shape == ()


def graph():
    x = np.arange(32, dtype="int8").reshape(2, 16) % 7
    first = np.eye(16, dtype="int8") * 2
    second = np.eye(16, dtype="int8") * 4
    bb = relax.BlockBuilder()
    incoming = relax.Var("activation", relax.TensorStructInfo(x.shape, "int8"))
    with bb.function("main", [incoming]):
        with bb.dataflow():
            value = incoming
            for weight in (first, second):
                product = bb.emit(relax.op.matmul(value, relax.const(weight), out_dtype="int32"))
                wide = bb.emit(relax.op.astype(product, "int64"))
                divided = bb.emit(relax.op.floor_divide(wide, relax.const(np.full((1, 16), 2, dtype="int64"), "int64")))
                clipped = bb.emit(relax.op.clip(divided, -127, 127))
                value = bb.emit(relax.op.astype(clipped, "int8"))
            output = bb.emit_output(value)
        bb.emit_func_output(output)
    prepared, tasks, _ = extract_inventory(bb.get(), GemminiCapabilities(), fusion_level=2)
    workloads = {str(task.dispatched[0]["main"].attrs[WORKLOAD_ATTR]): task.dispatched[0]["main"] for task in tasks}
    return prepared, workloads, x, first, second


def test_immutable_constant_domains_are_retained_without_boundary_synthesis():
    prepared, workloads, _, _, _ = graph()
    records = resolve_operator_inputs(prepared, workloads)
    assert records
    for record in records.values():
        assert record["input_overrides"]
        for array in record["input_overrides"].values():
            if array.dtype == np.int64:
                np.testing.assert_array_equal(array, np.full((1, 16), 2, dtype="int64"))
        assert all(not row["complete_inputs"] for row in record["provenance"]["occurrences"])
        assert all(item["origin"].startswith("immutable_prepared_constant")
                   for row in record["provenance"]["occurrences"] for item in row["inputs"])


def test_declared_boundaries_propagate_between_exact_mathematical_calls():
    prepared, workloads, x, first, second = graph()
    records = resolve_operator_inputs(prepared, workloads, [x])
    first_expected = np.clip((x.astype("int32") @ first.astype("int32")) // 2, -127, 127).astype("int8")
    second_expected = np.clip((first_expected.astype("int32") @ second.astype("int32")) // 2, -127, 127).astype("int8")
    occurrences = sorted((row for record in records.values() for row in record["provenance"]["occurrences"]),
                         key=lambda row: row["call_index"])
    assert len(occurrences) == 2
    assert occurrences[0]["outputs"][0] == array_identity(first_expected)
    assert occurrences[1]["outputs"][0] == array_identity(second_expected)
    consumer = next(item for item in occurrences[1]["inputs"] if item["parameter_index"] == 0)
    assert consumer["origin"] == "host_mathematical_producer_output"
    assert consumer["sha256"] == array_identity(first_expected)["sha256"]


def test_distinct_constant_occurrences_keep_representative_provenance():
    prepared, workloads, x, _, _ = graph()
    records = resolve_operator_inputs(prepared, workloads, [x])
    assert len(records) == 1  # Stock extraction deduplicates the mathematical task.
    provenance = next(iter(records.values()))["provenance"]
    assert provenance["different_constant_bindings"] is True
    assert provenance["selected_call_index"] == 0
    assert len(provenance["occurrences"]) == 2


def test_binding_uses_stock_extraction_normalization():
    prepared, workloads, x, _, _ = graph()
    for gv, func in list(prepared.functions.items()):
        if isinstance(func, tvm.tir.PrimFunc):
            prepared.update_func(gv, func.without_attr("tir.noalias"))
    assert resolve_operator_inputs(prepared, workloads, [x])


@pytest.mark.parametrize("value", [np.zeros((2, 16), "float32"), np.zeros((1, 16), "int8")])
def test_boundary_shape_and_dtype_are_not_coerced(value):
    prepared, workloads, _, _, _ = graph()
    with pytest.raises(ValueError, match="shape/dtype mismatch"):
        resolve_operator_inputs(prepared, workloads, [value])


def test_direct_input_zero_divisor_is_refused_before_native_host_execution():
    numerator = te.placeholder((4,), "int64", name="numerator")
    divisor = te.placeholder((4,), "int64", name="divisor")
    output = te.compute((4,), lambda i: numerator[i] // divisor[i], name="divide")
    func = te.create_prim_func([numerator, divisor, output])
    with pytest.raises(ValueError, match="Zero integer divisor.*divisor"):
        check_integer_divisors(func, [np.ones(4, "int64"), np.array([2, 2, 0, 2], "int64"), None])
    check_integer_divisors(func, [np.ones(4, "int64"), np.full(4, 2, "int64"), None])


def test_reshape_aliases_preserve_producer_values_and_parameter_abi():
    a = te.placeholder((4,), "int32", name="a")
    b = te.compute((4,), lambda i: a[i] + 1, name="increment")
    first = te.create_prim_func([a, b])
    a2 = te.placeholder((1, 4), "int32", name="a2")
    weights = te.placeholder((1, 4), "int32", name="weights")
    b2 = te.compute((1, 4), lambda i, j: a2[i, j] * weights[i, j], name="multiply")
    second = te.create_prim_func([a2, weights, b2])
    first = first.with_attr(WORKLOAD_ATTR, workload_key(first))
    second = second.with_attr(WORKLOAD_ATTR, workload_key(second))
    bb = relax.BlockBuilder()
    first_gv, second_gv = bb.add_func(first, "increment"), bb.add_func(second, "multiply")
    x = relax.Var("x", relax.TensorStructInfo((4,), "int32"))
    with bb.function("main", [x]):
        with bb.dataflow():
            produced = bb.emit(relax.call_tir(first_gv, (x,), relax.TensorStructInfo((4,), "int32")))
            view = bb.emit(relax.op.reshape(produced, (1, 4)))
            result = bb.emit(relax.call_tir(second_gv, (view, relax.const(np.full((1, 4), 2, "int32"))),
                                            relax.TensorStructInfo((1, 4), "int32")))
            output = bb.emit_output(result)
        bb.emit_func_output(output)
    normalize = tvm.get_global_func("tvm.meta_schedule.normalize_mod")
    workloads = {str(func.attrs[WORKLOAD_ATTR]): normalize(func.with_attr("global_symbol", "main"))["main"]
                 for func in (first, second)}
    records = resolve_operator_inputs(bb.get(), workloads, [np.arange(4, dtype="int32")])
    second_record = records[str(second.attrs[WORKLOAD_ATTR])]
    np.testing.assert_array_equal(second_record["input_overrides"][0], np.arange(1, 5, dtype="int32").reshape(1, 4))
    assert second_record["provenance"]["occurrences"][0]["inputs"][0]["origin"].endswith(":alias_view")
