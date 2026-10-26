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
"""Graph alternatives preserve arithmetic, change boundaries and replay exactly."""

import copy
from pathlib import Path
import sys

import numpy as np
import pytest
import tvm
from tvm import relax
from tvm.relax.backend.contrib.gemmini_tensorir import prepare_gemmini_tensorir_workloads
from tvm.relax.backend.contrib.gemmini_target import GemminiCapabilities

sys.path.insert(0, str(Path(tvm.__file__).resolve().parents[2] / "apps/gemmini"))
from graph_plans import GraphPlan, audit_plans, plans, replay


def graph():
    rng = np.random.default_rng(8)
    a = rng.integers(-3, 4, (2, 32), dtype="int8")
    w = rng.integers(-3, 4, (32, 48), dtype="int8")
    v = rng.integers(-3, 4, (48, 16), dtype="int8")
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo(a.shape, "int8"))
    with bb.function("main", [x]):
        with bb.dataflow():
            first = bb.emit(relax.op.matmul(x, relax.const(w), out_dtype="int32"))
            clipped = bb.emit(relax.op.clip(first, -127, 127))
            codes = bb.emit(relax.op.astype(clipped, "int8"))
            second = bb.emit(relax.op.matmul(codes, relax.const(v), out_dtype="int32"))
            result = bb.emit_output(relax.op.add(second, relax.const(7, "int32")))
        bb.emit_func_output(result)
    expected = np.clip(a.astype("int32") @ w.astype("int32"), -127, 127) @ v.astype("int32") + 7
    return bb.get(), a, expected


def call_count(mod):
    calls = []
    relax.analysis.post_order_visit(mod["main"].body,
                                   lambda e: calls.append(e) if isinstance(e, relax.Call) else None)
    return len(calls)


def test_plan_variants_preserve_connected_numerics_and_change_boundaries():
    mod, inputs, expected = graph()
    counts = {}
    identities = set()
    for plan in plans():
        trace = plan.trace(mod)
        prepared = trace.out_mod
        vm = relax.VirtualMachine(relax.build(prepared, target="llvm"), tvm.cpu())
        np.testing.assert_array_equal(vm["main"](tvm.nd.array(inputs)).numpy(), expected)
        counts[plan.name] = call_count(prepared)
        document = plan.identity(mod, prepared, trace=trace)
        identities.add(document["prepared_ir_sha256"])
        assert document["cross_function_sram_residency"] is False
    assert counts["fuse0-pack1"] > counts["fuse2-pack1"]
    assert len(identities) == 4


def test_graph_trace_replays_and_binds_the_actual_input_and_preparation():
    mod, _, _ = graph()
    plan = GraphPlan(0, False)
    trace = plan.trace(mod)
    document = plan.identity(mod, trace.out_mod, trace=trace)
    actual = replay(mod, document)
    tvm.ir.assert_structural_equal(actual.out_mod, trace.out_mod)
    loaded = tvm.ir.load_json(tvm.ir.save_json(mod))
    tvm.ir.assert_structural_equal(replay(loaded, document).out_mod, trace.out_mod)
    assert str(actual.decisions[0]) == plan.name
    bad = copy.deepcopy(document)
    bad["semantic_ir_sha256"] = "bad"
    with pytest.raises(ValueError, match="input identity"):
        replay(mod, bad)
    bad = copy.deepcopy(document)
    bad["prepared_ir_sha256"] = "bad"
    with pytest.raises(ValueError, match="prepared identity"):
        replay(mod, bad)


def test_existing_default_preparation_is_preserved():
    mod, _, _ = graph()
    with tvm.transform.PassContext(opt_level=2):
        existing, _, _ = prepare_gemmini_tensorir_workloads(mod)
        explicit, _, _ = GraphPlan().prepare(mod)
        tvm.ir.assert_structural_equal(existing, explicit)
        assert GraphPlan(-1).preparation_options["fusion_level"] == 2


def test_audit_collapses_inactive_choices_and_prefers_existing_default():
    mod, _, _ = graph()
    result = audit_plans(mod)
    assert result["configurations"] == 32
    assert result["distinct_prepared_graphs"] == 4
    assert result["groups"][0]["representative"] == GraphPlan().name
    assert sum(len(group["plans"]) for group in result["groups"]) == 32
    for field in ("bias_preload", "eliminate_common_subexpr"):
        assert result["option_effects"][field] == {
            "compared_pairs": 16, "changed_prepared_graph": 0,
            "identical_prepared_graph": 16}
    assert result["option_effects"]["prepack_weights"]["changed_prepared_graph"] == 16
    assert result["performance_measured"] is False


def test_audit_does_not_merge_different_programs_on_hash_collision(monkeypatch):
    mod, _, _ = graph()
    monkeypatch.setattr(tvm.ir, "structural_hash", lambda _: 0)
    assert audit_plans(mod, plans())["distinct_prepared_graphs"] == 4
    with pytest.raises(ValueError, match="distinct"):
        audit_plans(mod, [GraphPlan(), GraphPlan()])


@pytest.mark.parametrize("kwargs", [dict(fusion_level=True), dict(fusion_level=4),
                                  dict(prepack_weights=1), dict(bias_preload="yes"),
                                  dict(eliminate_common_subexpr=1)])
def test_invalid_plan_options_are_refused(kwargs):
    with pytest.raises(ValueError):
        GraphPlan(**kwargs)


def test_explicit_fusion_cannot_be_silently_disabled():
    mod, _, _ = graph()
    with pytest.raises(ValueError, match="requires optimize"):
        prepare_gemmini_tensorir_workloads(mod, optimize=False, fusion_level=0)


def test_extraction_and_trace_use_the_same_prepared_graph():
    mod, _, _ = graph()
    for plan in (GraphPlan(0, False), GraphPlan()):
        prepared, tasks, inventory = plan.extract(mod, GemminiCapabilities())
        tvm.ir.assert_structural_equal(prepared, plan.trace(mod).out_mod)
        assert len(tasks) == sum(row["eligible"] for row in inventory)


def test_cse_shares_quantization_as_a_replayable_graph_choice():
    from workload_chunks import Chain
    c = Chain()
    with c.bb.function("main"):
        with c.bb.dataflow():
            x = c.input("x", (2, 32))
            a = c.projection(x, 32, 48)
            b = c.projection(x, 32, 48)
            out = c.bb.emit_output(relax.Tuple([a, b]))
        c.bb.emit_func_output(out, params=c.params)
    mod = c.bb.get()
    baseline = GraphPlan()
    choice = GraphPlan(eliminate_common_subexpr=True)
    ordinary = baseline.prepare(mod)[0]
    prepared = choice.prepare(mod)[0]
    assert call_count(prepared) < call_count(ordinary)
    assert len(plans()) == 4 and len(plans(include_cse=True)) == 8
    document = choice.identity(mod, prepared)
    tvm.ir.assert_structural_equal(replay(mod, document).out_mod, prepared)
    inputs = np.random.default_rng(7).integers(-16, 17, (2, 32)).astype("float32") / 16
    outputs = []
    for program in (ordinary, prepared):
        vm = relax.VirtualMachine(relax.build(program, target="llvm"), tvm.cpu())
        outputs.append([a.numpy() for a in vm["main"](tvm.nd.array(inputs))])
    for before, after in zip(*outputs):
        np.testing.assert_array_equal(before, after)


def test_historical_three_argument_graph_choice_still_replays():
    from tvm.relax.transform.tuning_api import Choice, Knob, Trace
    mod, _, _ = graph()
    old = Trace(mod, [Knob("legacy", {"default": Choice(
        "relax.ext.gemmini.prepare_graph_plan", [2, 1, 1])})], ["default"])
    tvm.ir.assert_structural_equal(old.out_mod, GraphPlan().prepare(mod)[0])


def test_broad_graph_options_are_generic_and_preserve_integer_chain():
    mod, inputs, expected = graph()
    alternatives = plans(include_cse=True, fusion_levels=(0, 1, 2, 3),
                         bias_preload_choices=(True, False))
    assert len(alternatives) == 32
    for plan in alternatives:
        assert GraphPlan.from_name(plan.name) == plan
        prepared = plan.prepare(mod)[0]
        vm = relax.VirtualMachine(relax.build(prepared, target="llvm"), tvm.cpu())
        np.testing.assert_array_equal(vm["main"](tvm.nd.array(inputs)).numpy(), expected)
    for name in ("model-fast", "fuse2-pack2", "fuse2-pack1-cse1-bias0", "fuse2-pack1-cse1-cse1"):
        with pytest.raises(ValueError):
            GraphPlan.from_name(name)
