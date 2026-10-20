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

"""Tests for bounded search machinery; synthetic labels imply no device speed."""

from dataclasses import replace
import json
import subprocess
import sys

import numpy as np
import pytest

from tvm.relax.backend.contrib.gemmini_tuning import BoundedGemminiSearch, TimingRecord


@pytest.fixture(scope="module")
def search():
    return BoundedGemminiSearch(64, 64, 32, seed=7, allow_diagnostic=True, compiler_id="test_compiler", adapter_id="test_adapter")


def test_legal_distinct_choices_and_semantic_features(search):
    assert {(c.schedule.metadata["tile_i"], c.schedule.metadata["tile_j"]) for c in search.candidates} == {(i, j) for i in (1, 2, 4) for j in (1, 2, 4)}
    assert len({c.candidate_id for c in search.candidates}) == 9
    features = search.features()
    assert all(f.shape == (4, 164) and np.all(np.isfinite(f)) for f in features)
    assert len({f.tobytes() for f in features}) == 9
    # Actual semantic TIR records less A traffic with wider J reuse, and less
    # B traffic with wider I reuse; the device module retains that mathematics.
    baseline, wider_j, wider_i = search.candidates[0], search.candidates[2], search.candidates[6]
    assert wider_j.schedule.metadata["load_a"] < baseline.schedule.metadata["load_a"]
    assert wider_i.schedule.metadata["load_b"] < baseline.schedule.metadata["load_b"]
    assert all(c.schedule.metadata["spad_rows"] <= 16384 and c.schedule.metadata["acc_rows"] <= 1024 for c in search.candidates)
    assert all("tvm_gemmini_compute" in c.schedule.scheduled_mod.script() for c in search.candidates)


def test_reproducible_identity_and_cold_order(search):
    repeated = BoundedGemminiSearch(*search.shape, seed=search.seed, compiler_id=search.compiler_id, adapter_id=search.adapter_id)
    assert [c.candidate_id for c in repeated.candidates] == [c.candidate_id for c in search.candidates]
    assert [r.candidate.candidate_id for r in repeated.rank()] == [r.candidate.candidate_id for r in search.rank()]
    assert all(r.origin == "untrained" and r.score == 0 for r in repeated.rank())
    assert BoundedGemminiSearch(64, 64, 33).workload_id != search.workload_id
    script = "import json; from tvm.relax.backend.contrib.gemmini_tuning import BoundedGemminiSearch; s=BoundedGemminiSearch(64,64,32,seed=7,compiler_id='test_compiler',adapter_id='test_adapter'); print(json.dumps([r.candidate.candidate_id for r in s.rank()]))"
    independent_order = json.loads(subprocess.check_output([sys.executable, "-c", script], text=True))
    assert independent_order == [r.candidate.candidate_id for r in search.rank()]


def test_semantic_tail_gate_does_not_imply_device_correctness():
    search = BoundedGemminiSearch(33, 17, 19, compiler_id="test_compiler", adapter_id="test_adapter")
    candidate = search.candidates[-1]
    assert search.validate_semantics(candidate.candidate_id)["passed"]
    assert candidate.candidate_id not in search.device_checks
    timing = TimingRecord(candidate.candidate_id, search.workload_id, "qualified_board", "hardware_seconds", (0.001, 0.002), "timer artifact", protocol_id="kernel_timer_v1", compiler_id=search.compiler_id, adapter_id=search.adapter_id)
    with pytest.raises(ValueError, match="device correctness"):
        search.update([timing])
    search.record_device_check(candidate.candidate_id, passed=False, evidence="target mismatch artifact", platform_id="qualified_board")
    with pytest.raises(ValueError, match="device correctness"):
        search.update([timing])
    assert not search.records


def test_reject_invalid_timing_batches(search):
    candidate = search.candidates[0]
    search.validate_semantics(candidate.candidate_id)
    search.record_device_check(candidate.candidate_id, passed=True, evidence="test-only external correctness attestation", platform_id="test_platform")
    valid = TimingRecord(candidate.candidate_id, search.workload_id, "test_platform", "hardware_seconds", (0.01, 0.02), "test-only timer artifact", protocol_id="kernel_timer_v1", compiler_id=search.compiler_id, adapter_id=search.adapter_id)
    invalid = [
        (replace(valid, candidate_id="stale"), "candidate identity"),
        (replace(valid, workload_id="other_workload"), "workload identity"),
        (replace(valid, platform_id=""), "platform identity"),
        (replace(valid, evidence=""), "measurement evidence"),
        (replace(valid, source="spike_walltime"), "Unsupported timing source"),
        (replace(valid, source="cycle_simulator_seconds"), "mix timing"),
        (replace(valid, platform_id="spike"), "Spike proxies"),
        (replace(valid, samples_seconds=(0.01,)), "two finite positive"),
        (replace(valid, samples_seconds=(0.01, np.nan)), "two finite positive"),
        (replace(valid, samples_seconds=(0.01, np.inf)), "two finite positive"),
        (replace(valid, samples_seconds=(0.01, 0)), "two finite positive"),
        (replace(valid, samples_seconds=(0.01, -1)), "two finite positive"),
        (replace(valid, platform_id="other"), "platform/build"),
        (replace(valid, protocol_id=None), "measurement protocol"),
        (replace(valid, protocol_id="different_scope"), "mix timing"),
        (replace(valid, compiler_id="different_build"), "compiler/adapter"),
        (replace(valid, adapter_id="different_adapter"), "compiler/adapter"),
    ]
    for record, message in invalid:
        with pytest.raises(ValueError, match=message):
            search.update([valid, record])
    assert search.model is None and not search.records
    with pytest.raises(ValueError, match="boolean result and evidence"):
        search.record_device_check(candidate.candidate_id, passed=True, evidence="", platform_id="test_platform")


def test_diagnostic_requires_opt_in():
    search = BoundedGemminiSearch(16, 16, 16)
    candidate = search.candidates[0]
    record = TimingRecord(candidate.candidate_id, search.workload_id, "diagnostic_fixture", "diagnostic_synthetic", (1.0, 1.0), "toy labels only")
    with pytest.raises(ValueError, match="diagnostic opt-in"):
        search.update([record])


def test_actual_xgb_update_and_reproducible_ranking(monkeypatch):
    pytest.importorskip("xgboost")
    search = BoundedGemminiSearch(64, 64, 32, seed=11, allow_diagnostic=True)
    # Toy labels test the isolated learned model only. They are not timings,
    # simulator estimates, a schedule-performance result or a paper winner.
    records = []
    for candidate in search.candidates:
        search.validate_semantics(candidate.candidate_id)
        product = candidate.schedule.metadata["tile_i"] * candidate.schedule.metadata["tile_j"]
        records.append(TimingRecord(candidate.candidate_id, search.workload_id, "diagnostic_fixture", "diagnostic_synthetic", (1 / product, 1.01 / product), "diagnostic toy labels; no device measurement"))
    search.update(records)
    ranked = search.rank()
    assert search.model.data_size == 9
    assert all(r.origin == "diagnostic" and np.isfinite(r.score) for r in ranked)
    assert np.ptp([r.score for r in ranked]) > 0.01
    assert [r.score for r in ranked] == sorted([r.score for r in ranked], reverse=True)
    # Apply a learned diagnostic proposal through the real graph pipeline.
    # This structural check does not turn toy labels into qualified timings.
    import tvm
    from tvm import relax, tir
    from tvm.relax.backend.contrib.gemmini import prepare_gemmini_graph_legacy as prepare_gemmini_graph

    proposal = ranked[0]
    selected = search.schedule_for(proposal.candidate.candidate_id)
    m, n, k = search.shape
    a = relax.Var("a", relax.TensorStructInfo((m, k), "int8"))
    b = relax.Var("b", relax.TensorStructInfo((k, n), "int8"))
    bias = relax.Var("bias", relax.TensorStructInfo((n,), "int32"))
    builder = relax.BlockBuilder()
    with builder.function("main", [a, b, bias]):
        with builder.dataflow():
            product = builder.emit(relax.op.matmul(a, b, out_dtype="int32"))
            result = builder.emit_output(relax.op.add(product, bias))
        builder.emit_func_output(result)
    prepared = prepare_gemmini_graph(builder.get(), selected.metadata["tile_i"], selected.metadata["tile_j"])
    assert relax.analysis.well_formed(prepared)
    device_functions = [func for func in prepared.functions.values() if isinstance(func, tir.PrimFunc) and func.attrs and "gemmini.m" in func.attrs]
    host_functions = [func for func in prepared.functions.values() if isinstance(func, tir.PrimFunc) and not (func.attrs and "gemmini.m" in func.attrs)]
    assert len(device_functions) == 1 and len(host_functions) == 1
    actual = device_functions[0].without_attr("global_symbol").without_attr("op_pattern")
    expected = selected.scheduled_mod["main"].without_attr("global_symbol").without_attr("op_pattern")
    tvm.ir.assert_structural_equal(actual, expected)
    assert proposal.origin == "diagnostic"
    assert proposal.candidate.candidate_id not in search.device_checks
    repeated = BoundedGemminiSearch(*search.shape, seed=11, allow_diagnostic=True)
    for candidate in repeated.candidates:
        repeated.validate_semantics(candidate.candidate_id)
    repeated.update(records)
    np.testing.assert_array_equal([r.score for r in repeated.rank()], [r.score for r in ranked])
    assert [r.candidate.candidate_id for r in repeated.rank()] == [r.candidate.candidate_id for r in ranked]
    # Another batch exercises cumulative updating, not a one-shot model fit.
    search.update([records[0]])
    assert search.model.data_size == 10
    assert len(search.records) == 10
    with pytest.raises(ValueError, match="mix timing"):
        search.update([replace(records[0], platform_id="other_diagnostic")])
    manifest = json.loads(json.dumps(search.manifest()))
    assert manifest["measurement_domain"] == ["diagnostic_fixture", "diagnostic_synthetic", None, None, None]
    assert len(manifest["timings"]) == 10
    assert all(c["device_check"] is None for c in manifest["candidates"])
    with pytest.raises(ValueError, match="device correctness"):
        search.schedule_for(records[0].candidate_id, require_device_check=True, platform_id="diagnostic_fixture")
    with pytest.raises(ValueError, match="compiler, adapter"):
        search.update([replace(records[0], source="hardware_seconds", protocol_id="timer_v1")])
    previous_model, previous_records, previous_domain = search.model, search.records, search.measurement_domain

    class FailedTraining:
        def update(self, *_args):
            self.partially_mutated = True
            raise RuntimeError("injected training failure")

    class InvalidPredictions:
        def __init__(self, scores):
            self.scores = scores

        def update(self, *_args):
            pass

        def predict(self, *_args):
            return self.scores

    monkeypatch.setattr(search, "_new_model", lambda: FailedTraining())
    with pytest.raises(RuntimeError, match="injected training failure"):
        search.update([records[0]])
    assert search.model is previous_model
    assert search.records == previous_records and search.measurement_domain == previous_domain
    for scores in (np.full(9, np.nan), np.full(9, np.inf), np.ones(1)):
        monkeypatch.setattr(search, "_new_model", lambda: InvalidPredictions(scores))
        with pytest.raises(ValueError, match="invalid or nonfinite"):
            search.update([records[0]])
        assert search.model is previous_model
        assert search.records == previous_records and search.measurement_domain == previous_domain
        search.model = InvalidPredictions(scores)
        with pytest.raises(ValueError, match="invalid or nonfinite"):
            search.rank()
        search.model = previous_model



def test_build_bindings_candidate_application_and_mutation():
    search = BoundedGemminiSearch(16, 16, 16, compiler_id="compiler_build_a", adapter_id="adapter_build_a")
    candidate = search.candidates[0]
    changed_compiler = BoundedGemminiSearch(16, 16, 16, compiler_id="compiler_build_b", adapter_id="adapter_build_a")
    changed_adapter = BoundedGemminiSearch(16, 16, 16, compiler_id="compiler_build_a", adapter_id="adapter_build_b")
    assert changed_compiler.workload_id == changed_adapter.workload_id == search.workload_id
    assert candidate.candidate_id != changed_compiler.candidates[0].candidate_id
    assert candidate.candidate_id != changed_adapter.candidates[0].candidate_id
    assert search.schedule_for(candidate.candidate_id) is candidate.schedule
    with pytest.raises(ValueError, match="semantic correctness"):
        search.schedule_for(candidate.candidate_id, require_device_check=True, platform_id="board_a")
    search.validate_semantics(candidate.candidate_id)
    search.record_device_check(candidate.candidate_id, passed=True, evidence="test-only device result", platform_id="board_a")
    assert search.schedule_for(candidate.candidate_id, require_device_check=True, platform_id="board_a") is candidate.schedule
    with pytest.raises(ValueError, match="platform/build"):
        search.schedule_for(candidate.candidate_id, require_device_check=True, platform_id="board_b")
    # External snapshots cannot forge/invalidate the actual stored CPU receipt.
    search.semantic_checks[candidate.candidate_id]["passed"] = False
    snapshot = search.manifest()
    snapshot["candidates"][0]["semantic_check"]["passed"] = False
    assert search.semantic_checks[candidate.candidate_id]["passed"]
    candidate.schedule.metadata["tile_i"] = 2
    for action in (lambda: search.schedule_for(candidate.candidate_id), search.features, search.rank, search.manifest):
        with pytest.raises(ValueError, match="contents changed"):
            action()
    candidate.schedule.metadata["tile_i"] = 1
    module = candidate.schedule.scheduled_mod
    original = module["main"]
    module.update_func(module.get_global_var("main"), original.with_attr("changed", True))
    with pytest.raises(ValueError, match="contents changed"):
        search.schedule_for(candidate.candidate_id)
    module.update_func(module.get_global_var("main"), original)
    assert search.schedule_for(candidate.candidate_id) is candidate.schedule


def test_semantic_failure_revokes_prior_gate(monkeypatch):
    search = BoundedGemminiSearch(16, 16, 16)
    candidate = search.candidates[0]
    search.validate_semantics(candidate.candidate_id)
    assert search.semantic_checks[candidate.candidate_id]["passed"]
    import tvm

    def failed_build(*_args, **_kwargs):
        raise RuntimeError("injected compiler failure")

    monkeypatch.setattr(tvm, "build", failed_build)
    with pytest.raises(RuntimeError, match="injected compiler failure"):
        search.validate_semantics(candidate.candidate_id)
    assert not search.semantic_checks[candidate.candidate_id]["passed"]


def test_model_workload_inventory_exposes_host_attention():
    import tvm
    from tvm import relax
    from tvm.relax.backend.contrib.gemmini_tuning import collect_gemmini_workloads
    a = relax.Var("a", relax.TensorStructInfo((2, 3, 17), "int8"))
    weights = relax.Var("weights", relax.TensorStructInfo((17, 19), "int8"))
    q = relax.Var("q", relax.TensorStructInfo((2, 3, 17), "float32"))
    k = relax.Var("k", relax.TensorStructInfo((2, 17, 3), "float32"))
    builder = relax.BlockBuilder()
    with builder.function("main", [a, weights, q, k]):
        with builder.dataflow():
            first = builder.emit(relax.op.matmul(a, weights, out_dtype="int32"))
            second = builder.emit(relax.op.matmul(a, weights, out_dtype="int32"))
            attention = builder.emit(relax.op.matmul(q, k))
            output = builder.emit_output(relax.Tuple([first, second, attention]))
        builder.emit_func_output(output)
    inventory = collect_gemmini_workloads(builder.get())
    assert inventory["workloads"] == [{"shape": [6, 19, 17], "sites": ["main:matmul:0", "main:matmul:1"]}]
    assert len(inventory["host_contractions"]) == 1
    assert inventory["execution_qualified"] is False
    image = relax.Var("image", relax.TensorStructInfo((1, 4, 5, 5), "int8"))
    grouped_weight = relax.Var("grouped_weight", relax.TensorStructInfo((4, 2, 3, 3), "int8"))
    builder = relax.BlockBuilder()
    with builder.function("main", [image, grouped_weight]):
        builder.emit_func_output(relax.op.nn.conv2d(image, grouped_weight, groups=2, out_dtype="int32"))
    grouped = collect_gemmini_workloads(builder.get())
    assert not grouped["workloads"] and len(grouped["host_contractions"]) == 1
    assert grouped["host_contractions"][0]["site"].startswith("main:nn.conv2d:")


def test_model_search_budget_identity_ingestion_and_atomic_failure():
    from tvm.relax.backend.contrib.gemmini_tuning import GemminiModelSearch
    inventory = {"workloads": [{"shape": [3, 5, 7], "sites": ["prefill:0"]}, {"shape": [1, 5, 7], "sites": ["decode:0"]}], "host_contractions": [], "execution_qualified": False}
    session = GemminiModelSearch(inventory, budget=3, compiler_id="compiler", adapter_id="adapter", seed=7, allow_diagnostic=True)
    second_search = tuple(session.searches.values())[1]
    original_rank = second_search.rank
    before = session.manifest()
    def fail_rank():
        raise RuntimeError("injected second-workload ranking failure")
    second_search.rank = fail_rank
    with pytest.raises(RuntimeError, match="ranking failure"):
        session.next_measurements(2)
    assert session.manifest() == before and session._cursor == 0
    second_search.rank = original_rank
    requests = session.next_measurements(2)
    assert [request["shape"] for request in requests] == [[3, 5, 7], [1, 5, 7]]
    search = session.search_for(requests[0]["workload_id"])
    record = TimingRecord(requests[0]["candidate_id"], search.workload_id, "fixture", "diagnostic_synthetic", (0.1, 0.11), "explicit synthetic machinery labels", compiler_id="compiler", adapter_id="adapter")
    before = session.manifest()
    with pytest.raises(ValueError, match="semantic correctness"):
        session.ingest(record)
    assert session.manifest() == before
    search.validate_semantics(record.candidate_id)
    session.ingest(record)
    assert session.measurement_domain[1] == "diagnostic_synthetic"
    with pytest.raises(ValueError, match="already has"):
        session.ingest(record)
    with pytest.raises(ValueError, match="requested model"):
        session.ingest(replace(record, candidate_id="unrequested"))
    requests[0]["candidate_id"] = "external mutation"
    assert session.manifest()["requests"][0]["candidate_id"] != "external mutation"
    assert len(session.next_measurements(10)) == 1
    assert session.next_measurements() == []
    assert session.manifest()["remaining_budget"] == 0
    assert session.manifest()["timing_qualified"] is False
    assert session.manifest()["performance_winner"] is False



def test_model_proposals_apply_per_shape_without_changing_semantics():
    import tvm
    from tvm import relax, tir
    from tvm.relax.backend.contrib.gemmini import prepare_gemmini_graph_legacy as prepare_gemmini_graph
    from tvm.relax.backend.contrib.gemmini_tuning import GemminiModelSearch, collect_gemmini_workloads
    a = relax.Var("a", relax.TensorStructInfo((3, 7), "int8"))
    b = relax.Var("b", relax.TensorStructInfo((1, 7), "int8"))
    weights = relax.const(np.arange(35, dtype="int8").reshape(7, 5))
    builder = relax.BlockBuilder()
    with builder.function("main", [a, b]):
        with builder.dataflow():
            first = builder.emit(relax.op.matmul(a, weights, out_dtype="int32"))
            second = builder.emit(relax.op.matmul(b, weights, out_dtype="int32"))
            output = builder.emit_output(relax.Tuple([first, second]))
        builder.emit_func_output(output)
    mod = builder.get()
    session = GemminiModelSearch(collect_gemmini_workloads(mod), budget=2, compiler_id="compiler", adapter_id="adapter")
    requests = session.next_measurements(2)
    choices = {tuple(request["shape"]): (request["tile_i"], request["tile_j"]) for request in requests}
    prepared = prepare_gemmini_graph(mod, tile_choices=choices)
    boundaries = [function for function in prepared.functions.values() if isinstance(function, tir.PrimFunc) and function.attrs and "gemmini.m" in function.attrs]
    assert len(boundaries) == 2
    for function in boundaries:
        shape = tuple(int(function.attrs["gemmini." + name]) for name in ("m", "n", "k"))
        assert (int(function.attrs["gemmini.tile_i"]), int(function.attrs["gemmini.tile_j"])) == choices[shape]
    for invalid in ({(3, 5, 7): (8, 1)}, {(True, 5, 7): (1, 1)}, {"shape": (1, 1)}):
        with pytest.raises(ValueError, match="Tile choices"):
            prepare_gemmini_graph(mod, tile_choices=invalid)


if __name__ == "__main__":
    import tvm.testing
    tvm.testing.main()
