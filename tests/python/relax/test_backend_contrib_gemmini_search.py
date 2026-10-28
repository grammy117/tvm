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
"""Stock task tuning, candidate failure isolation, features and reusable records."""

import json
from pathlib import Path
import threading

import numpy as np
import pytest
import tvm
from tvm import meta_schedule as ms, tir
from tvm.relax.backend.contrib.gemmini_search import (
    AnnotateGemminiWorkloads, GemminiBuilder, GemminiRunner,
    MathematicalFeatures, candidate_workload, tune_gemmini_tasks,
)
from tvm.relax.backend.contrib.gemmini_meta_schedule import gemmini_space_generator
from tvm.relax.backend.contrib.gemmini_target import GemminiCapabilities
from test_backend_contrib_gemmini_sketch import matmul


def task():
    caps = GemminiCapabilities()
    mod = AnnotateGemminiWorkloads()(tvm.IRModule({"main": matmul(32, 32, 48)}))
    return ms.ExtractedTask("projection", mod, caps.target(), [mod], 7)


def test_builder_and_runner_isolate_failures():
    work = task()
    def build(mod, key):
        assert key == candidate_workload(work.mod)
        raise ValueError("intentional build refusal")
    result = GemminiBuilder(build).build([ms.builder.BuilderInput(work.mod, work.target)])[0]
    assert result.artifact_path is None
    assert "intentional build refusal" in result.error_msg
    def run(path):
        if path == "bad":
            raise ValueError("intentional run refusal")
        return 123
    futures = GemminiRunner(run).run([
        ms.runner.RunnerInput(name, "cpu", []) for name in ("good", "bad")
    ])
    assert float(futures[0].result().run_secs[0]) == 123
    assert "intentional run refusal" in futures[1].result().error_msg


@pytest.mark.parametrize("strategy", ["evolutionary", "replay"])
def test_stock_tuning_persists_queryable_records_and_failure_log(tmp_path, strategy):
    work = task()
    built, lock = [], threading.Lock()
    def build(mod, key):
        assert key == candidate_workload(work.mod)
        with lock:
            index = len(built)
            built.append(mod)
        if index == 1:
            raise ValueError("test build failure")
        return str(index)
    # Deliberately diagnostic labels; these tests make no hardware claim.
    def run(path):
        return 100 + int(path)
    database = tune_gemmini_tasks(
        [work], work_dir=tmp_path, builder=GemminiBuilder(build, max_workers=2),
        runner=GemminiRunner(run, max_workers=2), max_trials_global=12,
        num_trials_per_iter=3, strategy=strategy, log=tmp_path / "trials.jsonl",
    )
    rows = [json.loads(line) for line in (tmp_path / "trials.jsonl").read_text().splitlines()]
    assert len(rows) == 12
    assert sum(row["build_error"] is not None for row in rows) == 1
    assert sum(row["cycles"] is not None for row in rows) == 11
    reopened = ms.database.JSONDatabase(work_dir=str(tmp_path))
    assert len(reopened.get_all_tuning_records()) == 12
    assert sum(float(record.run_secs[0]) < 1e10 for record in reopened.get_all_tuning_records()) == 11
    winner = reopened.query_tuning_record(work.mod, work.target, "main")
    assert winner is not None
    replay = tir.Schedule(work.mod)
    winner.trace.apply_to_schedule(replay, remove_postproc=False)
    best = min((row for row in rows if row["cycles"] is not None), key=lambda row: row["cycles"])
    tvm.ir.assert_structural_equal(replay.mod, built[int(best["artifact"])])


def test_features_remain_finite_after_tensorization():
    work = task()
    context = ms.TuneContext(mod=work.mod, target=work.target,
                             space_generator=gemmini_space_generator(), num_threads=1)
    candidates = []
    for space in context.generate_design_space():
        sch = space.copy()
        sch.enter_postproc()
        if all(post.apply(sch) for post in context.space_generator.postprocs):
            candidates.append(ms.search_strategy.MeasureCandidate(sch, []))
        if len(candidates) == 4:
            break
    assert candidates
    features = MathematicalFeatures().extract_from(context, candidates)
    assert all(np.isfinite(feature.numpy()).all() and feature.shape[0] > 0 for feature in features)


def test_single_trial_batches_record_warmup_boundary_and_per_task_cap(tmp_path):
    work = task()
    database = tune_gemmini_tasks(
        [work], work_dir=tmp_path, builder=GemminiBuilder(lambda mod, key: key, max_workers=1),
        runner=GemminiRunner(lambda path: 123, max_workers=1), max_trials_global=9,
        max_trials_per_task=5, num_trials_per_iter=1, warmup=3, eps_greedy=0.0,
        strategy="evolutionary",
        compact=True, include_cpu=False, log=tmp_path / "trials.jsonl",
    )
    rows = [json.loads(line) for line in (tmp_path / "trials.jsonl").read_text().splitlines()]
    assert len(rows) == 5
    assert [row["training_samples_before_update"] for row in rows] == list(range(5))
    assert [row["learned_ranking_available"] for row in rows] == [False]*3 + [True]*2
    assert len(database.get_all_tuning_records()) == 5
    state = json.loads((tmp_path / "cost-model.json").read_text())
    assert state["training_samples"] == 5 and state["eps_greedy"] == 0


def test_replay_never_imports_trains_or_saves_a_learned_model(tmp_path, monkeypatch):
    import builtins
    original = builtins.__import__

    def no_xgb(name, *args, **kwargs):
        if name == "xgboost" or name.startswith("xgboost."):
            raise AssertionError("ReplayTrace attempted to import XGBoost")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_xgb)
    monkeypatch.setattr(ms.cost_model, "XGBModel", lambda **kw: pytest.fail("XGB constructed"))
    monkeypatch.setattr(ms.measure_callback, "UpdateCostModel", lambda: pytest.fail("model update"))
    database = tune_gemmini_tasks(
        [task()], work_dir=tmp_path, builder=GemminiBuilder(lambda mod, key: key, max_workers=1),
        runner=GemminiRunner(lambda path: 100, max_workers=1), max_trials_global=3,
        num_trials_per_iter=1, strategy="replay", log=tmp_path / "trials.jsonl",
        compact=True, include_cpu=False,
    )
    assert len(database.get_all_tuning_records()) == 3
    assert not (tmp_path / "cost_model.xgb").exists()
    state = json.loads((tmp_path / "cost-model.json").read_text())
    assert state["ml_enabled"] is False and state["training_samples"] == 0
    assert state["warmup_samples"] == 0
    assert all(not json.loads(row)["learned_ranking_available"]
               for row in (tmp_path / "trials.jsonl").read_text().splitlines())
