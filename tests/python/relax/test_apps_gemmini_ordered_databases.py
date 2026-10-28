# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied. See the License for the
# specific language governing permissions and limitations
# under the License.
"""Partial current libraries retain explicit stock first-match fallbacks."""

from pathlib import Path
import sys

import pytest
import tvm
from tvm import meta_schedule as ms, tir

sys.path.insert(0, str(Path(tvm.__file__).resolve().parents[2] / "apps/gemmini"))
from compile_package import prepare_tuning_databases
from graph_plans import GraphPlan
from model_workloads import smoke
from tvm.relax.backend.contrib.gemmini_target import GemminiCapabilities


@pytest.fixture
def tasks():
    program, _ = smoke()
    prepared = GraphPlan(0, False).prepare(program)[0]
    caps = GemminiCapabilities()
    extracted = ms.relax_integration.extract_tasks(prepared, caps.target())
    assert len(extracted) >= 2
    return prepared, caps, extracted


def write_database(path, records, target):
    path.mkdir()
    database = ms.database.JSONDatabase(work_dir=str(path))
    for mod, cost in records:
        sch = tir.Schedule(mod)
        sch.annotate(sch.get_block("root"), "test.database_record", int(cost))
        database.commit_tuning_record(ms.database.TuningRecord(
            sch.trace, database.commit_workload(mod), [float(cost)], target))
    return path


def test_primary_precedes_cheaper_fallback_and_fallback_covers_other_tasks(tmp_path, tasks):
    prepared, caps, extracted = tasks
    first, second = (task.dispatched[0] for task in extracted[:2])
    primary = write_database(tmp_path / "primary", [(first, 100)], caps.target())
    fallback = write_database(tmp_path / "fallback", [(first, 1), (second, 2)], caps.target())
    output = tmp_path / "output"
    output.mkdir()
    database, selection = prepare_tuning_databases(primary, fallback, output, prepared)
    assert float(database.query_tuning_record(first, caps.target(), "main").run_secs[0]) == 100
    assert float(database.query_tuning_record(second, caps.target(), "main").run_secs[0]) == 2
    roles = {row["task"]: row["source"] for row in selection["tasks"]}
    assert roles[extracted[0].task_name] == "primary"
    assert roles[extracted[1].task_name] == "fallback"
    assert not selection["cross_database_cost_comparison"]
    assert not selection["fallback_labels_promoted_to_current_measurements"]
    assert (output / "fallback-database/database_tuning_record.json").read_bytes() == (
        fallback / "database_tuning_record.json").read_bytes()


def test_fallback_cannot_hide_zero_primary_matches(tmp_path, tasks):
    prepared, caps, extracted = tasks
    first = extracted[0].dispatched[0]
    unrelated = tvm.IRModule({"main": first["main"].with_attr("test.unrelated", True)})
    primary = write_database(tmp_path / "primary", [(unrelated, 100)], caps.target())
    fallback = write_database(tmp_path / "fallback", [(first, 1)], caps.target())
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(ValueError, match="Primary tuning database does not match"):
        prepare_tuning_databases(primary, fallback, output, prepared)


@pytest.mark.parametrize("role", ["primary", "fallback"])
def test_both_database_targets_are_validated(tmp_path, tasks, role):
    prepared, caps, extracted = tasks
    first = extracted[0].dispatched[0]
    paths = {name: write_database(tmp_path / name, [(first, 1)],
                                 tvm.target.Target("llvm") if name == role else caps.target())
             for name in ("primary", "fallback")}
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(ValueError, match=role.title() + " tuning database target"):
        prepare_tuning_databases(paths["primary"], paths["fallback"], output, prepared)


def test_fallback_requires_explicit_primary(tmp_path):
    with pytest.raises(ValueError, match="requires a primary"):
        prepare_tuning_databases(None, tmp_path / "fallback", tmp_path, tvm.IRModule())


def test_unmatched_fallback_trace_must_still_replay_against_its_workload(tmp_path, tasks):
    prepared, caps, extracted = tasks
    first, second = (task.dispatched[0] for task in extracted[:2])
    primary = write_database(tmp_path / "primary", [(first, 1)], caps.target())
    fallback = tmp_path / "fallback"
    fallback.mkdir()
    database = ms.database.JSONDatabase(work_dir=str(fallback))
    unrelated = tir.PrimFunc([], tir.BlockRealize([], True, tir.Block(
        [], [], [], "absent_from_stored_workload", tir.Evaluate(0))))
    schedule = tir.Schedule(tvm.IRModule({"main": unrelated}))
    schedule.get_block("absent_from_stored_workload")
    database.commit_tuning_record(ms.database.TuningRecord(
        schedule.trace, database.commit_workload(second), [1.0], caps.target()))
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(tvm.error.TVMError):
        prepare_tuning_databases(primary, fallback, output, prepared)
