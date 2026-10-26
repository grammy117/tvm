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
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied. See the License for the
# specific language governing permissions and limitations
# under the License.
"""Per-task incumbent updates and measured-region guard bindings."""
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import pytest
import tvm

sys.path.insert(0, str(Path(tvm.__file__).resolve().parents[2] / "apps/gemmini"))
import cycle_watchdog
import measure
import tune_models


def test_live_binding_is_monotone_and_does_not_rewrite_elf(tmp_path):
    folder = tmp_path / "candidates/first"
    folder.mkdir(parents=True)
    elf = folder / "valid.elf"
    elf.write_bytes(b"immutable executable")
    artifact = dict(folder=str(folder), workload=str(tmp_path / "workloads/task-a"),
                    cycle_watchdog={"enabled": True}, elfs={"valid": {"path": str(elf)}})
    cycle_watchdog.bind_limit(artifact, None, 100, None)
    path = cycle_watchdog.cutoff_path(artifact)
    assert struct.unpack("<Q", path.read_bytes())[0] == 100
    cycle_watchdog.write_limit(path, 80)
    cycle_watchdog.bind_limit(artifact, None, 100, None)
    assert struct.unpack("<Q", path.read_bytes())[0] == 80
    assert elf.read_bytes() == b"immutable executable"
    other = dict(artifact, workload=str(tmp_path / "workloads/task-b"))
    assert cycle_watchdog.cutoff_path(other) != path


def test_moving_incumbent_is_per_workload_and_rejects_slower_result(tmp_path, monkeypatch):
    args = SimpleNamespace(work_dir=tmp_path, max_cycles=10000, cap_factor=1,
                           cpu_max_cycles=10000, rtl_seconds=10, moving_cutoff=True)
    campaign = tune_models.Campaign(args, None)
    limits = []

    def bind(artifact, _toolchain, limit, _identity):
        limits.append(limit)
        return artifact

    monkeypatch.setattr(tune_models, "bind_limit", bind)
    timings = iter([100, 500, 80, 90])
    monkeypatch.setattr(measure, "run_candidate", lambda artifact, *_a, **_kw:
                        dict(folder=artifact["folder"], status="passed", cycles=next(timings)))
    paths = []
    for index, key in enumerate(["task-a", "task-b", "task-a", "task-a"]):
        folder = tmp_path / str(index)
        folder.mkdir()
        path = folder / "artifact.json"
        path.write_text(json.dumps(dict(folder=str(folder), workload=key,
                                         program_sha256=str(index), native=True)))
        paths.append(path)
    assert [campaign.run(p) for p in paths[:3]] == [100, 500, 80]
    with pytest.raises(RuntimeError, match="pruned_by_newer_incumbent"):
        campaign.run(paths[3])
    assert limits == [0, 0, 100, 80]
    assert {k: v["cycles"] for k, v in campaign.incumbents.items()} == {"task-a": 80, "task-b": 500}
    refused = json.loads((paths[3].parent / "measurement.json").read_text())
    assert refused["observed_cycles"] == 90 and "cycles" not in refused


def test_watchdog_only_instruments_fixed_warm_and_uses_no_interrupt():
    with pytest.raises(ValueError, match="requires fixed-warm"):
        measure.render_runner({}, "legacy", watchdog=True)
    guarded = measure.render_runner({}, "fixed-warm", watchdog=True)
    assert "csrw mscratch" in guarded
    assert "dse_watch_finish(rtl)-begin" in guarded
    assert "mtimecmp" not in guarded and "interrupt(" not in guarded
    assert "csrw mscratch" not in measure.render_runner({}, "fixed-warm")
    with pytest.raises(ValueError, match="layout changed"):
        cycle_watchdog.instrument_runner("unexpected guest")


def test_fresh_baseline_database_can_commit_and_reopen(tmp_path):
    from tvm import te, tir, meta_schedule as ms
    x = te.placeholder((16,), "int32", "x")
    y = te.compute((16,), lambda i: x[i] + 1, name="y")
    mod = tvm.IRModule({"main": te.create_prim_func([x, y])})
    database = tune_models.validated_baseline_database(tmp_path / "new-campaign")
    assert tune_models.retain_measured_baseline(database, mod, tir.Schedule(mod).trace,
                                               {"status": "passed", "cycles": 123}, tvm.target.Target("llvm"))
    reopened = ms.database.JSONDatabase(work_dir=str(tmp_path / "new-campaign/validated-baselines"))
    assert len(reopened) == 1


def test_compact_space_keeps_real_choices_and_fixes_epilogue():
    import model_workloads
    from tvm import meta_schedule as ms
    from tvm.relax.backend.contrib.gemmini_target import GemminiCapabilities
    from tvm.relax.backend.contrib.gemmini_meta_schedule import gemmini_space_generator
    program, _ = model_workloads.build("smoke", operators=[1])
    caps = GemminiCapabilities()
    _, tasks, _ = tune_models.extract_inventory(program, caps)
    mod = tasks[0].dispatched[0]
    spaces = {}
    for compact in [False, True]:
        spaces[compact] = ms.TuneContext(mod=mod, target=caps.target(), num_threads=1,
                                        space_generator=gemmini_space_generator(caps, include_cpu=False, compact=compact)).generate_design_space()
    names = [str(s.mod["main"].body.block.annotations["gemmini.sketch"]) for s in spaces[True]]
    assert len(spaces[True]) < len(spaces[False])
    assert all(".eroot." in name for name in names)
    assert any(".os." in name for name in names) and any(".ws." in name for name in names)
    assert any(".p0." in name for name in names) and any(".p1." in name for name in names)
