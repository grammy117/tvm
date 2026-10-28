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
"""Operator reuse binds the trace and measured artifact, not just a cost set."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import tvm
from tvm import meta_schedule as ms, te, tir
from tvm.relax.backend.contrib.gemmini_search import WORKLOAD_ATTR, workload_key
from tvm.relax.backend.contrib.gemmini_target import GemminiCapabilities

sys.path.insert(0, str(Path(tvm.__file__).resolve().parents[2] / "apps/gemmini"))
import tuning_library as library


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture(autouse=True)
def isolated_console_parser(monkeypatch):
    # The production parser's channel/knob checks have dedicated memory tests.
    # Keep these database tests independent of an installed Merlin checkout.
    monkeypatch.setattr(library, "confirm_memory_model", lambda bound, text:
                        {"model_confirmed": "TEST_FASED_CONFIRMED" in text})


def campaign(root, choices=((1, 120),), *, fingerprint="fased-test", timing="fixed-warm", index_map=False):
    root.mkdir()
    a = te.placeholder((1, 16), "int8", name="a")
    b = te.placeholder((16, 16), "int8", name="b")
    k = te.reduce_axis((0, 16), name="k")
    c = te.compute((1, 16), lambda i, j: te.sum(a[i, k].astype("int32") * b[k, j].astype("int32"), axis=k), name="matmul")
    func = te.create_prim_func([a, b, c]).with_attr("global_symbol", "main")
    key = workload_key(func)
    func = func.with_attr(WORKLOAD_ATTR, key)
    mod = tvm.IRModule({"main": func})
    payload = tvm.ir.save_json(mod)
    for name in ("program.json", "prepared.json"):
        (root / name).write_text(payload)
    (root / "compiler.so").write_bytes(b"test compiler")
    compiler = library.identity(root / "compiler.so")
    memory = {"kind": "fased_latency_bandwidth_pipe", "fingerprint": fingerprint}
    caps = GemminiCapabilities()
    write(root / "campaign.json", {"capabilities": caps.fingerprint, "timing_policy": timing,
                                    "memory_model": memory, "compiler": compiler,
                                    "program_sha256": library.identity(root / "program.json")["sha256"],
                                    "prepared_sha256": library.module_digest(mod)})
    write(root / "graph-plan.json", {"prepared_ir_sha256": library.module_digest(mod)})
    workload_dir = root / "workloads" / key
    workload_dir.mkdir(parents=True)
    (workload_dir / "workload.json").write_text(payload)
    (root / "database").mkdir()
    database = ms.database.JSONDatabase(work_dir=str(root / "database"))
    for index, (choice, cycles) in enumerate(choices):
        schedule = tir.Schedule(mod)
        if index_map:
            schedule.transform_layout(schedule.get_block("matmul"), ("read", 0), lambda i, j: (i, j))
        schedule.annotate(schedule.get_block("root"), "test.choice", choice)
        folder = root / "candidates" / f"{index:05d}-{key}"
        (folder / "package").mkdir(parents=True)
        folder.with_suffix(".module.json").write_text(tvm.ir.save_json(schedule.mod))
        (folder / "runner.c").write_text("test runner")
        (folder / "package/operators.ll").write_text(f"test program {choice}")
        for label in ("valid", "bad-oracle"):
            (folder / f"{label}.elf").write_bytes(f"test executable {choice} {label}".encode())
        elfs = {label: library.identity(folder / f"{label}.elf") for label in ("valid", "bad-oracle")}
        artifact = {"folder": str(folder), "workload": str(workload_dir), "compiler": compiler,
                    "memory_model": memory, "timing_policy": timing, "elfs": elfs,
                    "runner_sha256": library.identity(folder / "runner.c")["sha256"],
                    "program_sha256": library.identity(folder / "package/operators.ll")["sha256"],
                    "adapter_sha256": "test adapter"}
        write(folder / "artifact.json", artifact)
        (folder / "spike-valid.log").write_text("DSE_PASS\n")
        (folder / "spike-negative.log").write_text("TENSORIR_NUMERICAL_FAILURE\n")
        (folder / "rtl-timing.log").write_text(
            f"TEST_FASED_CONFIRMED\nDSE_CYCLES first={cycles + 4} second={cycles}\nDSE_PASS\nVerilog $finish\n")
        measurement = {"status": "passed", "folder": str(folder), "cycles": cycles,
                       "first_call_cycles": cycles + 4, "warm_call_cycles": cycles,
                       "memory_model": memory, "timing_policy": timing, "rtl_elf": elfs["valid"],
                       "rtl_log": library.identity(folder / "rtl-timing.log"),
                       "memory_confirmation": {"model_confirmed": True}}
        write(folder / "measurement.json", measurement)
        database.commit_tuning_record(ms.database.TuningRecord(
            schedule.trace, database.commit_workload(mod), [float(cycles)], caps.target(), []))
    return root, mod


def test_merge_reuses_structures_and_retains_each_exact_measurement(tmp_path):
    first, mod = campaign(tmp_path / "first", ((1, 120),))
    second, _ = campaign(tmp_path / "second", ((2, 90),))
    output = tmp_path / "library"
    context = library.merge_campaigns([first, second], output)
    combined = ms.database.JSONDatabase(work_dir=str(output))
    assert len(combined.get_all_tuning_records()) == context["records"] == 2
    assert combined.has_workload(mod)
    assert float(combined.query_tuning_record(mod, GemminiCapabilities().target(), "main").run_secs[0]) == 90
    assert {row["cycles"] for row in context["record_provenance"]} == {90, 120}
    assert all(row["witnesses"][0]["elfs"]["valid"]["sha256"] for row in context["record_provenance"])
    assert context["historical_costs_imported"] is False
    with pytest.raises(ValueError, match="must be fresh"):
        library.merge_campaigns([first], output)


def test_replay_campaign_keeps_trace_evidence_and_publishes_beside_database(tmp_path):
    root = tmp_path / "replay"
    campaign(root)
    (root / "campaign.json").rename(root / "replay-campaign.json")
    (root / "database").rename(root / "replay-database")
    path = library.publish_campaign_context(root)
    assert path == root / "replay-database/measurement-context.json"
    context = json.loads(path.read_text())
    assert context["records"] == 1
    assert Path(context["campaign"]["path"]).name == "replay-campaign.json"
    (root / "campaign.json").write_text((root / "replay-campaign.json").read_text())
    with pytest.raises(ValueError, match="one completed campaign"):
        library.validate_campaign(root)


def test_a_cycle_from_another_trace_is_not_a_valid_witness(tmp_path):
    root, _ = campaign(tmp_path / "campaign", ((1, 120), (2, 90)))
    records = root / "database/database_tuning_record.json"
    rows = [json.loads(line) for line in records.read_text().splitlines()]
    rows[0][1][1] = [90]  # Exists in the campaign, but belongs to another trace.
    records.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="No fresh workload/trace/executable witness"):
        library.publish_campaign_context(root)


def test_trace_identity_handles_native_index_map_arguments(tmp_path):
    root, _ = campaign(tmp_path / "campaign", index_map=True)
    first = library.publish_campaign_context(root)
    assert library.publish_campaign_context(root).read_bytes() == first.read_bytes()


def test_baseline_cannot_borrow_another_executables_cost(tmp_path):
    root, _ = campaign(tmp_path / "campaign", ((1, 120), (2, 90)))
    artifact = sorted((root / "candidates").glob("*/artifact.json"))[0]
    key = json.loads(artifact.read_text())["workload"].split("/")[-1]
    write(root / "baselines" / key / "current.json",
          {"status": "passed", "cycles": 90, "artifact": str(artifact)})
    with pytest.raises(ValueError, match="Baseline lacks its fresh executable/cycle witness"):
        library.publish_campaign_context(root)


@pytest.mark.parametrize("relative", ["valid.elf", "runner.c", "package/operators.ll", "rtl-timing.log"])
def test_changed_measurement_bytes_are_refused(tmp_path, relative):
    root, _ = campaign(tmp_path / "campaign")
    candidate = next((root / "candidates").glob("*/artifact.json")).parent
    (candidate / relative).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Changed evidence|changed"):
        library.publish_campaign_context(root)


@pytest.mark.parametrize("field,value", [("fingerprint", "other-memory"), ("timing", "legacy")])
def test_incompatible_campaigns_do_not_create_a_library(tmp_path, field, value):
    first, _ = campaign(tmp_path / "first")
    second, _ = campaign(tmp_path / "second", **{field: value})
    output = tmp_path / "library"
    with pytest.raises(ValueError, match="Incompatible campaign memory|Only fixed-warm"):
        library.merge_campaigns([first, second], output)
    assert not output.exists()


def test_failed_records_are_preserved_as_omissions(tmp_path):
    root, mod = campaign(tmp_path / "campaign")
    database = ms.database.JSONDatabase(work_dir=str(root / "database"))
    database.commit_tuning_record(ms.database.TuningRecord(
        tir.Schedule(mod).trace, database.commit_workload(mod), [1e10], GemminiCapabilities().target(), []))
    write(root / "candidates/failed/measurement.json", {"status": "rtl_timeout_or_failure"})
    path = library.publish_campaign_context(root)
    context = json.loads(path.read_text())
    assert context["records"] == 1
    assert len(context["omitted_records"]) == len(context["failed_measurements"]) == 1
    assert library.publish_campaign_context(root) == path
    context["records"] = 7
    path.write_text(json.dumps(context))
    with pytest.raises(ValueError, match="immutable context"):
        library.publish_campaign_context(root)


def bind_fixture(root, mod):
    key = str(mod["main"].attrs[WORKLOAD_ATTR])
    workload = root / "workloads" / key
    x, w = np.ones((1, 16), "int8"), np.ones((16, 16), "int8")
    expected = x.astype("int32") @ w.astype("int32")
    blob = x.tobytes() + bytes(48) + w.tobytes()
    (workload / "inputs.bin").write_bytes(blob)
    (workload / "expected.bin").write_bytes(expected.tobytes())
    np.savez(root / "boundary.npz", input_0=x)
    (root / "binder.py").write_text("# fixture binder identity\n")
    bound = {"0": {"shape": list(x.shape), "dtype": "int8",
                   "sha256": __import__("hashlib").sha256(x.tobytes()).hexdigest()}}
    write(workload / "workload-manifest.json", {
        "outputs": [2], "buffers": [{"shape": list(array.shape), "dtype": array.dtype.name}
                                     for array in (x, w, expected)],
        "inputs_sha256": library.identity(workload / "inputs.bin")["sha256"],
        "expected_sha256": library.identity(workload / "expected.bin")["sha256"],
        "bound_inputs": bound, "input_provenance": {"fixture": "test"}})
    write(root / "operator-input-bindings.json", {
        "prepared_ir_sha256": library.module_digest(mod),
        "fixture": library.identity(root / "boundary.npz"),
        "binder_source": library.identity(root / "binder.py"),
        "workloads": {key: {"bound_inputs": bound, "provenance": {"fixture": "test"}}}})
    document = json.loads((root / "campaign.json").read_text())
    document["input_bindings"] = library.identity(root / "operator-input-bindings.json")
    write(root / "campaign.json", document)
    return workload


def test_bound_input_context_links_fixture_manifest_and_numerical_blobs(tmp_path):
    root, mod = campaign(tmp_path / "campaign")
    workload = bind_fixture(root, mod)
    _, context = library.validate_campaign(root)
    assert context["boundary_fixture"] == library.identity(root / "boundary.npz")
    witness = context["record_provenance"][0]["witnesses"][0]["workload_data"]
    assert witness["inputs"] == library.identity(workload / "inputs.bin")
    assert witness["expected"] == library.identity(workload / "expected.bin")
    assert witness["bound_inputs"]["0"]["shape"] == [1, 16]


@pytest.mark.parametrize("changed", ["boundary.npz", "binder.py", "operator-input-bindings.json", "inputs.bin", "expected.bin"])
def test_changed_bound_input_evidence_is_refused(tmp_path, changed):
    root, mod = campaign(tmp_path / "campaign")
    workload = bind_fixture(root, mod)
    target = workload / changed if changed.endswith(".bin") else root / changed
    target.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Changed evidence|bytes changed"):
        library.validate_campaign(root)
