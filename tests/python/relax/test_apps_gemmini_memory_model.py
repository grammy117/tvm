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
"""Memory timing must be tied to the binary and actually confirmed at execution."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "apps/gemmini"))
from memory_model import bind_memory_model, confirm_memory_model, identity, runtime_environment


@pytest.fixture
def receipts(tmp_path):
    pytest.importorskip("merlin.targetgen.mem_perturb")
    base = tmp_path / "stock"
    base.write_bytes(b"qualified stock binary")
    variant = tmp_path / "variant"
    variant.write_bytes(b"variant binary")
    hardware = tmp_path / "hardware.json"
    hardware.write_text(json.dumps({"simulator": identity(base)}))
    receipt = tmp_path / "variant.json"
    receipt.write_text(json.dumps({
        "schema_version": "merlin.mem-perturb-engine.v1",
        "models": {"latency_pipe": "available"}, "binary": identity(variant),
        "model_sources": {"mm_latency_pipe.cc": "bound source digest"},
        "control_relink": {"performed": True, "digest": identity(base)["sha256"],
                           "base_on_disk_digest": identity(base)["sha256"]},
    }))
    registry = tmp_path / "pins.yaml"
    registry.write_text("""artifacts:
  board:
    path: bitstream.tar
    digest: "0000000000000000000000000000000000000000000000000000000000000000"
    memory_model:
      kind: fased_latency_bandwidth_pipe
      read_latency: 30
      write_latency: 30
      max_reads: 16
      max_writes: 16
""")
    return hardware, receipt, registry, variant


def test_variant_binds_base_and_pin_and_clears_inherited_overrides(receipts):
    hardware, receipt, registry, _ = receipts
    bound = bind_memory_model(hardware, receipt, "board", registry)
    assert bound["knobs"] == dict(read_latency=30, write_latency=30, max_reads=16, max_writes=16)
    assert bound["base_simulator"] != bound["simulator"]
    env = runtime_environment(bound, {"MERLIN_MEM_PERTURB_SEED": "4", "MERLIN_MEM_LATENCY_PIPE_READ_LATENCY": "1", "KEEP": "yes"})
    assert "MERLIN_MEM_PERTURB_SEED" not in env
    assert env["MERLIN_MEM_LATENCY_PIPE_READ_LATENCY"] == "30"
    assert env["KEEP"] == "yes"


def test_console_requires_every_channel_and_exact_parameters(receipts):
    hardware, receipt, registry, _ = receipts
    bound = bind_memory_model(hardware, receipt, "board", registry)
    line = "[mem-perturb] model=latency_pipe read_latency=30 write_latency=30 max_reads=16 max_writes=16\n"
    assert confirm_memory_model(bound, line * 2)["model_confirmed"]
    for text in ("DSE_PASS\n", line + "[mem-perturb] model=in_order (mm_magic_t)\n",
                 line.replace("read_latency=30", "read_latency=1"), line.replace("max_reads=16", "max_reads=invalid")):
        assert not confirm_memory_model(bound, text)["model_confirmed"]


def test_variant_changed_binary_or_wrong_base_is_refused(receipts):
    hardware, receipt, registry, binary = receipts
    original = binary.read_bytes()
    binary.write_bytes(b"changed")
    with pytest.raises(ValueError, match="binary changed"):
        bind_memory_model(hardware, receipt, "board", registry)
    binary.write_bytes(original)
    doc = json.loads(receipt.read_text())
    doc["control_relink"]["digest"] = "wrong"
    receipt.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="control relink"):
        bind_memory_model(hardware, receipt, "board", registry)


def test_registry_settings_change_measurement_identity(receipts):
    hardware, receipt, registry, _ = receipts
    original = bind_memory_model(hardware, receipt, "board", registry)
    registry.write_text(registry.read_text().replace("read_latency: 30", "read_latency: 31"))
    changed = bind_memory_model(hardware, receipt, "board", registry)
    assert original["fingerprint"] != changed["fingerprint"]
    assert changed["knobs"]["read_latency"] == 31


def test_stock_policy_has_no_merlin_dependency_and_clears_memory_env(tmp_path):
    binary = tmp_path / "stock"
    binary.write_bytes(b"stock")
    hardware = tmp_path / "hardware.json"
    hardware.write_text(json.dumps({"simulator": identity(binary)}))
    bound = bind_memory_model(hardware)
    assert bound["kind"] == "stock_mm_magic"
    assert runtime_environment(bound, {"MERLIN_MEM_LATENCY_PIPE": "1", "MERLIN_MEM_PERTURB_SEED": "2"}) == {}
    assert confirm_memory_model(bound, "DSE_PASS")["model_confirmed"]
    with pytest.raises(ValueError, match="together"):
        bind_memory_model(hardware, artifact="board")


def test_fixed_warm_policy_and_toolchain_serialization():
    from measure import Toolchain, render_runner
    toolchain = Toolchain(Path("gcc"), Path("adapter"), Path("spike"), Path("rtl"),
                          Path("variant"), "board", Path("registry"), "fixed-warm")
    assert Toolchain.from_json(toolchain.as_json()) == toolchain
    assert "first<UINT64_MAX" in render_runner({}, "fixed-warm")
    assert "first<400000" in render_runner({}, "legacy")
    with pytest.raises(ValueError, match="Unknown timing"):
        render_runner({}, "adaptive")


def test_measurement_cache_separates_memory_models():
    from tune_models import Campaign
    artifact = {"workload": "/workloads/example", "program_sha256": "program",
                "elfs": {"valid": {"sha256": "same elf"}}}
    stock = Campaign.measurement_key(artifact)
    artifact["memory_model"] = {"fingerprint": "FASED 30/30/16/16"}
    assert Campaign.measurement_key(artifact) != stock


@pytest.mark.parametrize("confirm,complete", [(False, True), (True, False), (True, True)])
def test_candidate_runner_refuses_incomplete_or_unconfirmed_timing(receipts, tmp_path, monkeypatch, confirm, complete):
    import measure
    from types import SimpleNamespace

    hardware, receipt, registry, _ = receipts
    bound = bind_memory_model(hardware, receipt, "board", registry)
    toolchain = SimpleNamespace(memory_configuration=lambda: bound, memory_variant_receipt=receipt,
                                timing_policy="fixed-warm")
    elf = tmp_path / "guest"
    elf.write_bytes(b"same bound executable")
    artifact = {"folder": str(tmp_path), "memory_model": bound, "timing_policy": "fixed-warm",
                "elfs": {"valid": identity(elf), "bad-oracle": identity(elf)}}
    spike = iter([(0, "DSE_PASS\n"), (22, "TENSORIR_NUMERICAL_FAILURE\n")])
    monkeypatch.setattr(measure, "_spike", lambda *args, **kwargs: next(spike))

    def run(command, **kwargs):
        assert kwargs["env"]["MERLIN_MEM_LATENCY_PIPE"] == "1"
        text = "DSE_CYCLES first=12 second=11\n"
        if complete:
            text += "DSE_PASS\nVerilog $finish\n"
        if confirm:
            text += "[mem-perturb] model=latency_pipe read_latency=30 write_latency=30 max_reads=16 max_writes=16\n"
        kwargs["stdout"].write(text.encode())
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(measure.subprocess, "run", run)
    result = measure.run_candidate(artifact, toolchain, max_cycles=1000, rtl_seconds=1)
    expected = "memory_model_not_confirmed" if not confirm else ("passed" if complete else "rtl_timeout_or_failure")
    assert result["status"] == expected
    if not complete:
        assert "cycles" not in result
    assert result["memory_confirmation"]["model_confirmed"] == confirm
    elf.write_bytes(b"changed after build")
    with pytest.raises(ValueError, match="ELF changed"):
        measure.run_candidate(artifact, toolchain, max_cycles=1000)


def test_retained_baseline_uses_only_fresh_successful_cost(tmp_path):
    import tvm
    from tvm import meta_schedule as ms, tir
    from tune_models import retain_measured_baseline

    mod = tvm.script.from_source('''
@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((16,), "int32"), B: T.Buffer((16,), "int32")):
        for i in range(16):
            with T.block("copy"):
                vi = T.axis.spatial(16, i)
                B[vi] = A[vi]
''')
    database = ms.database.JSONDatabase(work_dir=str(tmp_path))
    schedule = tir.Schedule(mod)
    target = tvm.target.Target("llvm")
    historical = ms.database.TuningRecord(schedule.trace, ms.database.Workload(mod), [1.0], target, [])
    assert not retain_measured_baseline(database, mod, historical.trace,
                                        {"status": "rtl_timeout_or_failure", "cycles": 1}, target)
    assert len(database.get_all_tuning_records()) == 0
    assert retain_measured_baseline(database, mod, historical.trace,
                                    {"status": "passed", "cycles": 123}, target)
    assert [float(x) for x in database.get_all_tuning_records()[0].run_secs] == [123.0]


def test_bound_divisor_is_preserved_and_invalid_bindings_refuse(tmp_path):
    import numpy as np
    import tvm
    import measure

    mod = tvm.script.from_source('''
@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((4,), "int64"), D: T.Buffer((4,), "int64"), B: T.Buffer((4,), "int64")):
        for i in range(4):
            with T.block("divide"):
                vi = T.axis.spatial(4, i)
                B[vi] = A[vi] // D[vi]
''')
    func = mod["main"]
    numerator = np.array([-2049, -1024, 1023, 2048], dtype="int64")
    divisor = np.full((4,), 1024, dtype="int64")
    workload = measure.prepare_workload(func, tmp_path / "bound", input_overrides={0: numerator, 1: divisor},
                                       input_provenance={"source": "immutable extracted divisor"})
    np.testing.assert_array_equal(workload.inputs[1], divisor)
    np.testing.assert_array_equal(workload.expected[0], numerator // divisor)
    assert set(workload.manifest["bound_inputs"]) == {"0", "1"}
    for invalid in ({2: divisor}, {-1: divisor}, {1: divisor.astype("int32")}, {1: divisor[:2]}):
        with pytest.raises(ValueError, match="Input binding"):
            measure.prepare_workload(func, tmp_path / "invalid", input_overrides=invalid)
    with pytest.raises(ValueError, match="Zero integer divisor"):
        measure.prepare_workload(func, tmp_path / "zero", input_overrides={0: numerator, 1: np.zeros_like(divisor)})


def test_retained_replay_copies_exact_elf_without_overwriting_failure(tmp_path):
    import measure
    from replay_candidate import clone_candidate

    original = tmp_path / "original"
    original.mkdir()
    (original / "valid.elf").write_bytes(b"exact retained guest")
    (original / "bad-oracle.elf").write_bytes(b"retained wrong oracle")
    failed = b'{"status":"rtl_timeout_or_failure","cycle_cap":4000000}\n'
    (original / "measurement.json").write_bytes(failed)
    (original / "rtl-timing.log").write_bytes(b"partial interval; no all-output pass")
    artifact = {"folder": str(original), "workload": "/source/workload",
                "elfs": {name: measure.identity(original / (name + ".elf")) for name in ("valid", "bad-oracle")}}
    path = original / "artifact.json"
    path.write_text(json.dumps(artifact))
    copied = clone_candidate(path, tmp_path / "fresh", tmp_path / "workload")
    assert copied["elfs"]["valid"]["sha256"] == artifact["elfs"]["valid"]["sha256"]
    assert (original / "measurement.json").read_bytes() == failed
    assert not (tmp_path / "fresh/measurement.json").exists()
    assert not (tmp_path / "fresh/rtl-timing.log").exists()
    (original / "valid.elf").write_bytes(b"mutated")
    with pytest.raises(ValueError, match="executable bytes changed"):
        clone_candidate(path, tmp_path / "refused", tmp_path / "workload")


def test_validation_allowance_changes_budget_without_becoming_a_cost_label():
    from types import SimpleNamespace
    from tune_models import Campaign

    campaign = object.__new__(Campaign)
    campaign.args = SimpleNamespace(max_cycles=10_000_000, cap_factor=3)
    campaign.default_cycles = {"large_output": 68_796}
    assert campaign.cap("large_output") == 300_000 + 6 * 68_796
    campaign.args.validation_overhead_cycles = 8_000_000
    assert campaign.cap("large_output") == 8_000_000 + 6 * 68_796
    campaign.args.max_cycles = 8_000_000
    assert campaign.cap("large_output") == 8_000_000
    assert campaign.default_cycles["large_output"] == 68_796


def test_replay_finalization_creates_database_and_uses_only_fresh_label(tmp_path):
    import tvm
    from tvm import meta_schedule as ms, tir
    from replay_candidate import commit_measurement

    mod = tvm.script.from_source('''
@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((8,), "int32"), B: T.Buffer((8,), "int32")):
        for i in range(8):
            with T.block("copy"):
                vi = T.axis.spatial(8, i)
                B[vi] = A[vi]
''')
    source = ms.database.TuningRecord(tir.Schedule(mod).trace, ms.database.Workload(mod),
                                      [1e10], tvm.target.Target("llvm"), [])
    path = tmp_path / "fresh" / "database"
    assert not path.exists()
    database = commit_measurement(path, source, {"status": "passed", "cycles": 68_796})
    assert path.is_dir()
    assert [float(row.run_secs[0]) for row in database.get_all_tuning_records()] == [68_796]
    assert len(commit_measurement(path, source, {"status": "passed", "cycles": 68_796}).get_all_tuning_records()) == 1
    failed = commit_measurement(tmp_path / "failed" / "database", source, {"status": "rtl_timeout_or_failure"})
    assert not failed.get_all_tuning_records()
