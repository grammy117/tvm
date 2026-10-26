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
"""Measurement deduplication and CPU database placement on identical prepared IR."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import unittest

import numpy as np
import pytest
import tvm
from tvm import meta_schedule as ms, relax, tir
from tvm.relax.backend.contrib.gemmini import prepare_gemmini_graph
from tvm.relax.backend.contrib.gemmini_meta_schedule import gemmini_space_generator
from tvm.relax.backend.contrib.gemmini_target import GemminiCapabilities
from tvm.relax.backend.contrib.gemmini_tensorir import prepare_gemmini_tensorir_workloads
import test_backend_contrib_gemmini_tensorir as reference

sys.path.insert(0, str(Path(tvm.__file__).resolve().parents[2] / "apps/gemmini"))
from tune_models import Campaign
import measure
from retry_capped import retry


@pytest.mark.parametrize("status,cap,new_cap", [
    ("passed", None, 200),
    ("rtl_timeout_or_failure", 100, 100),
])
def test_cap_retry_rejects_success_and_unchanged_allowance(tmp_path, status, cap, new_cap):
    (tmp_path / "campaign.json").write_text("{}")
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}")
    (tmp_path / "measurement.json").write_text(json.dumps({"status": status, "cycle_cap": cap}))
    with pytest.raises(ValueError, match="increased allowance"):
        retry(tmp_path, artifact, max_cycles=new_cap)


def test_cap_retry_rejects_changed_executable(tmp_path):
    (tmp_path / "campaign.json").write_text("{}")
    elf = tmp_path / "valid.elf"
    elf.write_bytes(b"original executable")
    identity = measure.identity(elf)
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps({"elfs": {"valid": identity}}))
    (tmp_path / "measurement.json").write_text(json.dumps({
        "status": "rtl_timeout_or_failure", "cycle_cap": 100}))
    elf.write_bytes(b"replaced executable")
    with pytest.raises(ValueError, match="executable bytes changed"):
        retry(tmp_path, artifact, max_cycles=200)


def test_concurrent_duplicate_measurements_run_once(tmp_path, monkeypatch):
    args = SimpleNamespace(work_dir=tmp_path, max_cycles=1000, cap_factor=3, cpu_max_cycles=1000,
                           rtl_seconds=None)
    campaign = Campaign(args, None)
    calls = []
    def run(artifact, *_args, **_kwargs):
        calls.append(artifact["folder"])
        time.sleep(0.02)
        return {"folder": artifact["folder"], "status": "passed", "cycles": 17}
    monkeypatch.setattr(measure, "run_candidate", run)
    paths = []
    for index in range(4):
        folder = tmp_path / str(index)
        folder.mkdir()
        path = folder / "artifact.json"
        path.write_text(json.dumps({"folder": str(folder), "workload": "same-workload",
                                    "program_sha256": f"different-llvm-{index}", "native": True,
                                    "elfs": {"valid": {"sha256": "same-guest-bytes"}}}))
        paths.append(path)
    with ThreadPoolExecutor(4) as pool:
        assert list(pool.map(campaign.run, paths)) == [17] * 4
    assert len(calls) == 1
    restarted = Campaign(args, None)
    assert restarted.run(paths[0]) == 17
    assert len(calls) == 1
    changed = json.loads(paths[0].read_text())
    changed["elfs"]["valid"]["sha256"] = "new-input-or-wrapper-bytes"
    paths[0].write_text(json.dumps(changed))
    assert restarted.run(paths[0]) == 17
    assert len(calls) == 2


def test_cache_migration_prefers_a_successful_repaired_measurement(tmp_path):
    args = SimpleNamespace(work_dir=tmp_path, max_cycles=1000, cap_factor=3, cpu_max_cycles=1000)
    cache = {}
    for index, status in enumerate(("failed", "passed")):
        folder = tmp_path / str(index)
        folder.mkdir()
        (folder / "artifact.json").write_text(json.dumps({
            "folder": str(folder), "workload": "workload", "program_sha256": str(index),
            "elfs": {"valid": {"sha256": "same-elf"}}}))
        cache[str(index)] = {"status": status, "folder": str(folder), "cycle_cap": 500}
    # A later, expanded-cap repair supersedes the old cached refusal.
    (tmp_path / "1/measurement.json").write_text(json.dumps({
        "status": "passed", "folder": str(tmp_path / "1"), "cycles": 17, "cycle_cap": 1000}))
    (tmp_path / "measurement-cache.json").write_text(json.dumps(cache))
    campaign = Campaign(args, None)
    key = Campaign.measurement_key(json.loads((tmp_path / "1/artifact.json").read_text()))
    assert campaign.cache[key]["cycles"] == 17


class GemminiDSETests(unittest.TestCase):
    setUp = reference.GemminiTensorIRTests.setUp
    build = reference.GemminiTensorIRTests.build

    def test_cpu_database_winner_preserves_scheduled_code_and_math(self):
        a = relax.Var("a", relax.TensorStructInfo((17, 33), "int8"))
        b = relax.Var("b", relax.TensorStructInfo((33, 19), "int8"))
        builder = relax.BlockBuilder()
        with builder.function("main", [a, b]):
            with builder.dataflow():
                output = builder.emit_output(relax.op.matmul(a, b, out_dtype="int32"))
            builder.emit_func_output(output)
        program = builder.get()
        prepared, _, _ = prepare_gemmini_tensorir_workloads(program)
        caps = GemminiCapabilities()
        task = ms.relax_integration.extract_tasks(prepared, caps.target())[0]
        context = ms.TuneContext(mod=task.dispatched[0], target=caps.target(),
                                 space_generator=gemmini_space_generator(), num_threads=1)
        designs = context.generate_design_space()
        cpu = next(design for design in designs if "local.gemmini_" not in design.mod.script())
        cpu.enter_postproc()
        self.assertTrue(all(post.apply(cpu) for post in context.space_generator.postprocs))
        (self.root / "database").mkdir()
        database = ms.database.JSONDatabase(work_dir=str(self.root / "database"))
        database.commit_tuning_record(ms.database.TuningRecord(
            cpu.trace, database.commit_workload(task.dispatched[0]), [10.0], caps.target(), []))
        report = []
        compiled = prepare_gemmini_graph(program, database=database, report=report)
        self.assertEqual(report[0]["placement"], "cpu")
        self.assertEqual(report[0]["origin"], "TVM MetaSchedule database")
        function = next(func for func in compiled.functions.values() if isinstance(func, tir.PrimFunc))
        self.assertNotIn("tvm_gemmini", function.script())
        runtime, control = self.build(tvm.IRModule({"main": function.with_attr("global_symbol", "main")}))
        av = np.resize(np.array([-128, 127, -3, 0, 7], "int8"), (17, 33))
        bv = np.resize(np.array([19, -128, 127, 0], "int8"), (33, 19))
        out = tvm.nd.empty((17, 19), "int32")
        runtime(tvm.nd.array(av), tvm.nd.array(bv), out)
        np.testing.assert_array_equal(out.numpy(), av.astype("int64") @ bv.astype("int64"))
        self.assertEqual(control.emulator_commands(), 0)

    def test_bias_preload_can_be_disabled_without_changing_results(self):
        for preload in (False, True):
            a = relax.Var("a", relax.TensorStructInfo((3, 5), "int8"))
            b = relax.Var("b", relax.TensorStructInfo((5, 4), "int8"))
            bias = relax.Var("bias", relax.TensorStructInfo((4,), "int32"))
            builder = relax.BlockBuilder()
            with builder.function("main", [a, b, bias]):
                with builder.dataflow():
                    product = builder.emit(relax.op.matmul(a, b, out_dtype="int32"))
                    output = builder.emit_output(relax.op.add(product, bias))
                builder.emit_func_output(output)
            report = []
            graph = prepare_gemmini_graph(builder.get(), bias_preload=preload, report=report)
            selected = [row for row in report if row["placement"] == "gemmini"]
            self.assertEqual(len(selected), 1, report)
            self.assertEqual(bool(selected[0]["metadata"].get("bias_preloaded")), preload)
            self.assertEqual("mathematical_normalization" in selected[0], preload)
            self.index += 1
            path = self.root / f"bias-policy-{self.index}.so"
            relax.build(graph, "llvm").export_library(
                str(path), addons=[str(self.emulator)], cc="cc", options=["-std=c11"]
            )
            vm = relax.VirtualMachine(tvm.runtime.load_module(str(path)), tvm.cpu())
            av = np.arange(15, dtype="int8").reshape(3, 5) - 7
            bv = np.arange(20, dtype="int8").reshape(5, 4) - 9
            dv = np.array([-19, 23, 0, 1], "int32")
            np.testing.assert_array_equal(
                vm["main"](tvm.nd.array(av), tvm.nd.array(bv), tvm.nd.array(dv)).numpy(),
                av.astype("int64") @ bv.astype("int64") + dv,
            )
