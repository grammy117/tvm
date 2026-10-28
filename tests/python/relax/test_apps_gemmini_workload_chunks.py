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
"""Chunk boundaries, workload identity, and honest SRAM/graph accounting."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import tvm
from tvm import relax

sys.path.insert(0, str(Path(tvm.__file__).resolve().parents[2] / "apps/gemmini"))
from workload_chunks import CASES, bindings, build_case, slice_program
from workload_tuning import digest, resource_summary, verify_case
from package_audit import audit_graph
from tvm.relax.backend.contrib.gemmini_target import GemminiCapabilities


def chain():
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo((2, 3), "float32"))
    with bb.function("main", [x]):
        with bb.dataflow():
            const = bb.emit(relax.const(np.ones((2, 3), dtype="float32")))
            a = bb.emit(relax.op.add(x, const), name_hint="residual")
            b = bb.emit(relax.op.multiply(a, relax.const(2., "float32")), name_hint="double")
            c = bb.emit(relax.op.add(a, b))
            out = bb.emit_output(c)
        bb.emit_func_output(out)
    return bb.get()


def test_slice_preserves_residual_liveout_and_clones_constant():
    mod, boundary = slice_program(chain(), 1, 3)
    assert boundary["live_inputs"] == ["x"]
    assert boundary["live_outputs"] == ["residual", "double"]
    assert len(mod["main"].params) == 1
    vm = relax.VirtualMachine(relax.build(mod, target="llvm"), tvm.cpu())
    x = np.arange(6, dtype="float32").reshape(2, 3)
    result = vm["main"](tvm.nd.array(x))
    np.testing.assert_array_equal(result[0].numpy(), x + 1)
    np.testing.assert_array_equal(result[1].numpy(), (x + 1) * 2)


def test_slice_exposes_unselected_producer_as_input():
    mod, boundary = slice_program(chain(), 2, 3)
    assert boundary["live_inputs"] == ["residual"]
    assert boundary["live_outputs"] == ["double"]
    assert [int(v) for v in mod["main"].params[0].struct_info.shape] == [2, 3]


@pytest.mark.parametrize("start,stop", [(-1, 2), (1, 1), (0, 99)])
def test_slice_refuses_invalid_intervals(start, stop):
    with pytest.raises(ValueError, match="interval"):
        slice_program(chain(), start, stop)


@pytest.mark.parametrize("name,shape,count", [
    ("tinyllama-decode-kv", [1, 2048], 1408),
    ("tinyllama-prefill-kv", [32, 2048], 44),
    ("smolvla-action-in", [50, 32], 10),
    ("resnet50-layer4-reduce", [1, 7, 7, 2048], 2),
])
def test_surrogates_keep_dimensions_but_execute_one_site(name, shape, count):
    mod, metadata = build_case(name)
    assert [int(v) for v in mod["main"].params[0].struct_info.shape] == shape
    assert len(bindings(mod)[-1].value.fields) == 1
    assert metadata["session_occurrences"] == count
    assert metadata["executed_sites"] == 1
    assert metadata["full_model_qualified"] is False


def test_wrapper_uses_total_replay_budget_and_compiles_its_database(tmp_path, monkeypatch):
    import workload_tuning as wrapper

    monkeypatch.setattr(wrapper, "verify_case", lambda root: None)
    commands = []
    monkeypatch.setattr(wrapper.subprocess, "run", lambda command, **kwargs: commands.append(command))
    common = ["--case-dir", str(tmp_path), "--gcc", "gcc", "--adapter-receipt", "adapter.json"]
    wrapper.main(["tune", *common, "--spike-receipt", "spike.json", "--hardware-receipt", "rtl.json"])
    command = commands[-1]
    assert command[command.index("--strategy") + 1] == "replay"
    assert command[command.index("--trials-total") + 1] == "25"
    assert "--trials-per-task" not in command
    database = tmp_path / "campaign/replay-database"
    database.mkdir(parents=True)
    with pytest.raises(ValueError, match="completed campaign"):
        wrapper.main(["compile", *common, "--variant", "tuned"])
    (tmp_path / "campaign/replay-campaign.json").write_text("{}")
    wrapper.main(["compile", *common, "--variant", "tuned"])
    command = commands[-1]
    assert command[command.index("--tuning-database") + 1] == str(database)


def test_current_knobs_are_bound_to_case_payload(tmp_path):
    caps = GemminiCapabilities()
    program = tmp_path / "program.json"
    program.write_text(tvm.ir.save_json(chain()))
    knobs = tmp_path / "current-knobs.json"
    knobs.write_text('{"matches": []}')
    (tmp_path / "case.json").write_text(json.dumps({
        "schema": "gemmini_workload_chunk_v1", "capability_fingerprint": caps.fingerprint,
        "program_sha256": digest(program),
        "files": {"program.json": digest(program), "current-knobs.json": digest(knobs)},
    }))
    verify_case(tmp_path)
    knobs.write_text('{"matches": [{"matched": true}]}')
    with pytest.raises(ValueError, match="chunk changed"):
        verify_case(tmp_path)


def test_sram_bytes_are_width_aware_and_graph_counts_calls_not_functions():
    caps = GemminiCapabilities()
    placement = [{"function": "device", "placement": "gemmini",
                  "metadata": {"spad_rows": 256, "acc_rows": 256}}]
    resources = resource_summary(placement, caps)
    row = resources["functions"][0]
    assert row["scratchpad_bytes"] == 4096
    assert row["accumulator_bytes"] == 16384
    assert resources["scratchpad_capacity_bytes"] == 262144
    assert resources["accumulator_capacity_bytes"] == 65536
    graph = {"intermediates": [{"name": "mid", "bytes": 128}], "memory_preflight": {},
             "calls": [{"symbol": "device", "inputs": ["x"], "outputs": ["mid"]},
                       {"symbol": "cpu", "inputs": ["mid"], "outputs": ["y"]},
                       {"symbol": "device", "inputs": ["mid", "mid"], "outputs": ["z"]}]}
    audit = audit_graph(graph, placement, caps)
    assert audit["calls_by_placement"] == {"gemmini": 2, "cpu": 1}
    assert audit["logical_edge_bytes"] == 256
    assert len(audit["host_materialized_edges"]) == 2
    assert audit["cross_function_sram_residency"] is False


@pytest.mark.parametrize("name", ["tinyllama-decode-mlp", "tinyllama-prefill-mlp",
                                 "smolvla-action-time", "resnet50-layer4-pair"])
def test_connected_surrogates_are_well_formed_with_real_dependencies(name):
    mod, metadata = build_case(name)
    assert relax.analysis.well_formed(mod)
    assert metadata["connected"] is True
    values = [b.value for b in bindings(mod)]
    calls = [v for v in values if isinstance(v, relax.Call)]
    if name.startswith("tinyllama"):
        assert [v.name_hint for v in mod["main"].params] == ["normalized_hidden", "residual"]
        assert sum(v.op.name == "relax.matmul" for v in calls) == 3
    if name.startswith("resnet"):
        convolutions = [v for v in calls if v.op.name == "relax.nn.conv2d"]
        assert len(convolutions) == 2
        assert convolutions[1].args[0] not in mod["main"].params


def test_slice_preserves_numerical_policy_attributes():
    mod = chain()
    mod.update_func(mod.get_global_var("main"), mod["main"].with_attr("merlin.fixed_order", 1).with_attr("num_input", 9))
    mod = mod.with_attr("host_math_policy", "portable_exp_trig")
    result, _ = slice_program(mod, 2, 3)
    assert int(result["main"].attrs["merlin.fixed_order"]) == 1
    assert int(result["main"].attrs["num_input"]) == 1
    assert str(result.attrs["host_math_policy"]) == "portable_exp_trig"


def test_slice_refuses_opaque_pure_packed_call():
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo((2, 3), "float32"))
    with bb.function("main", [x]):
        with bb.dataflow():
            y = bb.emit(relax.call_pure_packed("external", x, sinfo_args=x.struct_info))
            out = bb.emit_output(y)
        bb.emit_func_output(out)
    with pytest.raises(ValueError, match="opaque"):
        slice_program(bb.get(), 0, 1)
