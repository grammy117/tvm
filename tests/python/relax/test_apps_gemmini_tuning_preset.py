# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Portable preset integrity and stock structural replay across programs."""

import json
from pathlib import Path
import shutil
import sys

import pytest
import tvm

sys.path.insert(0, str(Path(tvm.__file__).resolve().parents[2] / "apps/gemmini"))
from tuning_preset import materialize_preset
from model_workloads import smoke, tinyllama
from tvm.relax.backend.contrib.gemmini import prepare_gemmini_graph
from tvm.relax.backend.contrib.gemmini_search import open_gemmini_database
from tvm.relax.backend.contrib.gemmini_target import GemminiCapabilities

PRESETS = Path(tvm.__file__).resolve().parents[2] / "apps/gemmini/presets"


def test_preset_integrity_and_target_are_checked_before_copy(tmp_path):
    caps = GemminiCapabilities()
    shutil.copytree(PRESETS, tmp_path / "source")
    source = tmp_path / "source"
    with pytest.raises(ValueError, match="capability mismatch"):
        materialize_preset("rtl-v1", tmp_path / "wrong", root=source,
                           capabilities=GemminiCapabilities(spad_rows=8192))
    payload = source / "rtl-v1/database_tuning_record.json"
    payload.write_text(payload.read_text() + "\n")
    with pytest.raises(ValueError, match="payload changed"):
        materialize_preset("rtl-v1", tmp_path / "bad", root=source, capabilities=caps)
    assert not (tmp_path / "bad").exists()


def test_union_preset_replays_smoke_but_does_not_match_new_prefill_shape(tmp_path):
    caps = GemminiCapabilities()
    folder = materialize_preset("rtl-v1", tmp_path / "portable", capabilities=caps)
    # No input PrimFuncs: database loading must register all stored families
    # itself, even in a fresh process compiling an unrelated program.
    database = open_gemmini_database(folder, tvm.IRModule(), capabilities=caps)
    assert len(database.get_all_tuning_records()) == 9
    manifest = json.loads((folder / "preset.json").read_text())
    assert len(manifest["records"]) == 9
    report = []
    mod, _ = smoke()
    prepare_gemmini_graph(mod, report=report, database=database)
    winners = [row for row in report if row.get("origin") == "TVM MetaSchedule database"]
    assert len(winners) == 2, report
    report = []
    mod, _ = tinyllama("prefill", chunk=16, operators=[1])
    prepare_gemmini_graph(mod, report=report, database=database)
    assert report
    assert not any(row.get("origin") == "TVM MetaSchedule database" for row in report)


def test_fased_preset_is_portable_and_binds_its_memory_model(tmp_path):
    caps = GemminiCapabilities()
    folder = materialize_preset("fased-v1", tmp_path / "fased", capabilities=caps)
    manifest = json.loads((folder / "preset.json").read_text())
    # Records must stay relocatable: provenance is hashes, never machine paths.
    assert "/home/" not in (folder / "preset.json").read_text()
    assert manifest["memory_model"]["knobs"] == {
        "max_reads": 16, "max_writes": 16, "read_latency": 30, "write_latency": 30}
    assert manifest["historical_costs_imported"] is False
    database = open_gemmini_database(folder, tvm.IRModule(), capabilities=caps)
    assert len(database.get_all_tuning_records()) == len(manifest["records"]) == 16
