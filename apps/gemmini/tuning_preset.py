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
"""Integrity-checked, portable stock MetaSchedule records for comparison runs."""

import hashlib
import json
from pathlib import Path
import shutil


def materialize_preset(name, destination, *, capabilities, root=None):
    """Copy reviewed records; stock TVM remains responsible for matching/replay."""
    if name != "rtl-v1":
        raise ValueError("Unknown tuning preset: " + name)
    source = (Path(root) if root is not None else Path(__file__).parent / "presets") / name
    manifest = json.loads((source / "preset.json").read_text())
    if manifest.get("schema") != "gemmini_tuning_preset_v1":
        raise ValueError("Unknown tuning preset schema")
    if manifest.get("capability_fingerprint") != capabilities.fingerprint:
        raise ValueError("Tuning preset capability mismatch")
    names = {"database_workload.json", "database_tuning_record.json"}
    if set(manifest["files"]) != names:
        raise ValueError("Tuning preset payload inventory mismatch")
    for name in sorted(names):
        data = (source / name).read_bytes()
        expected = manifest["files"][name]
        if len(data) != expected["bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
            raise ValueError("Tuning preset payload changed: " + name)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for name in sorted(names | {"preset.json"}):
        shutil.copy2(source / name, destination / name)
    return destination
