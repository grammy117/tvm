# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied. See the License for the
# specific language governing permissions and limitations
# under the License.
"""Reference admission and memory boundaries of connected package timing."""

from pathlib import Path
import json
import sys

import numpy as np
import pytest
import tvm

sys.path.insert(0, str(Path(tvm.__file__).resolve().parents[2] / "apps/gemmini"))
from chunk_measure import check_array, layout_blobs


def test_chunk_reference_does_not_cast_or_admit_nonfinite_values():
    spec = {"shape": [2, 3], "dtype": "float32"}
    original = np.arange(6, dtype="float32").reshape(3, 2).T
    admitted = check_array(original, spec)
    assert admitted.flags.c_contiguous
    np.testing.assert_array_equal(admitted, original)
    for invalid in (original.astype("float64"), original.reshape(3, 2),
                    np.full((2, 3), np.inf, dtype="float32"),
                    np.full((2, 3), np.nan, dtype="float32")):
        with pytest.raises(ValueError):
            check_array(invalid, spec)


def test_chunk_blobs_preserve_tail_bytes_and_disjoint_guard_regions():
    arrays = [np.arange(n, dtype="int8") for n in (1, 63, 64, 65, 257)]
    offsets, blob = layout_blobs(arrays)
    for offset, value in zip(offsets, arrays):
        assert offset % 64 == 0
        assert blob[offset:offset + value.nbytes] == value.tobytes()
    outputs, storage = layout_blobs(arrays, guards=True)
    previous_end = 0
    for offset, value in zip(outputs, arrays):
        assert offset % 64 == 0
        assert offset - 64 >= previous_end
        previous_end = offset + value.nbytes + 64
    assert previous_end <= len(storage)


def test_outer_search_rejects_wrong_memory_failed_numerics_and_mixed_timing():
    from tune_chunk import select_winner
    def record(cycles, *, status="passed", confirmed=True, timing="fixed-warm"):
        return {"status": status, "measurement": {"cycles": cycles,
                "memory_confirmation": {"model_confirmed": confirmed}, "timing_policy": timing}}
    valid = record(100)
    trials = [record(1, status="spike_positive_failed"), record(2, confirmed=False),
              record(3, timing="legacy"), record(0), record(200), valid]
    assert select_winner(trials) is valid
    assert select_winner(trials[:4]) is None


def test_outer_database_binds_actual_bytes_and_compatible_compiler(tmp_path):
    from tune_chunk import bind_inner_database
    from verify_baremetal_cpu import identity
    names = ("database_workload.json", "database_tuning_record.json")
    for name in names:
        (tmp_path / name).write_text("[]\n")
    context = {"schema": "gemmini_measurement_context_v1", "memory_fingerprint": "memory",
               "capabilities": "caps", "target": "target", "timing_policy": "fixed-warm",
               "compiler": {"sha256": "compiler"},
               "database_files": [identity(tmp_path / name) for name in names]}
    path = tmp_path / "measurement-context.json"
    path.write_text(json.dumps(context))
    kwargs = dict(memory_fingerprint="memory", capabilities="caps", target="target",
                  compiler_sha256="compiler")
    assert bind_inner_database(tmp_path, **kwargs)["context"] == context
    for key in kwargs:
        with pytest.raises(ValueError):
            bind_inner_database(tmp_path, **dict(kwargs, **{key: "wrong"}))
    (tmp_path / names[1]).write_text("[123]\n")
    with pytest.raises(ValueError, match="bytes differ"):
        bind_inner_database(tmp_path, **kwargs)
