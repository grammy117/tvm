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
"""Host limits must serialize work across threads and spare reserved memory."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import threading

import pytest
import tvm

sys.path.insert(0, str(Path(tvm.__file__).resolve().parents[2] / "apps/gemmini"))
import host_execution


def pool(tmp_path, *, cpus=1, builders=1, reserve=0):
    roster = sorted(os.sched_getaffinity(0))[:cpus]
    (tmp_path / "config.json").write_text(json.dumps(dict(
        cpus=roster, build_slots=builders, min_available_bytes=reserve)))
    return tmp_path


@pytest.mark.parametrize("kind,cpus", [("run", 1), ("build", 2)])
def test_shared_cpu_and_builder_admission(tmp_path, kind, cpus):
    root = pool(tmp_path, cpus=cpus)
    entered, release, second = threading.Event(), threading.Event(), threading.Event()
    original = os.sched_getaffinity(0)

    def first():
        with host_execution.execution_lease(root, kind):
            assert len(os.sched_getaffinity(0)) == 1
            entered.set()
            assert release.wait(5)
        assert os.sched_getaffinity(0) == original

    def following():
        assert entered.wait(5)
        with host_execution.execution_lease(root, kind):
            second.set()

    with ThreadPoolExecutor(2) as executor:
        a, b = executor.submit(first), executor.submit(following)
        assert entered.wait(5)
        assert not second.wait(0.1)
        release.set()
        a.result(timeout=5)
        b.result(timeout=5)
    assert second.is_set()


def test_memory_floor_blocks_then_admits(tmp_path, monkeypatch):
    root = pool(tmp_path, reserve=100)
    memory, entered = [99], threading.Event()
    monkeypatch.setattr(host_execution, "available_bytes", lambda: memory[0])

    def run():
        with host_execution.execution_lease(root, "run"):
            entered.set()

    with ThreadPoolExecutor(1) as executor:
        future = executor.submit(run)
        assert not entered.wait(0.1)
        memory[0] = 100
        future.result(timeout=5)
    assert entered.is_set()


def test_stop_refuses_and_exception_releases(tmp_path):
    root = pool(tmp_path)
    with pytest.raises(ValueError, match="candidate failed"):
        with host_execution.execution_lease(root, "build"):
            raise ValueError("candidate failed")
    with host_execution.execution_lease(root, "build"):
        pass
    (root / "STOP").touch()
    with pytest.raises(RuntimeError, match="stopped"):
        with host_execution.execution_lease(root, "run"):
            pytest.fail("stopped pool admitted a measurement")


def test_expired_deadline_refuses_admission(tmp_path):
    root = pool(tmp_path)
    config = json.loads((root / "config.json").read_text())
    config["deadline_epoch"] = 1
    (root / "config.json").write_text(json.dumps(config))
    with pytest.raises(RuntimeError, match="deadline reached"):
        with host_execution.execution_lease(root, "run"):
            pytest.fail("expired pool admitted work")
