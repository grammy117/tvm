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
"""Host admission for external builders/runners; never selects TVM candidates.

Processes sharing a pool directory lease the same CPU locks. Configuration is
read on each admission, so reducing the roster stops new starts without killing
measurements. Time spent waiting is outside candidate measurement timeouts.
"""

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import threading
import time


def available_bytes():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Host admission requires Linux MemAvailable")


def _try_lock(path):
    handle = path.open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def _event(root, **fields):
    payload = dict(time=time.time(), pid=os.getpid(), thread=threading.get_ident(), **fields)
    # Each line is a single append write, shared by all tuning processes.
    with (root / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(payload) + "\n")


@contextmanager
def execution_lease(pool, kind):
    """Lease one host CPU and (for compilation) one limited builder slot."""
    if pool is None:
        yield
        return
    if kind not in ("build", "run"):
        raise ValueError("Unknown execution phase")
    root = Path(pool).resolve()
    started = time.monotonic()
    handles = []
    original = os.sched_getaffinity(0)
    cpu = None
    _event(root, event="waiting", kind=kind)
    try:
        while cpu is None:
            if (root / "STOP").exists():
                raise RuntimeError("Execution pool stopped by operator")
            with (root / "admission.lock").open("a+") as admission:
                fcntl.flock(admission, fcntl.LOCK_EX)
                config = json.loads((root / "config.json").read_text())
                if config.get("deadline_epoch") is not None and time.time() >= config["deadline_epoch"]:
                    raise RuntimeError("DSE wall deadline reached")
                roster = config["cpus"]
                if not roster or not set(roster).issubset(original):
                    raise ValueError("Execution CPU roster must be within the caller's affinity")
                if config["build_slots"] < 1 or config["min_available_bytes"] < 0:
                    raise ValueError("Invalid host admission limits")
                if available_bytes() >= config["min_available_bytes"]:
                    build = None
                    if kind == "build":
                        for index in range(config["build_slots"]):
                            build = _try_lock(root / f"build-{index}.lock")
                            if build is not None:
                                break
                    if kind == "run" or build is not None:
                        for selected in roster:
                            handle = _try_lock(root / f"cpu-{selected}.lock")
                            if handle is not None:
                                cpu = selected
                                handles = [handle] + ([build] if build is not None else [])
                                break
                        if cpu is None and build is not None:
                            build.close()
            if cpu is None:
                time.sleep(0.5)
        os.sched_setaffinity(0, {cpu})
        _event(root, event="started", kind=kind, cpu=cpu,
               wait_seconds=time.monotonic() - started, available_bytes=available_bytes())
        yield
    finally:
        os.sched_setaffinity(0, original)
        for handle in handles:
            handle.close()
        if cpu is not None:
            _event(root, event="finished", kind=kind, cpu=cpu)
