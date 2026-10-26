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
"""Live measured-region cutoff; the simulator observes cycle/mscratch CSRs.

CSR markers bracket only the second inference. The native monitor refreshes a
task's atomic cutoff file during simulation, so improvements tighten in-flight
jobs too. No target interrupt or periodic target polling perturbs the kernel.
"""
import json
from pathlib import Path
import struct


PROFILE = {"kind": "verilator_live_csr_cutoff_v1", "poll_evaluations": 64,
           "refresh_evaluations": 256, "spike_watchdog": False,
           "scope": "second fixed-warm inference; startup/warmup/validation excluded"}

C_SOURCE = r'''
#define DSE_WATCH_READY UINT64_C(0xd5eabc0011223344)
#define DSE_WATCH_DONE UINT64_C(0xd5eabc0044332211)
static void dse_watch_prepare(int rtl){
  if(rtl)asm volatile("csrw mscratch,%0"::"r"(DSE_WATCH_READY):"memory");
}
static uint64_t dse_watch_start(int rtl){
  uint64_t begin=cycles();
  if(rtl)asm volatile("csrw mscratch,%0"::"r"(begin):"memory");
  return begin;
}
static uint64_t dse_watch_finish(int rtl){
  if(rtl)asm volatile("csrw mscratch,%0"::"r"(DSE_WATCH_DONE):"memory");
  return cycles();
}
'''


def instrument_runner(runner):
    replacements = [
        ("int main(void){", C_SOURCE + "\nint main(void){"),
        ("int rtl=arch==ROCKET_MARCHID;", "int rtl=arch==ROCKET_MARCHID;dse_watch_prepare(rtl);"),
        ("if(first<WARM_LIMIT){begin=cycles();", "if(first<WARM_LIMIT){begin=dse_watch_start(rtl);"),
        ("second=cycles()-begin;}", "second=dse_watch_finish(rtl)-begin;}"),
    ]
    for old, new in replacements:
        if runner.count(old) != 1:
            raise ValueError("Guest timing layout changed; watchdog instrumentation refused")
        runner = runner.replace(old, new, 1)
    return runner


def cutoff_path(artifact):
    workload = Path(artifact["workload"])
    return workload.parent.parent / "cutoffs" / (workload.name + ".bin")


def write_limit(path, limit):
    if type(limit) is not int or not 0 <= limit < 2**64:
        raise ValueError("Cycle limit must be a uint64; zero bootstraps without an incumbent")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(struct.pack("<Q", limit))
    temporary.replace(path)


def bind_limit(artifact, _toolchain, limit, _identity):
    """Bind a live cutoff without modifying the ELF Spike will validate."""
    if not artifact.get("cycle_watchdog", {}).get("enabled"):
        return artifact
    path = cutoff_path(artifact)
    previous = struct.unpack("<Q", path.read_bytes())[0] if path.exists() else 0
    effective = min(previous, limit) if previous and limit else previous or limit
    write_limit(path, effective)
    artifact["cycle_watchdog"].update(limit_cycles=effective, cutoff_file=str(path))
    (Path(artifact["folder"]) / "artifact.json").write_text(json.dumps(artifact, indent=2) + "\n")
    return artifact
