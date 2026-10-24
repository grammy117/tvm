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
"""Bind an opt-in memory variant without relabeling the base RTL qualification.

FASED knobs are read through Merlin's pinned_latency_pipe API. The variant's
control relink must match the qualified base binary, and every console memory
channel must confirm both the requested model and its exact parameters.
"""

import hashlib
import inspect
import json
import os
from pathlib import Path


def identity(path):
    path = Path(path).resolve(strict=True)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return {"path": str(path), "sha256": h.hexdigest(), "bytes": path.stat().st_size}


def bind_memory_model(hardware_receipt, variant_receipt=None, artifact=None, registry=None):
    """Return immutable identities and settings for a stock or FASED run."""
    if bool(variant_receipt) != bool(artifact) or (registry and not artifact):
        raise ValueError("Memory variant receipt and FireSim memory artifact must be supplied together")
    base = json.loads(Path(hardware_receipt).read_text())
    simulator = identity(base["simulator"]["path"])
    if simulator["sha256"] != base["simulator"]["sha256"]:
        raise ValueError("Base simulator differs from its hardware qualification")
    bound = {"schema": "tvm_verilator_memory_model_v1", "kind": "stock_mm_magic",
             "simulator": simulator, "base_simulator": simulator,
             "bindings": [identity(hardware_receipt), simulator, identity(__file__)], "knobs": {},
             "requires_console_confirmation": False,
             "scope": "Local RTL timing; not deployed FireSim qualification"}
    if variant_receipt:
        from merlin.common import provenance
        from merlin.targetgen import mem_perturb

        variant = json.loads(Path(variant_receipt).read_text())
        if variant.get("schema_version") != mem_perturb.SCHEMA:
            raise ValueError("Unknown memory variant receipt")
        if "latency_pipe" not in variant.get("models", {}):
            raise ValueError("Variant does not declare a latency-pipe implementation")
        control = variant.get("control_relink", {})
        if (not control.get("performed") or control.get("digest") != simulator["sha256"]
                or control.get("base_on_disk_digest") != simulator["sha256"]):
            raise ValueError("Variant control relink does not match the qualified base simulator")
        binary = identity(variant["binary"]["path"])
        if binary["sha256"] != variant["binary"]["sha256"]:
            raise ValueError("Memory variant binary changed")
        registry = Path(registry) if registry else provenance.pins_path()
        knobs = mem_perturb.pinned_latency_pipe(artifact, registry=registry)
        pin = provenance.load_artifacts(registry)[artifact]
        bound.update(kind=mem_perturb.FASED_LATENCY_PIPE, simulator=binary, knobs=knobs,
                     requires_console_confirmation=True,
                     firesim_artifact={"name": artifact, "digest": pin.digest,
                                       "declared_path": pin.path, "memory_model": pin.memory_model_dict()},
                     model_sources=variant["model_sources"],
                     deployment_limits=["runtime.conf inside bitstream tar not byte-verified",
                                        "same-ELF cycle agreement with FireSim not verified"])
        bound["bindings"] += [identity(variant_receipt), binary, identity(registry),
                              identity(inspect.getfile(mem_perturb)), identity(inspect.getfile(provenance))]
    stable = {key: value for key, value in bound.items() if key != "bindings"}
    # Paths are provenance as well as hashes; the current campaign cannot silently
    # switch registries, source helpers or qualification receipts.
    stable["bindings"] = bound["bindings"]
    bound["fingerprint"] = hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()
    return bound


def runtime_environment(bound, source=None):
    """Clear inherited model overrides, then apply only this bound policy."""
    source = os.environ if source is None else source
    env = {key: value for key, value in source.items()
           if not key.startswith(("MERLIN_MEM_PERTURB_", "MERLIN_MEM_LATENCY_PIPE"))}
    if bound["requires_console_confirmation"]:
        env["MERLIN_MEM_LATENCY_PIPE"] = "1"
        env.update({"MERLIN_MEM_LATENCY_PIPE_" + key.upper(): str(value)
                    for key, value in bound["knobs"].items()})
    return env


def confirm_memory_model(bound, console):
    """Stock engines need no factory line; variants must confirm every channel."""
    if not bound["requires_console_confirmation"]:
        return {"model_confirmed": True, "confirmation": "qualified stock binary; overrides cleared"}
    from merlin.targetgen import mem_perturb

    seen = mem_perturb.reported_models(console)
    rows = []
    for line in console.splitlines():
        _, separator, rest = line.partition(mem_perturb.MODEL_LINE)
        if separator:
            fields = rest.split()
            try:
                knobs = {key: int(value) for key, value in
                         (field.split("=", 1) for field in fields[1:])}
            except (ValueError, TypeError):
                knobs = {}
            rows.append(knobs)
    passed = bool(seen) and set(seen) == {"latency_pipe"} and all(row == bound["knobs"] for row in rows)
    return {"model_confirmed": passed, "models_reported": list(seen), "knobs_reported": rows,
            "memory_fingerprint": bound["fingerprint"]}


def add_memory_arguments(parser):
    parser.add_argument("--memory-variant-receipt", type=Path)
    parser.add_argument("--firesim-memory-artifact")
    parser.add_argument("--memory-pin-registry", type=Path)
