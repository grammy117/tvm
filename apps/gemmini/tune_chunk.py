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
"""Bounded outer search of compiler graph choices using connected-chunk cycles.

Each alternative is a stock Relax Knob/Choice/Trace. Inner choices are existing
TIR records, explicit ordered fallback traces or reported defaults; use
tune_models separately to improve those records. Historical costs never score
graph plans. This finite initial search
does not provide a cross-function SRAM ABI or full-model performance guarantee.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from chunk_measure import build_chunk
from compile_package import digest
from graph_plans import GraphPlan, audit_plans, plans
from measure import Toolchain, run_candidate
from program_io import import_program
from tuning_preset import PRESET_NAMES
from verify_baremetal_cpu import identity

APP = Path(__file__).resolve().parent


def select_winner(results):
    """Only correctness-passing, confirmed, fixed-policy whole-chunk labels win."""
    valid = [r for r in results if r.get("status") == "passed"
             and r.get("measurement", {}).get("memory_confirmation", {}).get("model_confirmed")
             and r["measurement"].get("timing_policy") == "fixed-warm"
             and r["measurement"].get("cycles", 0) > 0]
    return min(valid, key=lambda r: r["measurement"]["cycles"]) if valid else None


def bind_inner_database(database, *, memory_fingerprint, capabilities, target, compiler_sha256):
    """Refuse stale context, altered labels, or an incompatible measured library."""
    database = Path(database).resolve(strict=True)
    binding = json.loads((database / "measurement-context.json").read_text())
    expected = {"schema": "gemmini_measurement_context_v1", "memory_fingerprint": memory_fingerprint,
                "timing_policy": "fixed-warm", "capabilities": capabilities, "target": target}
    if any(binding.get(key) != value for key, value in expected.items()):
        raise ValueError("Inner database uses different memory/timing/capability/target settings")
    if binding.get("compiler", {}).get("sha256") != compiler_sha256:
        raise ValueError("Inner database uses a different native TVM compiler")
    required = {"database_workload.json", "database_tuning_record.json"}
    records = binding.get("database_files", [])
    if len(records) != 2 or {Path(item["path"]).name for item in records} != required:
        raise ValueError("Inner database context must bind both standard database files")
    for item in records:
        if digest(database / Path(item["path"]).name) != item["sha256"]:
            raise ValueError("Inner database bytes differ from their measured context")
    return {"path": str(database), "context": binding,
            "files": {name: digest(database / name) for name in sorted(required)},
            "context_identity": identity(database / "measurement-context.json")}


def bind_fallback_database(database):
    """Freeze compilation-only fallback traces without importing their costs."""
    database = Path(database).resolve(strict=True)
    files = {name: identity(database / name) for name in
             ("database_workload.json", "database_tuning_record.json")}
    return {"path": str(database), "files": files,
            "selection": "Stock OrderedUnionDatabase: primary match first, fallback only on a miss",
            "cost_policy": "Historical costs are not fresh labels or outer graph scores"}


def run(args):
    root = args.output_dir.resolve()
    toolchain = Toolchain.from_json(json.loads(args.toolchain.read_text()))
    provenance = toolchain.provenance()
    if toolchain.timing_policy != "fixed-warm" or not toolchain.memory_variant_receipt:
        raise ValueError("This sweep requires explicit FASED memory and fixed-warm timing")
    program, fixture, reference = [p.resolve() for p in (args.program, args.fixture, args.fixture_provenance)]
    choices = {name: GraphPlan.from_name(name) for name in args.plans}
    if len(choices) != len(args.plans):
        raise ValueError("Graph plans must be distinct")
    program_identity = identity(program)
    graph_audit = audit_plans(import_program(program), choices.values())
    if identity(program) != program_identity:
        raise ValueError("Program changed during graph-choice audit")
    representatives = [group["representative"] for group in graph_audit["groups"]]
    equivalent = {name: group["representative"] for group in graph_audit["groups"]
                  for name in group["plans"] if name != group["representative"]}
    document = {"schema": "gemmini_connected_graph_sweep_v1", "program": program_identity,
                "fixture": identity(fixture), "reference": identity(reference),
                "toolchain": provenance, "plans": args.plans,
                "graph_choice_audit": graph_audit,
                "cycle_limit": args.max_cycles, "rtl_wall_limit": args.rtl_seconds,
                "preset": args.tuning_preset, "tuning_database": None,
                "scope": "Connected chunk, fixed boundary fixture; no full-model or FireSim qualification",
                "cross_function_sram_residency": False}
    if args.tuning_database:
        import tvm
        from tvm.relax.backend.contrib.gemmini_target import GemminiCapabilities
        caps = GemminiCapabilities()
        document["tuning_database"] = bind_inner_database(
            args.tuning_database, memory_fingerprint=provenance["memory_model"]["fingerprint"],
            capabilities=caps.fingerprint, target=str(caps.target()),
            compiler_sha256=identity(tvm._ffi.base._LIB._name)["sha256"])
        document["preset"] = None
    fallback = getattr(args, "fallback_tuning_database", None)
    if fallback is not None:
        if not args.tuning_database:
            raise ValueError("A fallback database requires an explicit primary measured library")
        document["fallback_tuning_database"] = bind_fallback_database(fallback)
    if root.exists():
        if not args.resume or json.loads((root / "sweep.json").read_text()) != document:
            raise ValueError("Use a fresh output directory, or explicit resume with identical inputs/settings")
    else:
        root.mkdir(parents=True)
        (root / "sweep.json").write_text(json.dumps(document, indent=2) + "\n")
    results = []
    # Measure representatives first. Equivalent choices retain an explicit
    # unmeasured result rather than multiplying one observation into many costs.
    for name in [*representatives, *equivalent]:
        plan = choices[name]
        folder = root / name
        result_path = folder / "result.json"
        if result_path.exists():
            results.append(json.loads(result_path.read_text()))
            continue
        if folder.exists():
            raise ValueError("Incomplete retained trial; inspect its process/evidence before starting a fresh sweep: " + str(folder))
        folder.mkdir()
        if name in equivalent:
            result = {"plan": name, "status": "equivalent_prepared_graph",
                      "representative": equivalent[name], "performance_measured": False}
            result_path.write_text(json.dumps(result, indent=2) + "\n")
            results.append(result)
            winner = select_winner(results)
            (root / "results.json").write_text(json.dumps({"results": results, "winner": winner,
                "complete": len(results) == len(args.plans)}, indent=2) + "\n")
            print(json.dumps(result), flush=True)
            continue
        (folder / "started.json").write_text(json.dumps({"pid": os.getpid(), "start_unix": time.time()}) + "\n")
        result = {"plan": name, "preparation_options": plan.preparation_options,
                  "package": str(folder / "package"), "measurement_folder": str(folder / "measurement")}
        command = [sys.executable, str(APP / "compile_package.py"), "--program", str(program),
                   "--output-dir", str(folder / "package"), "--riscv-gcc", str(toolchain.gcc),
                   "--adapter-receipt", str(toolchain.adapter_receipt), "--tensor-memory-bytes", str(256 << 20),
                   "--graph-fusion-level", str(plan.effective_fusion_level)]
        if not plan.prepack_weights:
            command.append("--no-prepack-weights")
        if plan.eliminate_common_subexpr:
            command.append("--graph-cse")
        if not plan.bias_preload:
            command.append("--no-bias-preload")
        if args.tuning_database:
            command += ["--tuning-database", str(args.tuning_database.resolve())]
            if fallback is not None:
                if bind_fallback_database(fallback) != document["fallback_tuning_database"]:
                    raise ValueError("Fallback database changed during the sweep")
                command += ["--fallback-tuning-database", str(fallback.resolve())]
        elif args.tuning_preset:
            command += ["--tuning-preset", args.tuning_preset]
        result["compile_command"] = command
        started = time.monotonic()
        try:
            with (folder / "compile.log").open("w") as stream:
                process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                         timeout=args.build_seconds)
            if process.returncode:
                raise ValueError("Graph-plan compilation failed; see compile.log")
            result["graph_plan"] = json.loads((folder / "package/graph-plan.json").read_text())
            result["memory_audit"] = json.loads((folder / "package/memory-audit.json").read_text())
            artifact = build_chunk(folder / "package", fixture, reference, folder / "measurement", toolchain)
            measurement = run_candidate(artifact, toolchain, max_cycles=args.max_cycles,
                                        rtl_seconds=args.rtl_seconds)
            (folder / "measurement/measurement.json").write_text(json.dumps(measurement, indent=2) + "\n")
            result.update(status=measurement["status"], measurement=measurement)
        except Exception as error:  # preserve failed candidates, continue bounded alternatives
            result.update(status="failed", error=str(error))
            (folder / "failure.txt").write_text(traceback.format_exc())
        result["wall_seconds"] = time.monotonic() - started
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        results.append(result)
        winner = select_winner(results)
        (root / "results.json").write_text(json.dumps({"results": results, "winner": winner,
            "complete": len(results) == len(args.plans)}, indent=2) + "\n")
        print(json.dumps({"plan": name, "status": result["status"],
                          "cycles": result.get("measurement", {}).get("cycles")}), flush=True)
    winner = select_winner(results)
    print(json.dumps({"winner": None if winner is None else winner["plan"],
                      "complete": True, "passed_plans": sum(r["status"] == "passed" for r in results)}))
    return winner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("program", "fixture", "fixture-provenance", "output-dir", "toolchain"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--plans", nargs="+", default=[p.name for p in plans()],
                        help="Canonical fuse{0,1,2,3}-pack{0,1}[-bias0][-cse1] configurations")
    parser.add_argument("--max-cycles", type=int, default=100_000_000)
    parser.add_argument("--rtl-seconds", type=int, default=3600)
    parser.add_argument("--build-seconds", type=int, default=900)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--tuning-database", type=Path)
    parser.add_argument("--fallback-tuning-database", type=Path,
                        help="Frozen traces for primary-library misses; costs are not fresh labels")
    parser.add_argument("--tuning-preset", choices=(*PRESET_NAMES, "none"), default="rtl-v1",
                        help="Initial records, or none for an unseeded/default graph control")
    args = parser.parse_args()
    if args.tuning_preset == "none":
        args.tuning_preset = None
    if args.max_cycles < 1 or not 1 <= args.rtl_seconds <= 14400 or args.build_seconds < 1:
        parser.error("Positive bounded cycle/build/wall budgets required")
    if len(args.plans) != len(set(args.plans)):
        parser.error("Graph alternatives must be distinct")
    try:
        for name in args.plans:
            GraphPlan.from_name(name)
    except ValueError as error:
        parser.error(str(error))
    if run(args) is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
