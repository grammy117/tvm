#!/usr/bin/env python3
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
"""Prepare isolated per-workload chunks, then delegate to stock tuning/package tools.

Preparation does not run a search or simulator. Current knobs come from the
verified rtl-v1 records, matched structurally. A case gets its own immutable
program, database seed, source/capability identities, CPU exclusions and future
campaign directory. Connected packages measure the whole chunk, including glue.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import tvm
from tvm.relax.backend.contrib.gemmini import prepare_gemmini_graph
from tvm.relax.backend.contrib.gemmini_search import open_gemmini_database
from tvm.relax.backend.contrib.gemmini_target import GemminiCapabilities

from program_io import import_program
from tune_models import extract_inventory
from tuning_preset import materialize_preset
from workload_chunks import CASES, build_case, inventory, slice_program
from memory_model import add_memory_arguments

APP = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def prepare_case(mod, destination, provenance):
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=False)
    caps = GemminiCapabilities()
    program = root / "program.json"
    program.write_text(tvm.ir.save_json(mod))
    prepared, tasks, task_inventory = extract_inventory(mod, caps)
    write(root / "bindings.json", inventory(mod))
    (root / "prepared.json").write_text(tvm.ir.save_json(prepared))
    seed = materialize_preset("rtl-v1", root / "seed", capabilities=caps)
    database = open_gemmini_database(seed, prepared, capabilities=caps)
    matches = []
    for task in tasks:
        record = database.query_tuning_record(task.dispatched[0], caps.target(), "main")
        matches.append({"task": task.task_name, "matched": record is not None,
                        "historical_operator_cycles": [float(x) for x in record.run_secs] if record else None,
                        "trace": record.trace.as_json() if record else None})
    for variant, db in (("default", None), ("current", database)):
        placement = []
        prepare_gemmini_graph(mod, report=placement, database=db)
        write(root / f"placement-{variant}.json", placement)
        write(root / f"resources-{variant}.json", resource_summary(placement, caps))
    write(root / "extraction.json", {"tasks": task_inventory, "eligible_tasks": len(tasks),
                                    "total_tasks": len(task_inventory)})
    write(root / "current-knobs.json", {
        "preset": "rtl-v1", "matches": matches,
        "scope": "historical operator labels; not connected-chunk or FireSim timing",
        "fallback": "default schedule, unmeasured for this case",
    })
    files = {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*")) if p.is_file()}
    document = dict(provenance, schema="gemmini_workload_chunk_v1", program_sha256=digest(program),
                    capabilities=asdict(caps), capability_fingerprint=caps.fingerprint,
                    files=files, compiler_library={"path": tvm._ffi.base._LIB._name,
                                                  "sha256": digest(tvm._ffi.base._LIB._name)},
                    source_files={str(p.relative_to(APP)): digest(p) for p in sorted(APP.glob("*.py"))},
                    tuning_status="not_run", qualification_status="not_run",
                    optimization_scope="per prepared program and capability; no universal model knobs")
    write(root / "case.json", document)
    return document


def resource_summary(placement, caps):
    result = []
    for row in placement:
        metadata = row.get("metadata", {})
        spad, acc = metadata.get("spad_rows", 0), metadata.get("acc_rows", 0)
        if spad > caps.spad_rows or acc > caps.acc_rows:
            raise ValueError("Lowered allocation exceeds selected capacities")
        result.append({"function": row["function"], "placement": row["placement"],
                       "origin": row.get("origin"), "reason": row.get("reason"),
                       "scratchpad_bytes": spad * caps.dim * caps.input_bits // 8,
                       "accumulator_bytes": acc * caps.dim * caps.accumulator_bits // 8,
                       "scratchpad_fraction": spad / caps.spad_rows,
                       "accumulator_fraction": acc / caps.acc_rows,
                       "schedule": metadata.get("schedule_parameters"),
                       "host_private_buffers": metadata.get("host_private_buffers"),
                       "extra_completion_fences": metadata.get("completion_fences"),
                       "waited_host_buffers": metadata.get("waited_host_buffers")})
    return {"functions": result, "scope": "per-function peak SRAM; host workspace is in compiled graph.json",
            "scratchpad_capacity_bytes": caps.spad_rows * caps.dim * caps.input_bits // 8,
            "accumulator_capacity_bytes": caps.acc_rows * caps.dim * caps.accumulator_bits // 8,
            "cross_function_sram_residency": False}


def verify_case(root):
    root = Path(root).resolve()
    doc = json.loads((root / "case.json").read_text())
    if doc.get("schema") != "gemmini_workload_chunk_v1":
        raise ValueError("Unknown chunk manifest")
    if doc["capability_fingerprint"] != GemminiCapabilities().fingerprint:
        raise ValueError("Chunk capability mismatch")
    for name, expected in doc["files"].items():
        path = root / name
        if not path.resolve().is_relative_to(root) or path.is_symlink() or digest(path) != expected:
            raise ValueError("Prepared chunk changed: " + name)
    if doc["program_sha256"] != digest(root / "program.json"):
        raise ValueError("Chunk program identity mismatch")
    return doc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    prep = actions.add_parser("prepare", help="Prepare surrogates; no search or simulator")
    prep.add_argument("--cases", nargs="+", choices=sorted(CASES), default=list(CASES))
    prep.add_argument("--output-dir", type=Path, required=True)
    inv = actions.add_parser("inventory", help="List bindings of a supplied exact prepared program")
    inv.add_argument("--program", type=Path, required=True)
    inv.add_argument("--output", type=Path, required=True)
    cut = actions.add_parser("slice", help="Extract exact connected bindings from an actual program")
    cut.add_argument("--program", type=Path, required=True)
    cut.add_argument("--start", type=int, required=True)
    cut.add_argument("--stop", type=int, required=True)
    cut.add_argument("--workload", required=True)
    cut.add_argument("--stage-calls", type=int, required=True)
    cut.add_argument("--sites", type=int, required=True)
    cut.add_argument("--output-dir", type=Path, required=True)
    for action in ("compile", "tune"):
        run = actions.add_parser(action)
        run.add_argument("--case-dir", type=Path, required=True)
        run.add_argument("--gcc", type=Path, required=True)
        run.add_argument("--adapter-receipt", type=Path, required=True)
        run.add_argument("--graph-fusion-level", type=int, choices=(-1, 0, 1, 2, 3), default=-1)
        run.add_argument("--graph-cse", action="store_true")
        run.add_argument("--no-bias-preload", dest="bias_preload", action="store_false", default=False)
        run.add_argument("--no-prepack-weights", action="store_true")
        if action == "compile":
            run.add_argument("--variant", choices=("cpu", "default", "current", "tuned"), default="current")
            run.add_argument("--tensor-memory-bytes", type=int, default=256 << 20)
        else:
            run.add_argument("--spike-receipt", type=Path, required=True)
            run.add_argument("--hardware-receipt", type=Path, required=True)
            run.add_argument("--trials-per-task", type=int, default=16)
            run.add_argument("--trials-per-iteration", type=int, default=4)
            run.add_argument("--builders", type=int, default=1)
            run.add_argument("--baseline-workers", type=int, default=1)
            run.add_argument("--input-fixture", type=Path)
            run.add_argument("--validation-overhead-cycles", type=int, default=300000)
            run.add_argument("--max-cycles", type=int, default=40_000_000)
            run.add_argument("--parallel", type=int, default=1)
            run.add_argument("--rtl-seconds", type=int, default=3600)
            run.add_argument("--timing-policy", choices=("legacy", "fixed-warm"), default="fixed-warm")
            add_memory_arguments(run)
    args = parser.parse_args(argv)
    if args.action == "prepare":
        args.output_dir.mkdir(parents=True, exist_ok=False)
        for name in args.cases:
            mod, provenance = build_case(name)
            prepare_case(mod, args.output_dir / name, provenance)
            print(name, flush=True)
    elif args.action == "inventory":
        with args.output.open("x") as stream:
            json.dump(inventory(import_program(args.program)), stream, indent=2)
    elif args.action == "slice":
        if args.stage_calls < 1 or args.sites < 1:
            parser.error("stage-calls and sites must be positive")
        mod, boundary = slice_program(import_program(args.program), args.start, args.stop)
        prepare_case(mod, args.output_dir, dict(boundary, case=args.workload,
                     source_kind="exact_program_slice", source_program=str(args.program.resolve()),
                     source_sha256=digest(args.program), stage_calls=args.stage_calls, sites=args.sites,
                     session_occurrences=args.stage_calls * args.sites, full_model_qualified=False))
    else:
        root = args.case_dir.resolve()
        verify_case(root)
        command = [sys.executable, str(APP / ("compile_package.py" if args.action == "compile" else "tune_models.py")),
                   "--program", str(root / "program.json"), "--adapter-receipt", str(args.adapter_receipt.resolve())]
        command += ["--graph-fusion-level", str(args.graph_fusion_level)]
        if args.graph_cse:
            command.append("--graph-cse")
        if not args.bias_preload:
            command.append("--no-bias-preload")
        if args.no_prepack_weights:
            command.append("--no-prepack-weights")
        if args.action == "compile":
            command += ["--riscv-gcc", str(args.gcc.resolve()), "--output-dir", str(root / ("package-" + args.variant)),
                        "--tensor-memory-bytes", str(args.tensor_memory_bytes)]
            if args.variant == "cpu":
                command += ["--cpu-only"]
            elif args.variant in ("current", "tuned"):
                # Use the frozen per-case seed, not a moving preset. Explicit DB
                # rejects zero hits; zero-hit current cases use reported defaults.
                if args.variant == "tuned":
                    command += ["--tuning-database", str(root / "campaign/database")]
                elif any(r["matched"] for r in json.loads((root / "current-knobs.json").read_text())["matches"]):
                    command += ["--tuning-database", str(root / "seed")]
        else:
            if min(args.trials_per_task, args.trials_per_iteration, args.parallel,
                   args.builders, args.baseline_workers, args.max_cycles) < 1:
                parser.error("Tuning budgets must be positive")
            if args.validation_overhead_cycles < 0:
                parser.error("Validation overhead must be nonnegative")
            campaign = root / "campaign"
            if campaign.exists():
                raise ValueError("Use a fresh prepared case for a new campaign; evidence is immutable")
            command += ["--gcc", str(args.gcc.resolve()), "--work-dir", str(campaign),
                        "--spike-receipt", str(args.spike_receipt.resolve()),
                        "--hardware-receipt", str(args.hardware_receipt.resolve()),
                        "--trials-per-task", str(args.trials_per_task), "--parallel", str(args.parallel),
                        "--builders", str(args.builders), "--baseline-workers", str(args.baseline_workers),
                        "--trials-per-iteration", str(args.trials_per_iteration),
                        "--max-cycles", str(args.max_cycles), "--rtl-seconds", str(args.rtl_seconds),
                        "--validation-overhead-cycles", str(args.validation_overhead_cycles),
                        "--timing-policy", args.timing_policy, "--baseline-database", str(root / "seed")]
            if args.input_fixture is not None:
                command += ["--input-fixture", str(args.input_fixture.resolve())]
            for name in ("memory_variant_receipt", "firesim_memory_artifact", "memory_pin_registry"):
                value = getattr(args, name)
                if value is not None:
                    command += ["--" + name.replace("_", "-"), str(value)]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
