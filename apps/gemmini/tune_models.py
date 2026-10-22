# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Per-operator MetaSchedule tuning of a semantic program on RTL.

Flow: representative Relax program -> the compiler's own preparation passes ->
``extract_tasks`` (deduplicated, call-count weights) -> ``tune_tasks`` with the
Gemmini/CPU design space -> a JSONDatabase that compile_package replays with
``MetaScheduleApplyDatabase``.

Before search, each task's untuned default Gemmini schedule and untuned CPU
lowering are measured for reporting and to bound simulation time: a candidate
is stopped once it exceeds ``--cap-factor`` times the default's cycles. These
baselines are not entered into the database, so the search never starts from
them; an identical program proposed by the search reuses the stored
measurement instead of simulating again.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP))

import tvm  # pylint: disable=wrong-import-position
from tvm import meta_schedule as ms, tir  # pylint: disable=wrong-import-position
from tvm.relax.backend.contrib.gemmini_meta_schedule import _supported  # pylint: disable=wrong-import-position
from tvm.relax.backend.contrib.gemmini_search import (  # pylint: disable=wrong-import-position
    GemminiBuilder,
    GemminiRunner,
    WORKLOAD_ATTR,
    tune_gemmini_tasks,
)
from tvm.relax.backend.contrib.gemmini_sketch import default_schedule  # pylint: disable=wrong-import-position
from tvm.relax.backend.contrib.gemmini_target import GemminiCapabilities  # pylint: disable=wrong-import-position
from tvm.relax.backend.contrib.gemmini_tensorir import (  # pylint: disable=wrong-import-position
    prepare_gemmini_tensorir_workloads,
)

import measure  # pylint: disable=wrong-import-position
import model_workloads  # pylint: disable=wrong-import-position
from program_io import import_program  # pylint: disable=wrong-import-position

REPO = APP.parents[1]


class Campaign:
    """Builds and measures candidates for every task of one tuning run."""

    def __init__(self, args, toolchain):
        self.args, self.toolchain = args, toolchain
        self.root = args.work_dir
        self.lock = threading.Lock()
        self.counter = len(list((self.root / "candidates").glob("*.request.json")))
        self.program_locks = {}
        self.default_cycles = {}
        cache = self.root / "measurement-cache.json"
        self.cache = {}
        if cache.exists():
            for record in json.loads(cache.read_text()).values():
                artifact_path = Path(record["folder"]) / "artifact.json"
                if artifact_path.exists():
                    artifact = json.loads(artifact_path.read_text())
                    measurement = artifact_path.parent / "measurement.json"
                    if measurement.exists():
                        latest = json.loads(measurement.read_text())
                        if latest.get("status") == "passed":
                            record = latest
                    key = self.measurement_key(artifact)
                    previous = self.cache.get(key)
                    if previous is None or (
                        previous["status"] != "passed"
                        and (record["status"] == "passed"
                             or record.get("cycle_cap", 0) > previous.get("cycle_cap", 0))
                    ):
                        self.cache[key] = record

    @staticmethod
    def measurement_key(artifact):
        # Bind the actual guest bytes: changes to inputs, wrappers, adapters or
        # the compiler must not reuse a label merely because LLVM text matches.
        # Conversely, distinct LLVM metadata can produce exactly the same ELF.
        binary = artifact.get("elfs", {}).get("valid", {}).get(
            "sha256", artifact["program_sha256"])
        return Path(artifact["workload"]).name + ":" + binary

    def workload_folder(self, key):
        return self.root / "workloads" / key

    def build(self, mod, key):
        with self.lock:
            index = self.counter
            self.counter += 1
        folder = self.root / "candidates" / f"{index:05d}-{key}"
        folder.parent.mkdir(parents=True, exist_ok=True)
        request = folder.with_suffix(".request.json")
        module = folder.with_suffix(".module.json")
        result = folder.with_suffix(".result.json")
        module.write_text(tvm.ir.save_json(mod))
        request.write_text(
            json.dumps(
                {
                    "module": str(module),
                    "workload": str(self.workload_folder(key)),
                    "folder": str(folder),
                    "toolchain": self.toolchain.as_json(),
                    "result": str(result),
                }
            )
        )
        process = subprocess.run(
            [sys.executable, str(APP / "measure.py"), "build", str(request)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            timeout=self.args.build_seconds,
        )
        (folder.with_suffix(".build.log")).write_text(process.stdout)
        outcome = json.loads(result.read_text()) if result.exists() else {"error": process.stdout[-2000:]}
        if process.returncode or "error" in outcome:
            raise RuntimeError(outcome.get("error", "build failed"))
        return Path(outcome["folder"]) / "artifact.json"

    def cap(self, key):
        default = self.default_cycles.get(key)
        if default is None:
            return self.args.max_cycles
        return min(self.args.max_cycles, int(2 * self.args.cap_factor * default + 300000))

    def run(self, artifact_path):
        artifact = json.loads(Path(artifact_path).read_text())
        key = Path(artifact["workload"]).name
        program = self.measurement_key(artifact)
        with self.lock:
            program_lock = self.program_locks.setdefault(program, threading.Lock())
        # Search strategies can submit identical generated code concurrently.
        # Serialize only those duplicates; distinct programs still run in parallel.
        with program_lock:
            cap = self.cap(key)
            if not artifact["native"]:
                cap = min(cap, self.args.cpu_max_cycles)
            with self.lock:
                cached = self.cache.get(program)
            if (cached is not None and cached.get("cycle_cap") is not None
                    and cached["status"] != "passed" and cached["cycle_cap"] < cap):
                cached = None
            if cached is not None:
                record = dict(cached, reused_from=cached["folder"])
            else:
                record = measure.run_candidate(artifact, self.toolchain, max_cycles=cap)
                with self.lock:
                    self.cache[program] = record
                    (self.root / "measurement-cache.json").write_text(json.dumps(self.cache))
        Path(artifact["folder"], "measurement.json").write_text(json.dumps(record, indent=2) + "\n")
        if record["status"] != "passed":
            raise RuntimeError(record["status"])
        return record["cycles"]


def describe(artifact_path):
    if not artifact_path:
        return {}
    folder = Path(artifact_path).parent
    measurement = folder / "measurement.json"
    record = json.loads(measurement.read_text()) if measurement.exists() else {}
    return {
        "native": json.loads(Path(artifact_path).read_text())["native"],
        "status": record.get("status"),
        "reused": "reused_from" in record,
        "rtl_seconds": record.get("rtl_seconds"),
    }


def baseline(campaign, key, label, mod):
    path = campaign.root / "baselines" / key / f"{label}.json"
    if path.exists():
        return json.loads(path.read_text())
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = None
    try:
        artifact = campaign.build(mod, key)
        cycles = campaign.run(artifact)
        record = {"status": "passed", "cycles": cycles, "artifact": str(artifact)}
    except Exception as error:  # pylint: disable=broad-except
        record = {"status": "failed", "error": str(error)}
        if artifact is not None:
            record["artifact"] = str(artifact)
    path.write_text(json.dumps(record, indent=2) + "\n")
    return record


def bind_program(root, program):
    """Refuse a changed semantic program before overwriting campaign evidence."""
    payload = tvm.ir.save_json(program)
    path = root / "program.json"
    if path.exists() and not tvm.ir.structural_equal(tvm.ir.load_json(path.read_text()), program):
        raise ValueError("A campaign cannot reuse measurements from a different program")
    if not path.exists():
        path.write_text(payload)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_inventory(program, capabilities):
    """Include every extracted task, with explicit placement exclusions."""
    prepared, _, _ = prepare_gemmini_tensorir_workloads(program)
    extracted = sorted(
        ms.relax_integration.extract_tasks(prepared, capabilities.target()),
        key=lambda task: task.task_name,
    )
    tasks, inventory = [], []
    for task in extracted:
        func = task.dispatched[0]["main"]
        kind, blocks = _supported(func)
        row = {"task": task.task_name, "weight": int(task.weight),
               "kind": kind, "blocks": blocks, "eligible": kind is not None}
        if kind is not None:
            row["tuning_index"] = len(tasks)
            tasks.append(task)
        else:
            row["reason"] = "No Gemmini rule; retained CPU fallback, outside this tuning campaign"
        inventory.append(row)
    return prepared, tasks, inventory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", choices=sorted(model_workloads.MODELS),
                        help="Synthetic representative program; not a full-model benchmark")
    source.add_argument("--program", type=Path,
                        help="Real semantic Relax JSON or torch.export .pt2 archive")
    parser.add_argument("--extract-only", action="store_true",
                        help="Write complete task inventory without a toolchain or simulation")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--trials-per-task", type=int, default=16,
                        help="Global budget multiplier; the stock scheduler distributes this times the task count")
    parser.add_argument("--parallel", type=int, default=14)
    parser.add_argument("--trials-per-iteration", type=int,
                        help="Stock search batch size; defaults to the runner concurrency")
    parser.add_argument("--builders", type=int, default=6)
    parser.add_argument("--strategy", choices=("evolutionary", "replay"), default="evolutionary")
    parser.add_argument("--tasks", type=int, nargs="*", help="Restrict to these task indices")
    parser.add_argument("--operators", type=int, nargs="+", help="Select semantic operators before task extraction")
    parser.add_argument("--baseline-workers", type=int, default=2)
    parser.add_argument("--cpu-max-cycles", type=int, default=2_000_000)
    parser.add_argument("--cap-factor", type=float, default=3.0)
    parser.add_argument("--max-cycles", type=int, default=40_000_000)
    parser.add_argument("--build-seconds", type=int, default=900)
    parser.add_argument("--no-cpu", action="store_true", help="Omit the CPU placement from the space")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gcc", type=Path)
    parser.add_argument("--adapter-receipt", type=Path)
    parser.add_argument("--spike-receipt", type=Path)
    parser.add_argument("--hardware-receipt", type=Path)
    args = parser.parse_args(argv)
    if args.program and args.operators is not None:
        parser.error("--operators selects synthetic --model operators only")
    if not args.extract_only and any(getattr(args, name) is None for name in
                                   ("gcc", "adapter_receipt", "spike_receipt", "hardware_receipt")):
        parser.error("Tuning requires gcc, adapter-receipt, spike-receipt and hardware-receipt")
    args.work_dir = args.work_dir.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    capabilities = GemminiCapabilities()
    started = time.monotonic()
    if args.program:
        program, operators = import_program(args.program), None
    else:
        program, operators = model_workloads.build(args.model, operators=args.operators)
    program_sha256 = bind_program(args.work_dir, program)
    prepared, tasks, extracted_inventory = extract_inventory(program, capabilities)
    extraction = {"schema": "gemmini_task_inventory_v1", "model": args.model,
                  "program": str(args.program.resolve()) if args.program else None,
                  "program_sha256": program_sha256,
                  "scope": "real_program" if args.program else "synthetic_operator_program",
                  "capabilities": capabilities.fingerprint,
                  "extracted_tasks": extracted_inventory,
                  "eligible_tasks": len(tasks), "total_tasks": len(extracted_inventory)}
    (args.work_dir / "extraction.json").write_text(json.dumps(extraction, indent=2) + "\n")
    (args.work_dir / "prepared.json").write_text(tvm.ir.save_json(prepared))
    if args.extract_only:
        print(json.dumps({"eligible_tasks": len(tasks), "total_tasks": len(extracted_inventory),
                          "inventory": str(args.work_dir / "extraction.json")}))
        return
    if args.tasks is not None:
        if not args.tasks or any(index < 0 or index >= len(tasks) for index in args.tasks):
            parser.error("--tasks must contain valid eligible-task indices from extraction.json")
        if len(set(args.tasks)) != len(args.tasks):
            parser.error("--tasks contains duplicate indices")
        tasks = [tasks[index] for index in args.tasks]
    if not tasks:
        parser.error("Program has no eligible Gemmini tasks; see extraction.json for CPU fallbacks")
    toolchain = measure.Toolchain(args.gcc, args.adapter_receipt, args.spike_receipt, args.hardware_receipt)
    provenance = toolchain.provenance()
    provenance_path = args.work_dir / "hardware-provenance.json"
    if provenance_path.exists():
        previous = json.loads(provenance_path.read_text())
        if previous["binaries"] != provenance["binaries"]:
            raise ValueError("A campaign cannot reuse measurements from different hardware/toolchain bytes")
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    campaign = Campaign(args, toolchain)
    inventory = []
    for task in tasks:
        func = task.dispatched[0]["main"]
        key = str(func.attrs[WORKLOAD_ATTR])
        folder = campaign.workload_folder(key)
        if not (folder / "workload-manifest.json").exists():
            measure.prepare_workload(func, folder, seed=args.seed)
        elif json.loads((folder / "workload-manifest.json").read_text())["seed"] != args.seed:
            raise ValueError("A campaign cannot reuse measurements from different input seeds")
        inventory.append({"task": task.task_name, "workload": key, "weight": int(task.weight),
                          "buffers": [[int(x) for x in func.buffer_map[p].shape] + [func.buffer_map[p].dtype] for p in func.params]})
    (args.work_dir / "tasks.json").write_text(json.dumps({"model": args.model, "operators": operators, "tasks": inventory}, indent=2) + "\n")

    # Baselines: untuned default Gemmini schedule, then the untuned CPU
    # lowering bounded by the default's cycles.
    def measure_baselines(pair):
        task, item = pair
        func = task.dispatched[0]["main"]
        key = item["workload"]
        try:
            default_mod = default_schedule(func).mod
            record = baseline(campaign, key, "default", default_mod)
        except (ValueError, tvm.TVMError) as error:
            record = {"status": "failed", "error": str(error)}
        item["default"] = record
        if record.get("status") == "passed":
            campaign.default_cycles[key] = record["cycles"]
        item["cpu_untuned"] = baseline(campaign, key, "cpu-untuned", tvm.IRModule({"main": func}))
        print(json.dumps({"task": item["task"], "default": record.get("cycles"), "cpu": item["cpu_untuned"].get("cycles")}), flush=True)
        with campaign.lock:
            (args.work_dir / "tasks.json").write_text(json.dumps({"model": args.model, "operators": operators, "tasks": inventory}, indent=2) + "\n")
    with ThreadPoolExecutor(args.baseline_workers) as pool:
        list(pool.map(measure_baselines, zip(tasks, inventory)))
    (args.work_dir / "tasks.json").write_text(json.dumps({"model": args.model, "operators": operators, "tasks": inventory}, indent=2) + "\n")

    builder = GemminiBuilder(campaign.build, max_workers=args.builders)
    runner = GemminiRunner(campaign.run, max_workers=args.parallel)
    database = tune_gemmini_tasks(
        tasks,
        work_dir=args.work_dir / ("database" if args.strategy == "evolutionary" else "replay-database"),
        builder=builder,
        runner=runner,
        max_trials_global=args.trials_per_task * len(tasks),
        num_trials_per_iter=args.trials_per_iteration or args.parallel,
        capabilities=capabilities,
        strategy=args.strategy,
        seed=args.seed,
        log=args.work_dir / ("trials.jsonl" if args.strategy == "evolutionary" else "replay-trials.jsonl"),
        describe=describe,
        include_cpu=not args.no_cpu,
    )
    # Add the verified default only after the search has finished. This gives
    # deployment a non-regressing fallback without seeding EvolutionarySearch
    # with baseline records it would otherwise propose measuring again.
    for task, item in zip(tasks, inventory):
        record = item["default"]
        if record.get("status") == "passed":
            sch = default_schedule(task.dispatched[0]["main"])
            database.commit_tuning_record(ms.database.TuningRecord(
                sch.trace, database.commit_workload(task.dispatched[0]),
                [float(record["cycles"])], capabilities.target(), [],
            ))
    (args.work_dir / ("campaign.json" if args.strategy == "evolutionary" else "replay-campaign.json")).write_text(
        json.dumps({"model": args.model, "program_sha256": program_sha256,
                    "scope": extraction["scope"], "strategy": args.strategy, "wall_seconds": time.monotonic() - started,
                    "trials_per_task": args.trials_per_task, "parallel": args.parallel, "cap_factor": args.cap_factor,
                    "trials_per_iteration": args.trials_per_iteration or args.parallel,
                    "max_cycles": args.max_cycles, "cpu_max_cycles": args.cpu_max_cycles,
                    "compiler": measure.identity(tvm._ffi.base._LIB._name),
                    "toolchain": toolchain.as_json(), "capabilities": capabilities.fingerprint}, indent=2) + "\n")


if __name__ == "__main__":
    main()
