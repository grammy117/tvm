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
from host_execution import execution_lease
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP))

import tvm  # pylint: disable=wrong-import-position
from tvm import meta_schedule as ms, tir  # pylint: disable=wrong-import-position
from tvm.relax.backend.contrib.gemmini_meta_schedule import _supported  # pylint: disable=wrong-import-position
from tvm.relax.backend.contrib.gemmini_search import (  # pylint: disable=wrong-import-position
    GemminiBuilder,
    GemminiRunner,
    WORKLOAD_ATTR,
    open_gemmini_database,
    tune_gemmini_tasks,
)
from tvm.relax.backend.contrib.gemmini_meta_schedule import default_device_schedule  # pylint: disable=wrong-import-position
from tvm.relax.backend.contrib.gemmini_target import GemminiCapabilities  # pylint: disable=wrong-import-position
from tvm.relax.backend.contrib.gemmini_tensorir import (  # pylint: disable=wrong-import-position
    prepare_gemmini_tensorir_workloads,
)

import measure  # pylint: disable=wrong-import-position
from cycle_watchdog import bind_limit, cutoff_path, write_limit
import model_workloads  # pylint: disable=wrong-import-position
from program_io import import_program  # pylint: disable=wrong-import-position
from memory_model import add_memory_arguments
from graph_plans import GraphPlan, module_digest

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
        self.incumbents = {}
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
        incumbent_path = self.root / "incumbents.json"
        if getattr(args, "moving_cutoff", False) and incumbent_path.exists():
            for key, entry in json.loads(incumbent_path.read_text()).items():
                artifact = json.loads(Path(entry["artifact"]).read_text())
                measurement = json.loads((Path(artifact["folder"]) / "measurement.json").read_text())
                if (measurement["status"] != "passed" or measurement["cycles"] != entry["cycles"]
                        or Path(artifact["workload"]).name != key):
                    raise ValueError("Incumbent is not a validated measurement of this workload")
                for elf in artifact["elfs"].values():
                    if measure.identity(elf["path"])["sha256"] != elf["sha256"]:
                        raise ValueError("Incumbent executable changed")
                self.incumbents[key] = entry

    @staticmethod
    def measurement_key(artifact):
        # Bind the actual guest bytes: changes to inputs, wrappers, adapters or
        # the compiler must not reuse a label merely because LLVM text matches.
        # Conversely, distinct LLVM metadata can produce exactly the same ELF.
        binary = artifact.get("elfs", {}).get("valid", {}).get(
            "sha256", artifact["program_sha256"])
        memory = artifact.get("memory_model", {}).get("fingerprint", "legacy-stock")
        return Path(artifact["workload"]).name + ":" + binary + ":" + memory

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
        with execution_lease(getattr(self.args, "execution_pool", None), "build"):
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
        overhead = getattr(self.args, "validation_overhead_cycles", 300000)
        return min(self.args.max_cycles, int(2 * self.args.cap_factor * default + overhead))

    def run(self, artifact_path):
        artifact = json.loads(Path(artifact_path).read_text())
        key = Path(artifact["workload"]).name
        program = self.measurement_key(artifact)
        with self.lock:
            program_lock = self.program_locks.setdefault(program, threading.Lock())
        # Search strategies can submit identical generated code concurrently.
        # Serialize only those duplicates; distinct programs still run in parallel.
        with program_lock:
            with execution_lease(getattr(self.args, "execution_pool", None), "run"):
                moving = getattr(self.args, "moving_cutoff", False)
                if moving:
                    with self.lock:
                        limit = self.incumbents.get(key, {}).get("cycles", 0)
                        artifact = bind_limit(artifact, self.toolchain, int(limit), measure.identity)
                    program = self.measurement_key(artifact)
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
                    wall = getattr(self.args, "rtl_seconds", None)
                    deadline = getattr(self.args, "deadline_epoch", None)
                    if deadline is not None:
                        remaining = deadline - time.time()
                        if remaining <= 0:
                            raise RuntimeError("DSE wall deadline reached")
                        wall = min(wall, remaining) if wall is not None else remaining
                    record = measure.run_candidate(artifact, self.toolchain, max_cycles=cap,
                                                   rtl_seconds=wall)
                if moving and record["status"] == "passed":
                    with self.lock:
                        previous = self.incumbents.get(key)
                        if previous is not None and record["cycles"] > previous["cycles"]:
                            record = dict(record, status="pruned_by_newer_incumbent",
                                          observed_cycles=record["cycles"],
                                          incumbent_cycles=previous["cycles"])
                            record.pop("cycles")
                        else:
                            self.incumbents[key] = {"cycles": int(record["cycles"]),
                                                    "artifact": str(artifact_path),
                                                    "first_call_cycles": record.get("first_call_cycles"),
                                                    "updated_at": time.time()}
                            path = self.root / "incumbents.json"
                            temporary = path.with_suffix(".tmp")
                            temporary.write_text(json.dumps(self.incumbents, indent=2) + "\n")
                            temporary.replace(path)
                            if artifact.get("cycle_watchdog", {}).get("enabled"):
                                write_limit(cutoff_path(artifact), int(record["cycles"]))
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


def baseline_database(source, work_dir, prepared, capabilities):
    """Freeze historical traces for remeasurement; never import their costs."""
    if source is None:
        return None, None
    source, root = Path(source).resolve(), Path(work_dir)
    binding = {"source": str(source), "cost_policy": "historical costs ignored; remeasure before use",
               "capabilities": capabilities.fingerprint,
               "files": [measure.identity(source / name) for name in
                         ("database_workload.json", "database_tuning_record.json")]}
    path = root / "baseline-input.json"
    if path.exists() and json.loads(path.read_text()) != binding:
        raise ValueError("A campaign cannot replace its baseline database")
    copy = root / "baseline-seed"
    copy.mkdir(exist_ok=True)
    for item in binding["files"]:
        destination = copy / Path(item["path"]).name
        if destination.exists() and measure.identity(destination)["sha256"] != item["sha256"]:
            raise ValueError("Frozen baseline trace payload changed")
        if not destination.exists():
            destination.write_bytes(Path(item["path"]).read_bytes())
    path.write_text(json.dumps(binding, indent=2) + "\n")
    database = open_gemmini_database(copy, prepared, capabilities=capabilities)
    if any(str(record.target) != str(capabilities.target()) for record in database.get_all_tuning_records()):
        raise ValueError("Baseline database capability target mismatch")
    return database, binding


def retain_measured_baseline(database, workload, trace, measurement, target):
    """Commit only this campaign's successful measured cost, never source labels."""
    if measurement.get("status") != "passed":
        return False
    database.commit_tuning_record(ms.database.TuningRecord(
        trace, database.commit_workload(workload), [float(measurement["cycles"])], target, []))
    return True


def validated_baseline_database(root):
    folder = Path(root) / "validated-baselines"
    folder.mkdir(parents=True, exist_ok=True)
    return ms.database.JSONDatabase(work_dir=str(folder))


def bind_program(root, program):
    """Refuse a changed semantic program before overwriting campaign evidence."""
    payload = tvm.ir.save_json(program)
    path = root / "program.json"
    if path.exists() and not tvm.ir.structural_equal(tvm.ir.load_json(path.read_text()), program):
        raise ValueError("A campaign cannot reuse measurements from a different program")
    if not path.exists():
        path.write_text(payload)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_inventory(program, capabilities, *, fusion_level=-1, prepack_weights=False, bias_preload=False,
                      eliminate_common_subexpr=False):
    """Include every extracted task, with explicit placement exclusions."""
    prepared, _, _ = prepare_gemmini_tensorir_workloads(
        program, fusion_level=fusion_level, prepack_weights=prepack_weights, bias_preload=bias_preload,
        eliminate_common_subexpr=eliminate_common_subexpr)
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


def bind_operator_workloads(prepared, tasks, root, *, input_fixture=None):
    """Bind extracted constants and optional exact graph boundary inputs."""
    from operator_inputs import resolve_operator_inputs
    import operator_inputs

    boundary = None
    if input_fixture is not None:
        with np.load(input_fixture, allow_pickle=False) as arrays:
            boundary = [arrays[f"input_{i}"].copy() for i in range(len(prepared["main"].params))]
        for value, param in zip(boundary, prepared["main"].params):
            info = param.struct_info
            if value.shape != tuple(int(x) for x in info.shape) or value.dtype.name != info.dtype:
                raise ValueError("Input fixture differs from the prepared graph boundary")
    workloads = {str(task.dispatched[0]["main"].attrs[WORKLOAD_ATTR]): task.dispatched[0]["main"]
                 for task in tasks}
    bindings = resolve_operator_inputs(prepared, workloads, boundary_inputs=boundary)
    document = {"schema": "gemmini_operator_input_bindings_v1", "prepared_ir_sha256": module_digest(prepared),
                "fixture": measure.identity(input_fixture) if input_fixture else None,
                "binder_source": measure.identity(operator_inputs.__file__),
                "workloads": {key: {"bound_inputs": measure.input_binding_identity(workloads[key], row["input_overrides"]),
                                     "provenance": row["provenance"]} for key, row in bindings.items()}}
    path = Path(root) / "operator-input-bindings.json"
    if path.exists() and json.loads(path.read_text()) != document:
        raise ValueError("Operator input bindings changed; use a fresh campaign root")
    for key, row in document["workloads"].items():
        manifest = Path(root) / "workloads" / key / "workload-manifest.json"
        if manifest.exists() and json.loads(manifest.read_text()).get("bound_inputs", {}) != row["bound_inputs"]:
            raise ValueError("Existing workload inputs differ from extracted bindings; use a fresh campaign root")
    path.write_text(json.dumps(document, indent=2) + "\n")
    return bindings, measure.identity(path)


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
    parser.add_argument("--trials-per-task", type=int, default=None,
                        help="Global budget multiplier; the stock scheduler distributes this times the task count")
    parser.add_argument("--trials-total", type=int, default=25,
                        help="Total proposals across extracted tasks unless --trials-per-task is explicit")
    parser.add_argument("--parallel", type=int, default=14)
    parser.add_argument("--trials-per-iteration", type=int,
                        help="Stock search batch size; defaults to the runner concurrency")
    parser.add_argument("--builders", type=int, default=6)
    parser.add_argument("--strategy", choices=("evolutionary", "replay"), default="replay")
    parser.add_argument("--tasks", type=int, nargs="*", help="Restrict to these task indices")
    parser.add_argument("--operators", type=int, nargs="+", help="Select semantic operators before task extraction")
    parser.add_argument("--baseline-workers", type=int, default=2)
    parser.add_argument("--moving-cutoff", action="store_true",
                        help="Abort measured inference above the fastest validated exact-task cycle count")
    parser.add_argument("--incumbent-baseline-only", action="store_true",
                        help="Remeasure matching current trace, falling back to default; defer CPU/extra baselines")
    parser.add_argument("--compact-search", action="store_true",
                        help="Bounded native search profile: root epilogue, short unrolling, smaller stock population")
    parser.add_argument("--deadline-epoch", type=float,
                        help="Absolute shared wall deadline; admission and supervisor enforce the same deadline")
    parser.add_argument("--baselines-after-search", action="store_true",
                        help="Start stock DSE before baseline comparisons; candidate correctness gates still apply")
    parser.add_argument("--execution-pool", type=Path,
                        help="Shared Linux CPU/memory admission directory for external builds and measurements")
    parser.add_argument("--cpu-max-cycles", type=int, default=2_000_000)
    parser.add_argument("--cap-factor", type=float, default=3.0)
    parser.add_argument("--max-cycles", type=int, default=40_000_000)
    parser.add_argument("--validation-overhead-cycles", type=int, default=300000,
                        help="Untimed validation/startup allowance in adaptive total-cycle caps; never a cost label")
    parser.add_argument("--build-seconds", type=int, default=900)
    parser.add_argument("--rtl-seconds", type=int, help="Explicit per-candidate RTL wall limit")
    parser.add_argument("--timing-policy", choices=("legacy", "fixed-warm"), default="legacy",
                        help="fixed-warm always measures after one warmup, independent of operator duration")
    parser.add_argument("--graph-fusion-level", type=int, choices=(-1, 0, 1, 2, 3), default=-1)
    parser.add_argument("--no-prepack-weights", dest="prepack_weights", action="store_false", default=False)
    parser.add_argument("--no-bias-preload", dest="bias_preload", action="store_false", default=False)
    parser.add_argument("--graph-cse", action="store_true",
                        help="Expose stock Relax common-subexpression elimination as a graph preparation choice")
    parser.add_argument("--baseline-database", type=Path,
                        help="Remeasure structurally matching historical traces with the baselines; costs are not imported")
    parser.add_argument("--input-fixture", type=Path,
                        help="NPZ input_0,... in prepared main parameter order; propagate mathematical boundary inputs")
    parser.add_argument("--no-cpu", action="store_true", help="Omit the CPU placement from the space")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gcc", type=Path)
    parser.add_argument("--adapter-receipt", type=Path)
    parser.add_argument("--spike-receipt", type=Path)
    parser.add_argument("--hardware-receipt", type=Path)
    add_memory_arguments(parser)
    args = parser.parse_args(argv)
    if args.moving_cutoff and args.timing_policy != "fixed-warm":
        parser.error("--moving-cutoff requires --timing-policy fixed-warm")
    if args.moving_cutoff and args.baselines_after_search:
        parser.error("Moving cutoffs require a freshly validated incumbent before search")
    if args.rtl_seconds is not None and args.rtl_seconds < 1:
        parser.error("rtl-seconds must be positive")
    if args.validation_overhead_cycles < 0:
        parser.error("validation-overhead-cycles must be nonnegative")
    if args.program and args.operators is not None:
        parser.error("--operators selects synthetic --model operators only")
    if not args.extract_only and any(getattr(args, name) is None for name in
                                   ("gcc", "adapter_receipt", "spike_receipt", "hardware_receipt")):
        parser.error("Tuning requires gcc, adapter-receipt, spike-receipt and hardware-receipt")
    if args.trials_total < 1 or (args.trials_per_task is not None and args.trials_per_task < 1):
        parser.error("Trial counts must be positive")
    args.work_dir = args.work_dir.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    capabilities = GemminiCapabilities()
    started = time.monotonic()
    if args.program:
        program, operators = import_program(args.program), None
    else:
        program, operators = model_workloads.build(args.model, operators=args.operators)
    program_sha256 = bind_program(args.work_dir, program)
    graph_plan = GraphPlan.from_args(args)
    preparation = graph_plan.preparation_options
    prepared, tasks, extracted_inventory = extract_inventory(program, capabilities, **preparation)
    prepared_sha256 = module_digest(prepared)
    graph_document = graph_plan.identity(program, prepared)
    extraction = {"schema": "gemmini_task_inventory_v1", "model": args.model,
                  "program": str(args.program.resolve()) if args.program else None,
                  "program_sha256": program_sha256,
                  "prepared_sha256": prepared_sha256, "preparation_options": preparation,
                  "scope": "real_program" if args.program else "synthetic_operator_program",
                  "capabilities": capabilities.fingerprint,
                  "extracted_tasks": extracted_inventory,
                  "eligible_tasks": len(tasks), "total_tasks": len(extracted_inventory)}
    extraction_path = args.work_dir / "extraction.json"
    if extraction_path.exists():
        previous = json.loads(extraction_path.read_text())
        previous_options = previous.get("preparation_options", GraphPlan(-1).preparation_options)
        if previous_options != preparation or previous.get("prepared_sha256", prepared_sha256) != prepared_sha256:
            raise ValueError("A campaign cannot change its graph preparation or prepared IR")
    (args.work_dir / "extraction.json").write_text(json.dumps(extraction, indent=2) + "\n")
    (args.work_dir / "graph-plan.json").write_text(json.dumps(graph_document, indent=2) + "\n")
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
    toolchain = measure.Toolchain(args.gcc, args.adapter_receipt, args.spike_receipt, args.hardware_receipt,
                                  args.memory_variant_receipt, args.firesim_memory_artifact,
                                  args.memory_pin_registry, args.timing_policy, args.moving_cutoff)
    provenance = toolchain.provenance()
    provenance_path = args.work_dir / "hardware-provenance.json"
    if provenance_path.exists():
        previous = json.loads(provenance_path.read_text())
        old_memory = previous.get("memory_model")
        memory_changed = (old_memory != provenance["memory_model"] if old_memory is not None
                          else provenance["memory_model"]["kind"] != "stock_mm_magic")
        if (previous["binaries"] != provenance["binaries"]
                or memory_changed
                or previous.get("timing_policy", "legacy") != provenance["timing_policy"]):
            raise ValueError("A campaign cannot reuse measurements from different hardware/toolchain bytes")
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    campaign = Campaign(args, toolchain)
    seed_database, seed_binding = baseline_database(args.baseline_database, args.work_dir, prepared, capabilities)
    input_bindings, input_binding_record = bind_operator_workloads(
        prepared, tasks, args.work_dir, input_fixture=args.input_fixture)
    current_traces = {}
    fallback_database = validated_baseline_database(args.work_dir)
    inventory = []
    for task in tasks:
        func = task.dispatched[0]["main"]
        key = str(func.attrs[WORKLOAD_ATTR])
        folder = campaign.workload_folder(key)
        if not (folder / "workload-manifest.json").exists():
            bound = input_bindings[key]
            measure.prepare_workload(func, folder, seed=args.seed,
                                     input_overrides=bound["input_overrides"],
                                     input_provenance=bound["provenance"])
        elif json.loads((folder / "workload-manifest.json").read_text())["seed"] != args.seed:
            raise ValueError("A campaign cannot reuse measurements from different input seeds")
        inventory.append({"task": task.task_name, "workload": key, "weight": int(task.weight),
                          "buffers": [[int(x) for x in func.buffer_map[p].shape] + [func.buffer_map[p].dtype] for p in func.params]})
    (args.work_dir / "tasks.json").write_text(json.dumps({"model": args.model, "operators": operators, "tasks": inventory}, indent=2) + "\n")
    (args.work_dir / "live-campaign.json").write_text(json.dumps({
        "status": "running", "program_sha256": program_sha256,
        "prepared_sha256": prepared_sha256, "scope": extraction["scope"],
        "preparation_options": preparation, "strategy": args.strategy,
        "proposal_budget": (args.trials_per_task * len(tasks) if args.trials_per_task is not None else args.trials_total),
        "moving_cutoff": args.moving_cutoff, "compact_search": args.compact_search,
        "deadline_epoch": args.deadline_epoch, "toolchain": toolchain.as_json(),
        "compiler": measure.identity(tvm._ffi.base._LIB._name),
        "input_bindings": input_binding_record, "baseline_database": seed_binding,
    }, indent=2) + "\n")

    # Remeasure current first so a known schedule sets a practical cap for the
    # default and CPU baselines. Historical magic-memory labels are never used.
    def measure_baselines(pair):
        task, item = pair
        func = task.dispatched[0]["main"]
        key = item["workload"]
        if seed_database is not None:
            current = seed_database.query_tuning_record(task.dispatched[0], capabilities.target(), "main")
            if current is None:
                item["current"] = {"status": "unmatched"}
            else:
                try:
                    schedule = tir.Schedule(task.dispatched[0])
                    current.trace.apply_to_schedule(schedule, remove_postproc=False)
                    current_traces[key] = current.trace
                    item["current"] = baseline(campaign, key, "current", schedule.mod)
                    item["current"]["historical_costs_imported"] = False
                    if item["current"].get("status") == "passed":
                        campaign.default_cycles[key] = item["current"]["cycles"]
                except (ValueError, tvm.TVMError) as error:
                    item["current"] = {"status": "failed", "error": str(error)}
        if args.incumbent_baseline_only and item.get("current", {}).get("status") == "passed":
            record = {"status": "deferred", "reason": "matching current trace freshly validated"}
        else:
            try:
                default_mod = default_device_schedule(func).mod
                record = baseline(campaign, key, "default", default_mod)
            except (ValueError, tvm.TVMError) as error:
                record = {"status": "failed", "error": str(error)}
        item["default"] = record
        if record.get("status") == "passed":
            campaign.default_cycles[key] = min(campaign.default_cycles.get(key, record["cycles"]), record["cycles"])
        item["cpu_untuned"] = ({"status": "deferred", "reason": "four-hour search policy"}
                               if args.incumbent_baseline_only else
                               baseline(campaign, key, "cpu-untuned", tvm.IRModule({"main": func})))
        print(json.dumps({"task": item["task"], "current": item.get("current", {}).get("cycles"),
                          "default": record.get("cycles"), "cpu": item["cpu_untuned"].get("cycles")}), flush=True)
        with campaign.lock:
            if record.get("status") == "passed":
                retain_measured_baseline(fallback_database, task.dispatched[0],
                                         default_device_schedule(func).trace, record, capabilities.target())
            if item.get("current", {}).get("status") == "passed":
                retain_measured_baseline(fallback_database, task.dispatched[0], current_traces[key],
                                         item["current"], capabilities.target())
            (args.work_dir / "tasks.json").write_text(json.dumps({"model": args.model, "operators": operators, "tasks": inventory}, indent=2) + "\n")
    def run_baselines():
        with ThreadPoolExecutor(args.baseline_workers) as pool:
            list(pool.map(measure_baselines, zip(tasks, inventory)))
    if not args.baselines_after_search:
        run_baselines()
    if args.moving_cutoff:
        missing = [item["workload"] for item in inventory if item["workload"] not in campaign.incumbents]
        if missing:
            (args.work_dir / "deferred.json").write_text(json.dumps({
                "status": "deferred_no_valid_incumbent", "workloads": missing,
                "reason": "Bootstrap did not establish a validated cycle limit within the bounded local policy",
                "next": "Continue with retained inputs, traces and failures during FireSim handoff",
            }, indent=2) + "\n")
            return
    (args.work_dir / "tasks.json").write_text(json.dumps({"model": args.model, "operators": operators, "tasks": inventory}, indent=2) + "\n")

    builder = GemminiBuilder(campaign.build, max_workers=args.builders)
    runner = GemminiRunner(campaign.run, max_workers=args.parallel)
    database = tune_gemmini_tasks(
        tasks,
        work_dir=args.work_dir / ("database" if args.strategy == "evolutionary" else "replay-database"),
        builder=builder,
        runner=runner,
        max_trials_global=(args.trials_per_task * len(tasks) if args.trials_per_task is not None else args.trials_total),
        num_trials_per_iter=args.trials_per_iteration or args.parallel,
        capabilities=capabilities,
        strategy=args.strategy,
        seed=args.seed,
        log=args.work_dir / ("trials.jsonl" if args.strategy == "evolutionary" else "replay-trials.jsonl"),
        describe=describe,
        include_cpu=not args.no_cpu,
        compact=args.compact_search,
    )
    if args.baselines_after_search:
        run_baselines()
    # Add freshly verified baselines only after the search has finished. This gives
    # deployment a non-regressing fallback without seeding EvolutionarySearch
    # with baseline records it would otherwise propose measuring again.
    for task, item in zip(tasks, inventory):
        record = item["default"]
        if record.get("status") == "passed":
            sch = default_device_schedule(task.dispatched[0]["main"])
            retain_measured_baseline(database, task.dispatched[0], sch.trace, record, capabilities.target())
        if item["workload"] in current_traces:
            retain_measured_baseline(database, task.dispatched[0], current_traces[item["workload"]],
                                     item["current"], capabilities.target())
    (args.work_dir / ("campaign.json" if args.strategy == "evolutionary" else "replay-campaign.json")).write_text(
        json.dumps({"model": args.model, "program_sha256": program_sha256,
                    "prepared_sha256": prepared_sha256, "preparation_options": preparation,
                    "scope": extraction["scope"], "strategy": args.strategy, "wall_seconds": time.monotonic() - started,
                    "trials_per_task": args.trials_per_task, "trials_total": args.trials_total,
                    "proposal_budget": (args.trials_per_task * len(tasks) if args.trials_per_task is not None else args.trials_total),
                    "parallel": args.parallel, "cap_factor": args.cap_factor,
                    "baselines_after_search": args.baselines_after_search,
                    "execution_pool": str(args.execution_pool) if args.execution_pool else None,
                    "moving_cutoff": args.moving_cutoff, "compact_search": args.compact_search,
                    "deadline_epoch": args.deadline_epoch,
                    "trials_per_iteration": args.trials_per_iteration or args.parallel,
                    "max_cycles": args.max_cycles, "cpu_max_cycles": args.cpu_max_cycles,
                    "validation_overhead_cycles": args.validation_overhead_cycles,
                    "rtl_seconds": args.rtl_seconds, "timing_policy": args.timing_policy,
                    "memory_model": provenance["memory_model"],
                    "baseline_database": seed_binding,
                    "input_bindings": input_binding_record,
                    "compiler": measure.identity(tvm._ffi.base._LIB._name),
                    "toolchain": toolchain.as_json(), "capabilities": capabilities.fingerprint}, indent=2) + "\n")


if __name__ == "__main__":
    main()
