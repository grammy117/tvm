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
"""Publish and combine measured operator records using stock TVM databases.

Workload structure, target, trace replay and retained measured executable bind
each label. Model names do not participate in matching. This library preserves
operator timing scope; connected programs must still be measured separately.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import tvm
from tvm import meta_schedule as ms, tir
from tvm.relax.backend.contrib.gemmini_search import WORKLOAD_ATTR, open_gemmini_database
from tvm.relax.backend.contrib.gemmini_target import GemminiCapabilities

from graph_plans import module_digest
from memory_model import confirm_memory_model


DATABASE_FILES = ("database_workload.json", "database_tuning_record.json")


def identity(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def _read(path):
    return json.loads(Path(path).read_text())


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _unchanged(item):
    actual = identity(item["path"])
    _require(actual["sha256"] == item["sha256"], f"Changed evidence: {item['path']}")
    return actual


def _write_immutable(path, document):
    payload = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if path.exists():
        _require(path.read_text() == payload, f"Refusing to replace immutable context: {path}")
    else:
        path.write_text(payload)
    return path


def _trace_digest(record):
    # TuningRecord.as_json's Python conversion cannot encode IndexMap on this
    # pinned TVM. Native TVM JSON preserves all standard trace argument types.
    return hashlib.sha256(tvm.ir.save_json(record.trace.as_json()).encode()).hexdigest()


def _workload_data(folder, bindings=None):
    """Bind actual measurement inputs/oracle bytes, including revision2 overrides."""
    key = folder.name
    result = {"mathematical_ir": identity(folder / "workload.json")}
    manifest_path = folder / "workload-manifest.json"
    if not manifest_path.exists():
        _require(bindings is None, "Bound-input workload is missing its manifest")
        return result
    manifest = _read(manifest_path)
    result["manifest"] = identity(manifest_path)
    for label in ("inputs", "expected"):
        item = identity(folder / (label + ".bin"))
        _require(item["sha256"] == manifest[label + "_sha256"], "Workload " + label + " bytes changed")
        result[label] = item
    if bindings is not None:
        _require(key in bindings["workloads"], "Workload has no declared input binding")
        declared = bindings["workloads"][key]
        _require(manifest.get("bound_inputs", {}) == declared["bound_inputs"],
                 "Workload input overrides differ from campaign bindings")
        _require(manifest.get("input_provenance") == declared["provenance"],
                 "Workload input provenance differs from campaign bindings")
        result["bound_inputs"] = declared["bound_inputs"]
    # Check each bound parameter against its actual aligned input blob region,
    # not just against a second metadata document carrying the same digest.
    blob, offset = (folder / "inputs.bin").read_bytes(), 0
    for index, spec in enumerate(manifest["buffers"]):
        if index in manifest["outputs"]:
            continue
        size = math.prod(spec["shape"]) * np.dtype(spec["dtype"]).itemsize
        _require(offset + size <= len(blob), "Truncated workload input blob")
        bound = manifest.get("bound_inputs", {}).get(str(index))
        if bound is not None:
            _require(bound["shape"] == spec["shape"] and bound["dtype"] == spec["dtype"],
                     "Bound parameter ABI differs from workload")
            _require(hashlib.sha256(blob[offset:offset + size]).hexdigest() == bound["sha256"],
                     "Bound parameter bytes differ from workload input blob")
        offset = (offset + size + 63) // 64 * 64
    return result


def _successful_candidate(root, path, campaign, bindings=None):
    measurement = _read(path)
    artifact_path = path.parent / "artifact.json"
    artifact = _read(artifact_path)
    key = Path(artifact["workload"]).name
    _require(Path(artifact["folder"]).resolve() == path.parent, "Candidate folder mismatch")
    _require(Path(artifact["workload"]).resolve() == root / "workloads" / key,
             "Candidate workload lies outside its campaign")
    memory = campaign["memory_model"]
    for item in (artifact, measurement):
        _require(item.get("timing_policy") == "fixed-warm", "Candidate timing policy mismatch")
        _require(item.get("memory_model", {}).get("fingerprint") == memory["fingerprint"],
                 "Candidate memory fingerprint mismatch")
    _require(artifact["compiler"]["sha256"] == campaign["compiler"]["sha256"],
             "Candidate compiler identity mismatch")
    elfs = {name: _unchanged(item) for name, item in artifact["elfs"].items()}
    _require(measurement["rtl_elf"]["sha256"] == elfs["valid"]["sha256"],
             "Timing label refers to a different executable")
    runner, llvm = identity(path.parent / "runner.c"), identity(path.parent / "package/operators.ll")
    _require(runner["sha256"] == artifact["runner_sha256"], "Candidate runner changed")
    _require(llvm["sha256"] == artifact["program_sha256"], "Candidate LLVM changed")
    log = _unchanged(measurement["rtl_log"])
    # A cache hit may refer to the first candidate's measured identical ELF.
    measured_folder = Path(measurement["folder"]).resolve()
    _require(measured_folder.parent == root / "candidates", "Measurement belongs to another campaign")
    measured_artifact = _read(measured_folder / "artifact.json")
    _require(measured_artifact["elfs"]["valid"]["sha256"] == elfs["valid"]["sha256"],
             "Cached measurement executable mismatch")
    _require(Path(log["path"]).parent == measured_folder, "Measurement console folder mismatch")
    console = Path(log["path"]).read_text(errors="replace")
    _require(confirm_memory_model(memory, console)["model_confirmed"],
             "FASED console does not confirm the bound memory model")
    _require(measurement.get("memory_confirmation", {}).get("model_confirmed") is True,
             "Measurement did not confirm memory settings")
    counts = []
    for line in console.splitlines():
        if line.startswith("DSE_CYCLES "):
            counts.append(dict(field.split("=", 1) for field in line.split()[1:]))
    _require(len(counts) == 1 and "DSE_PASS" in console.splitlines() and "Verilog $finish" in console,
             "Incomplete RTL numerical/timing evidence")
    first, warm = int(counts[0]["first"]), int(counts[0]["second"])
    _require(first > 0 and warm > 0 and warm == measurement["cycles"]
             and warm == measurement["warm_call_cycles"]
             and first == measurement["first_call_cycles"], "Fixed-warm cycle label mismatch")
    spike_valid, spike_bad = measured_folder / "spike-valid.log", measured_folder / "spike-negative.log"
    _require("DSE_PASS" in spike_valid.read_text().splitlines()
             and "TENSORIR_NUMERICAL_FAILURE" in spike_bad.read_text(),
             "Missing positive/wrong-oracle functional evidence")
    module = path.parent.with_suffix(".module.json")
    scheduled = tvm.ir.load_json(module.read_text())
    _require(str(scheduled["main"].attrs[WORKLOAD_ATTR]) == key, "Scheduled workload key mismatch")
    return {"workload": key, "cycles": warm, "module": scheduled,
            "provenance": {"measurement": identity(path), "artifact": identity(artifact_path),
                           "scheduled_module": identity(module), "elfs": elfs,
                           "llvm": llvm, "runner": runner, "rtl_log": log,
                           "spike_valid": identity(spike_valid), "spike_negative": identity(spike_bad),
                           "adapter_sha256": artifact["adapter_sha256"],
                           "workload_data": _workload_data(Path(artifact["workload"]), bindings),
                           "reused": "reused_from" in measurement}}


def _failed_candidate(path, campaign, bindings=None):
    measurement = _read(path)
    row = {"measurement": identity(path), "status": measurement["status"],
           "cost_admitted": False, "cycle_cap": measurement.get("cycle_cap"),
           "wall_limit_seconds": measurement.get("wall_limit_seconds")}
    artifact_path = path.parent / "artifact.json"
    if artifact_path.exists():
        artifact = _read(artifact_path)
        _require(Path(artifact["folder"]).resolve() == path.parent, "Failed candidate folder mismatch")
        _require(artifact.get("memory_model", {}).get("fingerprint")
                 == campaign["memory_model"]["fingerprint"], "Failed candidate memory mismatch")
        row.update(artifact=identity(artifact_path),
                   elfs={name: _unchanged(item) for name, item in artifact["elfs"].items()},
                   workload_data=_workload_data(Path(artifact["workload"]), bindings))
    if "rtl_log" in measurement:
        row["rtl_log"] = _unchanged(measurement["rtl_log"])
    row["functional_logs"] = [identity(path.parent / name)
                              for name in ("spike-valid.log", "spike-negative.log")
                              if (path.parent / name).exists()]
    if (path.parent / "budget-pruning.json").exists():
        row["budget_pruning"] = identity(path.parent / "budget-pruning.json")
    return row


def _baseline_evidence(root, successful):
    rows = []
    for path in sorted((root / "baselines").glob("*/*.json")):
        record = _read(path)
        row = {"label": path.stem, "workload": path.parent.name, "status": record["status"],
               "baseline": identity(path), "cost_admitted": False}
        if "artifact" in record:
            artifact_path = Path(record["artifact"]).resolve(strict=True)
            _require(artifact_path.parent.parent == root / "candidates", "Baseline artifact outside campaign")
            row["artifact"] = identity(artifact_path)
            measurement = artifact_path.parent / "measurement.json"
            if measurement.exists():
                row["measurement"] = identity(measurement)
        if record["status"] == "passed":
            _require("artifact" in row and any(
                item["provenance"]["artifact"] == row["artifact"]
                and item["workload"] == row["workload"] and item["cycles"] == record["cycles"]
                for item in successful), "Baseline lacks its fresh executable/cycle witness")
            row["cycles"] = record["cycles"]
            row["cost_admission"] = "Only a separately bound standard database trace can admit this cost"
        rows.append(row)
    return rows


def validate_campaign(root, *, capabilities=None):
    """Return its standard database and a complete per-record evidence binding.

    A matching cycle value alone is never evidence: every accepted trace must
    replay to the exact retained candidate module measured with that value.
    """
    root = Path(root).resolve(strict=True)
    capabilities = capabilities or GemminiCapabilities()
    campaign = _read(root / "campaign.json")  # Written only when search is complete.
    _require(campaign["capabilities"] == capabilities.fingerprint, "Campaign capability mismatch")
    _require(campaign["timing_policy"] == "fixed-warm", "Only fixed-warm campaigns can be combined")
    memory = campaign["memory_model"]
    _require(memory["kind"] == "fased_latency_bandwidth_pipe", "Only FASED measurements can be combined")
    _unchanged(campaign["compiler"])
    _require(identity(root / "program.json")["sha256"] == campaign["program_sha256"],
             "Campaign semantic program changed")
    prepared = tvm.ir.load_json((root / "prepared.json").read_text())
    _require(module_digest(prepared) == campaign["prepared_sha256"], "Campaign prepared IR changed")
    _require(_read(root / "graph-plan.json")["prepared_ir_sha256"] == campaign["prepared_sha256"],
             "Campaign graph plan mismatch")
    binding_identity = fixture_identity = binder_identity = None
    bindings = None
    if campaign.get("input_bindings") is not None:
        binding_identity = _unchanged(campaign["input_bindings"])
        _require(Path(binding_identity["path"]).parent == root, "Input bindings belong to another campaign")
        bindings = _read(binding_identity["path"])
        _require(bindings["prepared_ir_sha256"] == campaign["prepared_sha256"],
                 "Input bindings describe a different prepared program")
        binder_identity = _unchanged(bindings["binder_source"])
        if bindings.get("fixture") is not None:
            fixture_identity = _unchanged(bindings["fixture"])
    _require(all((root / "database" / name).is_file() for name in DATABASE_FILES),
             "Completed campaign database is missing")
    successful, failed = [], []
    for path in sorted((root / "candidates").glob("*/measurement.json")):
        status = _read(path)["status"]
        if status == "passed":
            successful.append(_successful_candidate(root, path, campaign, bindings))
        else:
            failed.append(_failed_candidate(path, campaign, bindings))
    baselines = _baseline_evidence(root, successful)
    database = open_gemmini_database(root / "database", prepared, capabilities=capabilities)
    records, omitted = [], []
    for index, record in enumerate(database.get_all_tuning_records()):
        costs = [float(cost) for cost in record.run_secs] if record.run_secs is not None else []
        if not costs or any(not math.isfinite(cost) or cost <= 0 or cost >= 1e10 for cost in costs):
            omitted.append({"source_record": index, "reason": "invalid/failed measurement cost"})
            continue
        _require(len(costs) == 1, "Expected one fixed-warm label per tuning record")
        _require(str(record.target) == str(capabilities.target()), "Record capability target mismatch")
        key = str(record.workload.mod["main"].attrs[WORKLOAD_ATTR])
        workload_file = root / "workloads" / key / "workload.json"
        _require(tvm.ir.structural_equal(record.workload.mod, tvm.ir.load_json(workload_file.read_text())),
                 "Database workload differs from measured mathematical workload")
        schedule = tir.Schedule(record.workload.mod)
        record.trace.apply_to_schedule(schedule, remove_postproc=False)
        witnesses = [item for item in successful if item["workload"] == key
                     and item["cycles"] == costs[0]
                     and tvm.ir.structural_equal(item["module"], schedule.mod)]
        _require(witnesses, f"No fresh workload/trace/executable witness for record {index}")
        records.append({"source_record": index, "workload": key, "cycles": costs[0],
                        "trace_sha256": _trace_digest(record),
                        "scheduled_ir_sha256": module_digest(schedule.mod),
                        "witnesses": [item["provenance"] for item in witnesses]})
    _require(records, "No verified fresh records in campaign")
    source_files = [root / name for name in
                    ("campaign.json", "program.json", "prepared.json", "graph-plan.json",
                     "hardware-provenance.json", "tasks.json", "trials.jsonl") if (root / name).exists()]
    app = Path(__file__).resolve().parent
    backend = Path(tvm.__file__).resolve().parent / "relax/backend/contrib"
    publication_sources = sorted(app.glob("*.py")) + sorted(backend.glob("gemmini*.py"))
    context = {"schema": "gemmini_measurement_context_v1", "memory_fingerprint": memory["fingerprint"],
               "memory_model": memory, "timing_policy": "fixed-warm", "capabilities": capabilities.fingerprint,
               "target": str(capabilities.target()), "compiler": campaign["compiler"],
               "campaign": identity(root / "campaign.json"),
               "input_bindings": binding_identity, "boundary_fixture": fixture_identity,
               "input_binder_source": binder_identity,
               "source_files": [identity(path) for path in source_files],
               "publication_sources": {"phase": "captured during publication, not at candidate build time",
                                       "files": [identity(path) for path in publication_sources]},
               "database_files": [identity(root / "database" / name) for name in DATABASE_FILES],
               "records": len(records), "record_provenance": records,
               "source_record_order": "stock Database.get_all_tuning_records (ascending mean cost)",
               "trace_identity_encoding": "TVM JSON of standard trace.as_json()",
               "failed_measurements": failed, "baseline_evidence": baselines, "omitted_records": omitted,
               "historical_costs_imported": False,
               "scope": "Reusable per-operator labels; no connected-model or deployed FireSim qualification"}
    return database, context


def publish_campaign_context(root, *, capabilities=None):
    _, context = validate_campaign(root, capabilities=capabilities)
    return _write_immutable(Path(root).resolve() / "database/measurement-context.json", context)


def merge_campaigns(roots, output, *, capabilities=None):
    """Create a fresh standard JSONDatabase with compatible verified records."""
    capabilities = capabilities or GemminiCapabilities()
    output = Path(output).resolve()
    _require(not output.exists(), f"Library output must be fresh: {output}")
    _require(bool(roots), "At least one completed campaign is required")
    sources = [validate_campaign(root, capabilities=capabilities) for root in roots]
    first = sources[0][1]
    for _, context in sources[1:]:
        for field in ("memory_fingerprint", "capabilities", "target", "timing_policy"):
            _require(context[field] == first[field], f"Incompatible campaign {field}")
        _require(context["compiler"]["sha256"] == first["compiler"]["sha256"],
                 "Incompatible compiler binaries")
    adapters = {witness["adapter_sha256"] for _, context in sources
                for row in context["record_provenance"] for witness in row["witnesses"]}
    _require(len(adapters) == 1, "Incompatible primitive adapter binaries")
    output.mkdir(parents=True)
    combined = ms.database.JSONDatabase(work_dir=str(output))
    provenance = []
    for database, context in sources:
        records = database.get_all_tuning_records()
        for row in context["record_provenance"]:
            record = records[row["source_record"]]
            combined.commit_tuning_record(ms.database.TuningRecord(
                record.trace, combined.commit_workload(record.workload.mod),
                record.run_secs, record.target, record.args_info))
            provenance.append(dict(row, campaign=context["campaign"], library_commit=len(provenance)))
    context = {"schema": "gemmini_measurement_context_v1", "library_schema": "gemmini_operator_library_v1",
               **{key: first[key] for key in ("memory_fingerprint", "memory_model", "capabilities",
                                               "target", "timing_policy", "compiler")},
               "records": len(provenance), "record_provenance": provenance,
               "source_campaigns": [item[1] for item in sources],
               "database_files": [identity(output / name) for name in DATABASE_FILES],
               "historical_costs_imported": False,
               "scope": "Structurally matched operator reuse; graph-plan latency requires connected measurement"}
    _write_immutable(output / "measurement-context.json", context)
    return context


def coverage(database_path, programs, *, capabilities=None, graph_plans=None):
    """Report stock structural matches for each semantic program and graph plan."""
    from graph_plans import plans
    from program_io import import_program

    capabilities = capabilities or GemminiCapabilities()
    database_path = Path(database_path).resolve(strict=True)
    _require(all((database_path / name).is_file() for name in DATABASE_FILES),
             "Coverage requires an existing complete database")
    rows = []
    for program_path in programs:
        program = import_program(program_path)
        for plan in plans() if graph_plans is None else graph_plans:
            prepared = plan.prepare(program)[0]
            database = open_gemmini_database(database_path, prepared, capabilities=capabilities)
            tasks = ms.relax_integration.extract_tasks(prepared, capabilities.target())
            matches = []
            for task in tasks:
                record = database.query_tuning_record(task.dispatched[0], capabilities.target(), "main")
                matches.append({"task": task.task_name, "weight": int(task.weight),
                                "matched": record is not None,
                                "cycles": float(record.run_secs[0]) if record is not None else None})
            rows.append({"program": identity(program_path), "plan": plan.name,
                         "prepared_ir_sha256": module_digest(prepared), "tasks": matches,
                         "matched_tasks": sum(item["matched"] for item in matches),
                         "total_tasks": len(matches)})
    return {"schema": "gemmini_operator_library_coverage_v1", "programs_and_plans": rows,
            "scope": "Exact structural matches only; this does not predict connected latency"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    publish = commands.add_parser("context", aliases=["publish"])
    publish.add_argument("--campaign", type=Path, required=True)
    merge = commands.add_parser("merge")
    merge.add_argument("--campaigns", type=Path, nargs="+", required=True)
    merge.add_argument("--output-dir", "--output", dest="output", type=Path, required=True)
    report = commands.add_parser("coverage")
    report.add_argument("--database", type=Path, required=True)
    report.add_argument("--program", "--programs", dest="programs", type=Path, nargs="+", required=True)
    report.add_argument("--graph-fusion-level", type=int, choices=(-1, 0, 1, 2, 3))
    report.add_argument("--no-prepack-weights", dest="prepack_weights", action="store_false", default=None)
    report.add_argument("--no-bias-preload", dest="bias_preload", action="store_false", default=None)
    report.add_argument("--graph-cse", action="store_true",
                        help="Query the explicitly selected plan with stock Relax CSE enabled")
    report.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command in ("context", "publish"):
        print(publish_campaign_context(args.campaign))
    elif args.command == "merge":
        context = merge_campaigns(args.campaigns, args.output)
        print(json.dumps({"database": str(args.output.resolve()), "records": context["records"]}))
    else:
        from graph_plans import GraphPlan

        selected = None
        if (args.graph_fusion_level is not None or args.prepack_weights is not None
                or args.bias_preload is not None or args.graph_cse):
            selected = [GraphPlan(args.graph_fusion_level if args.graph_fusion_level is not None else 2,
                                  args.prepack_weights if args.prepack_weights is not None else False,
                                  bias_preload=args.bias_preload if args.bias_preload is not None else False,
                                  eliminate_common_subexpr=args.graph_cse)]
        _write_immutable(args.output, coverage(args.database, args.programs, graph_plans=selected))
        print(args.output)


if __name__ == "__main__":
    main()
