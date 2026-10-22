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
"""Summarize verified per-operator campaigns using the standard tuning database."""

import argparse
import csv
import json
from pathlib import Path

import tvm
from tvm import tir
from tvm.relax.backend.contrib.gemmini_search import open_gemmini_database
from tvm.relax.backend.contrib.gemmini_target import GemminiCapabilities
from tvm.relax.backend.contrib.gemmini_tensorir import prepare_gemmini_tensorir_workloads


def summarize(root):
    root = Path(root).resolve()
    inventory = json.loads((root / "tasks.json").read_text())
    program = tvm.ir.load_json((root / "program.json").read_text())
    prepared, _, _ = prepare_gemmini_tensorir_workloads(program)
    caps = GemminiCapabilities()
    database = open_gemmini_database(root / "database", prepared, capabilities=caps)
    def read_trials(name):
        path = root / name
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    trials = read_trials("trials.jsonl")
    replay_trials = read_trials("replay-trials.jsonl")
    retry_trials = read_trials("retry-trials.jsonl") + read_trials("replay-retry-trials.jsonl")
    tasks = []
    for item in inventory["tasks"]:
        key = item["workload"]
        workload = tvm.ir.load_json((root / "workloads" / key / "workload.json").read_text())
        winner = database.query_tuning_record(workload, caps.target(), "main")
        row = dict(item)
        for kind, name in (("default", "default"), ("cpu_untuned", "cpu-untuned")):
            path = root / "baselines" / key / (name + ".json")
            if path.exists():
                row[kind] = json.loads(path.read_text())
        measured = [trial for trial in trials if trial["workload"] == key]
        replay = [trial for trial in replay_trials if trial["workload"] == key]
        retries = [trial for trial in retry_trials if trial["workload"] == key]
        row["post_search_retry_trials"] = len(retries)
        row["post_search_retry_best_cycles"] = min(
            (trial["cycles"] for trial in retries if trial.get("cycles") is not None), default=None)
        cpu = [trial for trial in measured + replay + retries
               if trial.get("cycles") is not None and not trial.get("native")]
        row["cpu_scheduled_best_cycles"] = min((trial["cycles"] for trial in cpu), default=None)
        row["cpu_scheduled_scope"] = "Successful stock LLVM candidates from both searches and any separate cap retries"
        row["search_trials"] = len(measured)
        row["successful_search_trials"] = sum(trial.get("cycles") is not None for trial in measured)
        row["reused_measurements"] = sum(bool(trial.get("reused")) for trial in measured)
        row["evolutionary_search_best_cycles"] = min(
            (trial["cycles"] for trial in measured if trial.get("cycles") is not None), default=None)
        row["replay_trials"] = len(replay)
        row["replay_successful_trials"] = sum(trial.get("cycles") is not None for trial in replay)
        row["replay_best_cycles"] = min(
            (trial["cycles"] for trial in replay if trial.get("cycles") is not None), default=None)
        matched_budget = min(len(measured), len(replay))
        row["matched_trial_budget"] = matched_budget
        for label, proposals in (("evolutionary", measured), ("replay", replay)):
            row[f"matched_budget_{label}_best_cycles"] = min(
                (trial["cycles"] for trial in proposals[:matched_budget]
                 if trial.get("cycles") is not None), default=None)
        if winner is not None:
            sch = tir.Schedule(workload)
            winner.trace.apply_to_schedule(sch, remove_postproc=False)
            cycles = float(winner.run_secs[0])
            row["best_cycles"] = cycles
            matching = [trial for trial in measured + replay + retries if trial.get("cycles") == cycles]
            trace_json = json.loads(tvm.ir.save_json(winner.trace.as_json()))
            match = next((trial for trial in matching if trial.get("trace") == trace_json),
                         matching[0] if matching else None)
            row["best_measurement_artifact"] = (
                match["artifact"] if match else row.get("default", {}).get("artifact"))
            row["best_measurement_stage"] = ("post-search cap retry" if match and
                                              match.get("post_search_retry") else
                                              "search" if match else "default")
            row["best_sketch"] = str(sch.mod["main"].body.block.annotations.get("gemmini.sketch", "cpu"))
            row["best_decisions"] = []
            decisions = dict(winner.trace.as_json()[1])
            for index, inst in enumerate(winner.trace.insts):
                if inst.kind.name.startswith("Sample"):
                    row["best_decisions"].append({"instruction": inst.kind.name,
                                                 "decision": decisions[index]})
            if row["best_sketch"] != "cpu":
                tiles = [item["decision"] for item in row["best_decisions"]
                         if item["instruction"] == "SamplePerfectTile"]
                row["sampled_tile_factors_outer_reuse"] = dict(zip(
                    ("M", "N", "K", "tail_M", "tail_N", "tail_K"), tiles))
            trace_path = root / "best" / key / "trace.py"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text(str(winner.trace) + "\n")
            (trace_path.parent / "scheduled.json").write_text(tvm.ir.save_json(sch.mod))
            row["best_trace"] = str(trace_path.relative_to(root))
            default = row.get("default", {}).get("cycles")
            row["speedup_vs_default"] = default / cycles if default else None
            cpu_cycles = row["cpu_scheduled_best_cycles"]
            row["speedup_vs_scheduled_cpu"] = cpu_cycles / cycles if cpu_cycles else None
        tasks.append(row)
    weighted = {}
    for field, getter in (("default_cycles", lambda row: row.get("default", {}).get("cycles")),
                          ("best_cycles", lambda row: row.get("best_cycles"))):
        values = [getter(row) for row in tasks]
        weighted[field] = sum(row["weight"] * value for row, value in zip(tasks, values)) if all(value is not None for value in values) else None
    result = {
        "schema": "gemmini_metaschedule_results_v1", "model": inventory["model"],
        "scope": "Selected integer operator recipes at model shapes with synthetic weights; not full-model latency or accuracy",
        "timing": "RTL cycles inside model_run; warm second call below 400000 cold cycles, otherwise first call",
        "database_metric": "target cycles stored in MetaSchedule run_secs for ranking; not wall seconds",
        "cpu_timeout_policy": "Unfinished bounded simulations have no cycle value and no speedup ratio",
        "database": "database", "operators": inventory["operators"], "tasks": tasks,
        "call_count_weighted_selected_ops": weighted,
        "replay_comparison": "Separate stock ReplayTrace search; compare successful trial counts and budgets. Default records are excluded from search-only minima.",
        "retry_scope": "Post-search cap retries run retained ELFs unchanged and remain outside search-only minima and matched budgets.",
    }
    for name in ("campaign.json", "replay-campaign.json", "database/cost-model.json",
                 "comparison-merge.json"):
        if (root / name).exists():
            result[name] = json.loads((root / name).read_text())
    provenance = root / "hardware-provenance.json"
    if provenance.exists():
        result["hardware_provenance"] = json.loads(provenance.read_text())
    (root / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    with (root / "results.csv").open("w", newline="") as stream:
        fields = ["task", "weight", "best_cycles", "default_cycles", "cpu_untuned_cycles",
                  "cpu_scheduled_best_cycles", "speedup_vs_default", "best_sketch",
                  "evolutionary_search_best_cycles", "replay_best_cycles"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in tasks:
            record = {key: row.get(key) for key in fields}
            record["default_cycles"] = row.get("default", {}).get("cycles")
            record["cpu_untuned_cycles"] = row.get("cpu_untuned", {}).get("cycles")
            writer.writerow(record)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    args = parser.parse_args()
    result = summarize(args.campaign)
    print(json.dumps({"model": result["model"], "tasks": len(result["tasks"]),
                      "results": str(args.campaign / "results.json")}))


if __name__ == "__main__":
    main()
