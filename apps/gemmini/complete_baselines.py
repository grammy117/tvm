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
"""Repair refused default baselines and retain them in a completed standard database.

The operator workloads and inputs stay fixed. Each repair records the compiler
bytes used; existing successful measurements are never simulated again.
"""
import argparse
import json
from pathlib import Path
import shutil

import tvm
from tvm import meta_schedule as ms
from tvm.relax.backend.contrib.gemmini_search import open_gemmini_database
from tvm.relax.backend.contrib.gemmini_sketch import default_schedule
from tvm.relax.backend.contrib.gemmini_target import GemminiCapabilities
from tvm.relax.backend.contrib.gemmini_tensorir import prepare_gemmini_tensorir_workloads
import measure


def complete(root, *, commit=False, commit_only=False, max_cycles=6_000_000):
    root = Path(root).resolve()
    inventory = json.loads((root / "tasks.json").read_text())
    provenance = json.loads((root / "hardware-provenance.json").read_text())
    receipts = provenance["receipts"]
    toolchain = measure.Toolchain(Path(provenance["binaries"]["gcc"]["path"]),
                                 Path(receipts[2]["path"]), Path(receipts[1]["path"]),
                                 Path(receipts[0]["path"]))
    repairs = []
    for item in inventory["tasks"]:
        key = item["workload"]
        baseline_path = root / "baselines" / key / "default.json"
        previous = {}
        if baseline_path.exists():
            record = json.loads(baseline_path.read_text())
            previous = record
            if record.get("status") == "passed":
                item["default"] = record
                continue
        if commit_only:
            item["default"] = previous
            continue
        # Do not race an ongoing baseline that has not reported a refusal.
        if not commit and item.get("default", {}).get("status") != "failed":
            continue
        workload = root / "workloads" / key
        func = tvm.ir.load_json((workload / "workload.json").read_text())["main"]
        sch = default_schedule(func)
        folder = root / "baseline-repairs" / key
        artifact_path = folder / "artifact.json"
        original = Path(previous.get("artifact", ""))
        if original.is_file():
            original_measurement = original.parent / "measurement.json"
            if original_measurement.exists():
                original_result = json.loads(original_measurement.read_text())
                if (original_result.get("status") != "passed"
                        and original_result.get("cycle_cap", max_cycles) < max_cycles):
                    # Expanding an allowance must run the same executable, not
                    # silently replace the default with freshly compiled code.
                    artifact_path, folder = original, original.parent
        if artifact_path.exists():
            artifact = json.loads(artifact_path.read_text())
        else:
            folder.parent.mkdir(parents=True, exist_ok=True)
            artifact = measure.build_candidate(sch.mod, workload, folder, toolchain)
        measured = folder / "measurement.json"
        outcome = None
        if measured.exists():
            outcome = json.loads(measured.read_text())
            if (outcome["status"] != "passed"
                    and outcome.get("cycle_cap", max_cycles) < max_cycles):
                suffix = f".cap{outcome['cycle_cap']}"
                for name in ("measurement.json", "rtl-timing.log"):
                    path = folder / name
                    if path.exists() and not path.with_name(name + suffix).exists():
                        shutil.copy2(path, path.with_name(name + suffix))
                outcome = None
        if outcome is None:
            outcome = measure.run_candidate(artifact, toolchain, max_cycles=max_cycles)
            measured.write_text(json.dumps(outcome, indent=2) + "\n")
        record = {"status": outcome["status"], "artifact": str(artifact_path),
                  "compiler": artifact.get("compiler", previous.get("compiler")), "repair": True}
        if outcome["status"] == "passed":
            record.update(status="passed", cycles=outcome["cycles"])
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(record, indent=2) + "\n")
        item["default"] = record
        repairs.append(record)
        print(json.dumps({"workload": key, "baseline": record}), flush=True)
    if commit:
        if not (root / "campaign.json").exists():
            raise ValueError("Wait for search to finish before adding baseline records")
        program = tvm.ir.load_json((root / "program.json").read_text())
        prepared, _, _ = prepare_gemmini_tensorir_workloads(program)
        caps = GemminiCapabilities()
        database = open_gemmini_database(root / "database", prepared, capabilities=caps)
        traces = {str(record.trace) for record in database.get_all_tuning_records()}
        for item in inventory["tasks"]:
            if item.get("default", {}).get("status") != "passed":
                continue
            mod = tvm.ir.load_json((root / "workloads" / item["workload"] / "workload.json").read_text())
            sch = default_schedule(mod["main"])
            if str(sch.trace) not in traces:
                database.commit_tuning_record(ms.database.TuningRecord(
                    sch.trace, database.commit_workload(mod), [float(item["default"]["cycles"])],
                    caps.target(), []))
        (root / "tasks.json").write_text(json.dumps(inventory, indent=2) + "\n")
    return repairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--commit", action="store_true")
    parser.add_argument("--commit-only", action="store_true",
                        help="Import completed baseline measurements without running repairs")
    parser.add_argument("--max-cycles", type=int, default=6_000_000)
    args = parser.parse_args()
    complete(args.campaign, commit=args.commit or args.commit_only,
             commit_only=args.commit_only, max_cycles=args.max_cycles)


if __name__ == "__main__":
    main()
