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
"""Retain verified comparison measurements in the deployment JSONDatabase."""
import argparse
import hashlib
import json
from pathlib import Path
import tvm
from tvm import meta_schedule as ms
from tvm.relax.backend.contrib.gemmini_search import open_gemmini_database, WORKLOAD_ATTR
from tvm.relax.backend.contrib.gemmini_target import GemminiCapabilities
from tvm.relax.backend.contrib.gemmini_tensorir import prepare_gemmini_tensorir_workloads


def trace_key(trace):
    return json.dumps(json.loads(tvm.ir.save_json(trace.as_json())), sort_keys=True)


def merge(root):
    root = Path(root).resolve()
    if not all((root / name).exists() for name in ("campaign.json", "replay-campaign.json")):
        raise ValueError("Both searches must finish before comparison records are aggregated")
    program = tvm.ir.load_json((root / "program.json").read_text())
    prepared, _, _ = prepare_gemmini_tensorir_workloads(program)
    caps = GemminiCapabilities()
    destination = open_gemmini_database(root / "database", prepared, capabilities=caps)
    source = ms.database.JSONDatabase(work_dir=str(root / "replay-database"))
    verified = set()
    lines = (root / "replay-trials.jsonl").read_text().splitlines()
    retries = root / "replay-retry-trials.jsonl"
    if retries.exists():
        lines += retries.read_text().splitlines()
    for line in lines:
        row = json.loads(line)
        if row.get("cycles") is None:
            continue
        artifact = Path(row["artifact"])
        outcome = json.loads((artifact.parent / "measurement.json").read_text())
        if outcome["status"] != "passed" or float(outcome["cycles"]) != row["cycles"]:
            raise ValueError("Comparison label is not backed by a successful measurement")
        verified.add((row["workload"], float(row["cycles"]), json.dumps(row["trace"], sort_keys=True)))

    def key(record):
        costs = record.run_secs
        return (str(record.workload.mod["main"].attrs[WORKLOAD_ATTR]),
                float(costs[0]) if costs else None, trace_key(record.trace))

    existing = {key(record) for record in destination.get_all_tuning_records()}
    matched, imported = set(), 0
    for record in source.get_all_tuning_records():
        identity = key(record)
        if identity not in verified:
            continue
        if str(record.target) != str(caps.target()):
            raise ValueError("Comparison record has a different target capability profile")
        matched.add(identity)
        if identity in existing:
            continue
        destination.commit_tuning_record(ms.database.TuningRecord(
            record.trace, destination.commit_workload(record.workload.mod), record.run_secs,
            record.target, record.args_info))
        existing.add(identity)
        imported += 1
    if matched != verified:
        raise ValueError("A verified comparison trace was not found in the standard database")
    record = {"schema": "gemmini_comparison_merge_v1", "verified_records": len(verified),
              "newly_imported_records": imported, "method": "JSONDatabase.commit_tuning_record",
              "source_records_sha256": hashlib.sha256(
                  (root / "replay-database/database_tuning_record.json").read_bytes()).hexdigest()}
    (root / "comparison-merge.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    args = parser.parse_args()
    print(json.dumps(merge(args.campaign)))


if __name__ == "__main__":
    main()
