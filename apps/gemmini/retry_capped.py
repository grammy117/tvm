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
"""Retry an existing capped ELF after search, without proposing or replaying schedules."""
import argparse
import json
from pathlib import Path

import tvm
from tvm import meta_schedule as ms
from tvm.relax.backend.contrib.gemmini_search import WORKLOAD_ATTR, open_gemmini_database
from tvm.relax.backend.contrib.gemmini_tensorir import prepare_gemmini_tensorir_workloads
import measure


def retry(root, artifact_path, *, max_cycles, replay=False):
    root, artifact_path = Path(root).resolve(), Path(artifact_path).resolve()
    prefix = "replay-" if replay else ""
    if not (root / (prefix + "campaign.json")).is_file():
        raise ValueError("Search must finish before a cap retry")
    artifact = json.loads(artifact_path.read_text())
    previous = json.loads((artifact_path.parent / "measurement.json").read_text())
    if (previous.get("status") != "rtl_timeout_or_failure"
            or not previous.get("cycle_cap") or previous["cycle_cap"] >= max_cycles):
        raise ValueError("Only a capped failed measurement with an increased allowance can retry")
    for elf in artifact["elfs"].values():
        actual = measure.identity(elf["path"])
        if actual["sha256"] != elf["sha256"] or actual["bytes"] != elf["bytes"]:
            raise ValueError("Retained executable bytes changed")
    trials = [json.loads(line) for line in
              (root / (prefix + "trials.jsonl")).read_text().splitlines()]
    trial = next((row for row in trials if row.get("artifact")
                  and Path(row["artifact"]).resolve() == artifact_path), None)
    if trial is None or trial.get("cycles") is not None:
        raise ValueError("Artifact must belong to a failed trial in this completed search")
    program = tvm.ir.load_json((root / "program.json").read_text())
    prepared, _, _ = prepare_gemmini_tensorir_workloads(program)
    database = open_gemmini_database(root / (prefix + "database"), prepared)
    trace = trial["trace"]
    source = next((record for record in database.get_all_tuning_records()
                   if str(record.workload.mod["main"].attrs[WORKLOAD_ATTR]) == trial["workload"]
                   and json.loads(tvm.ir.save_json(record.trace.as_json())) == trace), None)
    if source is None:
        raise ValueError("Original sampled trace is missing from the standard database")
    provenance = json.loads((root / "hardware-provenance.json").read_text())
    receipts = provenance["receipts"]
    toolchain = measure.Toolchain(Path(provenance["binaries"]["gcc"]["path"]),
                                 Path(receipts[2]["path"]), Path(receipts[1]["path"]),
                                 Path(receipts[0]["path"]))
    toolchain.provenance()
    folder = artifact_path.parent / "retries" / f"cap{max_cycles}"
    folder.mkdir(parents=True, exist_ok=True)
    for name, path in (("retry_capped.py", Path(__file__)),
                       ("measure.py", Path(measure.__file__))):
        snapshot = folder / name
        if not snapshot.exists():
            snapshot.write_bytes(path.read_bytes())
    provenance_path = folder / "retry-provenance.json"
    if not provenance_path.exists():
        provenance_path.write_text(json.dumps({
            "selection": "Explicit retained artifact; no new schedule or trace replay",
            "original_measurement": measure.identity(artifact_path.parent / "measurement.json"),
            "original_artifact": measure.identity(artifact_path),
            "source_files": [measure.identity(folder / name)
                             for name in ("retry_capped.py", "measure.py")],
            "hardware": toolchain.provenance(),
        }, indent=2) + "\n")
    retained = dict(artifact, folder=str(folder), original_artifact=str(artifact_path))
    retained_path = folder / "artifact.json"
    retained_path.write_text(json.dumps(retained, indent=2) + "\n")
    measurement_path = folder / "measurement.json"
    if measurement_path.exists():
        outcome = json.loads(measurement_path.read_text())
    else:
        outcome = measure.run_candidate(retained, toolchain, max_cycles=max_cycles)
        measurement_path.write_text(json.dumps(outcome, indent=2) + "\n")
    row = dict(trial, artifact=str(retained_path), original_artifact=str(artifact_path),
               original_cycle_cap=previous["cycle_cap"], retry_cycle_cap=max_cycles,
               post_search_retry=True, status=outcome["status"],
               original_run_error=trial.get("run_error"),
               run_error=None if outcome["status"] == "passed" else outcome["status"],
               record_writer=measure.identity(__file__),
               cycles=outcome.get("cycles"), rtl_seconds=outcome.get("rtl_seconds"))
    if outcome["status"] == "passed":
        exists = any(record.run_secs and float(record.run_secs[0]) == outcome["cycles"]
                     and str(record.workload.mod["main"].attrs[WORKLOAD_ATTR]) == trial["workload"]
                     and json.loads(tvm.ir.save_json(record.trace.as_json())) == trace
                     for record in database.get_all_tuning_records())
        if not exists:
            database.commit_tuning_record(ms.database.TuningRecord(
                source.trace, source.workload, [float(outcome["cycles"])],
                source.target, source.args_info))
        cache_path = root / "measurement-cache.json"
        cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        cache[trial["workload"] + ":" + artifact["elfs"]["valid"]["sha256"]] = outcome
        cache_path.write_text(json.dumps(cache) + "\n")
    log = root / (prefix + "retry-trials.jsonl")
    rows = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    rows = [old for old in rows if old["artifact"] != row["artifact"]] + [row]
    log.write_text("".join(json.dumps(old) + "\n" for old in rows))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--max-cycles", type=int, required=True)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    row = retry(args.campaign, args.artifact, max_cycles=args.max_cycles, replay=args.replay)
    print(json.dumps({key: row[key] for key in
                      ("workload", "artifact", "status", "cycles", "retry_cycle_cap")}))


if __name__ == "__main__":
    main()
