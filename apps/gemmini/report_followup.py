#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Report explicit follow-up receipts, keeping semantic prose outside observations."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bundles", type=Path, nargs="+", required=True)
    parser.add_argument("--searches", type=Path, nargs="*", default=[])
    parser.add_argument("--tests", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    bindings, seen = [], {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        key = str(path)
        if key not in seen:
            seen[key] = {
                "path": key,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
            bindings.append(seen[key])
        if expected is not None and seen[key]["sha256"] != expected:
            raise ValueError("Evidence changed: " + key)
        return seen[key]

    def load(path):
        bind(path)
        return json.loads(Path(path).read_text())

    bundles, searches, tests = [], [], []
    for folder in args.bundles:
        bundle = load(folder / "bundle.json")
        runs = []
        for path in sorted(folder.glob("*-result.json")):
            row = load(path)
            for item in [*row.get("bindings", []), row["ELF"], row["log"]]:
                bind(item["path"], item["sha256"])
            timing = row.get("timing") or {}
            runs.append(
                {
                    "receipt": str(path.resolve()),
                    **{
                        key: row.get(key)
                        for key in (
                            "status",
                            "backend",
                            "negative",
                            "timed_out",
                            "wall_seconds",
                            "cycle_limit",
                            "timing_error",
                        )
                    },
                    "samples": timing.get("samples_by_stage"),
                    "metric": timing.get("metric"),
                    "scope": timing.get("scope"),
                    "usable_for_dse": timing.get("usable_for_dse", False),
                }
            )
        # Argument and result types come from the exported buffer ABI, not an op
        # name or maintained datatype explanation.
        exports = {}
        manifests = list((folder / "package").glob("*/graph.json"))
        if not manifests:
            manifests = [
                Path(item["path"])
                for item in bundle["bindings"]
                if item["path"].endswith("/graph/graph.json")
            ]
        for path in manifests:
            manifest = load(path)
            tensors = {
                item["name"]: item
                for group in ("inputs", "outputs", "constants", "intermediates", "views")
                for item in manifest.get(group, [])
            }
            calls = []
            for call in manifest["calls"]:
                calls.append(
                    {
                        "symbol": call["symbol"],
                        "inputs": [tensors[name] for name in call["inputs"]],
                        "outputs": [tensors[name] for name in call["outputs"]],
                    }
                )
            exports[path.parent.name] = {
                "calls": calls,
                "memory_preflight": manifest["memory_preflight"],
            }
        bundles.append(
            {
                "directory": str(folder.resolve()),
                "cases": bundle["cases"],
                "scheduling": bundle.get("scheduling", {}),
                "exported_buffer_abis": exports,
                "runs": runs,
                "comparison": bundle.get("comparison"),
                "timing_repeats": bundle.get("timing_repeats", 0),
            }
        )
    for folder in args.searches:
        document = load(folder / "search-result.json")
        for record in document["records"]:
            evidence = record.get("measurement", {}).get("receipt") or record.get("evidence")
            if isinstance(evidence, dict):
                bind(evidence["path"], evidence["sha256"])
        searches.append(
            {
                "directory": str(folder.resolve()),
                "summary": document["summary"],
                "winner": document["winner"],
                "records": document["records"],
            }
        )
    for path in args.tests:
        bind(path)
        summaries = re.findall(r"^.*\b\d+ passed(?:[^\n]*)$", path.read_text(), re.MULTILINE)
        tests.append(
            {
                "log": str(path.resolve()),
                "summary": summaries[-1] if summaries else "no passing summary",
            }
        )
    bind(__file__)
    report = {
        "scope": "Explicit local compiler follow-up evidence; no FireSim or pretrained application-quality qualification",
        "bundles": bundles,
        "searches": searches,
        "tests": tests,
        "bindings": bindings,
        "semantic_explanations": "Maintained separately in EL4_STATUS.md; no source-op dispatch is inferred from an op spelling",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    lines = [
        "# Reviewer follow-up evidence",
        "",
        report["scope"],
        "",
        "Warm timing intervals cover model_run admission/completion after one warmup; oracle/startup/HTIF are excluded. Spike counters are functional only. Host affinity and load affect wall seconds, not simulated guest cycle intervals.",
        "",
        "| Bundle | Backend / control | Result | Guest samples / metric | Host seconds | Cycle ceiling |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    for bundle in bundles:
        for row in bundle["runs"]:
            samples = str(row["samples"]) if row["samples"] else "correctness only / none"
            lines.append(
                f"| {Path(bundle['directory']).name} | {row['backend']} / {'wrong oracle' if row['negative'] else 'positive'} | {row['status']} | {samples}; {row['metric']} | {row['wall_seconds']:.2f} | {row['cycle_limit']} |"
            )
    lines += [
        "",
        "## Measured function placement",
        "",
        "Every row is an isolated post-fusion function. Incomplete baselines cannot establish a CPU/native gain. Full-graph costs need their own matched comparison.",
        "",
        "| Search | Baselines complete | CPU median | Gemmini median | Selected |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for search in searches:
        costs = {
            kind: [
                statistics.median(row["measurement"]["samples"])
                for row in search["records"]
                if row["placement"] == kind and row.get("status") == "passed"
            ]
            for kind in ("cpu", "gemmini")
        }
        lines.append(
            f"| {Path(search['directory']).name} | {search['summary'].get('baseline_complete')} | {min(costs['cpu']) if costs['cpu'] else 'incomplete'} | {min(costs['gemmini']) if costs['gemmini'] else 'incomplete'} | {search['winner']['placement']} |"
        )
    lines += [
        "",
        "## Observed function dispatch",
        "",
        "Names and placement/proofs below come from compiler bundle reports; datatype/shape buffer ABIs are retained in the JSON. Static function sites are not dynamic RoCC command counts.",
        "",
        "| Bundle / function | Placement | Buffer datatypes in → out | Native operation / CPU reason | Bias preload | Scaled readout |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for bundle in bundles:
        for rows in bundle["scheduling"].values():
            for row in rows:
                meta = row.get("metadata", {})
                reason = meta.get("operation") or row.get("reason") or row.get("origin", "")
                reason = str(reason).replace("|", "/")
                calls = [
                    call
                    for export in bundle["exported_buffer_abis"].values()
                    for call in export["calls"]
                    if call["symbol"].endswith(row.get("function", "no_named_function"))
                ]
                types = (
                    "; ".join(
                        sorted(
                            set(
                                ",".join(item["dtype"] for item in call["inputs"])
                                + " → "
                                + ",".join(item["dtype"] for item in call["outputs"])
                                for call in calls
                            )
                        )
                    )
                    or "see fixture ABI"
                )
                lines.append(
                    f"| {Path(bundle['directory']).name} / {row.get('function', 'primitive fixture')} | {row['placement']} | {types} | {reason} | {meta.get('bias_preloaded', False)} | {meta.get('readout_fused', meta.get('scaled_readout', False))} |"
                )
    lines += ["", "## Tests", ""] + [
        f"- {Path(row['log']).name}: {row['summary']}" for row in tests
    ]
    lines += [
        "",
        "The JSON binds every execution ELF/log/source input and preserves all supplied failures. Historical reviewer and earlier source-freeze results remain separate; no deployment speedup is inferred.",
        "",
    ]
    with args.output.with_suffix(".md").open("x") as stream:
        stream.write("\n".join(lines))
    print(
        json.dumps({"status": "reported", "report": str(args.output.with_suffix(".md").resolve())})
    )


if __name__ == "__main__":
    main()
