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
"""Execute one measured operator from the actual exported package archive.

The small wrapper supplies the campaign's fixed generic operator inputs and
oracle. It links libmodel.a directly, without recompiling or replaying the
operator. Repeated model calls and unrelated operators need not be simulated.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from compile_package import verify_package
import measure
from verify_matmul import audit_instructions


def qualify(campaign, package, workload, *, max_cycles=6_000_000):
    source_text = Path(__file__).read_text()
    measurement_source = Path(measure.__file__).read_text()
    campaign, package = Path(campaign).resolve(), Path(package).resolve()
    document = verify_package(package)
    results = json.loads((campaign / "results.json").read_text())
    task = next(row for row in results["tasks"] if row["workload"] == workload)
    source = Path(task["best_measurement_artifact"]).parent
    artifact = json.loads((source / "artifact.json").read_text())
    measured = json.loads((source / "measurement.json").read_text())
    if measured.get("status") != "passed":
        raise ValueError("Package slice requires a previously verified operator fixture")
    symbol = task["task"]
    graph = json.loads((package / "graph/graph.json").read_text())
    original_graph = json.loads((source / "package/graph.json").read_text())
    if symbol not in {call["symbol"] for call in graph["calls"]}:
        raise ValueError("Extracted task does not name an operator in this package")
    if symbol not in document["tuned_functions"]:
        raise ValueError("Package operator did not use a measured database placement")
    required = graph["operator_workspaces"].get(symbol, {}).get("bytes", 0)
    if required != original_graph["workspace_bytes"]:
        raise ValueError("Exported operator workspace differs from its measured slice ABI")
    wrapper = (source / "package/model.c").read_text()
    if wrapper.count("kernel(") != 2:
        raise ValueError("Expected one kernel declaration and one call in the measured wrapper")
    wrapper = wrapper.replace("kernel(", symbol + "(")
    out = package.parent / (package.name + "-qualification") / workload
    out.mkdir(parents=True, exist_ok=False)
    (out / "qualification-source.py").write_text(source_text)
    (out / "measurement-source.py").write_text(measurement_source)
    for name in ("runner.c", "start.S", "link.ld", "data.S"):
        shutil.copy2(source / name, out / name)
    shutil.copy2(source / "package/constants.S", out / "constants.S")
    (out / "model.c").write_text(wrapper)
    provenance = json.loads((campaign / "hardware-provenance.json").read_text())
    receipts = provenance["receipts"]
    toolchain = measure.Toolchain(
        Path(provenance["binaries"]["gcc"]["path"]), Path(receipts[2]["path"]),
        Path(receipts[1]["path"]), Path(receipts[0]["path"]))
    verified_provenance = toolchain.provenance()
    adapter = json.loads((package / "build.json").read_text())["adapter"]
    policy = adapter["instruction_policy"]
    forbidden = {value for name, value in policy["derived_definitions"].items()
                 if name.startswith("k_LOOP_")}
    gcc = toolchain.gcc
    flags = [
        "-std=gnu11", "-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany",
        "-msmall-data-limit=0", "-O2", "-ffreestanding", "-fno-builtin",
        "-fno-stack-protector", "-fno-tree-loop-distribute-patterns", "-nostdlib",
        "-nostartfiles", "-static", "-Wl,--no-relax", "-Wl,-T," + str(out / "link.ld"),
    ]
    commands, elfs = [], {}
    for label, wrong in (("valid", 0), ("bad-oracle", 1)):
        elf = out / (label + ".elf")
        command = [str(gcc), *flags, f"-DBAD_ORACLE={wrong}",
                   *[str(out / name) for name in
                     ("start.S", "runner.c", "data.S", "model.c", "constants.S")],
                   "-Wl,--start-group",
                   *[str(package / name) for name in ("libmodel.a", "libm.a", "libgcc.a")],
                   "-Wl,--end-group", "-o", str(elf)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        (out / (label + "-link.log")).write_text(result.stdout + result.stderr)
        commands.append({"argv": command, "returncode": result.returncode})
        if result.returncode:
            raise RuntimeError(result.stderr[-4000:])
        audit = audit_instructions(elf.read_bytes(), policy["custom_opcode"], forbidden)
        if audit["prohibited_hits"]:
            raise ValueError("Package slice contains prohibited hardware-loop instructions")
        nm = gcc.with_name(gcc.name[:-3] + "nm")
        if subprocess.check_output([str(nm), "-u", str(elf)], text=True).strip():
            raise ValueError("Package slice contains unresolved symbols")
        elfs[label] = dict(measure.identity(elf), instruction_audit=audit)
    qualified = {
        "folder": str(out), "native": artifact["native"], "elfs": elfs,
        "workload": artifact["workload"],
        "program_sha256": hashlib.sha256((package / "libmodel.a").read_bytes()).hexdigest(),
        "package": measure.identity(package / "package.json"),
        "archive": measure.identity(package / "libmodel.a"), "operator": symbol,
    }
    (out / "artifact.json").write_text(json.dumps(qualified, indent=2) + "\n")
    with ThreadPoolExecutor(2) as pool:
        positive_future = pool.submit(measure.run_candidate, qualified, toolchain,
                                      max_cycles=max_cycles)
        negative_future = pool.submit(measure.run_negative_rtl, qualified, toolchain,
                                      max_cycles=max_cycles)
        positive, negative = positive_future.result(), negative_future.result()
    (out / "measurement.json").write_text(json.dumps(positive, indent=2) + "\n")
    record = {
        "status": "passed" if positive["status"] == negative["status"] == "passed" else "failed",
        "scope": "One operator linked from the exported libmodel.a; generic fixed inputs, not full-model accuracy or deployed FireSim timing",
        "workload": workload, "operator": symbol, "positive": positive, "negative": negative,
        "artifact": qualified, "commands": commands, "hardware_provenance": verified_provenance,
        "bindings": [measure.identity(source / "artifact.json"),
                     measure.identity(Path(artifact["workload"]) / "workload-manifest.json"),
                     measure.identity(out / "qualification-source.py"),
                     measure.identity(out / "measurement-source.py")],
    }
    record["bindings"].extend(measure.identity(out / name) for name in
                              ("runner.c", "model.c", "start.S", "link.ld", "data.S", "constants.S"))
    (out / "qualification.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": record["status"], "qualification": str(out / "qualification.json")}),
          flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("package", type=Path)
    parser.add_argument("workload")
    parser.add_argument("--max-cycles", type=int, default=6_000_000)
    args = parser.parse_args()
    result = qualify(args.campaign, args.package, args.workload, max_cycles=args.max_cycles)
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
