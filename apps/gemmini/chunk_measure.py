# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied. See the License for the
# specific language governing permissions and limitations
# under the License.
"""Measure a connected static package with the existing guarded DSE guest.

The exact same guest executes on Spike and RTL. Spike checks input hashes and
canaries; RTL checks every output, as in measure.py. One unconditional warmup
and one measured invocation include the complete package and CPU glue. This
search harness does not replace complete session/deployment qualification.
"""

import argparse
import json
import math
from pathlib import Path
import subprocess

import numpy as np

import measure
from compile_package import digest, verify_package
from qualify_package import elf_memory
from verify_baremetal_cpu import identity
from verify_matmul import audit_instructions


def check_array(value, specification):
    """Do not silently cast a reference or accept nonfinite expected outputs."""
    if list(value.shape) != specification["shape"] or value.dtype.name != specification["dtype"]:
        raise ValueError("Fixture shape/dtype differs from the exported package")
    if value.dtype.kind == "f" and (value.dtype.name != "float32" or not np.isfinite(value).all()):
        raise ValueError("Chunk guest requires finite FP32 floating tensors")
    return np.ascontiguousarray(value)


def layout_blobs(arrays, *, guards=False):
    """Construct aligned input/reference blobs or guarded output offsets."""
    offsets, blob = [], bytearray(64 if guards else 0)
    for value in arrays:
        offsets.append(len(blob))
        blob.extend(value.tobytes() if not guards else bytes(value.nbytes))
        size = measure._aligned(len(blob) + (64 if guards else 0)) + (64 if guards else 0)
        blob.extend(bytes(size - len(blob)))
    return offsets, blob


def build_chunk(package, fixture, provenance, folder, toolchain, *, rtol=1e-5, atol=2e-6):
    if not all(math.isfinite(v) and v >= 0 for v in (rtol, atol)):
        raise ValueError("Tolerances must be finite and nonnegative")
    if toolchain.timing_policy != "fixed-warm":
        raise ValueError("Connected chunk comparisons require fixed-warm timing")
    package, fixture, provenance, folder = map(lambda p: Path(p).resolve(),
                                              (package, fixture, provenance, folder))
    document = verify_package(package)
    graph = json.loads((package / "graph/graph.json").read_text())
    reference = json.loads(provenance.read_text())
    if reference.get("fixture_sha256") != digest(fixture) or not reference.get("reference_contract"):
        raise ValueError("Require an independently identified reference fixture")
    if not reference.get("reference_sources"):
        raise ValueError("Require reference source identities")
    for source in reference["reference_sources"]:
        if digest(source["path"]) != source["sha256"]:
            raise ValueError("Reference source changed: " + source["path"])
    comparison = reference.get("comparison")
    if comparison and (comparison["rtol"] != rtol or comparison["atol"] != atol):
        raise ValueError("Do not override the independent reference tolerance")
    build = json.loads((package / "build.json").read_text())
    if reference.get("program", {}).get("sha256") not in {s["sha256"] for s in build["sources"]}:
        raise ValueError("Reference program does not match package input")
    with np.load(fixture, allow_pickle=False) as data:
        inputs = [check_array(data[f"input_{i}"], spec) for i, spec in enumerate(graph["inputs"])]
        outputs = [check_array(data["expected" if len(graph["outputs"]) == 1 else f"expected_{i}"], spec)
                   for i, spec in enumerate(graph["outputs"])]
    if not outputs:
        raise ValueError("A measured chunk must have observable outputs")
    folder.mkdir(parents=True, exist_ok=False)
    input_offsets, input_blob = layout_blobs(inputs)
    expected_offsets, expected_blob = layout_blobs(outputs)
    output_offsets, output_blob = layout_blobs(outputs, guards=True)
    (folder / "inputs.bin").write_bytes(input_blob)
    (folder / "expected.bin").write_bytes(expected_blob)
    values = {
        "OUTPUT_AREA": len(output_blob) + 64,
        "WORKSPACE_AREA": measure._aligned(graph["workspace_bytes"]) + 128,
        "WORKSPACE_BYTES": graph["workspace_bytes"],
        "NUM_INPUTS": len(inputs), "NUM_OUTPUTS": len(outputs),
        "INPUT_OFFSETS": ",".join(map(str, input_offsets)) or "0",
        "INPUT_BYTES": ",".join(str(a.nbytes) for a in inputs) or "0",
        "INPUT_HASHES": ",".join(f"UINT64_C({measure._fnv(a.tobytes())})" for a in inputs) or "0",
        "OUTPUT_OFFSETS": ",".join(map(str, output_offsets)),
        "OUTPUT_BYTES": ",".join(str(a.nbytes) for a in outputs),
        "EXPECTED_OFFSETS": ",".join(map(str, expected_offsets)),
        "OUTPUT_KINDS": ",".join("1" if a.dtype.kind == "f" else "0" for a in outputs),
        "RTOL": float(rtol).hex() + "f", "ATOL": float(atol).hex() + "f",
    }
    (folder / "runner.c").write_text(measure.render_runner(values, timing_policy="fixed-warm"))
    (folder / "start.S").write_text(measure.STARTUP)
    (folder / "link.ld").write_text(measure.LINKER)
    assembly = ['.section .rodata.dse,"a",@progbits']
    for symbol, name in (("dse_inputs", "inputs.bin"), ("dse_expected", "expected.bin")):
        assembly += [".balign 64", f".globl {symbol}", f"{symbol}:",
                     ".incbin " + json.dumps(str(folder / name))]
    assembly += ['.section .note.GNU-stack,"",@progbits']
    (folder / "data.S").write_text("\n".join(assembly) + "\n")
    _, adapter = toolchain.adapter
    from tvm.relax.backend.contrib.gemmini_target import GemminiCapabilities
    if document["capability_fingerprint"] != GemminiCapabilities().fingerprint:
        raise ValueError("Package capability profile differs from the measurement compiler")
    if digest(package / "adapter.o") != adapter["object_sha256"]:
        raise ValueError("Package and measurement adapter differ")
    flags = ["-std=gnu11", "-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany",
             "-msmall-data-limit=0", "-O2", "-ffreestanding", "-fno-builtin",
             "-fno-stack-protector", "-fno-tree-loop-distribute-patterns", "-nostdlib",
             "-nostartfiles", "-static", "-Wl,--no-relax", "-Wl,-T," + str(folder / "link.ld")]
    forbidden = {v for k, v in adapter["instruction_policy"]["derived_definitions"].items()
                 if k.startswith("k_LOOP_")}
    nm = subprocess.check_output([str(toolchain.gcc), "-print-prog-name=nm"], text=True).strip()
    elfs, commands = {}, []
    for label, wrong in (("valid", 0), ("bad-oracle", 1)):
        elf = folder / (label + ".elf")
        command = [str(toolchain.gcc), *flags, f"-DBAD_ORACLE={wrong}",
                   *[str(folder / name) for name in ("start.S", "runner.c", "data.S")],
                   "-Wl,--start-group", *[str(package / name) for name in ("libmodel.a", "libm.a", "libgcc.a")],
                   "-Wl,--end-group", "-o", str(elf)]
        process = subprocess.run(command, capture_output=True, text=True, timeout=300)
        (folder / (label + "-build.log")).write_text(process.stdout + process.stderr)
        commands.append(command)
        if process.returncode:
            raise ValueError("Chunk guest link failed: " + process.stderr[-2000:])
        data = elf.read_bytes()
        audit = audit_instructions(data, adapter["instruction_policy"]["custom_opcode"], forbidden)
        if audit["prohibited_hits"] or subprocess.check_output([nm, "-u", str(elf)], text=True).strip():
            raise ValueError("Final guest instruction/symbol audit failed")
        elfs[label] = dict(identity(elf), memory_audit=elf_memory(data, 1 << 28), instruction_audit=audit)
    artifact = {"schema": "gemmini_connected_chunk_measurement_v1", "folder": str(folder),
                "elfs": elfs, "workload": str(package), "package": identity(package / "package.json"),
                "program_sha256": reference["program"]["sha256"],
                "fixture": identity(fixture), "reference_provenance": identity(provenance),
                "memory_model": toolchain.memory_configuration(), "timing_policy": "fixed-warm",
                "timing_scope": "whole model_run; includes CPU glue, routing within package, admission and completion",
                "guards": "Spike input hashes/canaries; both backends every output; same ELF",
                "runner_sha256": digest(folder / "runner.c"), "commands": commands,
                "comparison": {"rtol": rtol, "atol": atol, "integer": "exact", "finite": True},
                "sources": [identity(Path(__file__)), identity(Path(measure.__file__))]}
    (folder / "artifact.json").write_text(json.dumps(artifact, indent=2) + "\n")
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--fixture-provenance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--toolchain", type=Path, required=True)
    parser.add_argument("--max-cycles", type=int, default=100_000_000)
    parser.add_argument("--rtl-seconds", type=int, default=3600)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--negative-rtl", action="store_true")
    args = parser.parse_args()
    if args.max_cycles < 1 or not 1 <= args.rtl_seconds <= 14400:
        parser.error("Use positive cycles and an RTL wall cap between 1 and 14400 seconds")
    toolchain = measure.Toolchain.from_json(json.loads(args.toolchain.read_text()))
    toolchain.provenance()
    artifact = build_chunk(args.package, args.fixture, args.fixture_provenance, args.output_dir, toolchain)
    if not args.build_only:
        result = measure.run_candidate(artifact, toolchain, max_cycles=args.max_cycles,
                                       rtl_seconds=args.rtl_seconds)
        (args.output_dir / "measurement.json").write_text(json.dumps(result, indent=2) + "\n")
        if args.negative_rtl and result["status"] == "passed":
            measure.run_negative_rtl(artifact, toolchain, max_cycles=args.max_cycles)
        print(json.dumps(result), flush=True)
        if result["status"] != "passed":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
