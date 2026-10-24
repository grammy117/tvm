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
"""Link guarded positive/wrong-oracle boot ELFs from a portable compiler package.

No simulator or FireSim is launched. Use verify_tensorir.py run on the resulting
bundle for local execution. A final FireSim loader/DRAM/driver check is separate.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import struct

import numpy as np

from compile_package import verify_package, FLAGS, digest
from verify_baremetal_cpu import STARTUP, GUEST_BASE, STACK_BYTES, identity
from verify_matmul import audit_instructions
from verify_tensorir import GUEST, TIMING_GUEST, fixture_guest


WS_COUNTER = r"""
void __real_tvm_gemmini_compute_ws(uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,int32_t);
void __wrap_tvm_gemmini_compute_ws(uint32_t a,uint32_t b,uint32_t c,uint32_t m,uint32_t n,uint32_t k,int32_t add) {
  ++computes;__real_tvm_gemmini_compute_ws(a,b,c,m,n,k,add);
}
"""


def primitive_references(package, gcc):
    """Read actual compiled operator references, including database-scheduled TIR.

    A replayed schedule need not retain the old mathematical m/n/k metadata.
    The adapter archive defines all primitives, so inspect operators.o instead.
    """
    name = subprocess.check_output([str(gcc), "-print-prog-name=nm"], text=True).strip()
    resolved = shutil.which(name)
    if resolved is None:
        raise ValueError("Compiler's nm tool is unavailable: " + name)
    nm = Path(resolved).resolve(strict=True)
    obj = Path(package) / "graph/operators.o"
    output = subprocess.check_output([str(nm), "-u", str(obj)], text=True)
    symbols = sorted({line.split()[-1] for line in output.splitlines() if line.split()})
    return {"symbols": symbols, "object": identity(obj), "nm": identity(nm)}


def elf_memory(data, guest_bytes):
    """Audit every ELF load span and require entry in executable RV64/lp64d code."""
    if data[:6] != b"\x7fELF\x02\x01":
        raise ValueError("Require little-endian ELF64")
    header = struct.unpack_from("<16sHHIQQQIHHHHHH", data)
    if header[1] != 2 or header[2] != 243 or header[7] & 6 != 4 or header[9] != 56:
        raise ValueError("Require an executable RV64 double-float ABI ELF")
    spans = []
    for index in range(header[10]):
        segment = struct.unpack_from("<IIQQQQQQ", data, header[5] + index * 56)
        kind, flags, offset, virtual, physical, size, extent, _ = segment
        if kind != 1:
            continue
        if (
            physical != virtual
            or size > extent
            or offset + size > len(data)
            or not (GUEST_BASE <= physical < physical + extent <= GUEST_BASE + guest_bytes)
        ):
            raise ValueError("ELF load segment exceeds the declared bare-metal map")
        spans.append({"base": physical, "memory_bytes": extent, "file_bytes": size, "flags": flags})
    if not any(
        row["flags"] & 1 and row["base"] <= header[4] < row["base"] + row["memory_bytes"]
        for row in spans
    ):
        raise ValueError("ELF entry is outside executable load storage")
    return {"entry": header[4], "load_segments": spans, "stack_reserved_bytes": STACK_BYTES}


def prepare(args):
    package = args.package.resolve()
    document = verify_package(package)
    provenance = json.loads(args.fixture_provenance.read_text())
    if provenance.get("fixture_sha256") != digest(args.fixture):
        raise ValueError("Fixture provenance digest mismatch")
    if not provenance.get("reference_contract") or not provenance.get("reference_sources"):
        raise ValueError("Require an independent reference contract and source identities")
    app = Path(__file__).resolve().parent
    bindings = [identity(args.fixture), identity(args.fixture_provenance)]
    bindings.extend(identity(app / name) for name in (
        "qualify_package.py", "compile_package.py", "verify_tensorir.py",
        "verify_baremetal_cpu.py", "verify_exported_graph.py", "verify_matmul.py"))
    for item in provenance["reference_sources"]:
        if digest(item["path"]) != item["sha256"]:
            raise ValueError("Reference source identity changed")
        bindings.append(identity(item["path"]))
    manifest = json.loads((package / "graph/graph.json").read_text())
    with np.load(args.fixture, allow_pickle=False) as data:
        inputs = [data[f"input_{index}"].copy() for index in range(len(manifest["inputs"]))]
        expected = [
            data["expected" if len(manifest["outputs"]) == 1 else f"expected_{index}"].copy()
            for index in range(len(manifest["outputs"]))
        ]
    for array, info in zip(inputs + expected, manifest["inputs"] + manifest["outputs"]):
        if array.dtype.name != info["dtype"] or tuple(array.shape) != tuple(info["shape"]):
            raise ValueError("Fixture does not match the exported buffer ABI")
    guarded_bytes = sum(array.nbytes + 128 for array in inputs + expected) + sum(
        array.nbytes for array in expected
    )
    if (
        guarded_bytes + manifest["constants_bytes"] + manifest["workspace_bytes"] + (1 << 20)
        > args.guest_memory_bytes
    ):
        raise ValueError("Guarded fixture and explicit model storage exceed the guest budget")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = json.loads((package / "placement.json").read_text())
    native = [row for row in report if row["placement"] == "gemmini"]
    scaled = any(row.get("metadata", {}).get("readout_fused") for row in native)
    primitive_evidence = primitive_references(package, args.riscv_gcc.resolve(strict=True))
    symbols = set(primitive_evidence["symbols"])
    has_array = bool(symbols & {"tvm_gemmini_compute", "tvm_gemmini_compute_ws"})
    has_ws = "tvm_gemmini_compute_ws" in symbols
    bindings.extend([primitive_evidence["object"], primitive_evidence["nm"]])
    declarations, checks = fixture_guest(
        0,
        inputs,
        expected,
        manifest["workspace_bytes"],
        ("graph_array" if has_array else "native_graph") if native else None,
        "gemmini" if native else "host",
        required_scaled=scaled,
        timing_repeats=args.timing_repeats,
        rtol=args.rtol,
        atol=args.atol,
    )
    declarations.append("extern const uint8_t model_constants[];")
    checks.insert(0, f"uint64_t constants_before=hash(model_constants,{manifest['constants_bytes']});")
    checks.append(f"if(hash(model_constants,{manifest['constants_bytes']})!=constants_before)return 24;")
    checks.append(
        "extern uint8_t _stack_bottom[],_stack_top[]; uint8_t *first=_stack_bottom;"
        "while(first<_stack_top && *first==(uint8_t)(UINT64_C(0x55aa55aa55aa55aa)>>(((uintptr_t)first-(uintptr_t)_stack_bottom)%8*8)))++first;"
        "if(first==_stack_bottom || first==_stack_top)return 31;"
        "char stack_text[64];char *end=dse_u64(stack_text,(uint64_t)(_stack_top-first));*end=0;"
        'puts_htif("STACK_HIGH_WATER_BYTES=");puts_htif(stack_text);puts_htif("\\n");'
    )
    marker = (
        f"TENSORIR_PASS cases=1 exact={int(not (args.rtol or args.atol))} guards=1 primitives=1"
    )
    text = (
        GUEST + (WS_COUNTER if has_ws else "")
        + TIMING_GUEST
        + "\n"
        + "\n".join(declarations)
        + "\nint main(void){\n"
        + "\n".join(checks)
        + f'\nputs_htif("{marker}\\n");return 0;}}\n'
    )
    text = text.replace("stage_0_model_run", "model_run")
    (out / "runner.c").write_text(text)
    (out / "start.S").write_text(STARTUP)
    # Reuse the export verifier's exact shared memory-map/linker construction.
    from verify_exported_graph import guest_layout

    linker, memory_map = guest_layout(args.guest_memory_bytes)
    (out / "link.ld").write_text(linker)
    gcc = args.riscv_gcc.resolve(strict=True)
    commands, elfs = [], {}
    adapter = json.loads((package / "build.json").read_text())["adapter"]
    policy = adapter["instruction_policy"]
    forbidden = {
        value for name, value in policy["derived_definitions"].items() if name.startswith("k_LOOP_")
    }
    for negative, key, name in ((0, "valid", "valid.elf"), (1, "bad-oracle", "bad-oracle.elf")):
        cmd = [
            str(gcc),
            *FLAGS,
            "-msmall-data-limit=0",
            "-fno-stack-protector",
            "-nostdlib",
            "-nostartfiles",
            "-static",
            "-Wl,--no-relax",
            "-Wl,-T," + str(out / "link.ld"),
        ]
        cmd += [
            "-Wl,--wrap=" + symbol
            for symbol in (
                "tvm_gemmini_compute",
                "tvm_gemmini_zero_acc",
                "tvm_gemmini_dma_acc_write",
                "tvm_gemmini_dma_acc_add",
                "tvm_gemmini_dma_widen",
                "tvm_gemmini_dma_spad_out",
                "tvm_gemmini_dma_relu_out",
                "tvm_gemmini_dma_scaled_out",
            )
        ]
        if has_ws:
            cmd.append("-Wl,--wrap=tvm_gemmini_compute_ws")
        cmd += [
            f"-DBAD_ORACLE={negative}",
            str(out / "start.S"),
            str(out / "runner.c"),
            "-Wl,--start-group",
            *[str(package / lib) for lib in ("libmodel.a", "libm.a", "libgcc.a")],
            "-Wl,--end-group",
            "-o",
            str(out / name),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
        (out / (key + "-link.log")).write_text(result.stdout + result.stderr)
        commands.append({"argv": cmd, "exit_code": result.returncode})
        if result.returncode:
            raise RuntimeError("Final ELF link failed: " + result.stderr)
        elf = identity(out / name)
        elf["memory_audit"] = elf_memory((out / name).read_bytes(), args.guest_memory_bytes)
        elf["instruction_audit"] = audit_instructions(
            (out / name).read_bytes(), policy["custom_opcode"], forbidden
        )
        if elf["instruction_audit"]["prohibited_hits"]:
            raise ValueError("Final ELF violates the primitive-only instruction policy")
        elfs[key] = elf
    bindings += [identity(path) for path in sorted(package.rglob("*")) if path.is_file()]
    bindings += [
        identity(gcc),
        *[identity(out / name) for name in ("runner.c", "start.S", "link.ld")],
    ]
    bundle = {
        "status": "prepared",
        "scope": "Independent guarded package fixture; no deployed hardware or application-quality qualification",
        "ELFs": elfs,
        "commands": commands,
        "bindings": bindings,
        "case_count": 1,
        "cases": ["package"],
        "placements": {"package": "gemmini" if native else "host"},
        "scheduling": {"package": report},
        "primitive_evidence": primitive_evidence,
        "capabilities": document["capability_fingerprint"],
        "timing_repeats": args.timing_repeats,
        "comparison": {"rtol": args.rtol, "atol": args.atol, "integer": "exact"},
        "memory_map": memory_map,
        "guest_memory_bytes": args.guest_memory_bytes,
        "deployed_hardware_verified": False,
    }
    (out / "bundle.json").write_text(json.dumps(bundle, indent=2) + "\n")
    print(json.dumps({"status": "prepared", "bundle": str(out / "bundle.json")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("package", "fixture", "fixture-provenance", "output-dir", "riscv-gcc"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--guest-memory-bytes", type=int, default=256 << 20)
    parser.add_argument("--timing-repeats", type=int, default=0)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--atol", type=float, default=0.0)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
