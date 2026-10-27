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
"""Stage pinned Gemmini headers and cross-compile the actual C-library adapter.

This produces an object and a source/include receipt, not a device qualification.
No checkout, fetch, simulator launch, TVM rebuild, or hardware generation occurs.
Link only the audited matmul.o; matmul.raw.o is a noncompliant diagnostic that
can contain unused vendor FSM instructions. Audit the final linked ELF again.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import struct
import subprocess


HARDWARE_REVISION = "6ad65b90b1eb270c20ce4f04109e3dde0180b36f"
PARAMS_SHA256 = "3758ae967af3a179497660970201093a7fb624be00173990ce33d3f5c38da924"
OPERATOR_SHA256 = "18801f4eab0cd2e81e25a6c2aa97c7ed644b693e8d2c24cdeb41ab3a7208336d"
PUBLIC_SYMBOLS = (
    "tvm_gemmini_dma_acc_write", "tvm_gemmini_dma_acc_add",
    "tvm_gemmini_dma_widen", "tvm_gemmini_dma_spad_out", "tvm_gemmini_dma_relu_out",
    "tvm_gemmini_dma_scaled_out",
    "tvm_gemmini_dma_scaled_in",
    "tvm_gemmini_dma_pool_out",
    "tvm_gemmini_matmul_i8_i32", "tvm_gemmini_validate_matmul_i8_i32", "tvm_gemmini_begin",
    "tvm_gemmini_load_a", "tvm_gemmini_load_b", "tvm_gemmini_compute", "tvm_gemmini_store", "tvm_gemmini_end",
    "tvm_gemmini_zero_acc", "tvm_gemmini_dma_a", "tvm_gemmini_dma_b", "tvm_gemmini_dma_out",
    "tvm_gemmini_compute_ws", "tvm_gemmini_dma_a_blocks", "tvm_gemmini_dma_b_blocks",
    "tvm_gemmini_validate_tensor_buffers",
    "tvm_gemmini_validate_tensor_output",
)


def allowed(path):
    path = Path(path)
    for candidate in (path.absolute(),):
        if any(name in part.lower() for part in candidate.parts for name in ("hammer", "vlsi")):
            raise ValueError(f"Restricted path: {path}")
    resolved = path.resolve()
    if any(name in part.lower() for part in resolved.parts for name in ("hammer", "vlsi")):
        raise ValueError(f"Restricted path: {path}")
    return resolved


def digest(data):
    return hashlib.sha256(data).hexdigest()


def run(command, *, check=True):
    env = os.environ.copy()
    for name in ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "OBJC_INCLUDE_PATH", "LIBRARY_PATH", "COMPILER_PATH", "GCC_EXEC_PREFIX"):
        env.pop(name, None)
    result = subprocess.run([str(part) for part in command], capture_output=True, check=False, env=env)
    if check and result.returncode:
        raise RuntimeError(f"Command failed: {shlex.join(map(str, command))}\n{result.stderr.decode(errors='replace')}")
    return result


def git(repo, *args):
    return run(["git", "-C", allowed(repo), *args]).stdout


def gitlink(repo, revision, path):
    fields = git(repo, "ls-tree", revision, "--", path).decode().split()
    if len(fields) != 4 or fields[:2] != ["160000", "commit"] or fields[3] != path:
        raise ValueError(f"Expected exact gitlink {revision}:{path}")
    return fields[2]


def tool(name):
    resolved = shutil.which(name)
    if resolved is None:
        raise ValueError(f"Tool unavailable: {name}")
    return allowed(resolved)


def executable_sections(data):
    """Read every executable section of a little-endian ELF64 RISC-V object/ELF."""
    if data[:6] != b"\x7fELF\x02\x01" or len(data) < 64:
        raise ValueError("Instruction audit requires little-endian ELF64")
    header = struct.unpack_from("<16sHHIQQQIHHHHHH", data)
    if header[2] != 243 or header[11] != 64 or header[12] == 0:
        raise ValueError("Instruction audit requires RISC-V with an ordinary section table")
    offset, count, names_index = header[6], header[12], header[13]
    if offset + count * 64 > len(data) or names_index >= count:
        raise ValueError("Truncated ELF section table")
    sections = [struct.unpack_from("<IIQQQQIIQQ", data, offset + index * 64) for index in range(count)]
    names_section = sections[names_index]
    names = data[names_section[4]:names_section[4] + names_section[5]]
    result = []
    for section in sections:
        if section[2] & 4:  # ELF SHF_EXECINSTR, independent of Gemmini encodings.
            start, size = section[4], section[5]
            if section[1] != 1 or start + size > len(data):
                raise ValueError("Invalid executable ELF section")
            name = names[section[0]:].split(b"\0", 1)[0].decode()
            result.append((name, data[start:start + size]))
    if not result:
        raise ValueError("ELF has no executable sections")
    return result


def audit_instructions(data, custom_opcode, forbidden):
    """Decode RV64GC instruction lengths and reject source-derived loop functs."""
    instruction_count, hits = 0, []
    sections = executable_sections(data)
    for name, code in sections:
        offset = 0
        while offset < len(code):
            if offset + 2 > len(code):
                raise ValueError(f"Truncated instruction: {name}+{offset}")
            half = int.from_bytes(code[offset:offset + 2], "little")
            width = 2 if half & 3 != 3 else 4
            if width == 4:
                if half & 31 == 31 or offset + 4 > len(code):
                    raise ValueError(f"Unsupported/truncated instruction: {name}+{offset}")
                word = int.from_bytes(code[offset:offset + 4], "little")
                funct = (word >> 25) & 127
                if word & 127 == custom_opcode and funct in forbidden:
                    hits.append({"section": name, "offset": offset, "funct": funct, "word": f"{word:08x}"})
            offset += width
            instruction_count += 1
    return {"executable_sections": [name for name, _ in sections], "instructions": instruction_count, "prohibited_hits": hits}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gemmini-repo", type=Path, required=True)
    parser.add_argument("--hardware-revision", default=HARDWARE_REVISION)
    parser.add_argument("--rocc-tests-repo", type=Path, help="defaults to the Gemmini software/gemmini-rocc-tests checkout")
    parser.add_argument("--rocc-software-repo", type=Path, help="defaults to the nested rocc-software checkout")
    parser.add_argument("--build-root", type=Path, required=True, help="configured comparison build root ending in baselines/tvm-gemmini")
    parser.add_argument("--output-dir", type=Path, help="fresh directory inside --build-root")
    parser.add_argument("--cc", default="riscv64-unknown-elf-gcc")
    parser.add_argument("--objdump", default="riscv64-unknown-elf-objdump")
    parser.add_argument("--audit-elf", type=Path, action="append", default=[], help="also audit every executable section of a final ELF")
    args = parser.parse_args()
    if len(args.hardware_revision) != 40 or any(c not in "0123456789abcdef" for c in args.hardware_revision):
        parser.error("--hardware-revision must be a full lowercase commit SHA")

    hardware = allowed(args.gemmini_repo)
    rocc = allowed(args.rocc_tests_repo or hardware / "software/gemmini-rocc-tests")
    custom = allowed(args.rocc_software_repo or rocc / "rocc-software")
    revision = git(hardware, "rev-parse", "--verify", f"{args.hardware_revision}^{{commit}}")
    revision = revision.decode().strip()
    rocc_revision = gitlink(hardware, revision, "software/gemmini-rocc-tests")
    simulator_revision = gitlink(hardware, revision, "software/libgemmini")
    custom_revision = gitlink(rocc, rocc_revision, "rocc-software")
    hardware_config = git(hardware, "show", f"{revision}:src/main/scala/gemmini/Configs.scala")
    expected_fields = {"spatialArrayOutputType": "SInt(20.W)", "meshRows": "16", "meshColumns": "16", "tileRows": "1", "tileColumns": "1"}
    observed_fields = {}
    for line in hardware_config.decode().splitlines():
        key, separator, value = line.strip().partition(" = ")
        if separator and key in expected_fields and key not in observed_fields:
            observed_fields[key] = value.rstrip(",")
    if observed_fields != expected_fields:
        raise ValueError(f"Hardware source does not support the bounded OS schedule: {observed_fields}")
    build_root = allowed(args.build_root)
    if build_root.parts[-2:] != ("baselines", "tvm-gemmini"):
        parser.error("--build-root must be the configured baselines/tvm-gemmini build root")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = allowed(args.output_dir or build_root / f"matmul-library-{timestamp}")
    if not output.is_relative_to(build_root) or output == build_root:
        parser.error("--output-dir must be a fresh child of --build-root")
    output.mkdir(parents=True, exist_ok=False)
    include_root = output / "headers"

    headers = {}
    for relative in ("include/gemmini_params.h", "include/gemmini.h", "include/gemmini_counter.h"):
        data = git(rocc, "show", f"{rocc_revision}:{relative}")
        destination = include_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        headers[relative] = {"sha256": digest(data), "revision": rocc_revision}
    relative = "rocc-software/src/xcustom.h"
    data = git(custom, "show", f"{custom_revision}:src/xcustom.h")
    destination = include_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    headers[relative] = {"sha256": digest(data), "revision": custom_revision}
    if headers["include/gemmini_params.h"]["sha256"] != PARAMS_SHA256:
        raise ValueError("Selected hardware gitlink does not supply the canonical integer header")
    if headers["include/gemmini.h"]["sha256"] != OPERATOR_SHA256:
        raise ValueError("Selected operator header differs from the inspected C-library ABI")

    source_root = allowed(Path(__file__).parent)
    sources = {name: digest((source_root / name).read_bytes()) for name in ("matmul.c", "matmul.h", "verify_matmul.py")}
    cc, objdump = tool(args.cc), tool(args.objdump)
    flags = ["-std=gnu11", "-O3", "-finline-limit=10000", "-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany",
             "-ffunction-sections", "-fdata-sections", "-fno-common", "-fno-builtin-printf",
             "-fno-tree-loop-distribute-patterns", "-DBAREMETAL=1", "-DPRINT_TILE=0"]
    policy_flags = ["-DTVM_GEMMINI_FORBID_HW_LOOPS=1", "-DTVM_GEMMINI_PE_OUTPUT_BITS=20"]
    command = [cc, *flags, *policy_flags, "-I", include_root, source_root / "matmul.c"]
    preprocessing = run([*command, "-E", "-H"])
    (output / "matmul.i").write_bytes(preprocessing.stdout)
    (output / "includes.log").write_bytes(preprocessing.stderr)
    resolved = set()
    for line in preprocessing.stdout.decode().splitlines():
        if line.startswith("# "):
            fields = shlex.split(line)
            if len(fields) >= 3 and not fields[2].startswith("<"):
                resolved.add(allowed(fields[2]))
    for relative in headers:
        expected = allowed(include_root / relative)
        if expected not in resolved:
            raise ValueError(f"Compiler did not resolve the staged header: {relative}")

    object_path = output / "matmul.o"
    raw_object_path = output / "matmul.raw.o"
    compile_command = [*command, "-c", "-o", raw_object_path]
    compiled = run(compile_command)
    (output / "compile.log").write_bytes(compiled.stderr)
    # The vendor header can leave an unreferenced WS helper in the raw object.
    # Keep every public ABI and its full relocation closure, then audit every
    # remaining executable section. Never link the raw diagnostic object.
    link_command = [cc, "-march=rv64gc", "-mabi=lp64d", "-nostdlib", "-r",
                    "-Wl,--gc-sections,-e,tvm_gemmini_matmul_i8_i32",
                    *[f"-Wl,--undefined={name}" for name in PUBLIC_SYMBOLS], raw_object_path, "-o", object_path]
    linked = run(link_command)
    (output / "partial_link.log").write_bytes(linked.stderr)
    disassembly = run([objdump, "-dr", object_path]).stdout
    (output / "matmul.disassembly").write_bytes(disassembly)
    for name in PUBLIC_SYMBOLS:
        if f"<{name}>:".encode() not in disassembly:
            raise ValueError(f"Compiled object lacks the public symbol: {name}")

    definitions = {}
    for relative in ("include/gemmini_params.h", "include/gemmini.h"):
        for line in (include_root / relative).read_text().splitlines():
            fields = line.split()
            if len(fields) == 3 and fields[0] == "#define" and (fields[1] == "XCUSTOM_ACC" or fields[1].startswith("k_LOOP_")):
                definitions[fields[1]] = int(fields[2], 0)
    forbidden = {value for name, value in definitions.items() if name.startswith("k_LOOP_")}
    if not forbidden or "XCUSTOM_ACC" not in definitions:
        raise ValueError("Missing source-derived custom opcode/loop funct definitions")
    sample_funct = min(forbidden)
    encoding_source = output / "loop_encoding.S"
    encoding_source.write_text(f".text\n.insn r CUSTOM_{definitions['XCUSTOM_ACC']}, 3, {sample_funct}, x0, x0, x0\n")
    encoding_object = output / "loop_encoding.o"
    run([cc, "-march=rv64gc", "-mabi=lp64d", "-c", encoding_source, "-o", encoding_object])
    encoding_sections = [(name, code) for name, code in executable_sections(encoding_object.read_bytes()) if code]
    if len(encoding_sections) != 1 or len(encoding_sections[0][1]) != 4:
        raise ValueError("Unexpected assembler opcode probe")
    word = int.from_bytes(encoding_sections[0][1], "little")
    custom_opcode = word & 127
    if (word >> 25) & 127 != sample_funct:
        raise ValueError("Assembler probe does not encode the selected source funct")
    if not audit_instructions(encoding_object.read_bytes(), custom_opcode, forbidden)["prohibited_hits"]:
        raise ValueError("Instruction audit failed to reject an assembler-built loop instruction")
    audits = {}
    for path in [object_path, *map(allowed, args.audit_elf)]:
        data = path.read_bytes()
        result = audit_instructions(data, custom_opcode, forbidden)
        audits[str(path)] = {"sha256": digest(data), **result}
        if result["prohibited_hits"]:
            (output / "instruction_audit.json").write_text(json.dumps(audits, indent=2) + "\n")
            raise ValueError(f"Prohibited FSM instructions in {path}: {result['prohibited_hits']}")

    negative_checks = {}
    for name, old, new, marker in (
        ("policy_unspecified", None, None, "requires explicit TVM_GEMMINI_FORBID_HW_LOOPS=1"),
        ("small_readout_only", b"#define ACC_READ_FULL_WIDTH", b"", "full-width accumulator"),
        ("unsigned_operands", b"typedef int8_t elem_t;", b"typedef uint8_t elem_t;", "signed int8"),
        ("normalizations", b"#define GEMMINI_PARAMS_H", b"#define GEMMINI_PARAMS_H\n#define HAS_NORMALIZATIONS", "does not admit a normalization configuration"),
    ):
        case_root = output / "rejections" / name
        for relative in headers:
            data = (include_root / relative).read_bytes()
            if relative == "include/gemmini_params.h" and old is not None:
                if old not in data:
                    raise ValueError(f"Rejection fixture does not match canonical header: {name}")
                data = data.replace(old, new, 1)
            destination = case_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        defines = [] if name == "policy_unspecified" else policy_flags
        result = run([cc, *flags, *defines, "-I", case_root, source_root / "matmul.c", "-fsyntax-only"], check=False)
        (case_root / "compile.log").write_bytes(result.stderr)
        if result.returncode == 0 or marker.encode() not in result.stderr:
            raise ValueError(f"Adapter failed to reject {name} for the expected reason")
        negative_checks[name] = {"status": "rejected", "diagnostic": marker}

    for name, expected in sources.items():
        if digest((source_root / name).read_bytes()) != expected:
            raise ValueError(f"Source changed during verification: {name}")
    receipt = {
        "status": "cross_compiled", "device_execution": False, "paper_validation": False,
        "source_set": {"gemmini": revision, "gemmini_rocc_tests": rocc_revision,
                       "rocc_software": custom_revision, "libgemmini": simulator_revision},
        "hardware_source_check": {"configs_sha256": digest(hardware_config), "fields": observed_fields, "execution_verified": False},
        "headers": headers, "adapter_sources": sources, "public_symbols": list(PUBLIC_SYMBOLS),
        "resolved_headers": {str(path): digest(path.read_bytes()) for path in sorted(resolved)},
        "compiler": {"path": str(cc), "sha256": digest(cc.read_bytes()),
                     "version": run([cc, "--version"]).stdout.decode().splitlines()[0]},
        "compile_command": list(map(str, compile_command)),
        "partial_link_command": list(map(str, link_command)),
        "raw_object": {"sha256": digest(raw_object_path.read_bytes()), "linkable": False,
                       "instruction_audit": audit_instructions(raw_object_path.read_bytes(), custom_opcode, forbidden)},
        "object_sha256": digest(object_path.read_bytes()),
        "disassembly_sha256": digest(disassembly), "rejections": negative_checks,
        "instruction_policy": {"hardware_loops": "forbidden", "derived_definitions": definitions, "custom_opcode": custom_opcode},
        "instruction_audits": audits,
        "normalization_compatibility": {"fallback_stat_ids": 2, "hardware_enabled": False},
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "receipt": str(output / "receipt.json")}))


if __name__ == "__main__":
    main()
