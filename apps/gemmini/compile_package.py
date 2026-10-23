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
"""Compile a portable static RV64/Gemmini library without launching FireSim.

Input is a saved torch.export archive or semantic Relax JSON. Exported graph
storage, CPU fallback, adapter, math libraries and source identities are explicit.
The resulting archive needs a board-specific startup/loader and final ELF audit.
"""

import argparse
import hashlib
import json
import mmap
import os
from pathlib import Path
import shutil
import subprocess

import tvm
from tvm._ffi.base import _LIB
from tvm.relax.backend.contrib.gemmini import prepare_gemmini_graph
from tvm.relax.backend.contrib.gemmini_target import GemminiCapabilities
from tvm.relax.backend.contrib.gemmini_search import open_gemmini_database
from tvm.relax.backend.contrib.gemmini_tensorir import prepare_gemmini_tensorir_workloads

from program_io import import_program
from tuning_preset import materialize_preset
from baremetal import export_graph, RV64_TARGET
from verify_baremetal_cpu import identity
from verify_matmul import audit_instructions

FLAGS = [
    "-march=rv64imafdc",
    "-mabi=lp64d",
    "-mcmodel=medany",
    "-O2",
    "-ffreestanding",
    "-fno-builtin",
    "-fno-common",
    "-fno-pic",
    "-ffunction-sections",
    "-fdata-sections",
]

RUNTIME = r"""
#include <stdint.h>
#include <stddef.h>
/* One hart, no libc allocator, I/O or thread-local runtime. */
__attribute__((weak)) void *memcpy(void *d,const void *s,size_t n){uint8_t *a=d;const uint8_t *b=s;for(size_t i=0;i<n;++i)a[i]=b[i];return d;}
__attribute__((weak)) void *memmove(void *d,const void *s,size_t n){uint8_t *a=d;const uint8_t *b=s;if((uintptr_t)a<(uintptr_t)b){for(size_t i=0;i<n;++i)a[i]=b[i];}else{while(n){--n;a[n]=b[n];}}return d;}
__attribute__((weak)) void *memset(void *d,int x,size_t n){uint8_t *a=d;for(size_t i=0;i<n;++i)a[i]=(uint8_t)x;return d;}
__attribute__((weak)) int memcmp(const void *a,const void *b,size_t n){const uint8_t *x=a,*y=b;for(size_t i=0;i<n;++i)if(x[i]!=y[i])return (int)x[i]-(int)y[i];return 0;}
static int model_errno;
__attribute__((weak)) int *__errno(void){return &model_errno;}
/* Vendor error branches use printf/exit. A board may supply diagnostics;
 * the portable fallback suppresses text and traps rather than returning. */
__attribute__((weak)) int printf(const char *format,...){(void)format;return 0;}
__attribute__((weak,noreturn)) void exit(int status){(void)status;__builtin_trap();for(;;){}}
"""


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def verify_package(folder):
    """Verify relocatable payload identities; no original source paths are needed."""
    folder = Path(folder).resolve()
    document = json.loads((folder / "package.json").read_text())
    if document.get("schema") != "gemmini_static_package_v1":
        raise ValueError("Unknown static package schema")
    for item in document["files"]:
        path = folder / item["path"]
        if not path.resolve().is_relative_to(folder) or path.is_symlink():
            raise ValueError("Package file escapes its root")
        if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise ValueError("Package payload changed: " + item["path"])
    return document


def compile_package(args):
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    app = Path(__file__).resolve().parent
    gcc = args.riscv_gcc.resolve(strict=True)
    commands = []
    clean_env = os.environ.copy()
    for key in (
        "CPATH",
        "C_INCLUDE_PATH",
        "CPLUS_INCLUDE_PATH",
        "LIBRARY_PATH",
        "COMPILER_PATH",
        "GCC_EXEC_PREFIX",
    ):
        clean_env.pop(key, None)

    def command(argv):
        commands.append(list(map(str, argv)))
        return subprocess.check_output(
            commands[-1], stderr=subprocess.STDOUT, env=clean_env, text=True, timeout=120
        )

    caps = GemminiCapabilities()
    adapter_path = args.adapter_receipt.resolve(strict=True)
    adapter = json.loads(adapter_path.read_text())
    if adapter.get("status") != "cross_compiled":
        raise ValueError("Require a cross-compiled audited adapter")
    for name, expected in adapter["adapter_sources"].items():
        if digest(app / name) != expected:
            raise ValueError("Adapter source mismatch: " + name)
    adapter_object = adapter_path.parent / "matmul.o"
    if digest(adapter_object) != adapter["object_sha256"]:
        raise ValueError("Adapter object mismatch")
    if adapter["headers"]["include/gemmini_params.h"]["sha256"] != caps.params_sha256:
        raise ValueError("Adapter capability header mismatch")
    mod = import_program(args.program)
    database = None
    database_path = getattr(args, "tuning_database", None)
    preset_name = getattr(args, "tuning_preset", None)
    if preset_name and (database_path is not None or args.cpu_only):
        raise ValueError("Tuning preset requires an accelerator build without another database")
    if preset_name:
        database_path = materialize_preset(preset_name, out / "preset", capabilities=caps)
    if database_path is not None:
        if args.cpu_only:
            raise ValueError("CPU-only build cannot accept a tuning database")
        # Carry the standard records in the relocatable package. Replay and
        # workload matching belong to MetaScheduleApplyDatabase, not this CLI.
        destination = out / "database"
        destination.mkdir()
        for name in ("database_workload.json", "database_tuning_record.json"):
            shutil.copy2(database_path / name, destination / name)
        prepared, _, _ = prepare_gemmini_tensorir_workloads(mod)
        database = open_gemmini_database(destination, prepared, capabilities=caps)
        records = database.get_all_tuning_records()
        if not records:
            raise ValueError("Tuning database contains no measurements")
        if any(str(record.target) != str(caps.target()) for record in records):
            raise ValueError("Tuning database target does not match Gemmini capabilities")
    report = []
    if args.cpu_only:
        selected, _, _ = prepare_gemmini_tensorir_workloads(
            mod, bias_preload=False, prepack_weights=False
        )
    else:
        selected = prepare_gemmini_graph(mod, report=report, database=database)
        if database is not None and not preset_name and not any(
            row.get("origin") == "TVM MetaSchedule database" for row in report
        ):
            raise ValueError("Tuning database does not match any function in this program")
    graph = out / "graph"
    manifest = export_graph(
        selected, graph, target=RV64_TARGET, memory_limit_bytes=args.tensor_memory_bytes
    )
    # Compile incbin now. The delivered archive contains its own constant object,
    # so it remains usable after the source/output directory has moved.
    constants = '.section .rodata.model_constants,"a",@progbits\n.balign 64\n.global model_constants\nmodel_constants:\n.incbin "constants.bin"\n'
    (graph / "constants.S").write_text(constants)
    # GCC takes an explicit assembler search directory for the relative incbin.
    command(
        [
            gcc,
            *FLAGS,
            "-Wa,-I," + str(graph),
            "-c",
            graph / "constants.S",
            "-o",
            graph / "constants.o",
        ]
    )
    command([gcc, *FLAGS, "-c", graph / "model.c", "-o", graph / "model.o"])
    (out / "runtime.c").write_text(RUNTIME)
    command([gcc, *FLAGS, "-c", out / "runtime.c", "-o", out / "runtime.o"])
    shutil.copy2(adapter_object, out / "adapter.o")
    shutil.copy2(app / "matmul.h", out / "matmul.h")
    (out / "model.h").write_text(
        '#pragma once\n#include <stdint.h>\n#ifdef __cplusplus\nextern "C" {\n#endif\nint32_t model_run(const void *const *inputs, void *const *outputs, uint8_t *workspace, uint64_t workspace_bytes);\nconst char *model_get_last_error(void);\n#ifdef __cplusplus\n}\n#endif\n'
    )
    libraries = []
    for name in ("libm.a", "libgcc.a"):
        path = Path(command([gcc, *FLAGS, "-print-file-name=" + name]).strip()).resolve(strict=True)
        shutil.copy2(path, out / name)
        libraries.append(identity(path))
    ar = Path(command([gcc, "-print-prog-name=ar"]).strip())
    if not ar.is_absolute():
        ar = Path(shutil.which(str(ar)) or str(ar)).resolve(strict=True)
    command(
        [
            ar,
            "rcsD",
            out / "libmodel.a",
            graph / "model.o",
            graph / "operators.o",
            graph / "constants.o",
            out / "runtime.o",
            out / "adapter.o",
        ]
    )
    # A whole-archive partial link exercises closure against the supplied math
    # libraries, exposes hidden TVM callbacks, and permits one ISA/FSM audit.
    command(
        [
            gcc,
            *FLAGS,
            "-nostdlib",
            "-nostartfiles",
            "-Wl,-r,--no-relax",
            "-Wl,--whole-archive",
            out / "libmodel.a",
            "-Wl,--no-whole-archive",
            "-Wl,--start-group",
            out / "libm.a",
            out / "libgcc.a",
            "-Wl,--end-group",
            "-o",
            out / "linked.o",
        ]
    )
    nm = Path(command([gcc, "-print-prog-name=nm"]).strip())
    undefined = command([nm, "-u", out / "linked.o"]).strip()
    if undefined:
        raise ValueError("Unresolved runtime dependencies: " + undefined)
    policy = adapter["instruction_policy"]
    forbidden = {
        value for name, value in policy["derived_definitions"].items() if name.startswith("k_LOOP_")
    }
    # Constant blobs can exceed host RAM headroom. The decoder reads only ELF
    # metadata and executable sections; do not copy the full object into Python.
    with (out / "linked.o").open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
        audit = audit_instructions(data, policy["custom_opcode"], forbidden)
    if audit["prohibited_hits"]:
        raise ValueError("Generated package contains prohibited hardware loop instructions")
    # Serialize relative constants for downstream tools. Its hash and storage
    # envelope remain the exporter's originals.
    manifest["constants_file"] = "constants.bin"
    (graph / "graph.json").write_text(json.dumps(manifest, indent=2) + "\n")
    profile = json.loads((app / "firesim_target.json").read_text())
    if profile["params_sha256"] != caps.params_sha256:
        raise ValueError("FireSim target and compiler capability profiles disagree")
    (out / "firesim_target.json").write_text(json.dumps(profile, indent=2) + "\n")
    (out / "placement.json").write_text(json.dumps(report, indent=2) + "\n")
    source = Path(tvm.__file__).resolve().parent
    source_bindings = [
        identity(_LIB._name),
        identity(args.program),
        identity(gcc),
        identity(adapter_path),
        identity(app / "firesim_target.json"),
    ]
    source_bindings.extend(
        identity(path)
        for folder in (
            source / "relax/backend/contrib",
            source / "relax/frontend/torch",
            source / "tir/transform",
            app,
        )
        for path in sorted(folder.glob("*.py"))
    )
    if database_path is not None:
        source_bindings.extend(identity(path) for path in sorted((out / "database").glob("*.json")))
    (out / "build.json").write_text(
        json.dumps(
            {
                "commands": commands,
                "sources": source_bindings,
                "libraries": libraries,
                "adapter": adapter,
            },
            indent=2,
        )
        + "\n"
    )
    document = {
        "schema": "gemmini_static_package_v1",
        "status": "compiled_offline",
        "target": RV64_TARGET,
        "capability_fingerprint": caps.fingerprint,
        "instruction_policy": "primitive_no_fsm",
        "instruction_audit": audit,
        "entrypoint": "model_run",
        "abi": "RV64 lp64d; separate 64-byte-aligned disjoint buffers",
        "heap_required": False,
        "harts": 1,
        "rounding_mode": "RNE; startup must set fcsr=0 and enable FS",
        "memory": manifest["memory_preflight"],
        "placement_policy": "CPU only"
        if args.cpu_only
        else "CPU copies; exact supported arithmetic; measured overrides when supplied",
        "tuning_database": "database" if database is not None else None,
        "tuning_preset": preset_name,
        "tuned_functions": [row["function"] for row in report
                            if row.get("origin") == "TVM MetaSchedule database"],
        "unmeasured_placement_is_provisional": any(
            row.get("origin") != "TVM MetaSchedule database" for row in report
        ),
        "timing_qualified_for_firesim": False,
        "deployed_hardware_verified": False,
        "deployment_requirements": [
            "Bind actual bitstream, elaboration, generated header, DTS and matching driver",
            "Qualify loader, HTIF completion and single-hart bare-metal startup",
            "Link final ELF; audit load segments, ISA/no-FSM, stack and complete reserved DRAM",
            "Validate independent inputs, canaries and wrong oracle on selected deployment",
            "Remeasure CPU/default/winner with identical deployed timing scope",
        ],
        "link": [
            "-nostdlib",
            "-nostartfiles",
            "-Wl,--no-relax",
            "-Wl,--start-group",
            "libmodel.a",
            "libm.a",
            "libgcc.a",
            "-Wl,--end-group",
        ],
    }
    document["files"] = [
        {"path": str(path.relative_to(out)), "sha256": digest(path), "bytes": path.stat().st_size}
        for path in sorted(out.rglob("*"))
        if path.is_file()
    ]
    (out / "package.json").write_text(json.dumps(document, indent=2) + "\n")
    verify_package(out)
    print(json.dumps({"status": "compiled_offline", "package": str(out / "package.json")}))
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", type=Path, help="Verify an existing package after relocation")
    parser.add_argument("--program", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--riscv-gcc", type=Path)
    parser.add_argument("--adapter-receipt", type=Path)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--tensor-memory-bytes", type=int, default=64 << 20)
    tuning = parser.add_mutually_exclusive_group()
    tuning.add_argument("--tuning-database", type=Path, help="Standard MetaSchedule JSONDatabase directory")
    tuning.add_argument("--tuning-preset", choices=["rtl-v1"],
                        help="Apply shipped RTL winners where structurally matching; report fallbacks")
    args = parser.parse_args()
    if args.verify:
        verify_package(args.verify)
        print('{"status":"integrity_passed"}')
        return
    if any(
        getattr(args, key) is None
        for key in ("program", "output_dir", "riscv_gcc", "adapter_receipt")
    ):
        parser.error("Compilation requires program, output-dir, riscv-gcc and adapter-receipt")
    compile_package(args)


if __name__ == "__main__":
    main()
