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
from dataclasses import replace
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

from program_io import import_program
from package_audit import audit_graph
from graph_plans import GraphPlan
from tuning_preset import PRESET_NAMES, materialize_preset
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


def prepare_tuning_databases(primary_path, fallback_path, output, prepared, *,
                             capabilities=None, require_primary_match=True):
    """Copy validated databases and query stock first-match precedence.

    Fallback costs never compete with primary costs and are not promoted to
    current measurements. The retained selection manifest names each source.
    """
    caps = capabilities or GemminiCapabilities()
    if primary_path is None:
        raise ValueError("A fallback tuning database requires a primary tuning database")
    databases = []
    sources = []
    for role, path, destination in (
        ("primary", primary_path, "database"),
        ("fallback", fallback_path, "fallback-database"),
    ):
        if path is None:
            continue
        folder = Path(output) / destination
        folder.mkdir()
        for name in ("database_workload.json", "database_tuning_record.json"):
            shutil.copy2(Path(path) / name, folder / name)
        for name in ("measurement-context.json", "preset.json"):
            if (Path(path) / name).is_file():
                shutil.copy2(Path(path) / name, folder / name)
        database = open_gemmini_database(folder, prepared, capabilities=caps)
        records = database.get_all_tuning_records()
        if not records:
            raise ValueError(f"{role.title()} tuning database contains no measurements")
        for record in records:
            if str(record.target) != str(caps.target()):
                raise ValueError(f"{role.title()} tuning database target does not match Gemmini capabilities")
            schedule = tvm.tir.Schedule(record.workload.mod)
            record.trace.apply_to_schedule(schedule, remove_postproc=False)
        databases.append(database)
        sources.append({"role": role, "directory": destination, "records": len(records),
                        "files": [{"path": str(item.relative_to(output)), "sha256": digest(item)}
                                  for item in sorted(folder.glob("*.json"))]})
    tasks = tvm.meta_schedule.relax_integration.extract_tasks(prepared, caps.target())
    selections = []
    for task in tasks:
        row = {"task": task.task_name, "weight": int(task.weight), "source": None}
        for database, source in zip(databases, sources):
            record = database.query_tuning_record(task.dispatched[0], caps.target(), "main")
            if record is not None:
                row.update(source=source["role"], database=source["directory"],
                           trace_sha256=hashlib.sha256(
                               tvm.ir.save_json(record.trace.as_json()).encode()).hexdigest(),
                           fallback_cost_is_current_measurement=False)
                break
        selections.append(row)
    if require_primary_match and not any(row["source"] == "primary" for row in selections):
        raise ValueError("Primary tuning database does not match any function in this program")
    selection = {"schema": "gemmini_ordered_database_selection_v1", "sources": sources,
                 "precedence": [source["role"] for source in sources], "tasks": selections,
                 "cross_database_cost_comparison": False,
                 "fallback_labels_promoted_to_current_measurements": False,
                 "fallback_timing_qualified_for_current_memory_model": False,
                 "fallback_hardware_qualification_inherited": False,
                 "scope": "Compile-time trace selection; connected latency requires fresh measurement"}
    database = (databases[0] if len(databases) == 1 else
                tvm.meta_schedule.database.OrderedUnionDatabase(*databases))
    return database, selection


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
    plan = GraphPlan.from_args(args)
    if args.cpu_only:
        plan = replace(plan, prepack_weights=False, bias_preload=False)
    graph_trace = plan.trace(mod)
    prepared = graph_trace.out_mod
    (out / "graph-plan.json").write_text(
        json.dumps(plan.identity(mod, prepared, trace=graph_trace), indent=2) + "\n"
    )
    database = None
    database_path = getattr(args, "tuning_database", None)
    fallback_database_path = getattr(args, "fallback_tuning_database", None)
    preset_name = getattr(args, "tuning_preset", None)
    fallback_preset = getattr(args, "fallback_tuning_preset", None)
    if fallback_preset is not None:
        if fallback_database_path is not None:
            raise ValueError("Choose either a fallback tuning database or a fallback preset")
        if fallback_preset == preset_name:
            raise ValueError("Fallback preset must differ from the primary preset")
    if preset_name and (database_path is not None or args.cpu_only):
        raise ValueError("Tuning preset requires an accelerator build without another database")
    if preset_name:
        database_path = materialize_preset(preset_name, out / "preset", capabilities=caps)
    if fallback_preset is not None:
        fallback_database_path = materialize_preset(
            fallback_preset, out / "fallback-preset", capabilities=caps)
    if fallback_database_path is not None and database_path is None:
        raise ValueError("A fallback tuning database requires an explicit primary tuning database")
    if database_path is not None:
        if args.cpu_only:
            raise ValueError("CPU-only build cannot accept a tuning database")
        # Carry the standard records in the relocatable package. Replay and
        # workload matching belong to MetaScheduleApplyDatabase, not this CLI.
        database, database_selection = prepare_tuning_databases(
            database_path, fallback_database_path, out, prepared,
            capabilities=caps, require_primary_match=not preset_name)
        (out / "database-selection.json").write_text(
            json.dumps(database_selection, indent=2) + "\n")
    report = []
    if args.cpu_only:
        selected = prepared
    else:
        selected = prepare_gemmini_graph(mod, report=report, database=database,
                                        **plan.preparation_options)
        if database is not None and not preset_name and not any(
            row.get("origin") == "TVM MetaSchedule database" for row in report
        ):
            raise ValueError("Tuning database does not match any function in this program")
        if database is not None:
            by_task = {row["task"]: row["source"] for row in database_selection["tasks"]}
            for row in report:
                if row.get("origin") == "TVM MetaSchedule database":
                    row["database_source"] = by_task.get(row["function"])
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
    (out / "memory-audit.json").write_text(
        json.dumps(audit_graph(manifest, report, caps), indent=2) + "\n"
    )
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
    if fallback_database_path is not None:
        source_bindings.extend(identity(path) for path in sorted((out / "fallback-database").glob("*.json")))
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
        "fallback_tuning_database": "fallback-database" if fallback_database_path is not None else None,
        "database_selection": "database-selection.json" if database is not None else None,
        "tuning_preset": preset_name,
        "fallback_tuning_preset": fallback_preset,
        "graph_plan": "graph-plan.json",
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
    parser.add_argument("--graph-fusion-level", type=int, choices=(-1, 0, 1, 2, 3), default=-1,
                        help="Relax fusion alternative before inner schedule search; default follows PassContext")
    parser.add_argument("--no-prepack-weights", action="store_false", dest="prepack_weights",
                        help="Compare unpacked constant operands without changing model arithmetic")
    parser.add_argument("--graph-cse", action="store_true",
                        help="Use stock Relax common-subexpression elimination before legalization")
    parser.add_argument("--no-bias-preload", dest="bias_preload", action="store_false", default=True,
                        help="Retain the mathematical bias epilogue instead of canonicalizing preload")
    parser.add_argument("--tensor-memory-bytes", type=int, default=64 << 20)
    tuning = parser.add_mutually_exclusive_group()
    tuning.add_argument("--tuning-database", type=Path, help="Standard MetaSchedule JSONDatabase directory")
    tuning.add_argument("--tuning-preset", choices=PRESET_NAMES,
                        help="Apply shipped winners where structurally matching; report fallbacks")
    fallback = parser.add_mutually_exclusive_group()
    fallback.add_argument("--fallback-tuning-database", type=Path,
                          help="Optional frozen JSONDatabase used only where the explicit primary has no match")
    fallback.add_argument("--fallback-tuning-preset", choices=PRESET_NAMES,
                          help="Shipped preset consulted only where the primary has no match")
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
