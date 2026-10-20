#!/usr/bin/env python3
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
"""Compile structural TensorIR fixtures and run bounded Spike/Verilator checks.

Numerical references are independently computed with INT64 NumPy arithmetic.
The guest executes original TVM operators, checks every output and input guard,
and observes actual primitive calls. These runs establish functional evidence,
not hardware timing, model quality, or coverage of unimplemented operators.
"""

import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import time

from verify_baremetal_cpu import STARTUP, LINKER, identity
from verify_matmul import audit_instructions


GUEST = r"""
#include <stdint.h>
#include <stddef.h>
volatile uint64_t tohost __attribute__((section(".tohost"),aligned(64)));
volatile uint64_t fromhost __attribute__((section(".tohost"),aligned(64)));
static void puts_htif(const char* text) {
  static volatile uint64_t request[4] __attribute__((aligned(64)));
  size_t n=0;while(text[n])++n;
  request[0]=64;request[1]=1;request[2]=(uintptr_t)text;request[3]=n;
  asm volatile("fence rw,rw":::"memory");tohost=(uintptr_t)request;
  while(!fromhost){}fromhost=0;asm volatile("fence rw,rw":::"memory");
}
void* memset(void* p,int c,size_t n) { volatile unsigned char* d=p;for(size_t i=0;i<n;++i)d[i]=c;return p; }
void* memcpy(void* d,const void* s,size_t n) { volatile unsigned char* o=d;const volatile unsigned char* p=s;for(size_t i=0;i<n;++i)o[i]=p[i];return d; }
int printf(const char* format,...) { (void)format;puts_htif("VENDOR_DIAGNOSTIC\n");return 0; }
__attribute__((noreturn)) void exit(int code) { tohost=((uint64_t)(code?code:99)<<1)|1;for(;;){} }
static unsigned computes,zeros,dmas,accumulates;
void __real_tvm_gemmini_compute(uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,int32_t);
void __real_tvm_gemmini_zero_acc(uint32_t,uint32_t,uint32_t);
void __wrap_tvm_gemmini_compute(uint32_t a,uint32_t b,uint32_t c,uint32_t m,uint32_t n,uint32_t k,int32_t add) {
  ++computes;__real_tvm_gemmini_compute(a,b,c,m,n,k,add);
}
void __wrap_tvm_gemmini_zero_acc(uint32_t c,uint32_t m,uint32_t n) { ++zeros;__real_tvm_gemmini_zero_acc(c,m,n); }
void __real_tvm_gemmini_dma_acc_write(const int32_t*,uint32_t,uint32_t,uint32_t,int64_t,int32_t);
void __real_tvm_gemmini_dma_acc_add(const int32_t*,uint32_t,uint32_t,uint32_t,int64_t,int32_t);
void __real_tvm_gemmini_dma_widen(const int8_t*,uint32_t,uint32_t,uint32_t,int64_t,int32_t);
void __real_tvm_gemmini_dma_spad_out(int8_t*,uint32_t,uint32_t,uint32_t,int64_t,int32_t);
void __wrap_tvm_gemmini_dma_acc_write(const int32_t* p,uint32_t c,uint32_t m,uint32_t n,int64_t s,int32_t w) {
  ++dmas;__real_tvm_gemmini_dma_acc_write(p,c,m,n,s,w);
}
void __wrap_tvm_gemmini_dma_acc_add(const int32_t* p,uint32_t c,uint32_t m,uint32_t n,int64_t s,int32_t w) {
  ++dmas;++accumulates;__real_tvm_gemmini_dma_acc_add(p,c,m,n,s,w);
}
void __wrap_tvm_gemmini_dma_widen(const int8_t* p,uint32_t c,uint32_t m,uint32_t n,int64_t s,int32_t w) {
  ++dmas;__real_tvm_gemmini_dma_widen(p,c,m,n,s,w);
}
void __wrap_tvm_gemmini_dma_spad_out(int8_t* p,uint32_t c,uint32_t m,uint32_t n,int64_t s,int32_t w) {
  ++dmas;__real_tvm_gemmini_dma_spad_out(p,c,m,n,s,w);
}
static uint64_t hash(const void* p,size_t n) {
  const volatile unsigned char* b=p;uint64_t h=1469598103934665603ULL;
  for(size_t i=0;i<n;++i)h=(h^b[i])*1099511628211ULL;return h;
}
"""


def fixtures():
    import numpy as np
    from tvm import relax, te

    result = []
    for name, m, n, k, batches, fused in (
        ("small", 3, 4, 5, 1, False),
        ("full", 16, 16, 16, 1, True),
        ("tail", 17, 3, 17, 1, True),
        ("batched", 3, 4, 5, 2, True),
        ("reuse", 16, 32, 8, 1, False),
    ):
        ashape = (batches, m, k) if batches > 1 else (m, k)
        bshape = (batches, k, n) if batches > 1 else (k, n)
        a = ((np.arange(np.prod(ashape)).reshape(ashape) * 13) % 256 - 128).astype("int8")
        b = ((np.arange(np.prod(bshape)).reshape(bshape) * 7) % 256 - 128).astype("int8")
        if name == "small":
            a.fill(-128)
            b.fill(-128)
        bias = np.arange(n, dtype="int32") - 9
        av = relax.Var("a", relax.TensorStructInfo(ashape, "int8"))
        bv = relax.Var("b", relax.TensorStructInfo(bshape, "int8"))
        parameters, arrays = [av, bv], [a, b]
        if fused:
            dv = relax.Var("bias", relax.TensorStructInfo((n,), "int32"))
            parameters.append(dv)
            arrays.append(bias)
        builder = relax.BlockBuilder()
        with builder.function("main", parameters):
            with builder.dataflow():
                value = builder.emit(relax.op.matmul(av, bv, out_dtype="int32"))
                if fused:
                    value = builder.emit(relax.op.nn.relu(relax.op.add(value, dv)))
                output = builder.emit_output(value)
            builder.emit_func_output(output)
        expected = a.astype("int64") @ b.astype("int64")
        if fused:
            expected = np.maximum(expected + bias, 0)
        result.append((name, builder.get(), arrays, expected.astype("int32")))
    a = (np.arange(9).reshape(1, 1, 3, 3) * 13 - 60).astype("int8")
    b = (np.arange(8).reshape(2, 1, 2, 2) * 11 - 30).astype("int8")
    av = relax.Var("image", relax.TensorStructInfo(a.shape, "int8"))
    bv = relax.Var("filter", relax.TensorStructInfo(b.shape, "int8"))
    builder = relax.BlockBuilder()
    with builder.function("main", [av, bv]):
        with builder.dataflow():
            value = builder.emit_output(relax.op.nn.conv2d(av, bv, out_dtype="int32"))
        builder.emit_func_output(value)
    expected = np.zeros((1, 2, 2, 2), "int64")
    for _, oc, y, x in np.ndindex(expected.shape):
        expected[0, oc, y, x] = sum(
            int(a[0, 0, y + ky, x + kx]) * int(b[oc, 0, ky, kx])
            for ky in range(2)
            for kx in range(2)
        )
    result.append(("conv", builder.get(), [a, b], expected.astype("int32")))
    extremes = np.array([0, 1, 2147483647, -2147483648, -1, 305419896], "int32")
    for kind in ("add_i32", "copy_i8", "copy_i32", "widen"):
        if kind in ("copy_i8", "widen"):
            a = (
                (np.arange(256).reshape(16, 16) - 128).astype("int8")
                if kind == "widen"
                else (np.arange(15).reshape(3, 5) * 17 - 128).astype("int8")
            )
        else:
            a = np.resize(extremes, (3, 4))
        av = relax.Var("a", relax.TensorStructInfo(a.shape, str(a.dtype)))
        parameters, arrays = [av], [a]
        builder = relax.BlockBuilder()
        copy_gv = None
        if kind.startswith("copy"):
            source = te.placeholder(a.shape, str(a.dtype))
            copied = te.compute(a.shape, lambda *i: source[i], name="copied")
            copy_gv = builder.add_func(te.create_prim_func([source, copied]), "copy")
        if kind == "add_i32":
            b = np.resize(
                np.array([-1, 2147483647, 1, -1, -2147483648, 2147483647], "int32"), a.shape
            )
            bv = relax.Var("b", relax.TensorStructInfo(b.shape, "int32"))
            parameters.append(bv)
            arrays.append(b)
            left = te.placeholder(a.shape, "int32", "left")
            right = te.placeholder(b.shape, "int32", "right")
            wrapped = te.compute(
                a.shape,
                lambda *i: (left[i].astype("int64") + right[i].astype("int64")).astype("int32"),
                name="wrapping_sum",
            )
            add_gv = builder.add_func(te.create_prim_func([left, right, wrapped]), "wrapping_add")
        with builder.function("main", parameters):
            with builder.dataflow():
                if kind == "add_i32":
                    value = relax.call_tir(
                        add_gv, (av, bv), relax.TensorStructInfo(a.shape, "int32")
                    )
                elif kind == "widen":
                    value = relax.op.astype(av, "int32")
                else:
                    value = relax.call_tir(
                        copy_gv, (av,), relax.TensorStructInfo(a.shape, str(a.dtype))
                    )
                output = builder.emit_output(value)
            builder.emit_func_output(output)
        expected = (
            (a.astype("int64") + b.astype("int64")).astype("int32")
            if kind == "add_i32"
            else a.astype("int32")
            if kind == "widen"
            else a
        )
        result.append((kind, builder.get(), arrays, expected))
    for kind in ("permute", "repeat"):
        a = (np.arange(6).reshape(2, 3) * 17 - 128).astype("int8")
        av = relax.Var("a", relax.TensorStructInfo(a.shape, "int8"))
        builder = relax.BlockBuilder()
        with builder.function("main", [av]):
            with builder.dataflow():
                value = (
                    relax.op.permute_dims(av, [1, 0])
                    if kind == "permute"
                    else relax.op.tile(av, [2, 2])
                )
                output = builder.emit_output(value)
            builder.emit_func_output(output)
        expected = a.T if kind == "permute" else np.tile(a, (2, 2))
        result.append((kind, builder.get(), [a], expected))
    for kind in ("fill_i8", "fill_i32", "fill_widen", "add_scalar"):
        source_dtype = "int8" if kind in ("fill_i8", "fill_widen") else "int32"
        scalar = np.array(
            1 if kind == "add_scalar" else -128 if source_dtype == "int8" else -2147483648,
            source_dtype,
        )
        sv = relax.Var("scalar", relax.TensorStructInfo((), source_dtype))
        parameters, arrays = [sv], [scalar]
        builder = relax.BlockBuilder()
        if kind == "add_scalar":
            x = np.resize(np.array([2147483647, -2147483648, -1, 0, 17, 1234], "int32"), (3, 4))
            xv = relax.Var("x", relax.TensorStructInfo(x.shape, "int32"))
            parameters, arrays = [xv, sv], [x, scalar]
            a = te.placeholder(x.shape, "int32", "x")
            b = te.placeholder((), "int32", "scalar")
            c = te.compute(
                x.shape,
                lambda *i: (a[i].astype("int64") + b[()].astype("int64")).astype("int32"),
                name="scalar_sum",
            )
            gv = builder.add_func(te.create_prim_func([a, b, c]), "scalar_add")
        with builder.function("main", parameters):
            with builder.dataflow():
                value = (
                    relax.call_tir(gv, (xv, sv), relax.TensorStructInfo(x.shape, "int32"))
                    if kind == "add_scalar"
                    else relax.op.full((3, 4), sv, "int32" if kind != "fill_i8" else "int8")
                )
                output = builder.emit_output(value)
            builder.emit_func_output(output)
        expected = (
            (x.astype("int64") + int(scalar)).astype("int32")
            if kind == "add_scalar"
            else np.full((3, 4), int(scalar), "int8" if kind == "fill_i8" else "int32")
        )
        result.append((kind, builder.get(), arrays, expected))
    return result


def prepare(args):
    import numpy as np
    from tvm.relax.backend.contrib.gemmini import prepare_gemmini_graph
    from tvm.relax.backend.contrib.gemmini_target import GemminiCapabilities
    from baremetal import export_graphs

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    adapter_path = args.adapter_receipt.resolve(strict=True)
    adapter = json.loads(adapter_path.read_text())
    app = Path(__file__).resolve().parent
    for name, digest in adapter["adapter_sources"].items():
        if identity(app / name)["sha256"] != digest:
            raise ValueError("Adapter source changed: " + name)
    adapter_object = adapter_path.parent / "matmul.o"
    if identity(adapter_object)["sha256"] != adapter["object_sha256"]:
        raise ValueError("Adapter object changed")
    cases = [case for case in fixtures() if case[0] in args.cases]
    if len(cases) != len(set(args.cases)) or not cases:
        raise ValueError("Unknown or empty verification case selection")
    modules, reports = {}, {}
    for name, mod, _, _ in cases:
        report = []
        options = {"col_tiles": 2, "reduction_chunk": 8} if name == "reuse" else None
        modules[name] = prepare_gemmini_graph(mod, report=report, schedule_options=options)
        if sum(row["placement"] == "gemmini" for row in report) != 1:
            raise ValueError("Fixture did not use the structural accelerator rule: " + name)
        reports[name] = report
    package = out / "package"
    manifest = export_graphs(modules, package, memory_limit_bytes=16 << 20)
    guest = [GUEST, '#include "package/model.h"']
    checks = []
    for index, (name, _, arrays, expected) in enumerate(cases):
        output_ctype = "int8_t" if expected.dtype == np.int8 else "int32_t"
        for arg, array in enumerate(arrays):
            ctype = "int8_t" if array.dtype == np.int8 else "int32_t"
            values = ",".join(str(int(x)) for x in array.flat)
            guest.append(
                f"static const {ctype} in_{index}_{arg}[] __attribute__((aligned(64)))={{{values}}};"
            )
        values = ",".join(str(int(x)) for x in expected.flat)
        guest.append(f"static const {output_ctype} expected_{index}[]={{{values}}};")
        size = expected.size
        workspace = manifest["stages"][name]["workspace_bytes"]
        guest.append(
            f"static struct {{ uint8_t head[64]; {output_ctype} data[{size}]; uint8_t tail[64]; }} out_{index} __attribute__((aligned(64)));"
        )
        guest.append(
            f"static uint8_t work_{index}[{max(workspace, 1)}] __attribute__((aligned(64)));"
        )
        pointers = ",".join(f"in_{index}_{arg}" for arg in range(len(arrays)))
        checks.extend(
            [
                f'puts_htif("TENSORIR_CASE {name}\\n");',
                f"const void* inputs_{index}[]={{{pointers}}}; void* outputs_{index}[]={{out_{index}.data}};",
                f"memset(&out_{index},0x6d,sizeof(out_{index}));",
                f"unsigned before_{index}=computes, zero_before_{index}=zeros;",
                f"unsigned dma_before_{index}=dmas, add_before_{index}=accumulates;",
            ]
        )
        for arg in range(len(arrays)):
            checks.append(
                f"uint64_t hash_{index}_{arg}=hash(in_{index}_{arg},sizeof(in_{index}_{arg}));"
            )
        operation = next(
            row["metadata"].get("operation")
            for row in reports[name]
            if row["placement"] == "gemmini"
        )
        primitive_check = (
            f"if(dmas==dma_before_{index})return 21;"
            if operation
            else f"if(computes==before_{index} || zeros==zero_before_{index})return 21;"
        )
        if operation is not None and operation.startswith("add_i32"):
            primitive_check += f"if(accumulates==add_before_{index})return 21;"
        checks.extend(
            [
                f"if(stage_{index}_model_run(inputs_{index},outputs_{index},work_{index},{workspace})) return 20;",
                primitive_check,
                f'for(unsigned i=0;i<{size};++i)if((int64_t)out_{index}.data[i]!=(int64_t)expected_{index}[i]+BAD_ORACLE){{puts_htif("TENSORIR_NUMERICAL_FAILURE\\n");return 22;}}',
                f"for(unsigned i=0;i<64;++i)if(out_{index}.head[i]!=0x6d || out_{index}.tail[i]!=0x6d)return 23;",
            ]
        )
        for arg in range(len(arrays)):
            checks.append(
                f"if(hash_{index}_{arg}!=hash(in_{index}_{arg},sizeof(in_{index}_{arg})))return 24;"
            )
    guest.append(
        "int main(void){\n"
        + "\n".join(checks)
        + f'\nputs_htif("TENSORIR_PASS cases={len(cases)} exact=1 guards=1 primitives=1\\n");return 0;}}\n'
    )
    (out / "runner.c").write_text("\n".join(guest))
    (out / "start.S").write_text(STARTUP)
    (out / "link.ld").write_text(LINKER)
    gcc = args.riscv_gcc.resolve(strict=True)
    flags = [
        "-std=gnu11",
        "-march=rv64gc",
        "-mabi=lp64d",
        "-mcmodel=medany",
        "-msmall-data-limit=0",
        "-O2",
        "-ffreestanding",
        "-fno-builtin",
        "-fno-stack-protector",
        "-fno-tree-loop-distribute-patterns",
        "-nostdlib",
        "-nostartfiles",
        "-static",
        "-Wl,--no-relax",
        "-Wl,-T," + str(out / "link.ld"),
        "-Wl,--wrap=tvm_gemmini_compute",
        "-Wl,--wrap=tvm_gemmini_zero_acc",
        "-Wl,--wrap=tvm_gemmini_dma_acc_write",
        "-Wl,--wrap=tvm_gemmini_dma_acc_add",
        "-Wl,--wrap=tvm_gemmini_dma_widen",
        "-Wl,--wrap=tvm_gemmini_dma_spad_out",
        "-I" + str(out),
    ]
    sources = [out / "start.S", out / "runner.c", package / "runtime.c", package / "constants.S"]
    for name in modules:
        sources.extend([package / name / "model.c", package / name / "operators.o"])
    policy = adapter["instruction_policy"]
    forbidden = {
        value for key, value in policy["derived_definitions"].items() if key.startswith("k_LOOP_")
    }
    commands, elfs = [], {}
    for label, wrong in (("valid", 0), ("bad-oracle", 1)):
        elf = out / (label + ".elf")
        command = [
            str(gcc),
            *flags,
            "-DBAD_ORACLE=" + str(wrong),
            *map(str, sources),
            str(adapter_object),
            "-o",
            str(elf),
        ]
        process = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=45
        )
        (out / (label + "-link.log")).write_text(process.stdout)
        if process.returncode:
            raise RuntimeError(process.stdout)
        audit = audit_instructions(elf.read_bytes(), policy["custom_opcode"], forbidden)
        if audit["prohibited_hits"]:
            raise ValueError("Hardware-loop instructions in final ELF")
        nm = gcc.with_name(gcc.name[:-3] + "nm")
        unresolved = subprocess.check_output([nm, "-u", elf], text=True)
        if unresolved.strip():
            raise ValueError("Unresolved guest symbols: " + unresolved)
        commands.append({"argv": command, "exit_code": process.returncode})
        elfs[label] = dict(identity(elf), instruction_audit=audit)
    source = app.parents[1]
    bindings = [identity(adapter_path), identity(adapter_object), identity(gcc)]
    from tvm._ffi.base import _LIB  # pylint: disable=import-outside-toplevel

    bindings.append(identity(_LIB._name))
    bindings.extend(identity(path) for path in sorted(package.rglob("*")) if path.is_file())
    bindings.extend(identity(out / name) for name in ("runner.c", "start.S", "link.ld"))
    for relative in (
        "apps/gemmini/verify_tensorir.py",
        "apps/gemmini/baremetal.py",
        "apps/gemmini/matmul.c",
        "apps/gemmini/matmul.h",
        *[
            "python/tvm/relax/backend/contrib/" + name + ".py"
            for name in (
                "gemmini",
                "gemmini_tensorir",
                "gemmini_intrin",
                "gemmini_memory",
                "gemmini_target",
                "gemmini_meta_schedule",
                "gemmini_elementwise",
            )
        ],
    ):
        bindings.append(identity(source / relative))
    bundle = {
        "status": "prepared",
        "scope": "Bounded integer fixtures; no model/performance qualification",
        "capabilities": GemminiCapabilities().fingerprint,
        "ELFs": elfs,
        "scheduling": reports,
        "commands": commands,
        "bindings": bindings,
        "case_count": len(cases),
        "cases": [case[0] for case in cases],
    }
    (out / "bundle.json").write_text(json.dumps(bundle, indent=2) + "\n")
    print(json.dumps({"status": "prepared", "bundle": str(out / "bundle.json")}))


def classify_output(backend, text, returncode, timed_out, negative=False, expected_cases=4):
    """Require guest evidence and exact decoded exit status, including RTL padding."""
    if timed_out:
        return False
    if negative:
        if backend == "spike":
            decoded = returncode == 22
        else:
            codes = []
            for line in text.splitlines():
                if "Assertion failed:" in line and "exit code =" in line:
                    value = line.partition("exit code =")[2].partition(")")[0].strip()
                    if value.isdecimal():
                        codes.append(int(value))
            decoded = returncode == 255 and codes == [22]
        return decoded and "TENSORIR_NUMERICAL_FAILURE" in text
    marker = f"TENSORIR_PASS cases={expected_cases} exact=1 guards=1 primitives=1"
    passed = returncode == 0 and marker in text.splitlines()
    return passed and (backend == "spike" or "Verilog $finish" in text)


def run(args):
    out = args.bundle.resolve(strict=True)
    bundle = json.loads((out / "bundle.json").read_text())
    result = out / (args.name + "-result.json")
    if result.exists():
        raise ValueError("Refusing to overwrite an execution receipt")
    label = "bad-oracle" if args.negative else "valid"
    elf = bundle["ELFs"][label]
    bindings = list(bundle["bindings"]) + [elf]
    env = dict(os.environ)
    if args.backend == "spike":
        receipt = json.loads(args.simulator_receipt.read_text())
        bindings.extend(
            [
                identity(args.simulator_receipt),
                {"path": receipt["plugin_path"], "sha256": receipt["plugin_sha256"]},
                {"path": receipt["spike_path"], "sha256": receipt["spike_sha256"]},
            ]
        )
        for key, value in receipt["runtime_environment"].items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = str(value)
        command = [
            receipt["spike_path"],
            "--extlib=" + receipt["plugin_path"],
            "--extension=gemmini",
            "-m0x80000000:0x10000000",
            elf["path"],
        ]
    else:
        receipt = json.loads(args.hardware_receipt.read_text())
        simulator = receipt["simulator"]
        bindings.extend([identity(args.hardware_receipt), simulator])
        command = [
            simulator["path"],
            "+permissive",
            "+max-cycles=" + str(args.cycles),
            "+loadmem=" + elf["path"],
            "+permissive-off",
            elf["path"],
        ]
    for binding in bindings:
        if identity(binding["path"])["sha256"] != binding["sha256"]:
            raise ValueError("Execution input changed: " + binding["path"])

    def limits():
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
        resource.setrlimit(resource.RLIMIT_FSIZE, (16 << 20, 16 << 20))
        if hasattr(os, "sched_getaffinity"):
            os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})

    start, timed_out = time.monotonic(), False
    with (out / (args.name + ".log")).open("wb") as log:
        child = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, env=env, preexec_fn=limits
        )
        try:
            code = child.wait(timeout=args.seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            child.kill()
            code = child.wait()
    text = (out / (args.name + ".log")).read_text(errors="replace")
    passed = classify_output(
        args.backend, text, code, timed_out, args.negative, bundle["case_count"]
    )
    record = {
        "status": "passed" if passed else "failed_or_incomplete",
        "backend": args.backend,
        "negative": args.negative,
        "command": command,
        "returncode": code,
        "timed_out": timed_out,
        "wall_seconds": time.monotonic() - start,
        "cycle_limit": args.cycles if args.backend == "verilator" else None,
        "ELF": elf,
        "bindings": bindings,
        "log": identity(out / (args.name + ".log")),
        "functional_only": True,
        "performance_qualified": False,
    }
    result.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": record["status"], "receipt": str(result)}))
    if not passed:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    build = sub.add_parser("prepare")
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--adapter-receipt", type=Path, required=True)
    build.add_argument("--riscv-gcc", type=Path, required=True)
    build.add_argument(
        "--cases",
        nargs="+",
        default=["small", "full", "tail", "batched"],
        choices=[
            "small",
            "full",
            "tail",
            "batched",
            "conv",
            "reuse",
            "add_i32",
            "copy_i8",
            "copy_i32",
            "widen",
            "permute",
            "repeat",
            "fill_i8",
            "fill_i32",
            "fill_widen",
            "add_scalar",
        ],
    )
    execute = sub.add_parser("run")
    execute.add_argument("--bundle", type=Path, required=True)
    execute.add_argument("--backend", choices=("spike", "verilator"), required=True)
    execute.add_argument("--simulator-receipt", type=Path)
    execute.add_argument("--hardware-receipt", type=Path)
    execute.add_argument("--name", required=True)
    execute.add_argument("--negative", action="store_true")
    execute.add_argument("--seconds", type=int, default=60)
    execute.add_argument("--cycles", type=int, default=300000)
    args = parser.parse_args()
    if args.action == "run":
        if not 1 <= args.seconds <= 180 or not 1 <= args.cycles <= 1000000:
            parser.error("Use a bounded run: seconds [1,180], cycles [1,1000000]")
        if args.backend == "spike" and args.simulator_receipt is None:
            parser.error("Spike requires --simulator-receipt")
        if args.backend == "verilator" and args.hardware_receipt is None:
            parser.error("Verilator requires --hardware-receipt")
        run(args)
    else:
        prepare(args)


if __name__ == "__main__":
    main()
