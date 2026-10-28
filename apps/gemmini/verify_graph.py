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
"""Check scheduled TVM matmul or convolution/residual graphs in Gemmini Spike.

This is a bounded compiler/runtime integration check, not a model or timing
qualification. It reuses the pinned adapter/simulator receipt validation and
instruction audit. All generated arithmetic comes from the TVM graph; the guest
contains only independent expected-value checks and primitive-call counters.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback

from verify_baremetal_cpu import STARTUP, LINKER, GUEST_BASE, GUEST_BYTES, STACK_BYTES, checked_path, identity
from verify_matmul import audit_instructions
from verify_matmul_runtime import bind_runtime

RUNNER = r"""
#include <stdint.h>
#include <stddef.h>
#include "matmul.h"
#include "fixture.h"
volatile uint64_t tohost __attribute__((section(".tohost"), aligned(64)));
volatile uint64_t fromhost __attribute__((section(".tohost"), aligned(64)));
static void puts_htif(const char *text) {
  static volatile uint64_t request[4] __attribute__((aligned(8)));
  uint64_t size=0; while(text[size]) ++size;
  request[0]=64; request[1]=1; request[2]=(uintptr_t)text; request[3]=size;
  asm volatile("fence rw,rw" ::: "memory"); tohost=(uintptr_t)request;
  while(!fromhost) {} fromhost=0; asm volatile("fence rw,rw" ::: "memory");
}
int printf(const char *format, ...) { puts_htif("VENDOR_DIAGNOSTIC\n"); return 0; }
__attribute__((noreturn)) void exit(int code) { tohost=((uint64_t)(uint32_t)(code ? code : 99)<<1)|1; for(;;) {} }
extern int32_t model_run(const void *const*, void *const*, uint8_t*, uint64_t);
static unsigned loads_a, loads_b, computes, stores, overwrites;
void __real_tvm_gemmini_load_a(const int8_t*,uint32_t,uint32_t,uint32_t);
void __real_tvm_gemmini_load_b(const int8_t*,uint32_t,uint32_t,uint32_t);
void __real_tvm_gemmini_compute(uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,int32_t);
void __real_tvm_gemmini_store(int32_t*,uint32_t,uint32_t,uint32_t);
void __wrap_tvm_gemmini_load_a(const int8_t* p,uint32_t r,uint32_t m,uint32_t n) { ++loads_a; __real_tvm_gemmini_load_a(p,r,m,n); }
void __wrap_tvm_gemmini_load_b(const int8_t* p,uint32_t r,uint32_t m,uint32_t n) { ++loads_b; __real_tvm_gemmini_load_b(p,r,m,n); }
void __wrap_tvm_gemmini_compute(uint32_t a,uint32_t b,uint32_t c,uint32_t m,uint32_t n,uint32_t k,int32_t add) {
  ++computes; if(!add) ++overwrites; __real_tvm_gemmini_compute(a,b,c,m,n,k,add);
}
void __wrap_tvm_gemmini_store(int32_t* p,uint32_t r,uint32_t m,uint32_t n) { ++stores; __real_tvm_gemmini_store(p,r,m,n); }
struct input_buffer { uint8_t head[64]; int8_t value[INPUT_ELEMENTS]; uint8_t tail[64]; };
struct output_buffer { uint8_t head[64]; OUTPUT_TYPE value[M*N]; uint8_t tail[64]; } __attribute__((aligned(64)));
struct workspace_buffer { uint8_t head[64]; uint8_t value[WORKSPACE_BYTES]; uint8_t tail[64]; };
static struct input_buffer input __attribute__((aligned(64)));
static struct output_buffer output[2] __attribute__((aligned(64)));
static struct workspace_buffer workspace __attribute__((aligned(64)));
static OUTPUT_TYPE retained[M*N];
#if CONV
struct raw_buffer { uint8_t head[64]; int32_t value[M*N]; uint8_t tail[64]; } __attribute__((aligned(64)));
static struct raw_buffer raw[2] __attribute__((aligned(64)));
static int32_t raw_retained[M*N];
#endif
extern const uint8_t model_constants[];
static uint64_t hash(const void *p, uint64_t n) {
  const volatile uint8_t *bytes=p; uint64_t h=UINT64_C(1469598103934665603);
  for(uint64_t i=0;i<n;++i) h=(h^bytes[i])*UINT64_C(1099511628211); return h;
}
static int64_t contraction(unsigned row, unsigned col) {
  int64_t value=0;
#if CONV
  int y=(int)(row/W), x=(int)(row%W);
  for(unsigned channel=0;channel<CHANNELS;++channel)
    for(unsigned ky=0;ky<KERNEL;++ky) for(unsigned kx=0;kx<KERNEL;++kx) {
      int iy=y+(int)ky-PAD, ix=x+(int)kx-PAD;
      if(iy>=0 && iy<H && ix>=0 && ix<W) {
        unsigned weight=((col*CHANNELS+channel)*KERNEL+ky)*KERNEL+kx;
        value+=(int64_t)input.value[(channel*H+iy)*W+ix]*((int)(weight*7%255)-127);
      }
    }
#else
  for(unsigned k=0;k<K;++k) value+=(int64_t)input.value[row*K+k]*((int)((k*N+col)*7%255)-127);
#endif
  return value;
}
static int guards(void) {
  for(unsigned i=0;i<64;++i) {
    if(input.head[i]!=0x6d || input.tail[i]!=0x6d || workspace.head[i]!=0x6d || workspace.tail[i]!=0x6d) return 0;
    for(unsigned b=0;b<2;++b) if(output[b].head[i]!=0x6d || output[b].tail[i]!=0x6d) return 0;
#if CONV
    for(unsigned b=0;b<2;++b) if(raw[b].head[i]!=0x6d || raw[b].tail[i]!=0x6d) return 0;
#endif
  }
  return 1;
}
int main(void) {
  for(unsigned i=0;i<64;++i) {
    input.head[i]=input.tail[i]=workspace.head[i]=workspace.tail[i]=0x6d;
    for(unsigned b=0;b<2;++b) output[b].head[i]=output[b].tail[i]=0x6d;
#if CONV
    for(unsigned b=0;b<2;++b) raw[b].head[i]=raw[b].tail[i]=0x6d;
#endif
  }
#if CONV
  const void *inputs[1]={input.value}; void *outputs[2]={output[0].value,raw[0].value};
#else
  const void *inputs[1]={input.value}; void *outputs[1]={output[0].value};
#endif
  uint64_t constants_before=hash(model_constants,CONSTANT_BYTES);
  if(model_run(inputs,outputs,workspace.value,0)!=-1 || computes) return 10;
  inputs[0]=input.value+1;
  if(model_run(inputs,outputs,workspace.value,WORKSPACE_BYTES)!=-1 || computes) return 11;
  inputs[0]=input.value;
  if(model_run(inputs,outputs,(uint8_t*)output[0].value,WORKSPACE_BYTES)!=-2 || computes) return 12;
  puts_htif("GRAPH_INPUT_REJECTION_PASS\n");
  for(unsigned call=0;call<6;++call) {
    unsigned slot=call%2;
    for(unsigned i=0;i<INPUT_ELEMENTS;++i) input.value[i]=(int8_t)((int)((i*13+call*7)%256)-128);
    for(unsigned i=0;i<M*N;++i) output[slot].value[i]=37;
    for(unsigned i=0;i<WORKSPACE_BYTES;++i) workspace.value[i]=(uint8_t)(0xa5+call);
    uint64_t before=hash(input.value,sizeof(input.value));
    loads_a=loads_b=computes=stores=overwrites=0;
    outputs[0]=output[slot].value;
#if CONV
    outputs[1]=raw[slot].value;
    for(unsigned i=0;i<M*N;++i) raw[slot].value[i]=0x12345678;
#endif
    if(model_run(inputs,outputs,workspace.value,WORKSPACE_BYTES)) return 20;
    if(loads_a!=LOADS_A || loads_b!=LOADS_B || computes!=COMPUTES || stores!=STORES || overwrites!=STORES) return 21;
    for(unsigned row=0;row<M;++row) for(unsigned col=0;col<N;++col) {
      int64_t expected=contraction(row,col);
#if CONV
      unsigned index=col*M+row;
      if(raw[slot].value[index]!=expected) { puts_htif("GRAPH_RAW_CONV_FAILURE\n"); return 27; }
      expected+=(int)col-9+(int)input.value[index]*3-5;
      expected=expected>=0 ? expected/512 : -((-expected+511)/512);
      if(expected>127) expected=127;
#else
      unsigned index=row*N+col;
      expected+=(int)col-9;
#endif
      if(expected<0) expected=0;
      if(output[slot].value[index]!=expected+BAD_ORACLE) { puts_htif("GRAPH_NUMERICAL_FAILURE\n"); return 22; }
    }
    if(before!=hash(input.value,sizeof(input.value)) || constants_before!=hash(model_constants,CONSTANT_BYTES) || !guards()) return 23;
    if(call) for(unsigned i=0;i<M*N;++i) if(output[1-slot].value[i]!=retained[i]) return 24;
    for(unsigned i=0;i<M*N;++i) retained[i]=output[slot].value[i];
#if CONV
    if(call) for(unsigned i=0;i<M*N;++i) if(raw[1-slot].value[i]!=raw_retained[i]) return 28;
    for(unsigned i=0;i<M*N;++i) raw_retained[i]=raw[slot].value[i];
#endif
    if(model_run(inputs,outputs,workspace.value,0)!=-1) return 25;
  }
  extern uint64_t _stack_bottom[], _stack_top[];
  if(_stack_bottom[0]!=UINT64_C(0x55aa55aa55aa55aa)) return 26;
  uint64_t *first=_stack_bottom;
  while(first<_stack_top && *first==UINT64_C(0x55aa55aa55aa55aa)) ++first;
  uint64_t used=(uintptr_t)_stack_top-(uintptr_t)first;
  char digits[24]; unsigned count=0;
  do { digits[count++]=(char)('0'+used%10); used/=10; } while(used);
  puts_htif("STACK_HIGH_WATER_BYTES=");
  while(count) { char digit[2]={digits[--count],0}; puts_htif(digit); }
  puts_htif("\n");
  puts_htif("TVM_GEMMINI_GRAPH_PASS calls=6 exact=1 reuse=1 guards=1 retained=1 recovery=1\n");
  return 0;
}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = ("tvm-source", "tvm-build", "adapter-receipt", "simulator-receipt", "riscv-gcc", "spike", "dtc", "spike-library-dir", "plugin", "output-dir")
    for name in names:
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--tile-i", type=int, choices=(1, 2, 4))
    parser.add_argument("--tile-j", type=int, choices=(1, 2, 4))
    parser.add_argument("--graph-mode", choices=("baseline", "optimized"), default="baseline")
    parser.add_argument("--workload", choices=("matmul", "conv-residual"), default="matmul")
    parser.add_argument("--conv-kernel", type=int, choices=(1, 3, 7), help="Convolution fixture kernel size (default 3)")
    parser.add_argument("--search-seed", type=int, help="Execute a reproducible untrained search proposal; supplies no timing labels")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    args = parser.parse_args()
    if not 1 <= args.timeout_seconds <= 300:
        parser.error("timeout must be in [1,300]")
    if args.conv_kernel is not None and args.workload != "conv-residual":
        parser.error("--conv-kernel requires --workload conv-residual")
    if args.search_seed is not None and (not 0 <= args.search_seed < (1 << 31) - 1 or args.tile_i is not None or args.tile_j is not None):
        parser.error("search seed must be in [0,2^31-2] and cannot be combined with explicit tiles")
    paths = {name: checked_path(value) for name, value in vars(args).items() if isinstance(value, Path)}
    out = paths["output_dir"]
    out.mkdir(parents=True, exist_ok=False)
    app = checked_path(Path(__file__).parent)
    report = {"status": "running", "scope": "bounded scheduled TVM graph and primitive functional simulation", "invocation": [sys.executable, *sys.argv], "commands": [], "checks": [], "timing_qualified": False, "full_model_qualified": False, "deployment_qualified": False}
    immutable = {}

    def save():
        (out / "receipt.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    def bind(path, expected=None):
        record = identity(path)
        if expected is not None and record["sha256"] != expected:
            raise ValueError("identity mismatch: " + str(path))
        immutable[record["path"]] = record["sha256"]
        return record

    def command(argv, label, env=None, expected=0):
        entry = {"argv": list(map(str, argv)), "log": label + ".log", "expected": expected}
        report["commands"].append(entry)
        try:
            process = subprocess.run(entry["argv"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=args.timeout_seconds, env=env)
        except subprocess.TimeoutExpired as error:
            output = error.stdout or b""
            (out / entry["log"]).write_text(output.decode(errors="replace") if isinstance(output, bytes) else output)
            entry["timed_out"] = True
            save()
            raise RuntimeError(label + " timed out") from error
        (out / entry["log"]).write_text(process.stdout)
        entry["exit_code"] = process.returncode
        save()
        if (expected == "nonzero" and process.returncode == 0) or (expected != "nonzero" and process.returncode != expected):
            raise RuntimeError(label + " failed; see " + entry["log"])
        return process.stdout

    try:
        adapter, gcc, tools, opcode, forbidden = bind_runtime(paths, app, report, bind)
        report["runtime_helpers"].append(report["verifier_source"])
        report["verifier_source"] = bind(__file__)
        report["exporter_source"] = bind(app / "baremetal.py")
        source, build = paths["tvm_source"], paths["tvm_build"]
        os.environ.update(TVM_LIBRARY_PATH=str(build), TVM_FFI="ctypes", PYTHONDONTWRITEBYTECODE="1")
        sys.path.insert(0, str(checked_path(source / "python")))
        import numpy as np
        import tvm
        from tvm import relax
        from tvm._ffi.base import _LIB
        from tvm.relax.backend.contrib.gemmini import prepare_gemmini_graph_legacy as prepare_gemmini_graph
        from tvm.relax.backend.contrib.gemmini_schedule import make_gemmini_matmul
        from baremetal import export_graph
        if checked_path(tvm.__file__).parent != source / "python/tvm" or checked_path(_LIB._name).parent != build:
            raise ValueError("TVM source/library identity differs from explicit paths")
        report["compiler_library"] = bind(_LIB._name)
        report["build_info"] = dict(tvm.support.libinfo())
        report["source_head"] = command(["git", "-C", source, "rev-parse", "HEAD"], "source-head").strip()
        for relative in ("python/tvm/relax/backend/contrib/gemmini.py", "python/tvm/relax/backend/contrib/gemmini_legacy.py", "python/tvm/relax/backend/contrib/gemmini_schedule.py", "python/tvm/relax/backend/contrib/gemmini_conv.py"):
            bind(source / relative)
        convolution = args.workload == "conv-residual"
        kernel = args.conv_kernel or 3
        channels, height, width = 17, 7, 9
        m, n, k = (height * width, channels, channels * kernel * kernel) if convolution else (17, 19, 33)
        search, proposal = None, None
        tile_i, tile_j = args.tile_i or 2, args.tile_j or 2
        if args.search_seed is not None:
            from tvm.relax.backend.contrib.gemmini_tuning import BoundedGemminiSearch
            bind(source / "python/tvm/relax/backend/contrib/gemmini_tuning.py")
            compiler_parts = [args.graph_mode, report["compiler_library"]["sha256"], report["exporter_source"]["sha256"], report["verifier_source"]["sha256"], report["compiler"]["sha256"], *[immutable[str(source / relative)] for relative in ("python/tvm/relax/backend/contrib/gemmini.py", "python/tvm/relax/backend/contrib/gemmini_legacy.py", "python/tvm/relax/backend/contrib/gemmini_schedule.py", "python/tvm/relax/backend/contrib/gemmini_tuning.py")]]
            compiler_parts.append(immutable[str(source / "python/tvm/relax/backend/contrib/gemmini_conv.py")])
            if convolution:
                compiler_parts.extend([args.workload, kernel])
            compiler_id = hashlib.sha256(json.dumps(compiler_parts).encode()).hexdigest()
            search = BoundedGemminiSearch(m, n, k, seed=args.search_seed, compiler_id=compiler_id, adapter_id=report["adapter_object"]["sha256"])
            proposal = search.rank()[0]
            search.validate_semantics(proposal.candidate.candidate_id)
            schedule = search.schedule_for(proposal.candidate.candidate_id)
            tile_i, tile_j = schedule.metadata["tile_i"], schedule.metadata["tile_j"]
            report["search_proposal"] = {"candidate_id": proposal.candidate.candidate_id, "workload_id": search.workload_id, "origin": proposal.origin,
                                       "compiler_id": compiler_id, "adapter_id": report["adapter_object"]["sha256"], "timing_samples": 0, "performance_winner": False}
        else:
            schedule = make_gemmini_matmul(m, n, k, tile_i, tile_j)
        input_shape = (1, channels, height, width) if convolution else (m, k)
        weight_shape = (n, channels, kernel, kernel) if convolution else (k, n)
        a = relax.Var("input", relax.TensorStructInfo(input_shape, "int8"))
        weights = ((np.arange(k*n).reshape(weight_shape)*7 % 255)-127).astype("int8")
        builder = relax.BlockBuilder()
        with builder.function("main", [a]):
            with builder.dataflow():
                if convolution:
                    residual = builder.emit(relax.op.astype(a, "int32"))
                    residual = builder.emit(relax.op.multiply(residual, relax.const(3, "int32")))
                    residual = builder.emit(relax.op.subtract(residual, relax.const(5, "int32")))
                    raw_value = builder.emit(relax.op.nn.conv2d(a, relax.const(weights), padding=kernel // 2, data_layout="NCHW", kernel_layout="OIHW", out_dtype="int32"))
                    value = builder.emit(relax.op.add(raw_value, relax.const((np.arange(n, dtype="int32")-9).reshape(1, n, 1, 1))))
                    value = builder.emit(relax.op.add(value, residual))
                    value = builder.emit(relax.op.right_shift(value, relax.const(9, "int32")))
                    value = builder.emit(relax.op.clip(value, 0, 127))
                    value = builder.emit(relax.op.astype(value, "int8"))
                    value = builder.emit_output(relax.Tuple([value, raw_value]))
                else:
                    value = builder.emit(relax.op.matmul(a, relax.const(weights), out_dtype="int32"))
                    value = builder.emit(relax.op.add(value, relax.const(np.arange(n, dtype="int32")-9)))
                    value = builder.emit_output(relax.op.nn.relu(value))
            builder.emit_func_output(value)
        original = builder.get()
        (out / "input.relax.py").write_text(original.script(show_meta=True))
        lowered = prepare_gemmini_graph(original, tile_i=tile_i, tile_j=tile_j, optimize=args.graph_mode == "optimized")
        primitives = [func for _, func in lowered.functions_items() if isinstance(func, tvm.tir.PrimFunc) and func.attrs and "gemmini.m" in func.attrs]
        if len(primitives) != 1:
            raise ValueError("expected exactly one scheduled graph primitive")
        actual = primitives[0].without_attr("global_symbol").without_attr("op_pattern")
        expected = schedule.scheduled_mod["main"].without_attr("global_symbol").without_attr("op_pattern")
        tvm.ir.assert_structural_equal(actual, expected)
        (out / "schedule.semantic.py").write_text(schedule.semantic_mod.script())
        (out / "schedule.tensorized.py").write_text(schedule.scheduled_mod.script())
        (out / "schedule.trace.json").write_text(json.dumps(schedule.trace.as_json(), indent=2) + "\n")
        graph = export_graph(lowered, out, memory_limit_bytes=1 << 20)
        expected_outputs = [([1, n, height, width], "int8"), ([1, n, height, width], "int32")] if convolution else [([m, n], "int32")]
        if [(item["shape"], item["dtype"]) for item in graph["outputs"]] != expected_outputs or [(item["shape"], item["dtype"]) for item in graph["inputs"]] != [(list(input_shape), "int8")]:
            raise ValueError("exported fixture tensor ABI differs from the guest buffers")
        report["graph"] = graph
        report["graph_mode"] = args.graph_mode
        report["workload"] = args.workload
        if convolution:
            report["convolution"] = {"input_shape": list(input_shape), "weight_shape": list(weight_shape), "padding": kernel // 2, "stride": 1, "dilation": 1, "groups": 1,
                                     "epilogue": "int32 bias + (3*input - 5), floor divide by 512, clip [0,127], cast int8", "raw_output_checked": True,
                                     "scope": "explicit diagnostic integer block; no ResNet quantization or quality policy selected"}
        if not convolution and len(graph["calls"]) != (2 if args.graph_mode == "optimized" else 3):
            raise ValueError("fixture CPU fusion count differs from the selected graph mode")
        llvm_ir = (out / "operators.ll").read_text()
        if "@tvm_gemmini_compute(" not in llvm_ir or "@tvm_gemmini_matmul_i8_i32(" in llvm_ir:
            raise ValueError("generated operators do not use the scheduled primitive route")
        tiles_m, tiles_n, tiles_k = (m+15)//16, (n+15)//16, (k+15)//16
        counters = {"LOADS_A": tiles_m*((tiles_n+tile_j-1)//tile_j)*tiles_k, "LOADS_B": tiles_n*((tiles_m+tile_i-1)//tile_i)*tiles_k, "COMPUTES": tiles_m*tiles_n*tiles_k, "STORES": tiles_m*tiles_n}
        report["schedule"] = {"tile_i": tile_i, "tile_j": tile_j, "expected_calls_per_invocation": counters, "metadata": schedule.metadata,
                              "lowering_steps": schedule.lowering_steps, "trace_scope": "TensorIntrin substitution from the generated tiled semantic template"}
        fixture = {"M": m, "N": n, "K": k, "INPUT_ELEMENTS": int(np.prod(input_shape)), "OUTPUT_TYPE": "int8_t" if convolution else "int32_t", "CONV": int(convolution),
                   "CHANNELS": channels, "H": height, "W": width, "KERNEL": kernel, "PAD": kernel // 2, "WORKSPACE_BYTES": graph["workspace_bytes"], "CONSTANT_BYTES": graph["constants_bytes"], **counters}
        (out / "fixture.h").write_text("\n".join(f"#define {name} {value}" for name, value in fixture.items()) + "\n")
        for name, content in (("start.S", STARTUP), ("link.ld", LINKER), ("runner.c", RUNNER)):
            (out / name).write_text(content)
        env = dict(os.environ)
        for name in ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH", "COMPILER_PATH", "GCC_EXEC_PREFIX", "LD_PRELOAD", "LD_AUDIT"):
            env.pop(name, None)
        env["PATH"] = str(gcc.parent) + ":/usr/bin:/bin"
        flags = ["-std=gnu11", "-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany", "-msmall-data-limit=0", "-O2", "-ffreestanding", "-fno-builtin", "-fno-stack-protector", "-fno-tree-loop-distribute-patterns", "-nostdlib", "-nostartfiles", "-static", "-Wl,--no-relax", "-Wl,-T," + str(out / "link.ld"), "-I" + str(app)]
        wrappers = ["-Wl,--wrap=tvm_gemmini_" + name for name in ("load_a", "load_b", "compute", "store")]
        programs = {}
        for name, wrong in (("valid", 0), ("bad-oracle", 1)):
            elf = out / (name + ".elf")
            command([gcc, *flags, *wrappers, "-DBAD_ORACLE=" + str(wrong), "-Wl,-Map," + str(out / (name + ".map")), out / "start.S", out / "runner.c", out / "model.c", out / "constants.S", out / "operators.o", adapter, "-o", elf], "link-" + name, env)
            audit = audit_instructions(elf.read_bytes(), opcode, forbidden)
            if audit["prohibited_hits"]:
                raise ValueError("FSM instructions in final executable")
            report["checks"].append({"name": name + "_no_fsm", "passed": True, **audit})
            if command([tools["nm"], "-u", elf], "undefined-" + name).strip():
                raise ValueError("guest retains unresolved runtime symbols")
            command([tools["objdump"], "-dr", elf], "disassembly-" + name)
            programs[name] = elf
        headers = command([tools["readelf"], "-W", "-l", programs["valid"]], "program-headers")
        symbols_text = command([tools["nm"], "-S", "--defined-only", programs["valid"]], "symbols")
        segments, symbols = [], {}
        for line in headers.splitlines():
            fields = line.split()
            if fields and fields[0] == "LOAD":
                segments.append({"start": int(fields[2], 16), "file_bytes": int(fields[4], 16), "memory_bytes": int(fields[5], 16)})
        for line in symbols_text.splitlines():
            fields = line.split()
            if len(fields) in (3, 4) and re.fullmatch(r"[0-9a-fA-F]+", fields[0]):
                symbols[fields[-1]] = int(fields[0], 16)
        if not segments or any(s["start"] < GUEST_BASE or s["start"] + s["memory_bytes"] > GUEST_BASE + GUEST_BYTES for s in segments):
            raise ValueError("ELF load extent exceeds requested guest memory")
        if symbols["_stack_top"] - symbols["_stack_bottom"] != STACK_BYTES or symbols["_image_end"] > GUEST_BASE + GUEST_BYTES:
            raise ValueError("stack/image reservation mismatch")
        if any(symbols[name] % 64 for name in ("model_constants", "input", "output", "workspace", *(("raw",) if convolution else ()))):
            raise ValueError("linked tensor storage is not aligned")
        report["memory"] = {"requested_guest_bytes": GUEST_BYTES, "load_segments": segments, "reserved_image_bytes": symbols["_image_end"] - GUEST_BASE,
                            "static_reservation_bytes": symbols["_stack_bottom"] - GUEST_BASE, "stack_reserved_bytes": STACK_BYTES,
                            "workspace_bytes": graph["workspace_bytes"], "constants_bytes": graph["constants_bytes"], "parameter_copies": 1,
                            "scope": "bounded fixture in requested generic Spike map; no deployed capacity or full-model fit claim"}
        simulator_env = dict(env, LD_LIBRARY_PATH=str(paths["spike_library_dir"]), PATH=str(paths["dtc"].parent) + ":/usr/bin:/bin")
        base = [paths["spike"], "--isa=rv64gc", "-m0x80000000:0x10000000", "-p1"]
        missing = command([*base, programs["valid"]], "without-extension", simulator_env, expected="nonzero")
        if "GRAPH_INPUT_REJECTION_PASS" not in missing:
            raise ValueError("missing-extension control did not reach valid device dispatch")
        extension = [*base, "--extlib=" + str(paths["plugin"]), "--extension=gemmini"]
        valid = command([*extension, programs["valid"]], "valid", simulator_env)
        if "TVM_GEMMINI_GRAPH_PASS calls=6 exact=1 reuse=1 guards=1 retained=1 recovery=1" not in valid:
            raise ValueError("graph numerical/lifetime/reuse gate absent")
        high_water = re.search(r"STACK_HIGH_WATER_BYTES=(\d+)", valid)
        if not high_water or not 0 < int(high_water.group(1)) < STACK_BYTES:
            raise ValueError("stack high-water evidence absent or exceeds reservation")
        report["memory"]["stack_high_water_bytes"] = int(high_water.group(1))
        wrong = command([*extension, programs["bad-oracle"]], "bad-oracle", simulator_env, expected=22)
        if "GRAPH_NUMERICAL_FAILURE" not in wrong:
            raise ValueError("wrong-oracle control did not reach its numerical gate")
        if search is not None:
            platform_parts = [report["plugin"]["sha256"], report["spike"]["sha256"], "rv64gc", GUEST_BASE, GUEST_BYTES]
            platform_id = "functional_spike:" + hashlib.sha256(json.dumps(platform_parts).encode()).hexdigest()
            search.record_device_check(proposal.candidate.candidate_id, passed=True, platform_id=platform_id, evidence=str(out / "receipt.json"))
            report["search"] = search.manifest()
        for path, digest in immutable.items():
            if identity(path)["sha256"] != digest:
                raise ValueError("source/tool changed during verification")
        report["immutable_bindings"] = immutable
        report["artifacts"] = [identity(out / name) for name in ("model.c", "constants.bin", "constants.S", "operators.o", "fixture.h", "runner.c", "valid.elf", "bad-oracle.elf", "schedule.semantic.py", "schedule.tensorized.py", "schedule.trace.json")]
        report["checks"].append({"name": "graph_runtime", "passed": True, "invocations": 6, "independent_i64_oracle": True, "primitive_counts_checked": True, "host_add_relu": not convolution,
                                 "host_residual_requantization": convolution, "raw_convolution_output": convolution, "input_preservation": True, "constant_preservation": True,
                                 "retained_output": True, "elf_stack_bounds": True, "bad_oracle_exit": 22})
        report.update(status="passed", device_execution=True)
    except Exception as error:
        report.update(status="failed", error=str(error))
        (out / "traceback.txt").write_text(traceback.format_exc())
        raise
    finally:
        save()
    print(json.dumps({"status": report["status"], "receipt": str(out / "receipt.json")}))


if __name__ == "__main__":
    main()
