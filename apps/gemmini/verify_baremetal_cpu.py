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
"""Prove a bounded TVM-generated CPU Relax graph on RV64GC baremetal Spike.

This diagnostic emits static orchestration for its own fixed FP32 graph. It is
not a general Relax VM, AOT frontend, microTVM CRT, or accelerator qualification.
All tool paths and the TVM host build are explicit; generated artifacts stay in
a fresh output directory. The guest links no libc, libtvm, or C++ runtime.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import traceback

GUEST_BASE = 0x80000000
GUEST_BYTES = 0x10000000
STACK_BYTES = 16384
TARGET_GATE = "BAREMETAL_TVM_CPU_PASS calls=8 inputs=5 exact=1 retained=1 guards=1 recovery=1"


def checked_path(value):
    """Reject restricted names before reading any path component or symlink target."""
    path = Path(os.path.abspath(value))
    for _ in range(40):
        if any(any(word in part.lower() for word in ("hammer", "vlsi")) for part in path.parts):
            raise ValueError("restricted path component")
        resolved = Path(path.anchor)
        restart = False
        for index, part in enumerate(path.parts[1:], 1):
            resolved /= part
            try:
                mode = resolved.lstat().st_mode
            except FileNotFoundError:
                return path
            if stat.S_ISLNK(mode):
                target = Path(os.readlink(resolved))
                if not target.is_absolute():
                    target = resolved.parent / target
                path = Path(os.path.abspath(target.joinpath(*path.parts[index + 1:])))
                restart = True
                break
        if not restart:
            return path
    raise ValueError("symlink chain exceeds limit")


def identity(path):
    path = checked_path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def _rv64_double_abi(headers):
    """Validate ELF headers, including every object reported for an archive."""
    fields = {name: re.findall(r"^\s*" + name + r":\s*(.+)$", headers, re.MULTILINE)
              for name in ("Class", "Data", "Machine", "Flags")}
    count = len(fields["Class"])
    if not count or any(len(values) != count for values in fields.values()):
        raise ValueError("Runtime requires complete ELF headers")
    if (any(value != "ELF64" for value in fields["Class"])
            or any("little endian" not in value for value in fields["Data"])
            or any(value != "RISC-V" for value in fields["Machine"])):
        raise ValueError("Runtime requires little-endian ELF64 RISC-V objects")
    flags = [int(value.split(",", 1)[0], 16) for value in fields["Flags"]]
    if any(value & 6 != 4 or value & ~5 for value in flags):
        raise ValueError("Runtime requires the RV64 double-float ABI without extra ELF flags")
    return {"elf_objects": count, "class": "ELF64", "endianness": "little", "machine": "RISC-V", "abi": "lp64d"}


def bind_cpu_runtime(libraries, tools, bind, command):
    """Bind explicitly selected ordinary static archives; never discover libraries."""
    selected = [checked_path(path) for path in libraries]
    if len(set(selected)) != len(selected):
        raise ValueError("CPU runtime archives must be distinct")
    bindings = []
    for index, path in enumerate(selected):
        if path.suffix != ".a" or not path.is_file():
            raise ValueError("CPU runtime requires explicit regular .a archive files")
        binding = bind(path)
        with path.open("rb") as stream:
            if stream.read(8) != b"!<arch>\n":
                raise ValueError("CPU runtime requires an ordinary archive; thin archives are not bound")
        binding["elf_abi"] = _rv64_double_abi(command([tools["readelf"], "-h", path], f"cpu-runtime-{index}-headers"))
        bindings.append(binding)
    arguments = ["-Wl,--gc-sections", "-Wl,--start-group", *map(str, selected), "-Wl,--end-group"] if selected else []
    return arguments, {"archives": bindings, "link_arguments": arguments, "requested_isa": "rv64gc", "abi": "lp64d",
                       "scope": "Explicit static CPU runtime only; final linked image is audited for ISA, ABI, initialization and absence of heap/file callbacks"}


def audit_cpu_runtime(elf, map_path, bindings, tools, command):
    """Audit the final runtime closure, not discarded archive code."""
    headers = command([tools["readelf"], "-h", "-A", elf], elf.stem + "-cpu-runtime-abi")
    abi = _rv64_double_abi(headers)
    attributes = re.findall(r'Tag_RISCV_arch:\s*"([^"]+)"', headers)
    if len(attributes) != 1:
        raise ValueError("CPU runtime requires one final RISC-V ISA attribute")
    parts = attributes[0].split("_")
    if not re.fullmatch(r"rv64i\d+p\d+", parts[0]):
        raise ValueError("CPU runtime requires an RV64I base ISA")
    extensions = set()
    for part in parts[1:]:
        match = re.fullmatch(r"([a-z]+)\d+p\d+", part)
        if not match:
            raise ValueError("Unknown CPU runtime ISA attribute")
        extensions.add(match.group(1))
    if not extensions <= {"m", "a", "f", "d", "c", "zicsr", "zifencei", "zmmul"}:
        raise ValueError("CPU runtime ISA exceeds the selected RV64GC machine")
    sections = command([tools["readelf"], "-W", "-S", elf], elf.stem + "-cpu-runtime-sections")
    section_pattern = r"^\s*\[\s*\d+\]\s+(\S+)\s+(\S+)\s+[0-9a-f]+\s+[0-9a-f]+\s+([0-9a-f]+)\s+[0-9a-f]+\s*(.*?)\s+\d+\s+\d+\s+\d+\s*$"
    parsed = [{"name": name, "type": kind, "bytes": int(size, 16), "flags": flags.strip()}
              for name, kind, size, flags in re.findall(section_pattern, sections, re.MULTILINE)]
    if not parsed:
        raise ValueError("CPU runtime section audit is empty")
    for section in parsed:
        initializer = any(section["name"] == prefix or section["name"].startswith(prefix + ".")
                          for prefix in (".init", ".fini", ".ctors", ".dtors"))
        if section["bytes"] and ("T" in section["flags"] or initializer
                or section["type"] in {"INIT_ARRAY", "PREINIT_ARRAY", "FINI_ARRAY"}):
            raise ValueError("CPU runtime needs unsupported TLS or constructor initialization")
    symbols = command([tools["nm"], "--defined-only", elf], elf.stem + "-cpu-runtime-symbols")
    defined = {line.split()[-1] for line in symbols.splitlines() if len(line.split()) >= 3}
    forbidden = {"malloc", "calloc", "realloc", "free", "aligned_alloc", "memalign", "_malloc_r", "_calloc_r", "_realloc_r", "_free_r",
                 "_sbrk", "_sbrk_r", "sbrk", "_open", "_open_r", "_close", "_close_r", "_read", "_read_r", "_write", "_write_r",
                 "_lseek", "_lseek_r", "_fstat", "_fstat_r", "_isatty", "_isatty_r", "open", "close", "read", "write", "lseek", "fstat", "isatty",
                 "TVMBackendAllocWorkspace", "TVMBackendFreeWorkspace"}
    if defined & forbidden:
        raise ValueError("CPU runtime contains unsupported heap/file callbacks: " + ", ".join(sorted(defined & forbidden)))
    text = checked_path(map_path).read_text()
    if "Linker script and memory map" not in text:
        raise ValueError("CPU runtime needs a complete linker map")
    allocated = text.split("Linker script and memory map", 1)[1]
    contribution = []
    for binding in bindings["archives"]:
        pattern = re.escape(binding["path"]) + r"\(([^()\r\n]+)\)"
        extracted = sorted(set(re.findall(pattern, text.split("Discarded input sections", 1)[0])))
        retained = []
        for line in allocated.splitlines():
            member = re.search(pattern, line)
            extent = re.search(r"\b(0x[0-9a-f]+)\s+(0x[0-9a-f]+)\b", line)
            if member and extent:
                address, size = (int(value, 16) for value in extent.groups())
                if size and GUEST_BASE <= address < GUEST_BASE + GUEST_BYTES:
                    retained.append({"member": member.group(1), "address": address, "bytes": size})
        contribution.append({"archive": binding["path"], "extracted_members": extracted, "retained_allocated_extents": retained})
    return {"final_elf_abi": abi, "final_isa_attribute": attributes[0], "final_sections": parsed,
            "archive_contribution": contribution, "no_tls_or_constructors": True, "no_heap_or_file_callbacks": True,
            "checked_forbidden_symbols": sorted(forbidden),
            "initialization": "Single-hart loader initializes .data; startup clears .bss. No TLS, library constructor or heap initialization is supplied."}


def export_static_probe(mod, out):
    func = mod['main']
    if not isinstance(func, relax.Function) or len(func.params) != 1:
        raise ValueError('require one-input Relax main')
    if not isinstance(func.body, relax.SeqExpr):
        raise ValueError('require straight-line bindings')
    def size(expr):
        si = expr.struct_info
        if not isinstance(si, relax.TensorStructInfo) or si.dtype != 'float32':
            raise ValueError('only FP32 tensors supported')
        if not isinstance(si.shape, relax.ShapeExpr) or not all(isinstance(d,tir.IntImm) and int(d)>0 for d in si.shape.values):
            raise ValueError('only positive static tensor shapes supported')
        n = math.prod(int(d) for d in si.shape.values)
        if n > 4096:
            raise ValueError('tensor exceeds bounded probe limit')
        return n
    size(func.params[0])
    names = {func.params[0]: 'input'}
    constants = []
    calls = []
    buffers = []
    used = {}
    def argument(expr):
        if expr in names:
            return names[expr]
        if isinstance(expr, relax.Constant):
            size(expr)
            value = expr.data.numpy()
            name = 'constant_' + str(len(constants))
            if not np.isfinite(value).all():
                raise ValueError('non-finite constants unsupported')
            constants.append((name,value))
            names[expr] = name
            return name
        raise ValueError('unsupported call argument')
    for block in func.body.blocks:
        if not isinstance(block, (relax.BindingBlock, relax.DataflowBlock)):
            raise ValueError('unsupported binding block')
        for binding in block.bindings:
            if not isinstance(binding, relax.VarBinding):
                raise ValueError('unsupported binding kind')
            value = binding.value
            if isinstance(value, relax.Var) and value in names:
                names[binding.var] = names[value]
                continue
            if not isinstance(value, relax.Call) or value.op != tvm.ir.Op.get('relax.call_tir') or len(value.args) != 2:
                raise ValueError('only call_tir and variable aliases supported')
            gv, args = value.args
            if not isinstance(gv,tvm.ir.GlobalVar) or not isinstance(args,relax.Tuple):
                raise ValueError('unsupported call_tir signature')
            prim = mod[gv]
            symbol = gv.name_hint
            if not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*',symbol) or not isinstance(prim,tir.PrimFunc):
                raise ValueError('invalid PrimFunc symbol')
            # Exact buffer-only call ABI, no scalar args/hidden workspace.
            if len(prim.params) != len(args.fields)+1 or len(prim.buffer_map) != len(prim.params):
                raise ValueError('require buffer-only one-output PrimFunc')
            for param, expr in zip(prim.params, [*args.fields,binding.var]):
                buf = prim.buffer_map[param]
                if (buf.dtype != 'float32' or not all(isinstance(d,tir.IntImm) for d in buf.shape)
                        or tuple(int(d) for d in buf.shape) != tuple(int(d) for d in expr.struct_info.shape.values)):
                    raise ValueError('PrimFunc tensor ABI mismatch')
            dest = 'buffer_' + str(len(buffers))
            buffers.append((dest,size(binding.var)))
            names[binding.var] = dest
            calls.append((symbol,[argument(arg) for arg in args.fields],dest))
            used[gv] = prim.with_attr('global_symbol',symbol)
    if func.body.body not in names or not calls:
        raise ValueError('require bound tensor return')
    result = names[func.body.body]
    if len(calls) > 16:
        raise ValueError('graph exceeds bounded call limit')
    if result != calls[-1][2]:
        raise ValueError('require final call result as output')
    lines = ['#include <stdint.h>', '#include <stddef.h>']
    for symbol in sorted({c[0] for c in calls}):
        nargs = next(len(c[1])+1 for c in calls if c[0]==symbol)
        lines.append('extern int32_t '+symbol+'('+','.join(['float*']*nargs)+');')
    for name,array in constants:
        values = ','.join(float(v).hex()+'f' for v in array.flat)
        lines.append('static const float '+name+'[] = {'+values+'};')
    lines.append('struct model_workspace {')
    for name,n in buffers[:-1]:
        lines.append('  float '+name+'['+str(n)+'];')
    lines.append('};')
    lines.append('static int model_run(const float *input, float *output, struct model_workspace *ws) {')
    lines.append('  if (!input || !output || !ws) return -1;')
    remap = {name:'ws->'+name for name,_ in buffers[:-1]}
    remap[result] = 'output'
    for symbol,args,dest in calls:
        args = [remap.get(a,a) for a in [*args,dest]]
        lines.append('  if ('+symbol+'('+','.join('(float*)'+a for a in args)+')) return -2;')
    lines.extend(['  return 0;','}'])
    (out/'model.h').write_text('\n'.join(lines)+'\n')
    primmod=tvm.IRModule(used).with_attr('executor',tvm.relay.backend.Executor('aot',{'unpacked-api':True,'interface-api':'c'}))
    target='llvm -mtriple=riscv64-unknown-elf -mcpu=generic-rv64 -mattr=+m,+a,+f,+d,+c -mabi=lp64d'
    lib=tvm.build(primmod,target=target)
    lib.save(str(out/'operators.o'))
    (out/'operators.ll').write_text(lib.get_source('ll'))
    (out/'operators.c').write_text(tvm.build(primmod,target='c -keys=cpu').get_source())
    return {
        'target': target, 'calls': [{'symbol':s,'arguments':a,'output':d} for s,a,d in calls],
        'constants_bytes': sum(a.nbytes for _,a in constants), 'workspace_bytes': sum(n*4 for _,n in buffers[:-1]),
        'input_bytes': size(func.params[0])*4, 'output_bytes': size(func.body.body)*4,
    }

STARTUP = r"""/* Single-hart RV64GC diagnostic startup, machine mode, Spike HTIF. */
.section .text.init
.globl _start
_start:
  .option push
  .option norelax
  la gp, __global_pointer$
  .option pop
  la sp, _stack_top
  la t0, trap
  csrw mtvec, t0
  li t0, (1 << 13)
  csrs mstatus, t0
  csrw fcsr, zero
  la t0, _bss_start
  la t1, _bss_end
1:
  bgeu t0, t1, 2f
  sd zero, 0(t0)
  addi t0, t0, 8
  j 1b
2:
  la t0, _stack_bottom
  la t1, _stack_top
  li t2, 0x55aa55aa55aa55aa
4:
  bgeu t0, t1, 5f
  sd t2, 0(t0)
  addi t0, t0, 8
  j 4b
5:
  call main
  slli a0, a0, 1
  ori a0, a0, 1
  la t0, tohost
  sd a0, 0(t0)
3:
  j 3b
  .balign 4
trap:
  csrr t1, mcause
  slli t1, t1, 1
  ori t1, t1, 1
  li t2, 0x1000
  or t1, t1, t2
  la t0, tohost
  sd t1, 0(t0)
  j 3b
"""

LINKER = r"""OUTPUT_ARCH(riscv)
ENTRY(_start)
MEMORY { DRAM (rwx) : ORIGIN = 0x80000000, LENGTH = 0x10000000 }
SECTIONS {
  . = ORIGIN(DRAM);
  .text : { *(.text.init) *(.text .text.*) } > DRAM
  .rodata : { *(.rodata .rodata.*) } > DRAM
  .data : { *(.data .data.*) *(.sdata .sdata.*) } > DRAM
  __global_pointer$ = . + 0x800;
  .tohost ALIGN(64) : { *(.tohost) } > DRAM
  .bss ALIGN(16) (NOLOAD) : {
    _bss_start = .;
    *(.bss .bss.* .sbss .sbss.* COMMON)
    . = ALIGN(16);
    _bss_end = .;
  } > DRAM
  .stack ALIGN(16) (NOLOAD) : {
    _stack_bottom = .;
    . += 16384;
    _stack_top = .;
  } > DRAM
  _image_end = .;
  ASSERT(_image_end <= ORIGIN(DRAM) + LENGTH(DRAM), "image exceeds guest DRAM")
}
"""

RUNNER = r"""#include <stdint.h>
#include "model.h"

volatile uint64_t tohost __attribute__((section(".tohost"), aligned(64)));
volatile uint64_t fromhost __attribute__((section(".tohost"), aligned(64)));
static __attribute__((always_inline)) inline void puts_htif(const char *s) {
  /* FESVR SYS_write request: device 0/command 0 is the request pointer.
   * The ABI is implemented directly; there is no external guest runtime. */
  static volatile uint64_t request[4] __attribute__((aligned(8)));
  uint64_t length=0;
  while(s[length]) ++length;
  request[0]=64; request[1]=1; request[2]=(uintptr_t)s; request[3]=length;
  __asm__ volatile("fence rw,rw" ::: "memory");
  tohost=(uintptr_t)request;
  while(!fromhost) {}
  fromhost=0;
  __asm__ volatile("fence rw,rw" ::: "memory");
}
static uint32_t bits(float f) { union { float f; uint32_t u; } x = {f}; return x.u; }
struct guarded_output { uint32_t head; float value[8]; uint32_t tail; };
struct guarded_workspace { uint32_t head; struct model_workspace value; uint32_t tail; };
_Static_assert(offsetof(struct guarded_output, value)==4 && sizeof(struct guarded_output)==40, "output layout");
_Static_assert(offsetof(struct guarded_workspace, value)==4 && sizeof(struct guarded_workspace)==72, "workspace layout");
static struct guarded_output output[2];
static struct guarded_workspace workspace;

int main(void) {
  const int steps[8] = {0,1,2,0,3,1,4,0};
  float input[6], saved_input[6], retained[8];
  workspace.head = workspace.tail = 0xa5b6c7d8;
  for (int b=0;b<2;++b) output[b].head=output[b].tail=0x12345678;
  const float *volatile rejected_input=0;
  if (model_run(rejected_input,output[0].value,&workspace.value) != -1) return 10;
  for (int call=0;call<8;++call) {
    for(int j=0;j<6;++j) input[j]=saved_input[j]=(float)(j-3)*.5f+(float)steps[call]*.25f;
    int current=call%2;
    if(model_run(input,output[current].value,&workspace.value)) return 11;
    /* Independent scalar oracle, not the operator under test. */
    for(int row=0;row<2;++row) for(int col=0;col<4;++col) {
      float wanted=0;
      for(int k=0;k<3;++k) wanted+=saved_input[row*3+k]*((float)(k*4+col-5)*.25f);
      wanted+=(float)(col-1)*.5f;
      if(wanted<0) wanted=0;
      if(bits(wanted)!=bits(output[current].value[row*4+col])) return 12;
    }
    for(int j=0;j<6;++j) if(bits(input[j])!=bits(saved_input[j])) return 13;
    if(call) for(int j=0;j<8;++j) if(bits(output[1-current].value[j])!=bits(retained[j])) return 14;
    for(int j=0;j<8;++j) retained[j]=output[current].value[j];
    if(workspace.head!=0xa5b6c7d8 || workspace.tail!=0xa5b6c7d8) return 15;
    for(int b=0;b<2;++b) if(output[b].head!=0x12345678 || output[b].tail!=0x12345678) return 16;
    for(int j=0;j<12;++j) if(bits(((volatile const float*)constant_0)[j])!=bits((float)(j-5)*.25f)) return 17;
    for(int j=0;j<4;++j) if(bits(((volatile const float*)constant_1)[j])!=bits((float)(j-1)*.5f)) return 18;
  }
  puts_htif("BAREMETAL_TVM_CPU_PASS calls=8 inputs=5 exact=1 retained=1 guards=1 recovery=1\n");
  extern uint64_t _stack_bottom[], _stack_top[];
  uint64_t *first=_stack_bottom;
  while(first<_stack_top && *first==UINT64_C(0x55aa55aa55aa55aa)) ++first;
  uint64_t used=(uintptr_t)_stack_top-(uintptr_t)first;
  char digits[24]; int count=0;
  do { digits[count++]=(char)('0'+used%10); used/=10; } while(used);
  puts_htif("STACK_HIGH_WATER_BYTES=");
  for(int j=0;j<count/2;++j) { char c=digits[j]; digits[j]=digits[count-1-j]; digits[count-1-j]=c; }
  digits[count++]='\n'; digits[count]=0;
  puts_htif(digits);
  return 0;
}
"""

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('tvm-source', 'tvm-build', 'riscv-gcc', 'spike', 'dtc', 'spike-library-dir', 'output-dir'):
        parser.add_argument('--' + flag, required=True, type=Path)
    parser.add_argument('--timeout-seconds', type=int, default=60)
    args = parser.parse_args()
    if not 1 <= args.timeout_seconds <= 300:
        parser.error('--timeout-seconds must be in [1,300]')
    try:
        paths = {name: checked_path(value) for name, value in vars(args).items() if isinstance(value, Path)}
        for name, path in paths.items():
            if name != 'output_dir' and not path.exists():
                raise ValueError(name + ' does not exist')
        out = paths['output_dir']
        out.mkdir(parents=True, exist_ok=False)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    report = {
        'status': 'running', 'scope': 'bounded TVM-generated CPU graph, static Relax orchestration, generic Spike only',
        'invocation': [sys.executable, *sys.argv], 'commands': [], 'checks': [],
        'baremetal_cpu_validated': False, 'gemmini_validated': False, 'timing_validated': False,
    }

    def save():
        (out / 'receipt.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')

    def command(argv, label, env=None, expected_code=0):
        argv = [str(arg) for arg in argv]
        record = {'argv': argv, 'log': label + '.log', 'expected_exit_code': expected_code}
        report['commands'].append(record)
        try:
            process = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired as error:
            output = error.stdout or b''
            if isinstance(output, bytes):
                output = output.decode(errors='replace')
            (out / record['log']).write_text(output)
            record.update(exit_code=None, timeout_seconds=args.timeout_seconds)
            save()
            raise RuntimeError(label + ' timed out') from error
        (out / record['log']).write_text(process.stdout)
        record['exit_code'] = process.returncode
        save()
        if process.returncode != expected_code:
            raise RuntimeError(label + ' failed; see ' + str(out / record['log']))
        return process.stdout

    try:
        source, build, gcc, spike = (paths[name] for name in ('tvm_source', 'tvm_build', 'riscv_gcc', 'spike'))
        if not gcc.name.endswith('gcc'):
            raise ValueError('RISC-V compiler filename must end with gcc to identify sibling tools')
        tools = {name: checked_path(gcc.with_name(gcc.name[:-3] + name)) for name in ('nm', 'readelf', 'objdump')}
        if not all(tool.is_file() for tool in tools.values()):
            raise ValueError('nm/readelf/objdump must be present beside the supplied GCC')
        # Bind the host compiler before importing TVM. No global Chipyard setup is sourced.
        os.environ.update(TVM_LIBRARY_PATH=str(build), TVM_FFI='ctypes', PYTHONDONTWRITEBYTECODE='1')
        sys.path.insert(0, str(checked_path(source / 'python')))
        global np, tvm, relax, tir
        import numpy as np
        import tvm
        from tvm import relax, tir
        from tvm._ffi.base import _LIB
        if not checked_path(tvm.__file__).is_relative_to(source / 'python') or checked_path(_LIB._name).parent != build:
            raise RuntimeError('TVM Python/library paths do not match the supplied source/build')
        report['source_head'] = command(['git', '-C', source, 'rev-parse', 'HEAD'], 'tvm-head').strip()
        report['build_info'] = dict(tvm.support.libinfo())
        report['python_version'], report['numpy_version'] = sys.version, np.__version__
        selected_sources = ['python/tvm/driver/build_module.py', 'src/driver/driver_api.cc', 'src/tir/transforms/make_unpacked_api.cc']
        report['selected_source_identities'] = [identity(source / name) for name in selected_sources]
        report['verifier_source'] = identity(__file__)
        report['tool_identities'] = [identity(gcc), identity(spike), identity(paths['dtc']), *[identity(tool) for tool in tools.values()]]
        report['compiler_library'] = identity(_LIB._name)
        cpp_library = checked_path(paths['spike_library_dir'] / 'libstdc++.so.6')
        report['spike_cpp_library'] = identity(cpp_library)
        command([gcc, '--version'], 'gcc-version')
        spike_env = dict(os.environ, LD_LIBRARY_PATH=str(paths['spike_library_dir']), PATH=str(paths['dtc'].parent) + os.pathsep + os.defpath)
        report['spike_environment'] = {key: spike_env[key] for key in ('LD_LIBRARY_PATH', 'PATH')}
        command([spike, '--help'], 'spike-version', spike_env)
        overrides = ('CPATH', 'C_INCLUDE_PATH', 'CPLUS_INCLUDE_PATH', 'OBJC_INCLUDE_PATH', 'LIBRARY_PATH', 'COMPILER_PATH', 'GCC_EXEC_PREFIX')
        compiler_env = {key: value for key, value in os.environ.items() if key not in overrides}
        compiler_env['PATH'] = str(gcc.parent) + os.pathsep + os.defpath
        report['compiler_environment'] = {'cleared_overrides': list(overrides), 'PATH': compiler_env['PATH']}

        def graph(shape=(2, 3), dtype='float32', tuple_output=False):
            x = relax.Var('input', relax.TensorStructInfo(shape, dtype))
            weights = (np.arange(12, dtype=dtype).reshape(3, 4) - 5) * .25
            bias = (np.arange(4, dtype=dtype) - 1) * .5
            builder = relax.BlockBuilder()
            with builder.function('main', [x]):
                with builder.dataflow():
                    value = builder.emit(relax.op.matmul(x, relax.const(weights, dtype=dtype)))
                    value = builder.emit(relax.op.add(value, relax.const(bias, dtype=dtype)))
                    value = builder.emit_output(relax.op.nn.relu(value))
                builder.emit_func_output(relax.Tuple([value]) if tuple_output else value)
            return builder.get()

        mod = graph()
        (out / 'input.relax.py').write_text(mod.script(show_meta=True))
        lowered = relax.transform.LegalizeOps()(mod)
        (out / 'legalized.relax.py').write_text(lowered.script(show_meta=True))
        report['graph'] = export_static_probe(lowered, out)
        llvm_ir = (out / 'operators.ll').read_text()
        float_alignments = [int(value) for value in re.findall(r'(?:load|store) float[^\n]*align (\d+)', llvm_ir)]
        if not float_alignments or max(float_alignments) > 4 or re.search(r'ptr[^,\n)]*\balign\s+\d+', llvm_ir):
            raise AssertionError('unexpected raw-pointer or FP32 access alignment assumption')
        report['graph']['raw_pointer_min_alignment_bytes'] = 4
        report['graph']['llvm_fp32_load_store_alignments'] = sorted(set(float_alignments))
        rejection_cases = {
            'unlegalized_graph': mod,
            'dynamic_shape': relax.transform.LegalizeOps()(graph((tir.Var('batch', 'int64'), 3))),
            'unsupported_dtype': relax.transform.LegalizeOps()(graph(dtype='float64')),
            'tuple_output': relax.transform.LegalizeOps()(graph(tuple_output=True)),
        }
        for label, rejected in rejection_cases.items():
            try:
                export_static_probe(rejected, out)
            except ValueError as error:
                report['checks'].append({'name': label + '_rejected', 'passed': True, 'reason': str(error)})
            else:
                raise AssertionError(label + ' was unexpectedly accepted')
        inputs = np.stack([(np.arange(6, dtype='float32').reshape(2, 3) - 3) * .5 + step * .25 for step in (0, 1, 2, 0, 3, 1, 4, 0)])
        weights = (np.arange(12, dtype='float32').reshape(3, 4) - 5) * .25
        bias = (np.arange(4, dtype='float32') - 1) * .5
        np.savez(out / 'fixture.npz', inputs=inputs, weights=weights, bias=bias, expected=np.maximum(inputs @ weights + bias, 0))
        for name, text in (('start.S', STARTUP), ('link.ld', LINKER), ('runner.c', RUNNER)):
            (out / name).write_text(text)
        flags = [
            '-march=rv64gc', '-mabi=lp64d', '-mcmodel=medany', '-msmall-data-limit=0', '-O2', '-ffreestanding',
            '-fno-builtin', '-fno-stack-protector', '-ffp-contract=off', '-nostdlib', '-nostartfiles', '-static', '-Wl,--no-relax',
            '-Wl,-T,' + str(out / 'link.ld'), '-Wl,-Map,' + str(out / 'model.map'),
        ]
        command([gcc, *flags, out / 'start.S', out / 'runner.c', out / 'operators.o', '-o', out / 'model.elf'], 'link', compiler_env)
        undefined = command([tools['nm'], '-u', out / 'model.elf'], 'undefined')
        if undefined.strip():
            raise AssertionError('ELF retains unresolved runtime symbols')
        program_headers = command([tools['readelf'], '-W', '-l', out / 'model.elf'], 'program-headers')
        command([tools['readelf'], '-W', '-a', out / 'model.elf'], 'elf')
        disassembly = command([tools['objdump'], '-d', out / 'model.elf'], 'disassembly')
        instructions = re.findall(r'^\s*[0-9a-f]+:\s+([0-9a-f]{4}|[0-9a-f]{8})\s', disassembly, re.MULTILINE)
        if not instructions or any(len(word) == 8 and int(word, 16) & 0x7f in (0x0b, 0x2b, 0x5b, 0x7b) for word in instructions):
            raise AssertionError('CPU ELF contains a custom instruction opcode or cannot be audited')
        report['checks'].append({'name': 'cpu_elf_no_custom_opcodes', 'passed': True, 'instructions_checked': len(instructions)})
        symbols_text = command([tools['nm'], '-S', '--defined-only', out / 'model.elf'], 'symbols')
        symbols = {}
        for line in symbols_text.splitlines():
            fields = line.split()
            if len(fields) in (3, 4) and re.fullmatch(r'[0-9a-fA-F]+', fields[0]):
                symbols[fields[-1]] = {'address': int(fields[0], 16), 'bytes': int(fields[1], 16) if len(fields) == 4 else 0}
        segments = []
        for line in program_headers.splitlines():
            fields = line.split()
            if fields and fields[0] == 'LOAD':
                segments.append({'start': int(fields[2], 16), 'file_bytes': int(fields[4], 16), 'memory_bytes': int(fields[5], 16)})
        image_end = symbols['_image_end']['address']
        buffer_bases = [symbols[name]['address'] for name in ('constant_0', 'constant_1')]
        buffer_bases.extend((symbols['workspace']['address'] + 4, symbols['workspace']['address'] + 36))
        buffer_bases.extend((symbols['output']['address'] + 4, symbols['output']['address'] + 44))
        if any(address % 4 for address in buffer_bases):
            raise AssertionError('generated FP32 buffer address is not aligned to four bytes')
        report['checks'].append({'name': 'raw_pointer_alignment', 'passed': True, 'static_buffer_addresses': buffer_bases, 'input_alignment': 'C float array on aligned stack'})
        if not segments or any(s['start'] < GUEST_BASE or s['start'] + s['memory_bytes'] > GUEST_BASE + GUEST_BYTES for s in segments):
            raise AssertionError('ELF load extent exceeds requested guest memory')
        if symbols['_stack_top']['address'] - symbols['_stack_bottom']['address'] != STACK_BYTES or image_end > GUEST_BASE + GUEST_BYTES:
            raise AssertionError('stack/image reservation mismatch')
        run = command([spike, '--isa=rv64gc', '-m0x80000000:0x10000000', '-p1', out / 'model.elf'], 'spike', spike_env)
        high_water = re.search(r'STACK_HIGH_WATER_BYTES=(\d+)', run)
        if TARGET_GATE not in run or not high_water or not 0 < int(high_water.group(1)) <= STACK_BYTES:
            raise AssertionError('target correctness/stack gate absent')
        # This negative target run establishes that numerical failure propagates through HTIF.
        (out / 'bad-runner.c').write_text(RUNNER.replace('if(wanted<0) wanted=0;', 'if(wanted<0) wanted=0; wanted+=1.0f;'))
        bad_flags = [flag.replace('model.map', 'bad-model.map') for flag in flags]
        command([gcc, *bad_flags, out / 'start.S', out / 'bad-runner.c', out / 'operators.o', '-o', out / 'bad-model.elf'], 'link-bad-oracle', compiler_env)
        command([spike, '--isa=rv64gc', '-m0x80000000:0x10000000', '-p1', out / 'bad-model.elf'], 'spike-bad-oracle', spike_env, expected_code=12)
        report['checks'].append({
            'name': 'baremetal_cpu', 'passed': True, 'calls': 8, 'distinct_inputs': 5, 'exact_scalar_oracle': True,
            'input_unchanged': True, 'prior_output_retained': True, 'workspace_output_canaries': True, 'constants_unchanged': True,
            'null_rejection_and_recovery': True, 'deliberately_wrong_oracle_exit_code': 12,
        })
        report['memory'] = {
            'requested_guest_base': GUEST_BASE, 'requested_guest_bytes': GUEST_BYTES, 'load_segments': segments,
            'image_end': image_end, 'reserved_image_bytes': image_end - GUEST_BASE,
            'static_end': symbols['_stack_bottom']['address'], 'static_reservation_bytes': symbols['_stack_bottom']['address'] - GUEST_BASE,
            'stack_start': symbols['_stack_bottom']['address'], 'stack_end': symbols['_stack_top']['address'],
            'stack_reserved_bytes': STACK_BYTES, 'stack_high_water_bytes': int(high_water.group(1)),
            'heap_reserved_bytes': 0, 'dynamic_allocation': False, 'parameter_copies': 1,
            'parameter_storage_bytes': report['graph']['constants_bytes'], 'workspace_bytes': report['graph']['workspace_bytes'],
            'symbols': {name: symbols[name] for name in ('constant_0', 'constant_1', 'workspace', 'output', 'tohost', 'fromhost')},
            'parameter_lifetime': 'single ELF .rodata copy retained across all calls',
            'workspace_lifetime': 'caller-owned static scratch reused each call; no allocation or reset',
            'output_lifetime': 'two caller-owned buffers; previous output survives next call',
            'capacity_scope': 'requested generic Spike map, no deployed capacity or full-model fit claim',
        }
        names = ['start.S', 'link.ld', 'runner.c', 'model.h', 'operators.o', 'operators.ll', 'operators.c', 'model.elf',
                 'input.relax.py', 'legalized.relax.py', 'fixture.npz', 'bad-runner.c', 'bad-model.elf']
        report['artifacts'] = [identity(out / name) for name in names]
        report['limitations'] = [
            'CPU-only generic Spike without Gemmini/plugin/device/timing qualification',
            'No Relax VM or CRT; restricted static orchestration reads legalized graph bindings/constants',
            'One input, fixed positive FP32 shapes, straight-line call_tir buffer-only one-output operations and variable aliases',
            'Only the synthetic matmul/add/ReLU graph is numerically qualified; no complete-model or arbitrary-operator support claim',
            'HTIF console/exit and machine-mode startup are functional diagnostic interfaces, not a deployed platform runtime',
        ]
        report.update(status='passed', baremetal_cpu_validated=True)
    except Exception as error:
        report.update(status='failed', error=str(error))
        (out / 'traceback.txt').write_text(traceback.format_exc())
        raise
    finally:
        save()
    print(json.dumps({'status': report['status'], 'checks': report['checks'], 'memory': report['memory']}))


if __name__ == '__main__':
    main()
