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
"""Verify the pinned primitive Gemmini C-library adapter on baremetal RV64 Spike.

This bounded functional diagnostic reuses the independent CPU proof's minimal
startup/linker and the adapter verifier's full executable-section no-FSM audit.
It requires explicit cross-compiler, Spike, plugin, support and build receipts.
No source checkout, rebuild, TVM graph integration, FPGA or timing claim occurs.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback

from verify_baremetal_cpu import STARTUP, LINKER, checked_path, identity
from verify_matmul import HARDWARE_REVISION, PARAMS_SHA256, OPERATOR_SHA256, audit_instructions

RUNNER = r"""/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include "matmul.h"

#ifndef INVALID_ONLY
#define INVALID_ONLY 0
#endif
#ifndef BAD_ORACLE
#define BAD_ORACLE 0
#endif
#ifndef INCLUDE_MAX_K
#define INCLUDE_MAX_K 1
#endif
#define INPUT_CAPACITY 131072
#define OUTPUT_CAPACITY 512
#define GUARD 0x6d
#define SENTINEL INT32_C(0x5a6b7c8d)

volatile uint64_t tohost __attribute__((section(".tohost"), aligned(64)));
volatile uint64_t fromhost __attribute__((section(".tohost"), aligned(64)));
static void puts_htif(const char *text) {
  static volatile uint64_t request[4] __attribute__((aligned(8)));
  uint64_t count=0;
  while(text[count]) ++count;
  request[0]=64; request[1]=1; request[2]=(uintptr_t)text; request[3]=count;
  asm volatile("fence rw,rw" ::: "memory");
  tohost=(uintptr_t)request;
  while(!fromhost) {}
  fromhost=0;
  asm volatile("fence rw,rw" ::: "memory");
}
static void number(int64_t value) {
  char digits[32]; unsigned count=0;
  uint64_t magnitude=value<0 ? -(uint64_t)value : (uint64_t)value;
  do { digits[count++]=(char)('0'+magnitude%10); magnitude/=10; } while(magnitude);
  if(value<0) digits[count++]='-';
  for(unsigned j=0;j<count/2;++j) { char tmp=digits[j]; digits[j]=digits[count-1-j]; digits[count-1-j]=tmp; }
  digits[count]=0; puts_htif(digits);
}
int printf(const char *format, ...) { puts_htif("VENDOR_DIAGNOSTIC "); puts_htif(format); return 0; }
__attribute__((noreturn)) void exit(int code) {
  puts_htif("VENDOR_EXIT\n");
  tohost=((uint64_t)(uint32_t)(code ? code : 99)<<1)|1;
  for(;;) {}
}
struct input_storage { uint8_t head[64]; int8_t data[INPUT_CAPACITY]; uint8_t tail[64]; };
struct output_storage { uint8_t head[64]; int32_t data[OUTPUT_CAPACITY]; uint8_t tail[64]; };
static struct input_storage a __attribute__((aligned(64))), b __attribute__((aligned(64)));
static struct output_storage c[2] __attribute__((aligned(64)));
static const int8_t values[]={-128,127,-17,-1,0,1,29};
static uint64_t hash(const void *pointer, size_t bytes) {
  const volatile uint8_t *data=pointer;
  uint64_t result=UINT64_C(1469598103934665603);
  for(size_t j=0;j<bytes;++j) result=(result^data[j])*UINT64_C(1099511628211);
  return result;
}
static int guards(void) {
  for(unsigned j=0;j<64;++j) {
    if(((volatile uint8_t*)a.head)[j]!=GUARD || ((volatile uint8_t*)a.tail)[j]!=GUARD ||
       ((volatile uint8_t*)b.head)[j]!=GUARD || ((volatile uint8_t*)b.tail)[j]!=GUARD) return 0;
    for(unsigned slot=0;slot<2;++slot) if(((volatile uint8_t*)c[slot].head)[j]!=GUARD || ((volatile uint8_t*)c[slot].tail)[j]!=GUARD) return 0;
  }
  return 1;
}
static void initialize(void) {
  for(unsigned j=0;j<64;++j) {
    a.head[j]=a.tail[j]=b.head[j]=b.tail[j]=GUARD;
    for(unsigned slot=0;slot<2;++slot) c[slot].head[j]=c[slot].tail[j]=GUARD;
  }
  for(unsigned j=0;j<INPUT_CAPACITY;++j) a.data[j]=b.data[j]=7;
  for(unsigned slot=0;slot<2;++slot) for(unsigned j=0;j<OUTPUT_CAPACITY;++j) c[slot].data[j]=SENTINEL;
}
static int invalid_calls(void) {
  uint64_t before_a=hash(&a,sizeof(a)), before_b=hash(&b,sizeof(b)), before_c=hash(c,sizeof(c));
#define EXPECT(status, aa, bb, cc, mm, nn, kk, as, bs, cs) \
  do { if(tvm_gemmini_matmul_i8_i32(aa,bb,cc,mm,nn,kk,as,bs,cs)!=(status)) return __LINE__; } while(0)
  EXPECT(-1,a.data,b.data,c[0].data,0,1,1,1,1,1);
  EXPECT(-1,a.data,b.data,c[0].data,1,0,1,1,1,1);
  EXPECT(-1,a.data,b.data,c[0].data,1,1,0,1,1,1);
  EXPECT(-1,a.data,b.data,c[0].data,1,1,131072,131072,1,1);
  EXPECT(-1,a.data,b.data,c[0].data,1,1,1,0,1,1);
  EXPECT(-1,a.data,b.data,(int32_t*)((uint8_t*)c[0].data+1),1,1,1,1,1,1);
  EXPECT(-2,0,b.data,c[0].data,1,1,1,1,1,1);
  EXPECT(-2,a.data,0,c[0].data,1,1,1,1,1,1);
  EXPECT(-2,a.data,b.data,0,1,1,1,1,1,1);
  EXPECT(-2,a.data,b.data,c[0].data,1,1,2,1,1,1);
  EXPECT(-2,a.data,b.data,c[0].data,1,2,1,1,1,2);
  EXPECT(-2,a.data,b.data,c[0].data,1,2,1,1,2,1);
  EXPECT(-2,a.data,b.data,c[0].data,1,1,1,UINT64_C(0x100000000),1,1);
  EXPECT(-2,a.data,b.data,c[0].data,1,1,1,1,1,UINT64_C(0x40000000));
  EXPECT(-2,(const int8_t*)(UINTPTR_MAX-3),b.data,c[0].data,1,1,8,8,1,1);
  EXPECT(-2,a.data,b.data,c[0].data,INT64_MAX,1,1,UINT32_MAX,1,1);
  EXPECT(-3,(const int8_t*)c[0].data,b.data,c[0].data,1,1,1,1,1,1);
  EXPECT(-3,a.data,(const int8_t*)c[0].data,c[0].data,1,1,1,1,1,1);
#undef EXPECT
  if(before_a!=hash(&a,sizeof(a)) || before_b!=hash(&b,sizeof(b)) || before_c!=hash(c,sizeof(c)) || !guards()) return 100;
  puts_htif("INVALID_ARGUMENTS_PASS cases=18 buffers_unchanged=1\n");
  return 0;
}
struct shape { unsigned m,n,k,as,bs,cs; };
static const struct shape shapes[]={
  {1,1,1,3,3,3}, {3,7,5,8,11,12}, {17,19,33,37,23,24}, {19,5,17,21,8,9}, {2,18,16,20,22,23},
#if INCLUDE_MAX_K
  {1,1,131071,131071,1,1},
#endif
  {1,1,1,3,3,3}
};
static int one_call(const struct shape *s, unsigned seed, unsigned call, uint64_t *retained) {
  const unsigned a_bytes=s->m*s->as, b_bytes=s->k*s->bs;
  if(a_bytes>INPUT_CAPACITY || b_bytes>INPUT_CAPACITY || s->m*s->cs>OUTPUT_CAPACITY) return 20;
  for(unsigned j=0;j<a_bytes+16 && j<INPUT_CAPACITY;++j) a.data[j]=7;
  for(unsigned j=0;j<b_bytes+16 && j<INPUT_CAPACITY;++j) b.data[j]=7;
  for(unsigned row=0;row<s->m;++row) for(unsigned k=0;k<s->k;++k) a.data[row*s->as+k]=s->k==131071 ? -128 : values[(row*3+k+seed)%7];
  for(unsigned k=0;k<s->k;++k) for(unsigned col=0;col<s->n;++col) b.data[k*s->bs+col]=s->k==131071 ? -128 : values[(k*5+col+seed*2)%7];
  unsigned slot=call%2;
  for(unsigned j=0;j<OUTPUT_CAPACITY;++j) c[slot].data[j]=SENTINEL;
  uint64_t a_before=hash(a.data,a_bytes+16<INPUT_CAPACITY ? a_bytes+16 : INPUT_CAPACITY);
  uint64_t b_before=hash(b.data,b_bytes+16<INPUT_CAPACITY ? b_bytes+16 : INPUT_CAPACITY);
  if(tvm_gemmini_matmul_i8_i32(a.data,b.data,c[slot].data,s->m,s->n,s->k,s->as,s->bs,s->cs)) return 21;
  for(unsigned row=0;row<s->m;++row) for(unsigned col=0;col<s->n;++col) {
    int64_t expected=0;
    for(unsigned k=0;k<s->k;++k) expected+=(int64_t)a.data[row*s->as+k]*(int64_t)b.data[k*s->bs+col];
    expected+=BAD_ORACLE;
    if(expected<INT32_MIN || expected>INT32_MAX || c[slot].data[row*s->cs+col]!=(int32_t)expected) {
      puts_htif("NUMERICAL_FAILURE call="); number(call); puts_htif(" row="); number(row); puts_htif(" col="); number(col);
      puts_htif(" actual="); number(c[slot].data[row*s->cs+col]); puts_htif(" expected="); number(expected); puts_htif("\n"); return 22;
    }
  }
  for(unsigned j=0;j<OUTPUT_CAPACITY;++j) if((j/s->cs>=s->m || j%s->cs>=s->n) && c[slot].data[j]!=SENTINEL) return 23;
  if(a_before!=hash(a.data,a_bytes+16<INPUT_CAPACITY ? a_bytes+16 : INPUT_CAPACITY)) return 24;
  if(b_before!=hash(b.data,b_bytes+16<INPUT_CAPACITY ? b_bytes+16 : INPUT_CAPACITY)) return 25;
  if(call && retained[0]!=hash(c[1-slot].data,sizeof(c[1-slot].data))) return 26;
  if(!guards()) return 27;
  retained[0]=hash(c[slot].data,sizeof(c[slot].data));
  puts_htif("MATMUL_CALL_PASS call="); number(call); puts_htif(" shape="); number(s->m); puts_htif("x"); number(s->n); puts_htif("x"); number(s->k); puts_htif("\n");
  return 0;
}
int main(void) {
  initialize();
  int status=invalid_calls(); if(status) return status;
#if INVALID_ONLY
  return 0;
#else
  unsigned call=0; uint64_t retained=0;
  for(unsigned index=0;index<sizeof(shapes)/sizeof(shapes[0]);++index) {
    unsigned repeats=shapes[index].k==131071 ? 1 : 2;
    for(unsigned seed=0;seed<repeats;++seed) {
      status=one_call(&shapes[index],seed,call,&retained); if(status) return status; ++call;
      if(call==1) { status=invalid_calls(); if(status) return status; puts_htif("VALID_REJECT_VALID_RECOVERY_ARMED\n"); }
    }
  }
  puts_htif("GEMMINI_MATMUL_RUNTIME_PASS calls="); number(call);
  puts_htif(" max_k="); number(INCLUDE_MAX_K ? 131071 : 33); puts_htif(" exact_i64=1 guards=1 inputs=1 retained=1 overwrite=1\n");
  return 0;
#endif
}
"""


def bind_runtime(paths, app, report, bind):
    """Validate the shared pinned adapter/simulator/tool inputs before execution."""
    adapter_path, simulator_path = paths['adapter_receipt'], paths['simulator_receipt']
    report['adapter_receipt'], report['simulator_receipt'] = bind(adapter_path), bind(simulator_path)
    adapter, simulator = json.loads(adapter_path.read_text()), json.loads(simulator_path.read_text())
    if adapter['status'] != 'cross_compiled' or simulator['status'] != 'built_and_load_verified':
        raise ValueError('Require passing adapter compilation and explicit simulator load receipts')
    if adapter['source_set']['gemmini'] != HARDWARE_REVISION:
        raise ValueError('Adapter hardware source differs from the inspected provisional source set')
    if adapter['headers']['include/gemmini_params.h']['sha256'] != PARAMS_SHA256:
        raise ValueError('Adapter parameter header differs from the canonical integer header')
    if adapter['headers']['include/gemmini.h']['sha256'] != OPERATOR_SHA256:
        raise ValueError('Adapter operator header differs from the pinned C-library ABI')
    if simulator['libgemmini_revision'] != adapter['source_set']['libgemmini'] or simulator['sources']['gemmini_params.h'] != PARAMS_SHA256:
        raise ValueError('Simulator revision/parameter header does not bind to the adapter source set')
    if checked_path(simulator['plugin_path']) != paths['plugin'] or checked_path(simulator['spike_path']) != paths['spike']:
        raise ValueError('Explicit plugin/Spike paths differ from simulator build receipt')
    if simulator['load_smoke']['returncode'] != 0:
        raise ValueError('Simulator receipt does not establish successful explicit plugin loading')
    if set(simulator['sources']) != {'gemmini.cc', 'gemmini.h', 'gemmini_params.h', 'Makefile'}:
        raise ValueError('Simulator receipt has an unexpected staged source set')
    object_path = checked_path(adapter_path.parent / 'matmul.o')
    report['adapter_object'] = bind(object_path, adapter['object_sha256'])
    for name, digest in adapter['adapter_sources'].items():
        bind(app / name, digest)
    for name, binding in adapter['headers'].items():
        bind(adapter_path.parent / 'headers' / name, binding['sha256'])
    for name, digest in simulator['sources'].items():
        bind(simulator_path.parent / 'source' / name, digest)
    for path, digest in simulator['resolved_compiler_dependencies'].items():
        bind(path, digest)
    bind(simulator['compiler'], simulator['compiler_sha256'])
    bind(checked_path(simulator['builder_path']), simulator['builder_sha256'])
    report['plugin'] = bind(paths['plugin'], simulator['plugin_sha256'])
    report['spike'] = bind(paths['spike'], simulator['spike_sha256'])
    gcc = paths['riscv_gcc']
    if not gcc.name.endswith('gcc'):
        raise ValueError('Compiler filename must end in gcc to identify sibling tools')
    report['compiler'] = bind(gcc, adapter['compiler']['sha256'])
    tools = {name: checked_path(gcc.with_name(gcc.name[:-3] + name)) for name in ('nm', 'objdump', 'readelf')}
    report['tool_identities'] = [bind(paths['dtc']), *[bind(tool) for tool in tools.values()]]
    if checked_path(simulator['dtc_path']) != paths['dtc']:
        raise ValueError('Selected dtc differs from simulator load receipt')
    bind(paths['dtc'], simulator['dtc_sha256'])
    bind(paths['spike_library_dir'] / 'libstdc++.so.6', simulator['runtime_libstdcpp_sha256'])
    report['runtime_helpers'] = [bind(app / 'verify_baremetal_cpu.py'), bind(app / 'verify_matmul.py')]
    report['verifier_source'] = bind(__file__)
    report['adapter_sources'], report['headers'], report['source_set'] = adapter['adapter_sources'], adapter['headers'], adapter['source_set']
    report['simulator_qualification'] = simulator.get('qualification', {})
    report['simulator_dependency_receipt'] = {'sources': simulator['sources'], 'resolved_compiler_dependencies': simulator['resolved_compiler_dependencies']}
    report['simulator_dependency_scope'] = simulator.get('dependency_scope', 'compiler -MMD records non-system dependencies, not every system header')
    definitions = adapter['instruction_policy']['derived_definitions']
    forbidden = {value for name, value in definitions.items() if name.startswith('k_LOOP_')}
    opcode = adapter['instruction_policy']['custom_opcode']
    if adapter['instruction_policy']['hardware_loops'] != 'forbidden' or not forbidden:
        raise ValueError('Adapter lacks source-derived no-FSM policy')
    object_audit = audit_instructions(object_path.read_bytes(), opcode, forbidden)
    if object_audit['prohibited_hits']:
        raise ValueError('Selected adapter object contains prohibited FSM instructions')
    report['instruction_policy'] = adapter['instruction_policy']
    return object_path, gcc, tools, opcode, forbidden


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('adapter-receipt', 'simulator-receipt', 'riscv-gcc', 'spike', 'dtc', 'spike-library-dir', 'plugin', 'output-dir'):
        parser.add_argument('--' + name, required=True, type=Path)
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
    app = checked_path(Path(__file__).parent)
    report = {
        'status': 'running', 'scope': 'bounded primitive C-library matmul functional simulation only',
        'invocation': [sys.executable, *sys.argv], 'commands': [], 'checks': [],
        'device_execution': False, 'rtl_bit_exact': False, 'timing_verified': False, 'deployed_hardware_verified': False,
    }

    def save():
        (out / 'receipt.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')

    def command(argv, label, env=None, expected=0):
        record = {'argv': [str(value) for value in argv], 'expected_exit_code': expected, 'log': label + '.log'}
        report['commands'].append(record)
        try:
            process = subprocess.run(record['argv'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired as error:
            output = error.stdout or b''
            if isinstance(output, bytes):
                output = output.decode(errors='replace')
            (out / record['log']).write_text(output)
            record.update(exit_code=None, timeout_seconds=args.timeout_seconds)
            save()
            raise RuntimeError(label + ' exceeded bounded timeout') from error
        record['exit_code'] = process.returncode
        (out / record['log']).write_text(process.stdout)
        save()
        if (expected == 'nonzero' and process.returncode == 0) or (expected != 'nonzero' and process.returncode != expected):
            raise RuntimeError(label + ' failed; see ' + str(out / record['log']))
        return process.stdout

    immutable = {}

    def bind(path, expected=None):
        info = identity(checked_path(path))
        if expected is not None and info['sha256'] != expected:
            raise ValueError('Identity mismatch: ' + str(path))
        immutable[info['path']] = info['sha256']
        return info

    try:
        object_path, gcc, tools, opcode, forbidden = bind_runtime(paths, app, report, bind)
        for name, text in (('start.S', STARTUP), ('link.ld', LINKER), ('runner.c', RUNNER)):
            (out / name).write_text(text)
        compiler_env = dict(os.environ)
        overrides = ('CPATH', 'C_INCLUDE_PATH', 'CPLUS_INCLUDE_PATH', 'OBJC_INCLUDE_PATH', 'LIBRARY_PATH', 'COMPILER_PATH', 'GCC_EXEC_PREFIX')
        for key in overrides:
            compiler_env.pop(key, None)
        compiler_env['PATH'] = str(gcc.parent) + ':/usr/bin:/bin'
        spike_env = dict(os.environ, LD_LIBRARY_PATH=str(paths['spike_library_dir']), PATH=str(paths['dtc'].parent) + ':/usr/bin:/bin')
        for key in ('LD_PRELOAD', 'LD_AUDIT'):
            compiler_env.pop(key, None)
            spike_env.pop(key, None)
        report['compiler_environment'] = {'PATH': compiler_env['PATH'], 'cleared_overrides': list(overrides) + ['LD_PRELOAD', 'LD_AUDIT']}
        report['spike_environment'] = {key: spike_env.get(key) for key in ('LD_LIBRARY_PATH', 'PATH', 'LD_PRELOAD', 'LD_AUDIT')}
        flags = [
            '-std=gnu11', '-march=rv64gc', '-mabi=lp64d', '-mcmodel=medany', '-msmall-data-limit=0', '-O2', '-ffreestanding',
            '-fno-builtin', '-fno-stack-protector', '-fno-tree-loop-distribute-patterns', '-nostdlib', '-nostartfiles', '-static',
            '-Wl,--no-relax', '-Wl,-T,' + str(out / 'link.ld'), '-I' + str(app),
        ]
        programs = {}
        for name, defines in (('invalid', ['-DINVALID_ONLY=1']), ('valid', []), ('bad-oracle', ['-DBAD_ORACLE=1'])):
            elf = out / (name + '.elf')
            command([gcc, *flags, *defines, '-Wl,-Map,' + str(out / (name + '.map')), out / 'start.S', out / 'runner.c', object_path, '-o', elf], 'link-' + name, compiler_env)
            audit = audit_instructions(elf.read_bytes(), opcode, forbidden)
            if audit['prohibited_hits']:
                raise ValueError('Prohibited FSM instructions in ' + name)
            report['checks'].append({'name': name + '_no_fsm', 'passed': True, **audit})
            undefined = command([tools['nm'], '-u', elf], 'undefined-' + name)
            if undefined.strip():
                raise ValueError('Unresolved runtime symbols in ' + name)
            command([tools['objdump'], '-dr', elf], 'disassembly-' + name)
            programs[name] = elf
        program_headers = command([tools['readelf'], '-W', '-l', programs['valid']], 'valid-program-headers')
        symbols_text = command([tools['nm'], '-S', '--defined-only', programs['valid']], 'valid-symbols')
        symbols = {}
        for line in symbols_text.splitlines():
            fields = line.split()
            if len(fields) in (3, 4) and re.fullmatch(r'[0-9a-fA-F]+', fields[0]):
                symbols[fields[-1]] = int(fields[0], 16)
        segments = []
        for line in program_headers.splitlines():
            fields = line.split()
            if fields and fields[0] == 'LOAD':
                segments.append({'start': int(fields[2], 16), 'file_bytes': int(fields[4], 16), 'memory_bytes': int(fields[5], 16)})
        if not segments or any(segment['start'] < 0x80000000 or segment['start'] + segment['memory_bytes'] > 0x90000000 for segment in segments):
            raise ValueError('ELF does not fit the requested generic Spike map')
        if symbols['_stack_top'] - symbols['_stack_bottom'] != 16384 or symbols['_image_end'] > 0x90000000:
            raise ValueError('Unexpected stack/image reservation extent')
        report['memory'] = {
            'requested_guest_base': 0x80000000, 'requested_guest_bytes': 0x10000000, 'load_segments': segments,
            'static_end': symbols['_stack_bottom'], 'static_reserved_bytes': symbols['_stack_bottom'] - 0x80000000,
            'stack_start': symbols['_stack_bottom'], 'stack_end': symbols['_stack_top'], 'stack_reserved_bytes': 16384,
            'image_end': symbols['_image_end'], 'reserved_image_bytes': symbols['_image_end'] - 0x80000000,
            'heap_reserved_bytes': 0, 'capacity_scope': 'functional requested map only; no deployed DRAM or complete-model fit claim',
        }
        base = [paths['spike'], '--isa=rv64gc', '-m0x80000000:0x10000000', '-p1']
        invalid = command([*base, programs['invalid']], 'invalid-generic-spike', spike_env)
        if 'INVALID_ARGUMENTS_PASS cases=18 buffers_unchanged=1' not in invalid:
            raise ValueError('Invalid argument numerical/status gate absent')
        report['checks'].append({'name': 'invalids_before_device_dispatch', 'passed': True, 'cases': 18})
        missing_extension = command([*base, programs['valid']], 'valid-without-extension', spike_env, expected='nonzero')
        if 'INVALID_ARGUMENTS_PASS cases=18 buffers_unchanged=1' not in missing_extension:
            raise ValueError('Missing-extension control did not reach the first valid device call')
        report['checks'].append({'name': 'same_valid_elf_requires_extension', 'passed': True})
        # Spike registers extension factories when processing --extlib; order matters.
        extension = [*base, '--extlib=' + str(paths['plugin']), '--extension=gemmini']
        valid = command([*extension, programs['valid']], 'valid-gemmini-spike', spike_env)
        gate = 'GEMMINI_MATMUL_RUNTIME_PASS calls=13 max_k=131071 exact_i64=1 guards=1 inputs=1 retained=1 overwrite=1'
        if gate not in valid or 'VALID_REJECT_VALID_RECOVERY_ARMED' not in valid or valid.count('MATMUL_CALL_PASS call=') != 13:
            raise ValueError('Numerical/runtime/recovery gate absent')
        wrong_oracle = command([*extension, programs['bad-oracle']], 'bad-oracle-gemmini-spike', spike_env, expected=22)
        if 'NUMERICAL_FAILURE' not in wrong_oracle:
            raise ValueError('Wrong-oracle control did not reach the numerical comparison')
        for path, digest in immutable.items():
            if identity(path)['sha256'] != digest:
                raise ValueError('Source/tool/dependency changed during verification: ' + path)
        report['immutable_bindings'] = immutable
        fixture_shapes = RUNNER.split('static const struct shape shapes[]={', 1)[1].split('};', 1)[0]
        report['fixture_cases'] = [
            dict(zip(('m', 'n', 'k', 'a_stride', 'b_stride', 'c_stride'), map(int, fields)))
            for fields in re.findall(r'\{(\d+),(\d+),(\d+),(\d+),(\d+),(\d+)\}', fixture_shapes)
        ]
        for case in report['fixture_cases']:
            case['invocations'] = 1 if case['k'] == 131071 else 2
        report['checks'].append({
            'name': 'numerical_runtime', 'passed': True, 'calls': 13, 'max_k': 131071, 'independent_i64_oracle': True,
            'wrong_oracle_exit_code': 22, 'same_process_valid_reject_valid': True, 'input_canaries': True, 'output_canaries': True,
            'output_padding_unchanged': True, 'retained_previous_output': True, 'output_overwritten_from_sentinel': True,
            'input_unchanged_scope': 'active strided storage plus up to 16 guard bytes, allocation boundary canaries',
        })
        names = ['runner.c', 'start.S', 'link.ld', 'invalid.elf', 'valid.elf', 'bad-oracle.elf']
        report['artifacts'] = [identity(out / name) for name in names]
        report['limitations'] = [
            'Provisional source/header-bound functional Gemmini model, not selected FireSim deployment qualification',
            'Plugin does not establish RTL cycle or bit-exact PE behavior; fixed K tiles are separately bounded from source',
            'C-library adapter numerical gate only; no Relax graph-to-device integration or complete-model support',
            'Exclusive single-hart accelerator ownership, functional memory accesses; no actual-platform coherence or timing claim',
            'One maximum admitted K case executed under per-process timeout; this is not an exhaustive arithmetic/shape suite',
        ]
        report.update(status='passed', device_execution=True)
    except Exception as error:
        report.update(status='failed', error=str(error))
        (out / 'traceback.txt').write_text(traceback.format_exc())
        raise
    finally:
        save()
    print(json.dumps({'status': report['status'], 'checks': report['checks'], 'receipt': str(out / 'receipt.json')}))


if __name__ == '__main__':
    main()
