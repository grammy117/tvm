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
"""Qualify an exported static graph against frozen independent fixture bytes.

This is a functional Spike/plugin check, not timing, RTL, deployed hardware or
application-quality qualification. The caller provides reference provenance;
this verifier does not generate expected answers or import the reference model.
"""

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

import numpy as np

from verify_baremetal_cpu import STARTUP, LINKER, GUEST_BASE, GUEST_BYTES, STACK_BYTES, checked_path, identity, bind_cpu_runtime, audit_cpu_runtime
from verify_matmul import audit_instructions
from verify_matmul_runtime import bind_runtime

STORAGE_DTYPES = {name: np.dtype(name) for name in ("int8", "uint8", "int32", "int64", "float32", "float64", "bool")}
STORAGE_DTYPES["bfloat16"] = np.dtype("uint16")
MAX_SESSION_CALLS = 64


RUNNER = r'''
#include <stdint.h>
#include <stddef.h>
#include "matmul.h"
#include "fixture.h"
#ifndef BAD_ORACLE
#define BAD_ORACLE 0
#endif
volatile uint64_t tohost __attribute__((section(".tohost"),aligned(64)));
volatile uint64_t fromhost __attribute__((section(".tohost"),aligned(64)));
static void puts_htif(const char*s){static volatile uint64_t request[4] __attribute__((aligned(8)));uint64_t n=0;while(s[n])++n;request[0]=64;request[1]=1;request[2]=(uintptr_t)s;request[3]=n;asm volatile("fence rw,rw" ::: "memory");tohost=(uintptr_t)request;while(!fromhost){}fromhost=0;asm volatile("fence rw,rw" ::: "memory");}
static void number(uint64_t n){char text[32];unsigned i=0;do{text[i++]=(char)('0'+n%10);n/=10;}while(n);for(unsigned j=0;j<i/2;++j){char t=text[j];text[j]=text[i-1-j];text[i-1-j]=t;}text[i]=0;puts_htif(text);}
int printf(const char*format,...){puts_htif("VENDOR_DIAGNOSTIC\n");return 0;}
__attribute__((noreturn)) void exit(int code){tohost=((uint64_t)(uint32_t)(code?code:99)<<1)|1;for(;;){}}
extern int32_t model_run(const void*const*,void*const*,uint8_t*,uint64_t);
extern const uint8_t model_constants[];
static uint64_t loads_a,loads_b,computes,stores,admissions;
int32_t __real_tvm_gemmini_validate_matmul_i8_i32(const int8_t*,const int8_t*,int32_t*,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);
int32_t __wrap_tvm_gemmini_validate_matmul_i8_i32(const int8_t*a,const int8_t*b,int32_t*c,int64_t m,int64_t n,int64_t k,int64_t as,int64_t bs,int64_t cs){++admissions;return __real_tvm_gemmini_validate_matmul_i8_i32(a,b,c,m,n,k,as,bs,cs);}
void __real_tvm_gemmini_load_a(const int8_t*,uint32_t,uint32_t,uint32_t);
void __real_tvm_gemmini_load_b(const int8_t*,uint32_t,uint32_t,uint32_t);
void __real_tvm_gemmini_compute(uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,int32_t);
void __real_tvm_gemmini_store(int32_t*,uint32_t,uint32_t,uint32_t);
void __wrap_tvm_gemmini_load_a(const int8_t*p,uint32_t r,uint32_t m,uint32_t n){++loads_a;__real_tvm_gemmini_load_a(p,r,m,n);}
void __wrap_tvm_gemmini_load_b(const int8_t*p,uint32_t r,uint32_t m,uint32_t n){++loads_b;__real_tvm_gemmini_load_b(p,r,m,n);}
void __wrap_tvm_gemmini_compute(uint32_t a,uint32_t b,uint32_t c,uint32_t m,uint32_t n,uint32_t k,int32_t add){++computes;__real_tvm_gemmini_compute(a,b,c,m,n,k,add);}
void __wrap_tvm_gemmini_store(int32_t*p,uint32_t r,uint32_t m,uint32_t n){++stores;__real_tvm_gemmini_store(p,r,m,n);}
static uint64_t hash(const void*p,uint64_t n){const volatile uint8_t*b=p;uint64_t h=UINT64_C(1469598103934665603);for(uint64_t i=0;i<n;++i)h=(h^b[i])*UINT64_C(1099511628211);return h;}
static void fill(uint8_t*p,uint64_t n,uint8_t value){for(uint64_t i=0;i<n;++i)p[i]=value;}
static int equal(const uint8_t*a,const uint8_t*b,uint64_t n,int wrong){for(uint64_t i=0;i<n;++i)if(a[i]!=(uint8_t)(b[i]^((wrong&&i==0)?1:0)))return 0;return 1;}
static int guard(const uint8_t*head,const uint8_t*tail){for(unsigned i=0;i<64;++i)if(head[i]!=0x6d||tail[i]!=0x6d)return 0;return 1;}
static int padding(const uint8_t*value,uint64_t n,uint64_t extent,uint8_t poison){for(uint64_t i=n;i<extent;++i)if(value[i]!=poison)return 0;return 1;}
DECLARATIONS
int main(void){
  INITIALIZE_GUARDS
  const void*inputs[INPUT_COUNT];void*outputs[OUTPUT_COUNT];
  INITIAL_POINTERS
  if(model_run(0,outputs,workspace.value,WORKSPACE_BYTES)!=-1||computes)return 10;
  if(WORKSPACE_BYTES&&model_run(inputs,outputs,workspace.value,0)!=-1)return 10;
  puts_htif("EXPORTED_GRAPH_INPUT_REJECTION_PASS\n");
  uint64_t constants_before=hash(model_constants,CONSTANT_BYTES);
  uint64_t retained[OUTPUT_COUNT];unsigned call=0;
  for(unsigned repeat=0;repeat<REPEATS;++repeat)for(unsigned sample=0;sample<SAMPLES;++sample,++call){
    unsigned slot=call%2;uint8_t poison=(uint8_t)(0xa5^call);
    PREPARE_INPUTS
    PREPARE_OUTPUTS
    fill(workspace.value,sizeof(workspace.value),poison);
    loads_a=loads_b=computes=stores=admissions=0;
    if(model_run(inputs,outputs,workspace.value,WORKSPACE_BYTES))return 21;
    CHECK_OUTPUTS
    CHECK_INPUTS
    CHECK_GUARDS
    if(!padding(workspace.value,WORKSPACE_BYTES,sizeof(workspace.value),poison))return 28;
    if(constants_before!=hash(model_constants,CONSTANT_BYTES))return 29;
    if(loads_a!=EXPECTED_LOADS_A||loads_b!=EXPECTED_LOADS_B||computes!=EXPECTED_COMPUTES||stores!=EXPECTED_STORES||admissions!=EXPECTED_ADMISSIONS)return 30;
    puts_htif("EXPORTED_GRAPH_CALL_PASS sample=");number(sample);puts_htif(" repeat=");number(repeat);puts_htif(" primitive_counts=1 exact_bytes=1\n");
  }
  extern uint8_t _stack_bottom[],_stack_top[];
  uint8_t*first=_stack_bottom;while(first<_stack_top&&*first==(uint8_t)(UINT64_C(0x55aa55aa55aa55aa)>>(((uintptr_t)first-(uintptr_t)_stack_bottom)%8*8)))++first;
  uint64_t used=(uint64_t)(_stack_top-first);if(!used||used>=STACK_BYTES)return 31;
  puts_htif("STACK_HIGH_WATER_BYTES=");number(used);puts_htif("\n");
  puts_htif("TVM_EXPORTED_GRAPH_PASS calls=");number(call);puts_htif(" exact_bytes=1 guards=1 inputs=1 constants=1 retained=1 poisoned_workspace=1 primitive_counts=1\n");return 0;
}
'''


def aligned_size(size):
    return (size + 63) // 64 * 64


def validate_tensor(tensor):
    """Refuse declared byte extents that could underallocate a typed buffer."""
    if not isinstance(tensor, dict) or type(tensor.get("dtype")) is not str or tensor["dtype"] not in STORAGE_DTYPES:
        raise ValueError("Unsupported manifest tensor dtype")
    shape = tensor.get("shape")
    if not isinstance(shape, list) or any(type(dim) is not int or dim <= 0 for dim in shape):
        raise ValueError("Manifest tensor shape must contain positive static integers")
    required = math.prod(shape) * STORAGE_DTYPES[tensor["dtype"]].itemsize
    if required > (1 << 63) - 1 or type(tensor.get("bytes")) is not int or tensor["bytes"] != required:
        raise ValueError("Manifest tensor bytes must equal dtype size times shape")
    return required


def validate_manifest(manifest):
    """Validate every declared tensor and the explicit storage envelope."""
    if not isinstance(manifest, dict):
        raise ValueError("Manifest must be an object")
    for group in ("inputs", "outputs", "constants", "intermediates"):
        tensors = manifest.get(group)
        if not isinstance(tensors, list) or (group in ("inputs", "outputs") and not tensors):
            raise ValueError("Manifest requires complete input/output/tensor lists")
        for tensor in tensors:
            validate_tensor(tensor)
    for name in ("workspace_bytes", "constants_bytes", "explicit_tensor_bytes"):
        if type(manifest.get(name)) is not int or not 0 <= manifest[name] <= (1 << 63) - 1:
            raise ValueError("Manifest storage bytes must be nonnegative integers")
    required = sum(tensor["bytes"] for group in ("inputs", "outputs") for tensor in manifest[group]) + manifest["workspace_bytes"] + manifest["constants_bytes"]
    if manifest["explicit_tensor_bytes"] != required:
        raise ValueError("Manifest explicit tensor bytes do not match its storage envelope")
    for group, limit in (("constants", manifest["constants_bytes"]), ("intermediates", manifest["workspace_bytes"])):
        for tensor in manifest[group]:
            if group == "intermediates" and tensor.get("origin") != "workspace":
                continue
            offset = tensor.get("offset")
            if type(offset) is not int or offset < 0 or offset % 64 or offset + tensor["bytes"] > limit:
                raise ValueError("Manifest tensor lies outside its aligned storage envelope")


def strict_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON field: " + key)
            result[key] = value
        return result
    return json.loads(checked_path(path).read_text(), object_pairs_hook=unique)


def guest_layout(guest_bytes):
    """Bind one explicit functional memory budget to linking and simulator admission."""
    if type(guest_bytes) is not int or guest_bytes <= STACK_BYTES or guest_bytes % 4096 or GUEST_BASE + guest_bytes > 1 << 64:
        raise ValueError("Guest memory requires a page-aligned RV64 span larger than the reserved stack")
    return LINKER.replace(f"LENGTH = {GUEST_BYTES:#x}", f"LENGTH = {guest_bytes:#x}"), f"-m{GUEST_BASE:#x}:{guest_bytes:#x}"


def validate_sequence(package, route, *, guest_bytes=GUEST_BYTES):
    """Admit a bounded explicit DAG; every invocation retains all its outputs."""
    guest_layout(guest_bytes)
    stages = package.get("stages") if isinstance(package, dict) else None
    if not isinstance(stages, dict) or not 1 <= len(stages) <= MAX_SESSION_CALLS or any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name) for name in stages):
        raise ValueError(f"Session requires one to {MAX_SESSION_CALLS} named static stages")
    if package.get("state_routing") != "caller_owned" or package.get("session_execution") is not False:
        raise ValueError("Require an export_graphs package with caller-owned routing")
    for field in ("constants_bytes", "duplicated_stage_constants_bytes", "serial_workspace_bytes", "maximum_stage_tensor_bytes"):
        if type(package.get(field)) is not int or not 0 <= package[field] <= (1 << 63) - 1:
            raise ValueError("Session storage summaries must be plain nonnegative integers")
    entries = set()
    for stage in stages.values():
        validate_manifest(stage)
        if len(stage["inputs"]) + len(stage["outputs"]) > 256:
            raise ValueError("Stage tensor slots exceed the bounded driver")
        entry = stage.get("entrypoint")
        if not isinstance(entry, str) or not re.fullmatch(r"stage_[0-9]+_model_run", entry) or entry in entries:
            raise ValueError("Session stage entrypoints must be distinct exported symbols")
        entries.add(entry)
        if stage.get("heap_required_by_orchestration") is not False or stage.get("runtime_workspace_allocator_required") is not False or stage.get("alignment_bytes") != 64 or not stage.get("target", "").startswith("llvm -mtriple=riscv64-unknown-elf"):
            raise ValueError("Require aligned RV64 stages without hidden allocation")
        if stage["constants_bytes"] != package.get("constants_bytes") or stage.get("constants_sha256") != package.get("constants_sha256"):
            raise ValueError("Stage constants must bind the shared pool")
        if not isinstance(stage.get("calls"), list) or any(not isinstance(call, dict) or not isinstance(call.get("symbol"), str) or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", call["symbol"]) for call in stage["calls"]):
            raise ValueError("Stage calls require explicit C symbols")
    if package.get("serial_workspace_bytes") != max(stage["workspace_bytes"] for stage in stages.values()) or package.get("maximum_stage_tensor_bytes") != max(stage["explicit_tensor_bytes"] for stage in stages.values()):
        raise ValueError("Session storage summary differs from admitted stages")
    if not isinstance(route, dict) or set(route) != {"version", "calls"} or type(route["version"]) is not int or route["version"] != 1 or not isinstance(route["calls"], list) or not 1 <= len(route["calls"]) <= MAX_SESSION_CALLS:
        raise ValueError(f"Require strict version-one routing with one to {MAX_SESSION_CALLS} calls")
    fixtures, outputs, calls, used = {}, [], [], set()
    same = lambda tensor: (tensor["dtype"], tensor["shape"], tensor["bytes"])
    for ordinal, call in enumerate(route["calls"]):
        if not isinstance(call, dict) or set(call) != {"stage", "inputs"} or not isinstance(call["stage"], str) or call["stage"] not in stages:
            raise ValueError("Unknown or malformed session stage")
        name, stage = call["stage"], stages[call["stage"]]
        if not isinstance(call["inputs"], list) or len(call["inputs"]) != len(stage["inputs"]):
            raise ValueError("Every stage input requires exactly one explicit route")
        used.add(name)
        pointers = []
        for binding, tensor in zip(call["inputs"], stage["inputs"]):
            if not isinstance(binding, dict) or len(binding) != 1:
                raise ValueError("Routes admit only fixture indices or earlier outputs")
            if "fixture" in binding:
                index = binding["fixture"]
                if type(index) is not int or not 0 <= index < 256:
                    raise ValueError("Fixture route requires a bounded integer index")
                if index in fixtures and same(fixtures[index]) != same(tensor):
                    raise ValueError("Fixture route dtype/shape differs between stages")
                fixtures[index] = tensor
                pointers.append(("input", index))
            elif "output" in binding:
                source = binding["output"]
                if not isinstance(source, list) or len(source) != 2 or any(type(index) is not int for index in source) or not 0 <= source[0] < ordinal or not 0 <= source[1] < len(calls[source[0]]["outputs"]):
                    raise ValueError("Output route must name an earlier invocation/output")
                index = calls[source[0]]["outputs"][source[1]]
                if same(outputs[index]) != same(tensor):
                    raise ValueError("Output route dtype/shape must match exactly")
                pointers.append(("output", index))
            else:
                raise ValueError("Routes admit no transforms or inferred state")
        indices = list(range(len(outputs), len(outputs) + len(stage["outputs"])))
        outputs.extend(stage["outputs"])
        calls.append({"stage": name, "inputs": pointers, "outputs": indices})
        if len(outputs) + len(fixtures) > 256:
            raise ValueError("Session tensor slots exceed the bounded driver")
    if used != set(stages) or set(fixtures) != set(range(len(fixtures))) or not fixtures:
        raise ValueError("Every stage must execute and fixture indices must be contiguous")
    inputs = [fixtures[index] for index in range(len(fixtures))]
    required = sum(tensor["bytes"] for tensor in [*inputs, *outputs]) + package["constants_bytes"] + package["serial_workspace_bytes"]
    if required > guest_bytes - STACK_BYTES:
        raise ValueError("Retained session storage exceeds the requested guest budget")
    aggregate = {"inputs": inputs, "outputs": outputs, "constants": [], "intermediates": [], "constants_bytes": package["constants_bytes"], "workspace_bytes": package["serial_workspace_bytes"], "explicit_tensor_bytes": required}
    validate_manifest(aggregate)
    return aggregate, calls


def sequence_glue(manifest, calls, stages, counts, checks):
    """Use original static entrypoints; routed earlier outputs remain immutable."""
    lines = ["static unsigned session_sample,session_call;"]
    for stage in stages.values():
        lines.append(f"extern int32_t {stage['entrypoint']}(const void*const*,void*const*,uint8_t*,uint64_t);")
    lines.extend(["static int sequence_span(const void*p,uint64_t n){return p&&!((uintptr_t)p%64)&&n<=UINTPTR_MAX-(uintptr_t)p;}",
                  "static int sequence_overlap(const void*a,uint64_t an,const void*b,uint64_t bn){return an&&bn&&(uintptr_t)a<(uintptr_t)b+bn&&(uintptr_t)b<(uintptr_t)a+an;}",
                  "int32_t model_run(const void*const*inputs,void*const*outputs,uint8_t*scratch,uint64_t bytes){",
                  "if(!inputs||!outputs||bytes<WORKSPACE_BYTES)return -1;",
                  "if(WORKSPACE_BYTES&&!sequence_span(scratch,WORKSPACE_BYTES))return -1;"])
    for group in ("inputs", "outputs"):
        sizes = ",".join(str(tensor["bytes"]) for tensor in manifest[group])
        lines.append(f"const uint64_t {group}_sizes[]={{{sizes}}};")
        lines.append(f"for(unsigned i=0;i<{len(manifest[group])};++i)if(!sequence_span({group}[i],{group}_sizes[i]))return -1;")
    lines.extend(["for(unsigned i=0;i<OUTPUT_COUNT;++i){",
                  "if(sequence_overlap(outputs[i],outputs_sizes[i],model_constants,CONSTANT_BYTES)||sequence_overlap(outputs[i],outputs_sizes[i],scratch,WORKSPACE_BYTES))return -2;",
                  "for(unsigned j=0;j<INPUT_COUNT;++j)if(sequence_overlap(outputs[i],outputs_sizes[i],inputs[j],inputs_sizes[j]))return -2;",
                  "for(unsigned j=0;j<i;++j)if(sequence_overlap(outputs[i],outputs_sizes[i],outputs[j],outputs_sizes[j]))return -2;}",
                  "if(sequence_overlap(scratch,WORKSPACE_BYTES,model_constants,CONSTANT_BYTES))return -2;",
                  "for(unsigned i=0;i<INPUT_COUNT;++i)if(sequence_overlap(scratch,WORKSPACE_BYTES,inputs[i],inputs_sizes[i]))return -2;"])
    external_checks = [f"if(!equal(inputs[{i}],fixture_input_{i}+(uint64_t)session_sample*{aligned_size(tensor['bytes'])},{tensor['bytes']},0)||!padding(inputs[{i}],{tensor['bytes']},{aligned_size(tensor['bytes'])},0x37))return 24;" for i, tensor in enumerate(manifest["inputs"])]
    for ordinal, call in enumerate(calls):
        stage = stages[call["stage"]]
        ins = ",".join(f"{kind}s[{index}]" for kind, index in call["inputs"])
        outs = ",".join(f"outputs[{index}]" for index in call["outputs"])
        input_checks = [f"if(!equal(stage_inputs[{i}],fixture_{kind}_{index}+(uint64_t)session_sample*{aligned_size(tensor['bytes'])},{tensor['bytes']},0))return 24;" for i, ((kind, index), tensor) in enumerate(zip(call["inputs"], stage["inputs"]))]
        retained = [(i, tensor) for i, tensor in enumerate(manifest["outputs"]) if i not in call["outputs"]]
        lines.extend(["{", f"const void*stage_inputs[]={{{ins}}};void*stage_outputs[]={{{outs}}};",
                      *external_checks, *input_checks,
                      *[f"uint64_t retained_{i}=hash(outputs[{i}],{aligned_size(tensor['bytes'])});" for i, tensor in retained],
                      "uint64_t constants_before=hash(model_constants,CONSTANT_BYTES);",
                      f"uint8_t poison=(uint8_t)(0xc0^{ordinal});fill(scratch,WORKSPACE_BYTES,poison);",
                      "uint64_t la=loads_a,lb=loads_b,co=computes,st=stores,ad=admissions;",
                      f"if({stage['entrypoint']}(stage_inputs,stage_outputs,scratch,WORKSPACE_BYTES))return 21;"])
        lines.extend([*external_checks, *input_checks])
        lines.append("if(constants_before!=hash(model_constants,CONSTANT_BYTES))return 29;")
        for i, tensor in retained:
            lines.append(f"if(retained_{i}!=hash(outputs[{i}],{aligned_size(tensor['bytes'])}))return 26;")
        for i in call["outputs"]:
            tensor = manifest["outputs"][i]
            lines.append(f"if(!equal(outputs[{i}],fixture_output_{i}+(uint64_t)session_sample*{aligned_size(tensor['bytes'])},{tensor['bytes']},BAD_ORACLE&&session_call==0&&{i}==0)){{puts_htif(\"EXPORTED_GRAPH_NUMERICAL_FAILURE\\n\");return 22;}}")
            lines.append(f"if(!padding(outputs[{i}],{tensor['bytes']},{aligned_size(tensor['bytes'])},(uint8_t)(0xa5^session_call)))return 23;")
        lines.extend(checks)
        lines.append(f"if(!padding(scratch,{stage['workspace_bytes']},WORKSPACE_BYTES,poison))return 28;")
        expected = counts[call["stage"]]
        comparisons = "||".join(f"{name}-{before}!={expected[key]}" for name, before, key in (("loads_a", "la", "loads_a"), ("loads_b", "lb", "loads_b"), ("computes", "co", "computes"), ("stores", "st", "stores"), ("admissions", "ad", "admissions")))
        lines.append(f"if({comparisons})return 30;")
        lines.append(f'puts_htif("EXPORTED_SESSION_STAGE_PASS invocation={ordinal} exact_bytes=1 inputs=1 retained=1 primitive_counts=1\\n");}}')
    lines.append("return 0;}")
    return "\n".join(lines)


def load_fixture(path, manifest):
    """Require explicit exact dtype/shape for each complete input/output stream."""
    for group in ("inputs", "outputs"):
        for tensor in manifest[group]:
            validate_tensor(tensor)
    arrays, count = {}, None
    with np.load(checked_path(path), allow_pickle=False) as fixture:
        expected = {f"{kind}_{index}" for kind in ("input", "output") for index in range(len(manifest[kind + "s"]))}
        if set(fixture.files) != expected:
            raise ValueError("Fixture must contain exactly the graph input_N and output_N arrays")
        for kind in ("input", "output"):
            for index, tensor in enumerate(manifest[kind + "s"]):
                name = f"{kind}_{index}"
                value = fixture[name]
                if value.dtype != STORAGE_DTYPES[tensor["dtype"]] or value.ndim != len(tensor["shape"]) + 1 or tuple(value.shape[1:]) != tuple(tensor["shape"]) or value.shape[0] < 1:
                    raise ValueError(f"Fixture dtype/shape mismatch: {name}")
                if count is None:
                    count = value.shape[0]
                nonfinite = np.any((value & np.uint16(0x7f80)) == np.uint16(0x7f80)) if tensor["dtype"] == "bfloat16" else value.dtype.kind == "f" and not np.isfinite(value).all()
                if value.shape[0] != count or nonfinite:
                    raise ValueError("Fixture streams must have equal positive counts and finite values")
                arrays[name] = np.ascontiguousarray(value)
    return arrays, count


def make_guest(out, manifest, arrays, samples, repeats, counts, *, sequence=None, guest_bytes=GUEST_BYTES):
    """Generate guarded caller storage and compare independent frozen byte blobs."""
    linker, _ = guest_layout(guest_bytes)
    validate_manifest(manifest)
    declarations, initialization, initial_pointers = [], [], []
    prepare_inputs, prepare_outputs, check_inputs, check_outputs, check_guards = [], [], [], [], []
    assembly = []
    for kind in ("input", "output"):
        for index, tensor in enumerate(manifest[kind + "s"]):
            name = f"{kind}_{index}"
            extent, size = aligned_size(tensor["bytes"]), tensor["bytes"]
            blob = out / (name + ".bin")
            if any(character in str(blob) for character in ("\n", "\r")):
                raise ValueError("Fixture assembly path contains an unsupported line break")
            with blob.open("wb") as stream:
                for value in arrays[name]:
                    stream.write(value.astype(value.dtype.newbyteorder("<"), copy=False).tobytes())
                    stream.write(bytes(extent - size))
            assembly.extend([f'.section .rodata.fixture_{name},"a",@progbits', '.balign 64', f'.globl fixture_{name}', f'.type fixture_{name},@object', f'fixture_{name}:', f'.incbin "{str(blob).replace(chr(92), chr(92)*2).replace(chr(34), chr(92)+chr(34))}"', f'.size fixture_{name},.-fixture_{name}'])
            declarations.extend([f"extern const uint8_t fixture_{name}[];", f"struct {name}_storage {{uint8_t head[64];uint8_t value[{extent}];uint8_t tail[64];}};", f"static struct {name}_storage {name}{'[2]' if kind == 'output' else ''} __attribute__((aligned(64)));"])
            if kind == "input":
                initialization.append(f"fill({name}.head,64,0x6d);fill({name}.tail,64,0x6d);")
                initial_pointers.append(f"inputs[{index}]={name}.value;")
                prepare_inputs.append(f"fill({name}.value,{extent},0x37);for(uint64_t i=0;i<{size};++i){name}.value[i]=fixture_{name}[(uint64_t)sample*{extent}+i];")
                check_inputs.append(f"if(!equal({name}.value,fixture_{name}+(uint64_t)sample*{extent},{size},0)||!padding({name}.value,{size},{extent},0x37))return 24;")
                check_guards.append(f"if(!guard({name}.head,{name}.tail))return 27;")
            else:
                initialization.append(f"for(unsigned b=0;b<2;++b){{fill({name}[b].head,64,0x6d);fill({name}[b].tail,64,0x6d);}}")
                initial_pointers.append(f"outputs[{index}]={name}[0].value;")
                prepare_outputs.append(f"outputs[{index}]={name}[slot].value;fill({name}[slot].value,{extent},poison);")
                check_outputs.extend([f"if(!equal({name}[slot].value,fixture_{name}+(uint64_t)sample*{extent},{size},BAD_ORACLE&&call==0&&{index}==0)){{puts_htif(\"EXPORTED_GRAPH_NUMERICAL_FAILURE\\n\");return 22;}}", f"if(!padding({name}[slot].value,{size},{extent},poison))return 23;", f"if(call&&retained[{index}]!=hash({name}[1-slot].value,{extent}))return 26;retained[{index}]=hash({name}[slot].value,{extent});"])
                check_guards.append(f"for(unsigned b=0;b<2;++b)if(!guard({name}[b].head,{name}[b].tail))return 27;")
    workspace = max(64, aligned_size(manifest["workspace_bytes"]))
    declarations.extend([f"struct workspace_storage {{uint8_t head[64];uint8_t value[{workspace}];uint8_t tail[64];}};", "static struct workspace_storage workspace __attribute__((aligned(64)));"])
    initialization.append("fill(workspace.head,64,0x6d);fill(workspace.tail,64,0x6d);")
    check_guards.append("if(!guard(workspace.head,workspace.tail))return 27;")
    if sequence is not None:
        calls, stages, stage_counts = sequence
        declarations.append(sequence_glue(manifest, calls, stages, stage_counts, check_guards))
        prepare_inputs.append("session_sample=sample;session_call=call;")
    replacements = {"DECLARATIONS": declarations, "INITIALIZE_GUARDS": initialization, "INITIAL_POINTERS": initial_pointers,
                    "PREPARE_INPUTS": prepare_inputs, "PREPARE_OUTPUTS": prepare_outputs, "CHECK_INPUTS": check_inputs,
                    "CHECK_OUTPUTS": check_outputs, "CHECK_GUARDS": check_guards}
    runner = RUNNER
    for name, lines in replacements.items():
        runner = runner.replace(name, "\n  ".join(lines))
    if sequence is not None:
        runner = runner.replace("if(model_run(inputs,outputs,workspace.value,WORKSPACE_BYTES))return 21;", "int32_t status=model_run(inputs,outputs,workspace.value,WORKSPACE_BYTES);if(status)return status==22?22:21;")
        # Admission controls execute before preparing fixtures or dispatching a
        # stage. Restore each pointer before testing the next invalid envelope.
        zero = "loads_a||loads_b||computes||stores||admissions"
        controls = ["void*saved=outputs[0];", f"outputs[0]=(void*)inputs[0];if(model_run(inputs,outputs,workspace.value,WORKSPACE_BYTES)!=-2||{zero})return 10;outputs[0]=saved;",
                    f"outputs[0]=(uint8_t*)saved+1;if(model_run(inputs,outputs,workspace.value,WORKSPACE_BYTES)!=-1||{zero})return 10;outputs[0]=saved;",
                    f"if(OUTPUT_COUNT>1){{void*other=outputs[1];outputs[1]=outputs[0];if(model_run(inputs,outputs,workspace.value,WORKSPACE_BYTES)!=-2||{zero})return 10;outputs[1]=other;}}",
                    f"if(CONSTANT_BYTES){{outputs[0]=(void*)model_constants;if(model_run(inputs,outputs,workspace.value,WORKSPACE_BYTES)!=-2||{zero})return 10;outputs[0]=saved;}}",
                    f"if(WORKSPACE_BYTES){{outputs[0]=workspace.value;if(model_run(inputs,outputs,workspace.value,WORKSPACE_BYTES)!=-2||{zero})return 10;outputs[0]=saved;}}",
                    'puts_htif("EXPORTED_SESSION_SPAN_REJECTION_PASS device_counts=0\\n");']
        runner = runner.replace('puts_htif("EXPORTED_GRAPH_INPUT_REJECTION_PASS\\n");', "\n".join(controls) + '\nputs_htif("EXPORTED_GRAPH_INPUT_REJECTION_PASS\\n");')
    assembly.append('.section .note.GNU-stack,"",@progbits')
    (out / "fixture.S").write_text("\n".join(assembly) + "\n")
    (out / "runner.c").write_text(runner)
    definitions = {"INPUT_COUNT": len(manifest["inputs"]), "OUTPUT_COUNT": len(manifest["outputs"]), "SAMPLES": samples, "REPEATS": repeats,
                   "WORKSPACE_BYTES": manifest["workspace_bytes"], "CONSTANT_BYTES": manifest["constants_bytes"], "STACK_BYTES": STACK_BYTES,
                   **{"EXPECTED_" + name.upper(): value for name, value in counts.items()}}
    (out / "fixture.h").write_text("\n".join(f"#define {name} UINT64_C({value})" for name, value in definitions.items()) + "\n")
    (out / "start.S").write_text(STARTUP)
    (out / "link.ld").write_text(linker)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("graph-dir", "fixture", "fixture-provenance", "tvm-source", "tvm-build", "adapter-receipt", "simulator-receipt", "riscv-gcc", "spike", "dtc", "spike-library-dir", "plugin", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--guest-memory-bytes", type=int, default=GUEST_BYTES, help="Explicit page-aligned functional Spike map; defaults to 256 MiB and does not declare deployed DRAM capacity")
    parser.add_argument("--session-route", type=Path, help="Opt-in strict fixed-sequence route JSON for an export_graphs package. Every invocation output requires independent fixture bytes.")
    parser.add_argument("--cpu-runtime-library", action="append", type=Path, default=[], help="Explicit ordinary RV64/lp64d static archive; repeat for each CPU runtime dependency. No libraries are selected by default.")
    args = parser.parse_args()
    if args.repeats < 2 or not 1 <= args.timeout_seconds <= 86400:
        parser.error("repeats must be at least two; timeout must be in [1,86400]")
    try:
        _, guest_map = guest_layout(args.guest_memory_bytes)
    except ValueError as error:
        parser.error(str(error))
    paths = {name: checked_path(value) for name, value in vars(args).items() if isinstance(value, Path)}
    out, app = paths["output_dir"], checked_path(Path(__file__).parent)
    out.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "scope": __doc__, "commands": [], "checks": [], "timing_qualified": False, "application_quality_qualified": False, "deployed_hardware_verified": False, "rtl_bit_exact": False, "invocation": [sys.executable, *sys.argv]}
    immutable = {}
    def save():
        (out / "receipt.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    def bind(path, expected=None):
        record = identity(checked_path(path))
        if expected is not None and record["sha256"] != expected:
            raise ValueError("Identity mismatch: " + str(path))
        immutable[record["path"]] = record["sha256"]
        return record
    def command(argv, label, env=None, expected=0):
        entry = {"argv": list(map(str, argv)), "expected": expected, "log": label + ".log"}
        report["commands"].append(entry)
        save()
        start = time.monotonic()
        print(label, "started", flush=True)
        try:
            with (out / entry["log"]).open("w") as stream:
                process = subprocess.run(entry["argv"], stdout=stream, stderr=subprocess.STDOUT, text=True, timeout=args.timeout_seconds, env=env)
            stdout = (out / entry["log"]).read_text()
        except subprocess.TimeoutExpired as error:
            entry.update(timed_out=True, wall_seconds=time.monotonic() - start)
            save()
            raise RuntimeError(label + " timed out") from error
        entry.update(exit_code=process.returncode, wall_seconds=time.monotonic() - start)
        save()
        print(label, "exit", process.returncode, flush=True)
        if (expected == "nonzero" and process.returncode == 0) or (expected != "nonzero" and process.returncode != expected):
            raise RuntimeError(label + " failed; see " + entry["log"])
        return stdout
    try:
        adapter, gcc, tools, opcode, forbidden = bind_runtime(paths, app, report, bind)
        report["runtime_helpers"].append(report["verifier_source"])
        report["verifier_source"] = bind(__file__)
        graph_dir = paths["graph_dir"]
        sequence = None
        if args.session_route is not None:
            report["route_artifact"] = bind(paths["session_route"])
            report["graph_artifacts"] = [bind(graph_dir / name) for name in ("graphs.json", "constants.S", "constants.bin", "runtime.c", "model.h")]
            package, route = strict_json(graph_dir / "graphs.json"), strict_json(paths["session_route"])
            manifest, calls = validate_sequence(package, route, guest_bytes=args.guest_memory_bytes)
            graphs = {}
            for name, stage in package["stages"].items():
                folder = checked_path(graph_dir / name)
                report["graph_artifacts"].extend(bind(folder / artifact) for artifact in ("graph.json", "graph.relax.py", "operators.tir.py", "operators.o", "operators.ll", "model.c"))
                if strict_json(folder / "graph.json") != stage or checked_path(stage["constants_file"]) != graph_dir / "constants.bin":
                    raise ValueError("Stage artifact/constant location differs from package declaration")
                graphs[name] = (folder, stage)
            link_sources = [graph_dir / "runtime.c", graph_dir / "constants.S", *[folder / artifact for folder, _ in graphs.values() for artifact in ("model.c", "operators.o")]]
            report["session"] = {"route": route, "calls": calls, "shared_constants": True, "serial_workspace": True,
                                 "reset": "Each sample/repeat starts the fixed sequence from external fixture inputs",
                                 "memory_scope": "Retained verification trace; not an optimized production-session plan",
                                 "automatic_model_session_adapter": False, "full_model_qualified": False}
        else:
            report["graph_artifacts"] = [bind(graph_dir / name) for name in ("graph.json", "graph.relax.py", "operators.tir.py", "operators.o", "operators.ll", "model.c", "constants.S", "constants.bin")]
            manifest = json.loads((graph_dir / "graph.json").read_text())
            validate_manifest(manifest)
            if not manifest.get("target", "").startswith("llvm -mtriple=riscv64-unknown-elf") or manifest.get("runtime_workspace_allocator_required") is not False:
                raise ValueError("Require a RV64 static export without runtime workspace allocation")
            graphs = {"graph": (graph_dir, manifest)}
            link_sources = [graph_dir / "model.c", graph_dir / "constants.S", graph_dir / "operators.o"]
        if (graph_dir / "constants.bin").stat().st_size != max(manifest["constants_bytes"], 1):
            raise ValueError("Constant blob size differs from its declared storage envelope")
        if bind(graph_dir / "constants.bin")["sha256"] != (package if args.session_route is not None else manifest)["constants_sha256"]:
            raise ValueError("Exported constants do not match their manifest")
        report["fixture"], report["fixture_provenance_artifact"] = bind(paths["fixture"]), bind(paths["fixture_provenance"])
        provenance = json.loads(paths["fixture_provenance"].read_text())
        if provenance.get("fixture_sha256") != report["fixture"]["sha256"] or not isinstance(provenance.get("reference_sources"), list) or not provenance["reference_sources"] or not isinstance(provenance.get("reference_contract"), str) or not provenance["reference_contract"].strip():
            raise ValueError("Fixture provenance must bind fixture_sha256, reference_sources and reference_contract")
        for source in provenance["reference_sources"]:
            bind(source["path"], source["sha256"])
        report["fixture_provenance"] = provenance
        arrays, samples = load_fixture(paths["fixture"], manifest)
        if args.session_route is not None:
            # Include complete frozen streams and guarded double outputs before
            # constructing artifacts; final ELF/stack extents are audited below.
            sizes = {group: [aligned_size(tensor["bytes"]) for tensor in manifest[group]] for group in ("inputs", "outputs")}
            guest_storage = sum(size + 128 for size in sizes["inputs"]) + 2 * sum(size + 128 for size in sizes["outputs"]) + max(64, aligned_size(manifest["workspace_bytes"])) + 128 + max(1, manifest["constants_bytes"]) + samples * sum(sizes["inputs"] + sizes["outputs"]) + STACK_BYTES
            if guest_storage > args.guest_memory_bytes:
                raise ValueError("Complete guarded fixture/trace storage exceeds the requested guest budget")
            report["session"]["reserved_trace_fixture_stack_bytes"] = guest_storage
        sample_ids = provenance.get("sample_ids")
        if not isinstance(sample_ids, list) or len(sample_ids) != samples or any(not isinstance(value, str) or not value.strip() for value in sample_ids) or len(set(sample_ids)) != samples:
            raise ValueError("Fixture provenance requires distinct sample_ids for the complete selected stream")
        report["samples"], report["repeats"] = samples, args.repeats
        source, build = paths["tvm_source"], paths["tvm_build"]
        os.environ.update(TVM_LIBRARY_PATH=str(build), TVM_FFI="ctypes", PYTHONDONTWRITEBYTECODE="1")
        sys.path.insert(0, str(checked_path(source / "python")))
        import tvm
        from tvm._ffi.base import _LIB
        if not checked_path(tvm.__file__).is_relative_to(source / "python") or checked_path(_LIB._name).parent != build:
            raise ValueError("TVM source/library do not match selected paths")
        report["compiler_library"] = bind(_LIB._name)
        for name in ("apps/gemmini/baremetal.py", "python/tvm/relax/backend/contrib/gemmini_schedule.py"):
            bind(source / name)
        counts = dict.fromkeys(("loads_a", "loads_b", "computes", "stores", "admissions"), 0)
        shapes, stage_counts = [], {}
        # Derive counts from the selected module's declared schedule, not model names.
        from tvm.relax.backend.contrib.gemmini_schedule import _DIM
        for stage_name, (folder, stage) in graphs.items():
            primitive_mod = tvm.script.from_source((folder / "operators.tir.py").read_text())
            stage_counts[stage_name] = dict.fromkeys(counts, 0)
            multiplicity = sum(call["stage"] == stage_name for call in calls) if args.session_route is not None else 1
            for call in stage["calls"]:
                function = primitive_mod[call["symbol"]]
                if function.attrs and "gemmini.m" in function.attrs:
                    m, n, k, ti, tj = (int(function.attrs["gemmini." + name]) for name in ("m", "n", "k", "tile_i", "tile_j"))
                    mt, nt, kt = ((value + _DIM - 1) // _DIM for value in (m, n, k))
                    row = {"symbol": call["symbol"], "m": m, "n": n, "k": k, "macs": m * n * k,
                           "loads_a": mt * ((n + tj * _DIM - 1) // (tj * _DIM)) * kt,
                           "loads_b": nt * ((m + ti * _DIM - 1) // (ti * _DIM)) * kt,
                           "computes": mt * nt * kt, "stores": mt * nt, "admissions": 1}
                    shapes.extend([row] * multiplicity)
                    for key in counts:
                        stage_counts[stage_name][key] += row[key]
                        counts[key] += row[key] * multiplicity
        if not shapes:
            raise ValueError("No scheduled contraction with declared primitive-count metadata")
        report["schedule"] = {"shapes": shapes, "expected_counts_per_invocation": counts, "semantic_macs": sum(row["macs"] for row in shapes)}
        if args.session_route is not None:
            sequence = (calls, package["stages"], stage_counts)
            report["session"]["expected_counts_per_stage"] = stage_counts
        make_guest(out, manifest, arrays, samples, args.repeats, counts, sequence=sequence, guest_bytes=args.guest_memory_bytes)
        report["generated_fixtures"] = [bind(out / f"{kind}_{index}.bin") for kind in ("input", "output") for index in range(len(manifest[kind + "s"]))]
        report["generated_guest_sources"] = [bind(out / name) for name in ("runner.c", "fixture.h", "fixture.S", "start.S", "link.ld")]
        env = dict(os.environ)
        for name in ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH", "COMPILER_PATH", "GCC_EXEC_PREFIX", "LD_PRELOAD", "LD_AUDIT"):
            env.pop(name, None)
        env["PATH"] = str(gcc.parent) + ":/usr/bin:/bin"
        runtime_arguments, runtime_binding = bind_cpu_runtime(args.cpu_runtime_library, tools, bind, command)
        if runtime_arguments:
            report["cpu_runtime"] = runtime_binding
        flags = ["-std=gnu11", "-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany", "-msmall-data-limit=0", "-O2", "-ffreestanding", "-fno-builtin", "-fno-stack-protector", "-fno-tree-loop-distribute-patterns", "-nostdlib", "-nostartfiles", "-static", "-Wl,--no-relax", "-Wl,-T," + str(out / "link.ld"), "-I" + str(app), "-I" + str(out)]
        wrappers = ["-Wl,--wrap=tvm_gemmini_" + name for name in ("load_a", "load_b", "compute", "store", "validate_matmul_i8_i32")]
        programs = {}
        for name, wrong in (("valid", 0), ("bad-oracle", 1)):
            elf = out / (name + ".elf")
            command([gcc, *flags, *wrappers, "-DBAD_ORACLE=" + str(wrong), "-Wl,-Map," + str(out / (name + ".map")), out / "start.S", out / "runner.c", out / "fixture.S", *link_sources, adapter, *runtime_arguments, "-o", elf], "link-" + name, env)
            audit = audit_instructions(elf.read_bytes(), opcode, forbidden)
            if audit["prohibited_hits"]:
                raise ValueError("FSM instructions in final executable")
            report["checks"].append({"name": name + "_no_fsm", "passed": True, **audit})
            if command([tools["nm"], "-u", elf], "undefined-" + name).strip():
                raise ValueError("Unresolved runtime symbols in guest")
            if runtime_arguments:
                report["cpu_runtime"].setdefault("linked_images", {})[name] = audit_cpu_runtime(elf, out / (name + ".map"), runtime_binding, tools, command)
            programs[name] = elf
        headers = command([tools["readelf"], "-W", "-l", programs["valid"]], "program-headers")
        symbols_text = command([tools["nm"], "-S", "--defined-only", programs["valid"]], "symbols")
        segments, symbols, symbol_sizes = [], {}, {}
        for line in headers.splitlines():
            fields = line.split()
            if fields and fields[0] == "LOAD":
                segments.append({"start": int(fields[2], 16), "file_bytes": int(fields[4], 16), "memory_bytes": int(fields[5], 16)})
        for line in symbols_text.splitlines():
            fields = line.split()
            if len(fields) in (3, 4):
                try:
                    symbols[fields[-1]] = int(fields[0], 16)
                    symbol_sizes[fields[-1]] = int(fields[1], 16) if len(fields) == 4 else 0
                except ValueError:
                    pass
        if not segments or any(segment["start"] < GUEST_BASE or segment["start"] + segment["memory_bytes"] > GUEST_BASE + args.guest_memory_bytes for segment in segments):
            raise ValueError("ELF load extent exceeds requested memory")
        if symbols["_stack_top"] - symbols["_stack_bottom"] != STACK_BYTES or symbols["_image_end"] > GUEST_BASE + args.guest_memory_bytes:
            raise ValueError("Stack/image reservation mismatch")
        storage = ["model_constants", "workspace", *(f"input_{i}" for i in range(len(manifest["inputs"]))), *(f"output_{i}" for i in range(len(manifest["outputs"])))]
        if any(symbols[name] % 64 for name in storage):
            raise ValueError("Linked tensor storage is not aligned")
        fixture_symbols = []
        for kind in ("input", "output"):
            for index, tensor in enumerate(manifest[kind + "s"]):
                name = f"fixture_{kind}_{index}"
                extent = samples * aligned_size(tensor["bytes"])
                address = symbols.get(name)
                if address is None or address % 64 or symbol_sizes[name] != extent or not any(segment["start"] <= address and address + extent <= segment["start"] + segment["file_bytes"] for segment in segments):
                    raise ValueError("Linked fixture symbol extent does not match its frozen blob")
                fixture_symbols.append({"name": name, "address": address, "bytes": extent})
        report["fixture_symbols"] = fixture_symbols
        report["memory"] = {"load_segments": segments, "reserved_image_bytes": symbols["_image_end"] - GUEST_BASE,
                            "requested_guest_bytes": args.guest_memory_bytes, "stack_reserved_bytes": STACK_BYTES,
                            "workspace_bytes": manifest["workspace_bytes"], "constants_bytes": manifest["constants_bytes"],
                            "explicit_tensor_bytes": manifest["explicit_tensor_bytes"], "parameter_copies": 1,
                            "scope": "requested generic Spike map; not deployed hardware capacity"}
        simulator_env = dict(env, LD_LIBRARY_PATH=str(paths["spike_library_dir"]), PATH=str(paths["dtc"].parent) + ":/usr/bin:/bin")
        base = [paths["spike"], "--isa=rv64gc", guest_map, "-p1"]
        missing = command([*base, programs["valid"]], "without-extension", simulator_env, expected="nonzero")
        if "EXPORTED_GRAPH_INPUT_REJECTION_PASS" not in missing:
            raise ValueError("Missing-extension control did not reach device dispatch")
        if args.session_route is not None and "EXPORTED_SESSION_SPAN_REJECTION_PASS device_counts=0" not in missing:
            raise ValueError("Missing-extension control lacks session pre-dispatch rejection evidence")
        extension = [*base, "--extlib=" + str(paths["plugin"]), "--extension=gemmini"]
        valid = command([*extension, programs["valid"]], "valid", simulator_env)
        gate = f"TVM_EXPORTED_GRAPH_PASS calls={samples * args.repeats} exact_bytes=1 guards=1 inputs=1 constants=1 retained=1 poisoned_workspace=1 primitive_counts=1"
        if gate not in valid or valid.count("EXPORTED_GRAPH_CALL_PASS") != samples * args.repeats:
            raise ValueError("Complete stream numerical/lifetime gate absent")
        if args.session_route is not None and valid.count("EXPORTED_SESSION_STAGE_PASS") != samples * args.repeats * len(calls):
            raise ValueError("Complete fixed-sequence stage execution evidence absent")
        if args.session_route is not None and "EXPORTED_SESSION_SPAN_REJECTION_PASS device_counts=0" not in valid:
            raise ValueError("Session pre-dispatch span rejection evidence absent")
        high_water = next((line.partition("=")[2] for line in valid.splitlines() if line.startswith("STACK_HIGH_WATER_BYTES=")), None)
        if high_water is None or not 0 < int(high_water) < STACK_BYTES:
            raise ValueError("Stack high-water evidence absent or out of bounds")
        report["memory"]["stack_high_water_bytes"] = int(high_water)
        wrong = command([*extension, programs["bad-oracle"]], "bad-oracle", simulator_env, expected=22)
        if "EXPORTED_GRAPH_NUMERICAL_FAILURE" not in wrong:
            raise ValueError("Wrong-oracle control did not reach numerical gate")
        report["checks"].append({"name": "exported_graph_runtime", "passed": True, "samples": samples, "repeats": args.repeats,
                                 "invocations": samples * args.repeats, "exact_bytes": True, "primitive_counts": True,
                                 "inputs_preserved": True, "constants_preserved": True, "retained_outputs": True,
                                 "workspace_poisoned_each_call": True, "storage_canaries": True, "bad_oracle_exit": 22})
        if args.session_route is not None:
            report["checks"].append({"name": "fixed_sequence_runtime", "passed": True, "sequences": samples * args.repeats,
                                     "stage_invocations": samples * args.repeats * len(calls), "every_output_compared": True,
                                     "all_external_inputs_preserved_each_stage": True, "all_other_outputs_preserved_each_stage": True,
                                     "workspace_poisoned_each_stage": True, "per_stage_primitive_counts": True,
                                     "span_rejections_before_dispatch": True, "reset_from_external_fixtures": True,
                                     "full_model_qualified": False, "production_memory_plan": False})
        for path, digest in immutable.items():
            if identity(path)["sha256"] != digest:
                raise ValueError("Source/tool/fixture changed during verification")
        report["immutable_bindings"] = immutable
        report["artifacts"] = [identity(out / name) for name in ("runner.c", "fixture.h", "fixture.S", "valid.elf", "bad-oracle.elf", "start.S", "link.ld")]
        report.update(status="passed", simulator_executed=True)
    except Exception as error:
        report.update(status="failed", error=str(error))
        (out / "traceback.txt").write_text(traceback.format_exc())
        raise
    finally:
        save()
    print(json.dumps({"status": report["status"], "receipt": str(out / "receipt.json")}), flush=True)


if __name__ == "__main__":
    main()
