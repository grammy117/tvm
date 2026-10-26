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
"""Per-operator cycle measurement for Gemmini MetaSchedule tuning.

Each candidate becomes one positive and one wrong-oracle ELF. The guest calls
the exported operator once, and a second time when the first call is short, and
reports both guest cycle intervals. It then verifies every output against a host
oracle with canaries and checks that inputs are unchanged.

Spike executes the verification: the positive ELF must pass and the
wrong-oracle ELF must fail numerically. Verilator executes the identical
positive ELF for timing and output verification; on the RTL core (marchid 1)
the guest skips input hashing and guard initialization. Inputs
and oracle values are linked as binary blobs, so operator size does not grow
generated C source.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import tvm
from tvm import relax, tir

APP = Path(__file__).resolve().parent
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from baremetal import RV64_TARGET, export_graph  # pylint: disable=wrong-import-position
from verify_baremetal_cpu import STARTUP, identity  # pylint: disable=wrong-import-position
from verify_matmul import audit_instructions  # pylint: disable=wrong-import-position
from memory_model import bind_memory_model, confirm_memory_model, runtime_environment
from cycle_watchdog import PROFILE as WATCHDOG_PROFILE, instrument_runner

ROCKET_MARCHID = 1
GUEST_BASE, GUEST_BYTES = 0x80000000, 0x10000000
# A first call shorter than this is repeated once warm; longer calls are timed
# cold, where instruction-cache misses are a negligible share of the interval.
WARM_LIMIT = 400000

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
  /* Operator outputs and workspace: written by the guest before use, so the
   * startup code does not spend simulated cycles zeroing them. */
  .noinit ALIGN(64) (NOLOAD) : { *(.noinit .noinit.*) } > DRAM
  .stack ALIGN(16) (NOLOAD) : {
    _stack_bottom = .;
    . += 16384;
    _stack_top = .;
  } > DRAM
  _image_end = .;
  ASSERT(_image_end <= ORIGIN(DRAM) + LENGTH(DRAM), "image exceeds guest DRAM")
}
"""

RUNNER = r"""
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
__attribute__((weak)) void *memcpy(void *d,const void *s,size_t n){uint8_t *a=d;const uint8_t *b=s;for(size_t i=0;i<n;++i)a[i]=b[i];return d;}
__attribute__((weak)) void *memmove(void *d,const void *s,size_t n){uint8_t *a=d;const uint8_t *b=s;if((uintptr_t)a<(uintptr_t)b){for(size_t i=0;i<n;++i)a[i]=b[i];}else{while(n){--n;a[n]=b[n];}}return d;}
__attribute__((weak)) void *memset(void *d,int x,size_t n){uint8_t *a=d;for(size_t i=0;i<n;++i)a[i]=(uint8_t)x;return d;}
__attribute__((weak)) int memcmp(const void *a,const void *b,size_t n){const uint8_t *x=a,*y=b;for(size_t i=0;i<n;++i)if(x[i]!=y[i])return (int)x[i]-(int)y[i];return 0;}
static int guest_errno;
int *__errno(void){return &guest_errno;}
int printf(const char *format,...){(void)format;puts_htif("VENDOR_DIAGNOSTIC\n");return 0;}
__attribute__((noreturn)) void exit(int code){tohost=((uint64_t)(code?code:99)<<1)|1;for(;;){}}
static uint64_t hash(const void* p,size_t n) {
  const unsigned char* b=p;uint64_t h=1469598103934665603ULL;
  for(size_t i=0;i<n;++i)h=(h^b[i])*1099511628211ULL;return h;
}
static inline uint64_t cycles(void){uint64_t v;asm volatile("fence rw,rw; csrr %0,cycle":"=r"(v)::"memory");return v;}
static char* put_u64(char* out,uint64_t value){
  char text[24];unsigned used=0;do{text[used++]=(char)('0'+value%10);value/=10;}while(value);
  while(used)*out++=text[--used];return out;
}
extern int32_t model_run(const void *const *inputs, void *const *outputs, uint8_t *workspace, uint64_t workspace_bytes);
extern const uint8_t dse_inputs[];
extern const uint8_t dse_expected[];
#define GUARD 64
static uint8_t outputs_area[OUTPUT_AREA] __attribute__((section(".noinit"),aligned(64)));
static uint8_t workspace_area[WORKSPACE_AREA] __attribute__((section(".noinit"),aligned(64)));
static const uint64_t input_offsets[]={INPUT_OFFSETS};
static const uint64_t input_bytes[]={INPUT_BYTES};
static const uint64_t input_hashes[]={INPUT_HASHES};
static const uint64_t output_offsets[]={OUTPUT_OFFSETS};
static const uint64_t output_bytes[]={OUTPUT_BYTES};
static const uint64_t expected_offsets[]={EXPECTED_OFFSETS};
static const uint8_t output_kinds[]={OUTPUT_KINDS};  /* 0 integer bytes, 1 float32 */
static int compare(unsigned index){
  const uint8_t* actual=outputs_area+output_offsets[index];
  const uint8_t* expected=dse_expected+expected_offsets[index];
  uint64_t n=output_bytes[index];
  if(output_kinds[index]==0){
    for(uint64_t i=0;i<n;++i)if(actual[i]!=(uint8_t)(expected[i]^(i==0?BAD_ORACLE:0)))return 1;
    return 0;
  }
  for(uint64_t i=0;i<n/4;++i){
    union{uint32_t u;float f;}a,e;
    a.u=((const uint32_t*)actual)[i];e.u=((const uint32_t*)expected)[i]^(BAD_ORACLE?UINT32_C(0x40000000):0);
    /* Reject nonfinite values before tolerance bounds can overflow. */
    if((a.u&UINT32_C(0x7f800000))==UINT32_C(0x7f800000) ||
       (e.u&UINT32_C(0x7f800000))==UINT32_C(0x7f800000))return 1;
    /* BAD_ORACLE has already changed e.u. Valid configured tolerances are
       nonnegative; preserve the old fallback for other configurations. */
    if(a.u==e.u && ATOL>=0 && RTOL>=0)continue;
    float tolerance=ATOL+RTOL*(e.f<0?-e.f:e.f);
    if(!(a.f>=e.f-tolerance && a.f<=e.f+tolerance))return 1;
  }
  return 0;
}
int main(void){
  uint64_t arch;asm volatile("csrr %0,marchid":"=r"(arch));
  int rtl=arch==ROCKET_MARCHID;
  const void* inputs[NUM_INPUTS+1];void* outputs[NUM_OUTPUTS+1];
  for(unsigned i=0;i<NUM_INPUTS;++i)inputs[i]=dse_inputs+input_offsets[i];
  for(unsigned i=0;i<NUM_OUTPUTS;++i)outputs[i]=outputs_area+output_offsets[i];
  if(!rtl){memset(outputs_area,0x6d,sizeof(outputs_area));memset(workspace_area,0x6d,sizeof(workspace_area));}
  uint64_t begin=cycles();
  if(model_run(inputs,outputs,workspace_area+GUARD,WORKSPACE_BYTES))return 20;
  uint64_t first=cycles()-begin,second=0;
  if(first<WARM_LIMIT){begin=cycles();if(model_run(inputs,outputs,workspace_area+GUARD,WORKSPACE_BYTES))return 20;second=cycles()-begin;}
  char line[96];char* out=line;const char* text="DSE_CYCLES first=";
  while(*text)*out++=*text++;out=put_u64(out,first);text=" second=";
  while(*text)*out++=*text++;out=put_u64(out,second);*out++='\n';*out=0;puts_htif(line);
  for(unsigned i=0;i<NUM_OUTPUTS;++i)if(compare(i)){puts_htif("TENSORIR_NUMERICAL_FAILURE\n");return 22;}
  if(rtl){puts_htif("DSE_PASS\n");return 0;}
  for(unsigned i=0;i<GUARD;++i)if(workspace_area[i]!=0x6d||workspace_area[sizeof(workspace_area)-1-i]!=0x6d)return 23;
  for(unsigned i=0;i<NUM_OUTPUTS;++i){
    const uint8_t* p=outputs_area+output_offsets[i];
    for(unsigned g=1;g<=GUARD;++g)if(p[-(int)g]!=0x6d||p[output_bytes[i]+g-1]!=0x6d)return 23;
  }
  for(unsigned i=0;i<NUM_INPUTS;++i)if(hash(dse_inputs+input_offsets[i],input_bytes[i])!=input_hashes[i])return 24;
  puts_htif("DSE_PASS\n");
  return 0;
}
"""


def render_runner(values, timing_policy="legacy", *, watchdog=False):
    """Share the checked guest with connected-package measurements."""
    if timing_policy not in ("legacy", "fixed-warm"):
        raise ValueError("Unknown timing policy")
    values = dict(values, ROCKET_MARCHID=ROCKET_MARCHID,
                  WARM_LIMIT="UINT64_MAX" if timing_policy == "fixed-warm" else WARM_LIMIT)
    runner = RUNNER
    if watchdog:
        if timing_policy != "fixed-warm":
            raise ValueError("Measured-region watchdog requires fixed-warm timing")
        runner = instrument_runner(runner)
    for key in sorted(values, key=len, reverse=True):
        runner = runner.replace(key, str(values[key]))
    return runner


def _aligned(size, alignment=64):
    return (size + alignment - 1) // alignment * alignment


def _fnv(data):
    value = 1469598103934665603
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


def written_parameters(func):
    """Parameter positions the function writes, by store or write access pointer."""
    written = []

    def visit(node):
        if isinstance(node, tir.BufferStore):
            written.append(node.buffer.data)
        elif (
            isinstance(node, tir.Call)
            and node.op.same_as(tvm.ir.Op.get("tir.tvm_access_ptr"))
            and int(node.args[4]) & 2
        ):
            written.append(node.args[1])

    tir.stmt_functor.post_order_visit(func.body, visit)
    return [
        index
        for index, param in enumerate(func.params)
        if any(func.buffer_map[param].data.same_as(var) for var in written)
    ]


def random_array(shape, dtype, rng):
    if dtype == "int8":
        return rng.integers(-128, 128, shape, dtype="int8")
    if dtype in ("int32", "int64"):
        return rng.integers(-(1 << 20), 1 << 20, shape).astype(dtype)
    if dtype == "uint8":
        return rng.integers(0, 256, shape, dtype="uint8")
    if dtype == "bool":
        return rng.integers(0, 2, shape).astype("bool")
    if dtype == "float32":
        # Positive scales in the ranges dequantization recipes use.
        return (rng.uniform(0.5, 2.0, shape) / 64).astype("float32")
    raise ValueError(f"No synthetic generator for {dtype}")


@dataclass
class OperatorWorkload:
    """One task's fixed inputs and host oracle, shared by all its candidates."""

    folder: Path
    inputs: list
    expected: list
    output_indices: list
    rtol: float
    atol: float

    @property
    def manifest(self):
        return json.loads((self.folder / "workload-manifest.json").read_text())


def input_binding_identity(func, input_overrides=None):
    """Validate exact parameter bindings and describe their numerical bytes."""
    outputs = written_parameters(func)
    bindings = {}
    for index, value in (input_overrides or {}).items():
        if type(index) is not int or index < 0 or index >= len(func.params) or index in outputs:
            raise ValueError("Input binding must name a read-only parameter index")
        buffer = func.buffer_map[func.params[index]]
        array = np.asarray(value)
        if array.shape != tuple(int(x) for x in buffer.shape) or array.dtype.name != buffer.dtype:
            raise ValueError("Input binding shape/dtype differs from the mathematical workload")
        bindings[str(index)] = {"shape": list(array.shape), "dtype": array.dtype.name,
                                "sha256": hashlib.sha256(array.tobytes()).hexdigest()}
    return bindings


def prepare_workload(func, folder, *, seed=0, rtol=1e-5, atol=1e-6,
                     input_overrides=None, input_provenance=None):
    """Generate unbound inputs and evaluate unscheduled mathematical TIR.

    Lifted immutable graph constants and supplied boundary activations retain
    their exact bytes. Randomizing a divisor or other domain constraint can
    make an otherwise legal extracted operator undefined.
    """
    bound_inputs = input_binding_identity(func, input_overrides)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    outputs = written_parameters(func)
    if not outputs:
        raise ValueError("Workload writes no parameter")
    buffers = [func.buffer_map[param] for param in func.params]
    rng = np.random.default_rng(seed)
    arrays = []
    for index, buffer in enumerate(buffers):
        shape = [int(x) for x in buffer.shape]
        if index in outputs:
            arrays.append(np.zeros(shape, buffer.dtype))
        elif input_overrides is not None and index in input_overrides:
            arrays.append(np.array(input_overrides[index], copy=True))
        else:
            arrays.append(random_array(shape, buffer.dtype, rng))
    from operator_inputs import check_integer_divisors
    check_integer_divisors(func, [None if i in outputs else array for i, array in enumerate(arrays)])
    reference = tvm.build(tvm.IRModule({"main": func.with_attr("global_symbol", "main")}), "llvm")
    devices = [tvm.nd.array(array) for array in arrays]
    reference["main"](*devices)
    inputs = [arrays[i] for i in range(len(arrays)) if i not in outputs]
    expected = [devices[i].numpy() for i in outputs]
    blob = bytearray()
    for array in inputs:
        blob += array.tobytes()
        blob += bytes(_aligned(len(blob)) - len(blob))
    (folder / "inputs.bin").write_bytes(bytes(blob) or b"\0")
    blob = bytearray()
    for array in expected:
        blob += array.tobytes()
        blob += bytes(_aligned(len(blob)) - len(blob))
    (folder / "expected.bin").write_bytes(bytes(blob) or b"\0")
    (folder / "workload.json").write_text(tvm.ir.save_json(tvm.IRModule({"main": func})))
    manifest = {
        "seed": seed,
        "outputs": outputs,
        "buffers": [{"shape": [int(x) for x in b.shape], "dtype": b.dtype} for b in buffers],
        "rtol": rtol,
        "atol": atol,
        "inputs_sha256": hashlib.sha256((folder / "inputs.bin").read_bytes()).hexdigest(),
        "expected_sha256": hashlib.sha256((folder / "expected.bin").read_bytes()).hexdigest(),
    }
    if bound_inputs or input_provenance is not None:
        manifest.update(bound_inputs=bound_inputs, input_provenance=input_provenance)
    (folder / "workload-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return OperatorWorkload(folder, inputs, expected, outputs, rtol, atol)


def load_workload(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "workload-manifest.json").read_text())
    return manifest


@dataclass(frozen=True)
class Toolchain:
    """Pinned toolchain, adapter object, simulators and their receipts."""

    gcc: Path
    adapter_receipt: Path
    spike_receipt: Path
    hardware_receipt: Path
    memory_variant_receipt: Path | None = None
    firesim_memory_artifact: str | None = None
    memory_pin_registry: Path | None = None
    timing_policy: str = "legacy"
    cycle_watchdog: bool = False

    @property
    def adapter(self):
        receipt = json.loads(self.adapter_receipt.read_text())
        obj = self.adapter_receipt.parent / "matmul.o"
        if identity(obj)["sha256"] != receipt["object_sha256"]:
            raise ValueError("Adapter object does not match its receipt")
        return obj, receipt

    def as_json(self):
        return {key: str(value) if isinstance(value, Path) else value
                for key, value in self.__dict__.items()}

    def memory_configuration(self):
        memory = bind_memory_model(self.hardware_receipt, self.memory_variant_receipt,
                                   self.firesim_memory_artifact, self.memory_pin_registry)
        if self.cycle_watchdog:
            receipt = json.loads(self.memory_variant_receipt.read_text())
            if receipt.get("dse_cycle_monitor", {}).get("kind") != WATCHDOG_PROFILE["kind"]:
                raise ValueError("Moving cutoff requires the source-bound native cycle monitor")
        return memory

    def provenance(self):
        """Verify the actual executable bytes against the retained qualifications."""
        from tvm.relax.backend.contrib.gemmini_target import GemminiCapabilities

        hardware = json.loads(self.hardware_receipt.read_text())
        spike = json.loads(self.spike_receipt.read_text())
        adapter_object, adapter = self.adapter
        caps = GemminiCapabilities()
        if adapter["headers"]["include/gemmini_params.h"]["sha256"] != caps.params_sha256:
            raise ValueError("Adapter header differs from the compiler capability profile")
        if adapter["hardware_source_check"]["configs_sha256"] != caps.config_sha256:
            raise ValueError("Adapter hardware differs from the compiler capability profile")
        identities = {
            "simulator": identity(hardware["simulator"]["path"]),
            "spike": identity(spike["spike_path"]),
            "plugin": identity(spike["plugin_path"]),
            "adapter": identity(adapter_object),
            "gcc": identity(self.gcc),
        }
        for name, expected in (("simulator", hardware["simulator"]["sha256"]),
                               ("spike", spike["spike_sha256"]),
                               ("plugin", spike["plugin_sha256"])):
            if identities[name]["sha256"] != expected:
                raise ValueError(f"{name} differs from its qualification receipt")
        memory = self.memory_configuration()
        if self.memory_variant_receipt:
            identities["base_simulator"] = identities["simulator"]
            identities["simulator"] = memory["simulator"]
        return {"hardware": hardware["hardware"], "binaries": identities,
                "memory_model": memory, "timing_policy": self.timing_policy,
                "cycle_watchdog": WATCHDOG_PROFILE if self.cycle_watchdog else None,
                "adapter_sources": adapter["source_set"],
                "receipts": [identity(p) for p in (self.hardware_receipt, self.spike_receipt,
                                                  self.adapter_receipt)]}

    @classmethod
    def from_json(cls, value):
        return cls(**{key: Path(item) if item is not None and key not in
                     ("firesim_memory_artifact", "timing_policy", "cycle_watchdog") else item
                      for key, item in value.items()})


def _relax_graph(func):
    buffers = [func.buffer_map[param] for param in func.params]
    outputs = written_parameters(func)
    builder = relax.BlockBuilder()
    gv = builder.add_func(func.with_attr("global_symbol", "kernel"), "kernel")
    inputs = [
        relax.Var(f"input_{i}", relax.TensorStructInfo(buffer.shape, buffer.dtype))
        for i, buffer in enumerate(buffers)
        if i not in outputs
    ]
    sinfo = [relax.TensorStructInfo(buffers[i].shape, buffers[i].dtype) for i in outputs]
    with builder.function("main", inputs):
        with builder.dataflow():
            result = builder.emit_output(
                relax.call_tir(gv, inputs, out_sinfo=sinfo if len(sinfo) > 1 else sinfo[0])
            )
        builder.emit_func_output(result)
    return builder.get()


def build_candidate(mod, workload_folder, folder, toolchain, *, timeout=300):
    """Lower one MetaSchedule candidate and link its positive/wrong-oracle ELFs."""
    from tvm.relax.backend.contrib.gemmini_tensorir import finalize_gemmini_function

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    workload_folder = Path(workload_folder)
    manifest = json.loads((workload_folder / "workload-manifest.json").read_text())
    func = mod["main"]
    native = "local.gemmini_" in func.script()
    if native:
        func = finalize_gemmini_function(func)
    graph = _relax_graph(func)
    package = export_graph(graph, folder / "package", target=RV64_TARGET, memory_limit_bytes=1 << 28)
    buffers = manifest["buffers"]
    outputs = manifest["outputs"]
    inputs = [i for i in range(len(buffers)) if i not in outputs]

    def nbytes(spec):
        return int(np.prod(spec["shape"], dtype=np.int64)) * np.dtype(spec["dtype"]).itemsize

    input_offsets, offset = [], 0
    for i in inputs:
        input_offsets.append(offset)
        offset = _aligned(offset + nbytes(buffers[i]))
    blob = (workload_folder / "inputs.bin").read_bytes()
    hashes = [
        _fnv(blob[o : o + nbytes(buffers[i])]) for o, i in zip(input_offsets, inputs)
    ]
    output_offsets, expected_offsets, offset, expected_offset = [], [], 64, 0
    for i in outputs:
        output_offsets.append(offset)
        offset = _aligned(offset + nbytes(buffers[i]) + 64) + 64
        expected_offsets.append(expected_offset)
        expected_offset = _aligned(expected_offset + nbytes(buffers[i]))
    kinds = [1 if buffers[i]["dtype"] == "float32" else 0 for i in outputs]
    workspace = package["workspace_bytes"]
    values = {
        "OUTPUT_AREA": offset + 64,
        "WORKSPACE_AREA": _aligned(workspace) + 128,
        "WORKSPACE_BYTES": workspace,
        "NUM_INPUTS": len(inputs),
        "NUM_OUTPUTS": len(outputs),
        "INPUT_OFFSETS": ",".join(map(str, input_offsets)) or "0",
        "INPUT_BYTES": ",".join(str(nbytes(buffers[i])) for i in inputs) or "0",
        "INPUT_HASHES": ",".join(f"UINT64_C({h})" for h in hashes) or "0",
        "OUTPUT_OFFSETS": ",".join(map(str, output_offsets)),
        "OUTPUT_BYTES": ",".join(str(nbytes(buffers[i])) for i in outputs),
        "EXPECTED_OFFSETS": ",".join(map(str, expected_offsets)),
        "OUTPUT_KINDS": ",".join(map(str, kinds)),
        "RTOL": f"{float(manifest['rtol']).hex()}f",
        "ATOL": f"{float(manifest['atol']).hex()}f",
        "WARM_LIMIT": WARM_LIMIT,
        "ROCKET_MARCHID": ROCKET_MARCHID,
    }
    runner = render_runner(values, toolchain.timing_policy, watchdog=toolchain.cycle_watchdog)
    (folder / "runner.c").write_text(runner)
    (folder / "start.S").write_text(STARTUP)
    (folder / "link.ld").write_text(LINKER)
    (folder / "data.S").write_text(
        '.section .rodata.dse,"a",@progbits\n.balign 64\n.globl dse_inputs\ndse_inputs:\n'
        f'.incbin "{(workload_folder / "inputs.bin").resolve()}"\n.balign 64\n.globl dse_expected\n'
        f'dse_expected:\n.incbin "{(workload_folder / "expected.bin").resolve()}"\n'
        '.section .note.GNU-stack,"",@progbits\n'
    )
    adapter, receipt = toolchain.adapter
    gcc = toolchain.gcc
    flags = [
        "-std=gnu11", "-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany", "-msmall-data-limit=0",
        "-O2", "-ffreestanding", "-fno-builtin", "-fno-stack-protector",
        "-fno-tree-loop-distribute-patterns", "-nostdlib", "-nostartfiles", "-static",
        "-Wl,--no-relax", "-Wl,-T," + str(folder / "link.ld"),
    ]
    libgcc = subprocess.check_output(
        [str(gcc), "-march=rv64gc", "-mabi=lp64d", "-print-libgcc-file-name"], text=True
    ).strip()
    sources = [
        folder / "start.S", folder / "runner.c", folder / "data.S",
        folder / "package" / "model.c", folder / "package" / "operators.o",
        folder / "package" / "constants.S",
    ]
    policy = receipt["instruction_policy"]
    forbidden = {v for k, v in policy["derived_definitions"].items() if k.startswith("k_LOOP_")}
    elfs = {}
    for label, wrong in (("valid", 0), ("bad-oracle", 1)):
        elf = folder / f"{label}.elf"
        process = subprocess.run(
            [str(gcc), *flags, f"-DBAD_ORACLE={wrong}", *map(str, sources), str(adapter), libgcc, "-o", str(elf)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout,
        )
        if process.returncode:
            raise RuntimeError(process.stdout[-4000:])
        audit = audit_instructions(elf.read_bytes(), policy["custom_opcode"], forbidden)
        if audit["prohibited_hits"]:
            raise ValueError("Hardware-loop instructions in final ELF")
        nm = gcc.with_name(gcc.name[:-3] + "nm")
        if subprocess.check_output([str(nm), "-u", str(elf)], text=True).strip():
            raise ValueError("Unresolved guest symbols")
        elfs[label] = identity(elf)
    artifact = {
        "folder": str(folder),
        "native": native,
        "elfs": elfs,
        "workload": str(workload_folder),
        "program_sha256": hashlib.sha256((folder / "package" / "operators.ll").read_bytes()).hexdigest(),
        "compiler": identity(tvm._ffi.base._LIB._name),
        "runner_sha256": hashlib.sha256((folder / "runner.c").read_bytes()).hexdigest(),
        "adapter_sha256": receipt["object_sha256"],
        "timing_policy": toolchain.timing_policy,
        "memory_model": toolchain.memory_configuration(),
        "cycle_watchdog": dict(WATCHDOG_PROFILE, enabled=True, limit_cycles=0)
                          if toolchain.cycle_watchdog else {"enabled": False},
    }
    (folder / "artifact.json").write_text(json.dumps(artifact, indent=2) + "\n")
    return artifact


def _spike(elf, toolchain, *, timeout):
    receipt = json.loads(toolchain.spike_receipt.read_text())
    env = dict(os.environ)
    for key, value in receipt["runtime_environment"].items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = str(value)
    command = [
        receipt["spike_path"], "--extlib=" + receipt["plugin_path"], "--extension=gemmini",
        f"-m{GUEST_BASE:#x}:{GUEST_BYTES:#x}", str(elf),
    ]
    try:
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    return process.returncode, process.stdout.decode(errors="replace")


def _parse_cycles(text):
    for line in text.splitlines():
        if line.startswith("DSE_CYCLES "):
            fields = dict(field.split("=", 1) for field in line.split()[1:])
            return int(fields["first"]), int(fields["second"])
    return None


def run_candidate(artifact, toolchain, *, max_cycles, cycles_per_second=1500, spike_seconds=900,
                  rtl_seconds=None):
    """Spike positive and wrong-oracle checks, then RTL timing of the positive ELF."""
    folder = Path(artifact["folder"])
    memory = toolchain.memory_configuration()
    if artifact.get("memory_model", {}).get("fingerprint", memory["fingerprint"]) != memory["fingerprint"]:
        raise ValueError("Candidate memory model differs from this measurement campaign")
    if toolchain.memory_variant_receipt and "memory_model" not in artifact:
        raise ValueError("FASED candidate must bind its memory model before measurement")
    if artifact.get("timing_policy", "legacy") != toolchain.timing_policy:
        raise ValueError("Candidate timing policy differs from this measurement campaign")
    for elf in artifact["elfs"].values():
        if identity(elf["path"])["sha256"] != elf["sha256"]:
            raise ValueError("Candidate ELF changed after compilation")
    record = {"folder": str(folder), "memory_model": memory,
              "timing_policy": toolchain.timing_policy}
    started = time.monotonic()
    code, text = _spike(artifact["elfs"]["valid"]["path"], toolchain, timeout=spike_seconds)
    (folder / "spike-valid.log").write_text(text)
    if code != 0 or "DSE_PASS" not in text.splitlines():
        record.update(status="spike_positive_failed", returncode=code)
        return record
    code, text = _spike(artifact["elfs"]["bad-oracle"]["path"], toolchain, timeout=spike_seconds)
    (folder / "spike-negative.log").write_text(text)
    if code != 22 or "TENSORIR_NUMERICAL_FAILURE" not in text:
        record.update(status="spike_negative_not_detected", returncode=code)
        return record
    record["spike_seconds"] = time.monotonic() - started
    simulator = memory["simulator"]["path"]
    elf = artifact["elfs"]["valid"]["path"]
    command = [simulator, "+permissive", f"+max-cycles={int(max_cycles)}", f"+loadmem={elf}", "+permissive-off", elf]
    log = folder / "rtl-timing.log"
    started = time.monotonic()
    wall = rtl_seconds if rtl_seconds is not None else 60 + max_cycles / cycles_per_second
    if wall <= 0:
        raise ValueError("RTL wall limit must be positive")
    environment = runtime_environment(memory)
    environment.pop("MERLIN_DSE_CYCLE_LIMIT_FILE", None)
    if getattr(toolchain, "cycle_watchdog", False):
        environment["MERLIN_DSE_CYCLE_LIMIT_FILE"] = artifact["cycle_watchdog"]["cutoff_file"]
    with log.open("wb") as stream:
        try:
            process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=wall,
                                     env=environment)
            code = process.returncode
        except subprocess.TimeoutExpired:
            code = None
    text = log.read_text(errors="replace")
    record["rtl_seconds"] = time.monotonic() - started
    confirmation = confirm_memory_model(memory, text)
    record.update(memory_confirmation=confirmation, command=command, wall_limit_seconds=wall,
                  cycle_cap=int(max_cycles), rtl_log=identity(log))
    record["cycle_watchdog"] = artifact.get("cycle_watchdog", {"enabled": False})
    if not confirmation["model_confirmed"]:
        record.update(status="memory_model_not_confirmed", returncode=code)
        return record
    if getattr(toolchain, "cycle_watchdog", False) and "[dse-cycle-monitor] enabled" not in text:
        record.update(status="cycle_monitor_not_confirmed", returncode=code)
        return record
    if "DSE_PRUNED limit=" in text:
        record.update(status="pruned_by_incumbent", returncode=code)
        return record
    cycles = _parse_cycles(text)
    if code != 0 or cycles is None or "Verilog $finish" not in text or "DSE_PASS" not in text:
        record.update(status="rtl_timeout_or_failure", returncode=code, cycle_cap=int(max_cycles))
        return record
    first, second = cycles
    if toolchain.timing_policy == "fixed-warm" and not second:
        record.update(status="missing_fixed_warm_measurement", returncode=code)
        return record
    record.update(
        status="passed",
        cycles=second or first,
        first_call_cycles=first,
        warm_call_cycles=second or None,
        timing="warm second call" if second else "single cold call (long operator)",
        rtl_elf=artifact["elfs"]["valid"],
    )
    return record


def run_negative_rtl(artifact, toolchain, *, max_cycles=6_000_000):
    """Require the RTL guest to detect its deliberately corrupted oracle."""
    folder = Path(artifact["folder"])
    elf = artifact["elfs"]["bad-oracle"]
    if identity(elf["path"])["sha256"] != elf["sha256"]:
        raise ValueError("Wrong-oracle ELF changed after compilation")
    provenance = toolchain.provenance()
    memory = provenance["memory_model"]
    simulator = provenance["binaries"]["simulator"]["path"]
    command = [simulator, "+permissive", f"+max-cycles={max_cycles}",
               f"+loadmem={elf['path']}", "+permissive-off", elf["path"]]
    log = folder / "rtl-negative.log"
    started = time.monotonic()
    with log.open("wb") as stream:
        try:
            process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                     timeout=60 + max_cycles / 1500, env=runtime_environment(memory))
            code = process.returncode
        except subprocess.TimeoutExpired:
            code = None
    text = log.read_text(errors="replace")
    confirmation = confirm_memory_model(memory, text)
    detected = (confirmation["model_confirmed"] and code not in (None, 0) and "TENSORIR_NUMERICAL_FAILURE" in text
                and _parse_cycles(text) is not None and "(timeout)" not in text)
    record = {"status": "passed" if detected else "negative_not_detected",
              "returncode": code, "elf": elf, "log": identity(log), "command": command,
              "wall_seconds": time.monotonic() - started,
              "memory_confirmation": confirmation,
              "hardware_provenance": provenance}
    (folder / "rtl-negative.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    """Build one candidate in a separate process (killable, isolated)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build",))
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    mod = tvm.ir.load_json(Path(request["module"]).read_text())
    toolchain = Toolchain.from_json(request["toolchain"])
    try:
        artifact = build_candidate(mod, request["workload"], request["folder"], toolchain)
    except Exception as error:  # pylint: disable=broad-except
        import traceback

        Path(request["result"]).write_text(
            json.dumps({"error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()})
        )
        raise SystemExit(1)
    Path(request["result"]).write_text(json.dumps(artifact))


if __name__ == "__main__":
    main()
