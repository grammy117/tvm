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
"""Export straight-line, static Relax call_tir graphs without a guest VM or heap.

Graph optimization and accelerator scheduling must precede export. This module
only derives calls, constants and buffer lifetimes; it does not implement tensor
operators. Unsupported control flow, dynamic tensors and hidden scalar ABIs fail
explicitly. The caller supplies aligned, disjoint input/output/workspace storage.
Each call_tir operator must synchronously complete its outputs, preserve inputs
and constants, and retain no buffer pointers after return. These are compiler/
operator contracts, not a static proof of arbitrary TIR or external-call effects.
"""

from dataclasses import dataclass
import hashlib
import json
import math
import re

import numpy as np
import tvm
from tvm import relax, tir
from verify_baremetal_cpu import checked_path

ALIGNMENT = 64
DTYPES = {"int8": "int8_t", "uint8": "uint8_t", "int16": "int16_t", "int32": "int32_t", "int64": "int64_t", "float16": "uint16_t", "float32": "float", "float64": "double", "bfloat16": "uint16_t", "bool": "uint8_t"}
# Float islands keep each multiply and add separately rounded, as unfused references compute them.
RV64_TARGET = "llvm -mtriple=riscv64-unknown-elf -mcpu=generic-rv64 -mattr=+m,+a,+f,+d,+c -mabi=lp64d -fp-contract=off"


def _aligned(size):
    return (size + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


@dataclass(eq=False)
class Tensor:
    name: str
    shape: tuple
    dtype: str
    origin: str
    birth: int = -1
    last_use: int = -1
    offset: int = 0
    value: object = None
    alias_of: object = None

    @property
    def storage(self):
        return self if self.alias_of is None else self.alias_of.storage

    @property
    def nbytes(self):
        return math.prod(self.shape) * (2 if self.dtype == "bfloat16" else np.dtype(self.dtype).itemsize)

    def metadata(self):
        record = {"name": self.name, "shape": list(self.shape), "dtype": self.dtype, "bytes": self.nbytes, "origin": self.origin, "offset": self.offset, "birth": self.birth, "last_use": self.last_use}
        if self.alias_of is not None:
            record["alias_of"] = self.storage.name
        return record


def _tensor(info, name, origin, birth=-1):
    if not isinstance(info, relax.TensorStructInfo) or info.dtype not in DTYPES:
        raise ValueError("unsupported static tensor dtype or structure")
    if not isinstance(info.shape, relax.ShapeExpr) or any(not isinstance(d, tir.IntImm) or int(d) <= 0 for d in info.shape):
        raise ValueError("tensor dimensions must be positive static integers")
    if info.vdevice is not None and info.vdevice.target.kind.name not in ("llvm", "c"):
        raise ValueError("only host-addressed tensor buffers are supported")
    result = Tensor(name, tuple(int(d) for d in info.shape), info.dtype, origin, birth=birth, last_use=birth)
    if result.nbytes > (1 << 63) - 1:
        raise ValueError("tensor byte extent overflows the target pointer ABI")
    return result


def _flatten(value):
    if isinstance(value, Tensor):
        return [value]
    if isinstance(value, tuple):
        return [tensor for item in value for tensor in _flatten(item)]
    raise ValueError("only tensor and nested tensor-tuple values are supported")


def _allocate_workspace(tensors):
    """Reuse aligned storage only after a tensor's final consuming call."""
    free, active, high_water = [], [], 0
    for tensor in tensors:
        survivors = []
        for previous in active:
            if previous.last_use < tensor.birth:
                free.append((previous.offset, _aligned(previous.nbytes)))
            else:
                survivors.append(previous)
        active = survivors
        merged = []
        for offset, size in sorted(free):
            if merged and merged[-1][0] + merged[-1][1] == offset:
                merged[-1] = (merged[-1][0], merged[-1][1] + size)
            else:
                merged.append((offset, size))
        free = merged
        size = _aligned(tensor.nbytes)
        suitable = [(extent, offset, index) for index, (offset, extent) in enumerate(free) if extent >= size]
        if suitable:
            extent, offset, index = min(suitable)
            free.pop(index)
            if extent > size:
                free.append((offset + size, extent - size))
        else:
            offset = high_water
            high_water += size
        tensor.offset = offset
        active.append(tensor)
    return high_water


def plan_graph(mod):
    """Validate the graph's raw-pointer ABI and derive a reusable storage plan."""
    function = mod["main"]
    if not isinstance(function, relax.Function):
        raise ValueError("require a straight-line Relax main function")
    values, inputs, constants, tensors, calls, functions = {}, [], [], [], [], {}
    views = []
    for index, param in enumerate(function.params):
        tensor = _tensor(param.struct_info, f"input_{index}", "input")
        values[param] = tensor
        inputs.append(tensor)

    def resolve(expr):
        if isinstance(expr, relax.Var) and expr in values:
            return values[expr]
        if isinstance(expr, relax.Constant):
            if expr not in values:
                tensor = _tensor(expr.struct_info, f"constant_{len(constants)}", "constant")
                tensor.value = expr.data.numpy()
                values[expr] = tensor
                constants.append(tensor)
            return values[expr]
        if isinstance(expr, relax.Tuple):
            return tuple(resolve(field) for field in expr.fields)
        if isinstance(expr, relax.TupleGetItem):
            source = resolve(expr.tuple_value)
            if not isinstance(source, tuple) or not 0 <= expr.index < len(source):
                raise ValueError("invalid static tuple selection")
            return source[expr.index]
        raise ValueError("unsupported graph value or unbound variable")

    def output_value(info, call_index):
        if isinstance(info, relax.TupleStructInfo):
            return tuple(output_value(field, call_index) for field in info.fields)
        tensor = _tensor(info, f"tensor_{len(tensors)}", "workspace", call_index)
        tensors.append(tensor)
        return tensor

    def mark_use(tensor, index):
        tensor.last_use = max(tensor.last_use, index)
        tensor.storage.last_use = max(tensor.storage.last_use, index)

    blocks = function.body.blocks if isinstance(function.body, relax.SeqExpr) else []
    result_expr = function.body.body if isinstance(function.body, relax.SeqExpr) else function.body
    for block in blocks:
        if not isinstance(block, (relax.BindingBlock, relax.DataflowBlock)):
            raise ValueError("unsupported graph binding block")
        for binding in block.bindings:
            if not isinstance(binding, relax.VarBinding):
                raise ValueError("unsupported graph binding")
            call = binding.value
            if not isinstance(call, relax.Call):
                values[binding.var] = resolve(call)
                continue
            if call.op == tvm.ir.Op.get("relax.reshape"):
                if len(call.args) != 2 or not isinstance(call.args[1], relax.ShapeExpr):
                    raise ValueError("export requires static reshape metadata")
                source = resolve(call.args[0])
                if not isinstance(source, Tensor):
                    raise ValueError("reshape requires one tensor value")
                view = _tensor(binding.var.struct_info, f"view_{len(views)}", "view", len(calls))
                if view.dtype != source.dtype or view.nbytes != source.nbytes:
                    raise ValueError("reshape must preserve dtype and tensor byte extent")
                view.alias_of = source.storage
                mark_use(source, len(calls))
                views.append(view)
                values[binding.var] = view
                continue
            if call.op != tvm.ir.Op.get("relax.call_tir") or len(call.args) != 2:
                raise ValueError("export requires legalized buffer-only call_tir operations")
            gv, arguments = call.args
            if not isinstance(gv, tvm.ir.GlobalVar) or not isinstance(arguments, relax.Tuple):
                raise ValueError("unsupported call_tir signature")
            symbol = gv.name_hint
            if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", symbol):
                raise ValueError("PrimFunc symbol is not a C identifier")
            primitive = mod[gv]
            if not isinstance(primitive, tir.PrimFunc) or len(primitive.params) != len(primitive.buffer_map):
                raise ValueError("PrimFunc requires an unsupported scalar or hidden workspace ABI")
            if primitive.attrs and int(primitive.attrs.get("calling_conv", 0)) != 0:
                raise ValueError("PrimFunc requires the default calling convention")
            if primitive.attrs and "target" in primitive.attrs:
                raise ValueError("PrimFunc target must be supplied by the exporter")
            actual_inputs = [resolve(arg) for arg in arguments.fields]
            if any(not isinstance(arg, Tensor) for arg in actual_inputs):
                raise ValueError("call_tir operands must be individual tensors")
            result = output_value(binding.var.struct_info, len(calls))
            actual_outputs = _flatten(result)
            if len(primitive.params) != len(actual_inputs) + len(actual_outputs):
                raise ValueError("PrimFunc buffer count differs from call_tir")
            for param, tensor in zip(primitive.params, [*actual_inputs, *actual_outputs]):
                buffer = primitive.buffer_map[param]
                if buffer.dtype != tensor.dtype or any(not isinstance(dim, tir.IntImm) for dim in buffer.shape) or tuple(int(dim) for dim in buffer.shape) != tensor.shape:
                    raise ValueError("PrimFunc tensor shape/dtype ABI mismatch")
                if buffer.strides or not isinstance(buffer.elem_offset, tir.IntImm) or int(buffer.elem_offset) != 0:
                    raise ValueError("only compact zero-offset PrimFunc buffers are supported")
                if buffer.scope() != "global":
                    raise ValueError("PrimFunc parameters must use host-addressed storage")
                if buffer.data_alignment <= 0 or ALIGNMENT % buffer.data_alignment:
                    raise ValueError("PrimFunc external-buffer alignment exceeds the storage guarantee")
            external_data = {buffer.data for buffer in primitive.buffer_map.values()}
            def check_alignment(node):
                if isinstance(node, tir.AttrStmt) and node.attr_key == "storage_alignment" and node.node in external_data:
                    if not isinstance(node.value, tir.IntImm) or int(node.value) <= 0 or ALIGNMENT % int(node.value):
                        raise ValueError("PrimFunc external-buffer storage alignment exceeds the storage guarantee")
            tir.stmt_functor.post_order_visit(primitive.body, check_alignment)
            for tensor in actual_inputs:
                mark_use(tensor, len(calls))
            values[binding.var] = result
            functions[gv] = primitive.with_attr("global_symbol", symbol)
            calls.append((symbol, actual_inputs, actual_outputs))
    outputs = _flatten(resolve(result_expr))
    if not outputs:
        raise ValueError("require a nonempty tensor output")
    unique_outputs = set(outputs)
    for index, tensor in enumerate(outputs):
        mark_use(tensor, len(calls))
        # Duplicate returns and input/constant returns are copied at completion.
        if tensor.origin == "workspace":
            tensor.origin = "output"
            tensor.offset = index
    workspace = [tensor for tensor in tensors if tensor not in unique_outputs]
    workspace_bytes = _allocate_workspace(workspace)
    constants_bytes = 0
    for tensor in constants:
        tensor.offset = constants_bytes
        constants_bytes = _aligned(constants_bytes + tensor.nbytes)
    return {"inputs": inputs, "outputs": outputs, "constants": constants, "tensors": tensors, "views": views, "calls": calls, "functions": functions, "workspace_bytes": workspace_bytes, "constants_bytes": constants_bytes}


def _memory_preflight(plan, memory_limit_bytes):
    """Report logical liveness separately from physically reserved tensor storage."""
    allocated = {
        "input_bytes": sum(t.nbytes for t in plan["inputs"]),
        # Every return slot requires disjoint caller storage, including aliases.
        "output_bytes": sum(t.nbytes for t in plan["outputs"]),
        "constants_bytes": plan["constants_bytes"],
        "workspace_bytes": plan["workspace_bytes"],
    }
    tensors_bytes = sum(allocated.values())
    allocated["explicit_tensor_bytes"] = tensors_bytes
    if isinstance(memory_limit_bytes, bool) or not isinstance(memory_limit_bytes, int) or memory_limit_bytes <= 0 or tensors_bytes > memory_limit_bytes:
        raise ValueError("explicit tensor storage exceeds the supplied memory budget")
    # Tensor identities, rather than graph aliases or repeated output slots,
    # describe logical values. Input/constant allocations remain resident even
    # after their last graph use; allocated above includes their full envelopes.
    tensors = list(dict.fromkeys([*plan["inputs"], *plan["constants"], *plan["tensors"]]))
    per_call = []
    for index, (symbol, _, _) in enumerate(plan["calls"]):
        live = [t for t in tensors if t.birth <= index <= t.last_use]
        workspace = [t for t in live if t.origin == "workspace"]
        private = plan.get("operator_workspaces", {}).get(symbol, {}).get("bytes", 0)
        per_call.append({
            "index": index, "symbol": symbol,
            "live_tensors": [t.name for t in live],
            "live_tensor_bytes": sum(t.nbytes for t in live) + private,
            "live_workspace_bytes": sum(t.nbytes for t in workspace) + private,
            "live_workspace_aligned_bytes": sum(_aligned(t.nbytes) for t in workspace) + _aligned(private),
            "operator_workspace_bytes": private,
        })
    completion = list(dict.fromkeys(t.storage for t in plan["outputs"]))
    return {
        "scope": "explicit tensor storage only; excludes ELF sections, operator stack, startup and platform reservations",
        "memory_limit_bytes": memory_limit_bytes,
        "allocated": allocated,
        "per_call": per_call,
        "completion": {"live_tensors": [t.name for t in completion], "live_tensor_bytes": sum(t.nbytes for t in completion)},
        "peak_live_tensor_bytes": max([entry["live_tensor_bytes"] for entry in per_call] + [sum(t.nbytes for t in completion)]),
        "peak_live_workspace_bytes": max((entry["live_workspace_bytes"] for entry in per_call), default=0),
        "peak_live_workspace_aligned_bytes": max((entry["live_workspace_aligned_bytes"] for entry in per_call), default=0),
        "largest_intermediates": [t.metadata() for t in sorted((t for t in plan["tensors"] if t.origin == "workspace"), key=lambda t: (-t.nbytes, t.name))[:10]],
        "output_bindings": [{"index": index, "tensor": t.name, "storage": t.storage.origin,
                             "backing_tensor": t.storage.name,
                             "copy_required": t.origin != "output" or t.offset != index}
                            for index, t in enumerate(plan["outputs"])],
    }


def preflight_graph(mod, *, memory_limit_bytes=1 << 30, target=RV64_TARGET):
    """Plan graph/operator storage and check capacity without machine-code emission."""
    plan = plan_graph(mod)
    _plan_operator_workspaces(plan, target)
    return _memory_preflight(plan, memory_limit_bytes)


def export_graph(mod, output_dir, *, target=RV64_TARGET, memory_limit_bytes=1 << 30,
                 profile_operators=False):
    """Write operators, one constant blob, C orchestration and a storage manifest.

    The limit covers explicit tensor storage only; executable sections, generated
    operator stack allocations, startup and platform reservations are additional.
    Caller-supplied buffers must each contain the manifest's full byte envelope.
    Emitted TVM runtime workspace allocation/free dependencies are rejected;
    this static exporter provides no guest allocator or callback initialization.
    """
    plan = plan_graph(mod)
    _plan_operator_workspaces(plan, target)
    return _export_plan(mod, plan, output_dir, target, memory_limit_bytes,
                        profile_operators=profile_operators)


def _plan_operator_workspaces(plan, target):
    """Use TVM USMP for static private allocations requiring runtime workspace.

    This is a deployment ABI transform after graph scheduling. The original
    call_tir buffer ABI is checked by plan_graph first. Only a planner-created,
    typed UINT8 pool parameter may be added; arbitrary hidden ABIs stay refused.
    Serial synchronous graph calls share one pool after the graph tensor region.
    """
    from tvm.tir.usmp import analysis, transform, utils  # pylint: disable=import-outside-toplevel

    target = tvm.target.Target(target)
    workspaces = {}
    for gv, function in list(plan["functions"].items()):
        symbol = gv.name_hint
        lowered = tvm.lower(function)["main"]
        allocations = []
        tir.stmt_functor.post_order_visit(lowered.body, lambda node: allocations.append(node)
                                        if isinstance(node, tir.Allocate) else None)
        # LowerTVMBuiltin keeps CPU global allocations strictly below 1024 bytes
        # on the stack. The remaining static allocations need an explicit pool.
        sizes = []
        for allocation in allocations:
            if not all(isinstance(extent, tir.IntImm) and int(extent) > 0 for extent in allocation.extents):
                raise ValueError("Static export refuses dynamically sized private workspace")
            dtype = tvm.DataType(allocation.dtype)
            sizes.append(math.prod(int(extent) for extent in allocation.extents) *
                         max(1, (dtype.bits * dtype.lanes + 7) // 8))
        if not sizes or max(sizes) < 1024:
            continue
        pool = tvm.WorkspacePoolInfo("operator_workspace", [target])

        def annotate(node):
            if isinstance(node, tir.Allocate):
                return tir.Allocate(node.buffer_var, node.dtype, node.extents, node.condition,
                                    node.body, {**node.annotations, utils.CANDIDATE_MEMORY_POOL_ATTR: [pool]})
            return None

        lowered = lowered.with_body(tir.stmt_functor.ir_transform(lowered.body, None, annotate))
        lowered = lowered.with_attr("target", target).with_attr("global_symbol", "__tvm_main__")
        module = tvm.IRModule({"__tvm_main__": lowered})
        info = analysis.extract_buffer_info(lowered, module)
        buffers = tvm.get_global_func("tir.usmp.CreateArrayBufferInfo")(info.buffer_info_stmts)
        placements = tvm.get_global_func("tir.usmp.algo.greedy_by_size")(buffers, info.memory_pressure)
        statements = tvm.get_global_func("tir.usmp.AssignStmtPoolAllocations")(info.buffer_info_stmts, placements)
        planned = transform.convert_pool_allocations_to_offsets(statements, emit_tvmscript_printable=True)(module)["__tvm_main__"]
        if len(planned.params) != len(function.params) + 1:
            raise ValueError("USMP produced an unsupported multi-pool operator ABI")
        buffer = planned.buffer_map[planned.params[-1]]
        if buffer.dtype != "uint8" or len(buffer.shape) != 1 or not isinstance(buffer.shape[0], tir.IntImm):
            raise ValueError("USMP did not produce a bounded UINT8 workspace parameter")
        remaining = []
        tir.stmt_functor.post_order_visit(planned.body, lambda node: remaining.append(node)
                                        if isinstance(node, tir.Allocate) else None)
        if remaining:
            raise ValueError("USMP left private allocations outside the explicit workspace pool")
        for key, value in function.attrs.items() if function.attrs else []:
            if key != "target":
                planned = planned.with_attr(key, value)
        planned = planned.with_attr("global_symbol", symbol)
        plan["functions"][gv] = planned
        workspaces[symbol] = {"bytes": int(buffer.shape[0]), "algorithm": "TVM USMP greedy_by_size",
                              "planned_allocations": len(buffers), "memory_pressure": int(info.memory_pressure)}
    offset = _aligned(plan["workspace_bytes"])
    size = max((record["bytes"] for record in workspaces.values()), default=0)
    plan["operator_workspaces"] = workspaces
    plan["operator_workspace_offset"] = offset
    plan["workspace_bytes"] = offset + _aligned(size)


def _export_plan(mod, plan, output_dir, target, memory_limit_bytes, *, prefix="", shared_constants=None,
                 profile_operators=False):
    preflight = _memory_preflight(plan, memory_limit_bytes)
    tensors_bytes = preflight["allocated"]["explicit_tensor_bytes"]
    output_dir = checked_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    constant_path = output_dir / "constants.bin" if shared_constants is None else shared_constants
    if shared_constants is None:
        _write_constants(plan["constants"], plan["constants_bytes"], constant_path)

    def address(tensor):
        tensor = tensor.storage
        if tensor.origin == "input":
            return f"inputs[{plan['inputs'].index(tensor)}]"
        if tensor.origin == "output":
            return f"outputs[{tensor.offset}]"
        if tensor.origin == "constant":
            return f"(model_constants + {tensor.offset})"
        return f"(workspace + {tensor.offset})"

    # The board linker may place large immutable blobs beyond medany's reach.
    # Recompile only this pointer-array wrapper with an explicit absolute base;
    # operator code and constant bytes remain the package's original payloads.
    lines = ["#include <stdint.h>", "#include <stddef.h>",
             "#ifdef TVM_MODEL_CONSTANTS_ADDRESS",
             "#define model_constants ((const uint8_t *)(uintptr_t)(TVM_MODEL_CONSTANTS_ADDRESS))",
             "#else", "extern const uint8_t model_constants[];", "#endif"]
    if shared_constants is None:
        lines.extend(["static const char *model_last_error;", "void TVMAPISetLastError(const char *message) { model_last_error = message; }"])
    else:
        lines.append("extern const char *model_last_error;")
    lines.append(f"const char *{prefix}model_get_last_error(void) {{ return model_last_error; }}")
    if profile_operators:
        count = len(plan["calls"])
        names = ", ".join(json.dumps(symbol) for symbol, _, _ in plan["calls"]) or "0"
        lines.extend([
            "#if defined(__riscv) && __riscv_xlen == 64",
            'static uint64_t model_profile_clock(void) { uint64_t v; __asm__ volatile("csrr %0,mcycle":"=r"(v)::"memory"); return v; }',
            "#else",
            "extern uint64_t model_profile_read_cycles(void);",
            "static uint64_t model_profile_clock(void) { return model_profile_read_cycles(); }",
            "#endif",
            f"static uint64_t model_operator_cycles[{max(1, count)}];",
            "static uint32_t model_profile_completed;",
            f"static const char *const model_operator_symbols[{max(1, count)}] = {{{names}}};",
            f"uint32_t {prefix}model_profile_count(void) {{ return model_profile_completed; }}",
            f"uint64_t {prefix}model_profile_cycles(uint32_t i) {{ return i < model_profile_completed ? model_operator_cycles[i] : 0; }}",
            f"const char *{prefix}model_profile_symbol(uint32_t i) {{ return i < {count} ? model_operator_symbols[i] : 0; }}",
        ])
    for symbol, args, results in plan["calls"]:
        ctypes = [DTYPES[tensor.dtype] + "*" for tensor in [*args, *results]]
        if symbol in plan.get("operator_workspaces", {}):
            ctypes.append("uint8_t*")
        declaration = "extern int32_t " + symbol + "(" + ", ".join(ctypes) + ");"
        if declaration not in lines:
            lines.append(declaration)
    lines.extend([
        "static int model_span(const void *pointer, uint64_t bytes) { return pointer && !((uintptr_t)pointer % 64) && bytes <= UINTPTR_MAX - (uintptr_t)pointer; }",
        "static int model_overlap(const void *a, uint64_t an, const void *b, uint64_t bn) { return an && bn && (uintptr_t)a < (uintptr_t)b + bn && (uintptr_t)b < (uintptr_t)a + an; }",
        f"int32_t {prefix}model_run(const void *const *inputs, void *const *outputs, uint8_t *workspace, uint64_t workspace_bytes) {{",
        "  model_last_error = 0;",
        *(["  model_profile_completed = 0;"] if profile_operators else []),
        "  if (!outputs" + (" || !inputs" if plan["inputs"] else "") + ") return -1;",
        f"  if (workspace_bytes < UINT64_C({plan['workspace_bytes']})) return -1;",
    ])
    if plan["workspace_bytes"]:
        lines.append(f"  if (!model_span(workspace, {plan['workspace_bytes']})) return -1;")
    for group in ("inputs", "outputs"):
        for index, tensor in enumerate(plan[group]):
            lines.append(f"  if (!model_span({group}[{index}], {tensor.nbytes})) return -1;")
    read_buffers = [(f"inputs[{i}]", t.nbytes) for i, t in enumerate(plan["inputs"])] + [("model_constants", plan["constants_bytes"])]
    write_buffers = [(f"outputs[{i}]", t.nbytes) for i, t in enumerate(plan["outputs"])]
    if plan["workspace_bytes"]:
        write_buffers.append(("workspace", plan["workspace_bytes"]))
    for index, (pointer, size) in enumerate(write_buffers):
        for other, other_size in [*read_buffers, *write_buffers[:index]]:
            if other_size:
                lines.append(f"  if (model_overlap({pointer}, {size}, {other}, {other_size})) return -2;")
    for call_index, (symbol, args, results) in enumerate(plan["calls"]):
        arguments = ["(" + DTYPES[tensor.dtype] + "*)" + address(tensor) for tensor in [*args, *results]]
        if symbol in plan.get("operator_workspaces", {}):
            arguments.append(f"(workspace + {plan['operator_workspace_offset']})")
        if profile_operators:
            lines.append("  { uint64_t start = model_profile_clock();")
            lines.append(f"    int32_t rc = {symbol}({', '.join(arguments)});")
            lines.append(f"    model_operator_cycles[{call_index}] = model_profile_clock() - start;")
            lines.append("    if (rc) return -3;")
            lines.append(f"    model_profile_completed = {call_index + 1}; }}")
        else:
            lines.append(f"  if ({symbol}({', '.join(arguments)})) return -3;")
    for index, tensor in enumerate(plan["outputs"]):
        if tensor.origin != "output" or tensor.offset != index:
            lines.append(f"  for (uint64_t i = 0; i < UINT64_C({tensor.nbytes}); ++i) ((uint8_t*)outputs[{index}])[i] = ((const uint8_t*){address(tensor)})[i];")
    lines.extend(["  return 0;", "}"])
    (output_dir / "model.c").write_text("\n".join(lines) + "\n")
    primitive_mod = tvm.IRModule(plan["functions"]).with_attr("executor", tvm.relay.backend.Executor("aot", {"unpacked-api": True, "interface-api": "c"}))
    if tvm.target.Target(target).kind.name == "llvm":
        from tvm.tir.transform.rounding import lower_bounded_rne
        for gv, func in list(primitive_mod.functions.items()):
            lowered, _ = lower_bounded_rne(func)
            primitive_mod.update_func(gv, lowered)
    if plan["functions"]:
        module = tvm.build(primitive_mod, target=target)
    else:
        # The ordinary build driver rejects an empty IRModule. There are no
        # operators to lower in a constant/view-only graph; ask the standard
        # LLVM target codegen for its empty object container instead.
        selected = tvm.target.Target(target)
        if selected.kind.name != "llvm":
            raise ValueError("Static export requires an LLVM target")
        module = tvm.get_global_func("target.build.llvm")(primitive_mod, selected)
    llvm_ir = module.get_source("ll")
    (output_dir / "operators.ll").write_text(llvm_ir)
    # The pinned LLVM backend may emit null callback globals instead of ordinary
    # undefined function symbols. A successful static link cannot prove these
    # calls are usable without TVM's runtime initializing the module context.
    allocators = sorted(set(re.findall(r'@"?((?:__)?TVMBackend(?:Alloc|Free)Workspace)"?(?=[\s(,])', llvm_ir)))
    if allocators:
        raise ValueError("unsupported TVM runtime workspace allocator dependency: " + ", ".join(allocators))
    module.save(str(output_dir / "operators.o"))
    (output_dir / "operators.tir.py").write_text(primitive_mod.script(show_meta=True))
    (output_dir / "graph.relax.py").write_text(mod.script(show_meta=True))
    manifest = {
        "entrypoint": prefix + "model_run", "constants_file": str(constant_path),
        "constants_binding": {"default": "link_symbol", "symbol": "model_constants",
                              "absolute_address_macro": "TVM_MODEL_CONSTANTS_ADDRESS"},
        "target": str(target), "alignment_bytes": ALIGNMENT, "heap_required_by_orchestration": False, "runtime_workspace_allocator_required": False,
        "workspace_bytes": plan["workspace_bytes"], "constants_bytes": plan["constants_bytes"], "explicit_tensor_bytes": tensors_bytes,
        "memory_limit_bytes": memory_limit_bytes, "memory_scope": "graph tensor storage and explicit USMP operator pools; add ELF sections, operator stack, startup and platform reservations",
        "memory_preflight": preflight,
        "operator_workspaces": plan.get("operator_workspaces", {}),
        "operator_workspace_offset": plan.get("operator_workspace_offset", 0),
        "inputs": [tensor.metadata() for tensor in plan["inputs"]], "outputs": [tensor.metadata() for tensor in plan["outputs"]],
        "constants": [tensor.metadata() for tensor in plan["constants"]], "intermediates": [tensor.metadata() for tensor in plan["tensors"]],
        "views": [tensor.metadata() for tensor in plan["views"]],
        "calls": [{"symbol": name, "inputs": [t.name for t in args], "outputs": [t.name for t in results]} for name, args, results in plan["calls"]],
        "constants_sha256": _file_sha256(constant_path), "full_model_qualified": False, "timing_qualified": False,
    }
    if profile_operators:
        manifest["operator_profile"] = {
            "count": len(plan["calls"]), "count_entrypoint": prefix + "model_profile_count",
            "cycles_entrypoint": prefix + "model_profile_cycles",
            "symbol_entrypoint": prefix + "model_profile_symbol",
            "counter_storage_bytes": max(1, len(plan["calls"])) * 8 + 4,
            "scope": "Last invocation, per call_tir; includes each function's existing completion handling",
            "resets_each_invocation": True, "prints_inside_model_run": False,
            "aggregate_includes_counter_overhead": True,
            "excludes": ["wrapper argument checks", "final output copies", "caller routing"],
        }
    (output_dir / "graph.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_constants(constants, size, path):
    with path.open("wb") as stream:
        for tensor in constants:
            stream.seek(tensor.offset)
            # TVM exposes BF16 constants as raw uint16 payloads, not numeric uint16 values.
            dtype = np.dtype("uint16" if tensor.dtype == "bfloat16" else tensor.dtype).newbyteorder("<")
            stream.write(tensor.value.astype(dtype, copy=False).tobytes())
        stream.truncate(max(size, 1))
    include_path = str(path.absolute()).replace("\\", "\\\\").replace('"', '\\"')
    if any(character in include_path for character in ("\n", "\r")):
        raise ValueError("constant artifact path contains an unsupported line break")
    path.with_suffix(".S").write_text(f'.section .rodata.model_constants,"a",@progbits\n.balign {ALIGNMENT}\n.globl model_constants\nmodel_constants:\n.incbin "{include_path}"\n.section .note.GNU-stack,"",@progbits\n')


def export_graphs(modules, output_dir, *, target=RV64_TARGET, memory_limit_bytes=1 << 30):
    """Export static stages with distinct symbols and one deduplicated constant pool.

    Stages execute serially and may reuse workspace. The reported bound covers a
    single invocation; retained session state, executable/stack and fixtures are
    additional. The caller still owns state routing and disjoint stage buffers.
    """
    if not isinstance(modules, dict) or not modules or any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name) for name in modules):
        raise ValueError("Stage names must be distinct C identifiers in a nonempty mapping")
    output_dir = checked_path(output_dir)
    destinations = {name: checked_path(output_dir / name) for name in modules}
    stages, pool, unique, size, duplicated = [], [], {}, 0, 0
    for index, (name, mod) in enumerate(modules.items()):
        renamed = mod.replace_global_vars({gv: f"stage_{index}_operator_" + gv.name_hint for gv in mod.get_global_vars() if isinstance(mod[gv], tir.PrimFunc)})
        plan = plan_graph(renamed)
        _plan_operator_workspaces(plan, target)
        duplicated += plan["constants_bytes"]
        for tensor in plan["constants"]:
            key = (tensor.dtype, tensor.shape, hashlib.sha256(tensor.value.tobytes()).hexdigest())
            previous = unique.get(key)
            if previous is not None:
                if not np.array_equal(np.ascontiguousarray(previous.value).view("uint8"), np.ascontiguousarray(tensor.value).view("uint8")):
                    raise ValueError("Constant digest collision")
                tensor.offset = previous.offset
            else:
                tensor.offset = size
                size = _aligned(size + tensor.nbytes)
                unique[key] = tensor
                pool.append(tensor)
        stages.append((name, renamed, plan))
    # Validate every stage before creating any artifacts. Each invocation maps
    # the complete shared pool, rather than claiming all stages fit in isolation.
    for _, _, plan in stages:
        plan["constants_bytes"] = size
        _memory_preflight(plan, memory_limit_bytes)
    output_dir = checked_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    constant_path = output_dir / "constants.bin"
    _write_constants(pool, size, constant_path)
    (output_dir / "runtime.c").write_text('const char *model_last_error;\nvoid TVMAPISetLastError(const char *message) { model_last_error = message; }\n')
    records = {}
    declarations = ["#include <stdint.h>"]
    for index, (name, mod, plan) in enumerate(stages):
        prefix = f"stage_{index}_"
        records[name] = _export_plan(mod, plan, destinations[name], target, memory_limit_bytes, prefix=prefix, shared_constants=constant_path)
        declarations.append(f"int32_t {prefix}model_run(const void *const *inputs, void *const *outputs, uint8_t *workspace, uint64_t workspace_bytes);")
        declarations.append(f"const char *{prefix}model_get_last_error(void);")
    (output_dir / "model.h").write_text("\n".join(declarations) + "\n")
    report = {"stages": records, "constants_bytes": size, "duplicated_stage_constants_bytes": duplicated,
              "constants_sha256": _file_sha256(constant_path),
              "maximum_stage_tensor_bytes": max(stage["explicit_tensor_bytes"] for stage in records.values()),
              "serial_workspace_bytes": max(stage["workspace_bytes"] for stage in records.values()),
              "memory_scope": "shared constants plus one stage invocation; retained session state, ELF, stack and fixtures are additional",
              "state_routing": "caller_owned", "session_execution": False, "full_model_qualified": False, "timing_qualified": False}
    (output_dir / "graphs.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
