# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Bind mathematical task arguments from the same prepared semantic program.

Constants retain their actual bytes. Optional boundary arrays propagate through
unscheduled host PrimFuncs to supply consumer activations. This propagation is
a host mathematical reference, distinct from an independent NumPy oracle and
from any scheduled candidate. No model identity chooses a value or a schedule.
"""

import hashlib

import numpy as np
import tvm
from tvm import tir
from tvm.relax.backend.contrib.gemmini_search import WORKLOAD_ATTR

from baremetal import plan_graph


def array_identity(array):
    return {"shape": list(array.shape), "dtype": array.dtype.name,
            "sha256": hashlib.sha256(array.tobytes()).hexdigest()}


def _checked(value, shape, dtype):
    value = np.asarray(value)
    if value.shape != tuple(int(v) for v in shape) or value.dtype.name != str(dtype):
        raise ValueError("Bound operator input shape/dtype mismatch")
    if value.dtype.kind == "f" and not np.isfinite(value).all():
        raise ValueError("Bound operator input is nonfinite")
    # ascontiguousarray promotes rank-zero constants to shape (1,), which
    # violates the extracted PrimFunc ABI during boundary propagation.
    return np.array(value, copy=True, order="C")


def check_integer_divisors(func, parameter_arrays):
    """Reject zero literal/direct-input integer divisors before host execution.

    This is a checked admission guard, not a rewrite or proof of arbitrary
    computed denominator expressions. It prevents known invalid fixture values
    from terminating the host process with SIGFPE.
    """
    inputs = {func.buffer_map[param].data: array for param, array in
              zip(func.params, parameter_arrays) if array is not None}

    def visit(node):
        if not isinstance(node, (tir.Div, tir.Mod, tir.FloorDiv, tir.FloorMod)):
            return
        if not str(node.dtype).startswith(("int", "uint")):
            return
        divisor = node.b
        while isinstance(divisor, tir.Cast):
            divisor = divisor.value
        if isinstance(divisor, tir.IntImm) and int(divisor) == 0:
            raise ValueError("Zero literal integer divisor in mathematical workload")
        if isinstance(divisor, tir.BufferLoad) and divisor.buffer.data in inputs:
            value = inputs[divisor.buffer.data]
            if np.any(value == 0):
                raise ValueError("Zero integer divisor in bound input buffer " + divisor.buffer.name)

    tir.stmt_functor.post_order_visit(func.body, visit)


def resolve_operator_inputs(prepared, workloads, boundary_inputs=None):
    """Return representative input overrides and provenance for each workload.

    ``workloads`` maps structural workload keys to extracted PrimFuncs.
    ``boundary_inputs`` follows prepared ``main`` parameter order. Without it,
    only immutable constants (including alias views) are bound; no synthetic
    arrays are invented here. Distinct occurrences retain their identities;
    one representative input set does not establish all-occurrence timing.
    """
    plan = plan_graph(prepared)
    values, origins = {}, {}
    if boundary_inputs is not None:
        if len(boundary_inputs) != len(plan["inputs"]):
            raise ValueError("Prepared graph boundary input count mismatch")
        for tensor, array in zip(plan["inputs"], boundary_inputs):
            values[tensor] = _checked(array, tensor.shape, tensor.dtype)
            origins[tensor] = "declared_boundary_input"
    for tensor in plan["constants"]:
        values[tensor] = _checked(tensor.value, tensor.shape, tensor.dtype)
        origins[tensor] = "immutable_prepared_constant"

    def resolve(tensor):
        if tensor in values:
            return values[tensor], origins[tensor]
        if tensor.alias_of is not None:
            value, origin = resolve(tensor.storage)
            if value is not None:
                values[tensor] = value.reshape(tensor.shape)
                origins[tensor] = origin + ":alias_view"
                return values[tensor], origins[tensor]
        return None, None

    occurrences = {str(key): [] for key in workloads}
    for index, (symbol, inputs, outputs) in enumerate(plan["calls"]):
        func = plan["functions"][prepared.get_global_var(symbol)]
        key = str(func.attrs[WORKLOAD_ATTR]) if func.attrs and WORKLOAD_ATTR in func.attrs else None
        args, sources, overrides = [], [], {}
        for position, tensor in enumerate(inputs):
            value, origin = resolve(tensor)
            args.append(value)
            if value is not None:
                overrides[position] = value.copy()
                sources.append({"parameter_index": position, "origin": origin, **array_identity(value)})
        row = {"call_index": index, "function": symbol, "inputs": sources,
               "complete_inputs": all(value is not None for value in args)}
        if key in workloads:
            # Match the exact stock extraction normalization, including its
            # noalias annotation, rather than weakening structural comparison.
            selected = tvm.get_global_func("tvm.meta_schedule.normalize_mod")(
                func.with_attr("global_symbol", "main"))["main"]
            if not tvm.ir.structural_equal(selected, workloads[key]):
                raise ValueError("Prepared argument binding differs from extracted workload structure")
            occurrences[key].append((overrides, row))
        if boundary_inputs is None or not row["complete_inputs"]:
            continue
        # A prepared mathematical task may contain CPU glue, but must never
        # contain already-lowered external accelerator commands.
        def check_call(node):
            if isinstance(node, tir.Call) and node.op == tvm.ir.Op.get("tir.call_extern"):
                raise ValueError("Cannot propagate inputs through external/scheduled commands")
        tir.stmt_functor.post_order_visit(func.body, check_call)
        arrays = args + [np.zeros(tensor.shape, tensor.dtype) for tensor in outputs]
        check_integer_divisors(func, args + [None] * len(outputs))
        reference = tvm.build(tvm.IRModule({"main": func.with_attr("global_symbol", "main")}), "llvm")
        devices = [tvm.nd.array(array) for array in arrays]
        reference["main"](*devices)
        row["propagation"] = "unscheduled mathematical PrimFunc on host LLVM; not an independent NumPy oracle"
        row["outputs"] = []
        for tensor, device in zip(outputs, devices[len(args):]):
            values[tensor] = _checked(device.numpy(), tensor.shape, tensor.dtype)
            origins[tensor] = "host_mathematical_producer_output"
            row["outputs"].append(array_identity(values[tensor]))

    result = {}
    for key, rows in occurrences.items():
        if not rows:
            raise ValueError("Extracted workload has no prepared call site: " + key)
        # Prefer a fully bound occurrence, then preserve original call order.
        selected = max(range(len(rows)), key=lambda i: (len(rows[i][0]), -i))
        overrides, chosen = rows[selected]
        constant_sets = {tuple((item["parameter_index"], item["sha256"]) for item in row["inputs"]
                               if item["origin"].startswith("immutable_prepared_constant"))
                         for _, row in rows}
        result[key] = {"input_overrides": overrides,
                       "provenance": {"schema": "gemmini_operator_input_bindings_v1",
                                      "selected_call_index": chosen["call_index"],
                                      "representative_policy": "most bound input parameters, ties by prepared call order",
                                      "occurrences": [row for _, row in rows],
                                      "different_constant_bindings": len(constant_sets) > 1,
                                      "boundary_inputs_supplied": boundary_inputs is not None,
                                      "scope": "One deterministic representative argument set per extracted workload; not all-occurrence or full-model timing"}}
    return result
