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
"""Exact scalar readout proofs and TensorIR scheduling helpers."""

import hashlib
import json
import math
from functools import lru_cache

import numpy as np
import tvm
from tvm import tir
from tvm.script import tir as T


def _constant_pair(expression, cls):
    if not isinstance(expression, cls):
        return None
    for value, constant in ((expression.a, expression.b), (expression.b, expression.a)):
        if isinstance(constant, tir.FloatImm):
            return value, float(constant.value)
    return None


def _readout_chain(expression, accumulator):
    if (
        isinstance(expression, tir.Cast)
        and expression.dtype == "float32"
        and tvm.ir.structural_equal(expression.value, accumulator)
    ):
        return (), False
    multiplication = _constant_pair(expression, tir.Mul)
    if multiplication is not None:
        inner, factor = multiplication
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError("Readout scales must be positive finite FP32 constants")
        factors, relu = _readout_chain(inner, accumulator)
        return factors + (float(np.float32(factor)),), relu
    if isinstance(expression, tir.Div) and isinstance(expression.b, tir.FloatImm):
        divisor = float(expression.b.value)
        # Only powers of two make division identical to multiplication by
        # its reciprocal across the full FP32 accumulator domain.
        mantissa, _ = math.frexp(divisor)
        if divisor <= 0 or not np.isfinite(divisor) or mantissa != 0.5:
            raise ValueError("Readout division requires a positive power of two")
        factors, relu = _readout_chain(expression.a, accumulator)
        return factors + (float(np.float32(1.0 / divisor)),), relu
    activation = _constant_pair(expression, tir.Max)
    if activation is not None and activation[1] == 0:
        factors, _ = _readout_chain(activation[0], accumulator)
        return factors, True
    raise ValueError("Readout is not a monotone scalar cast/positive-scale/ReLU chain")


def _thresholds(evaluate, *, relu):
    """Locate every output-code transition in the complete signed INT32 domain."""
    codes = np.arange(1 if relu else -127, 128, dtype="int64")
    low, high = np.full(codes.shape, -(1 << 31), "int64"), np.full(codes.shape, 1 << 31, "int64")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        for _ in range(33):
            active = low < high
            if not active.any():
                break
            midpoint = low + (high - low) // 2
            integers = np.minimum(midpoint, (1 << 31) - 1).astype("int32")
            at_or_above = evaluate(integers) >= codes
            high = np.where(active & at_or_above, midpoint, high)
            low = np.where(active & ~at_or_above, midpoint + 1, low)
    if np.any(low != high):
        raise ValueError("Readout threshold proof did not converge")
    return low


@lru_cache(None)
def _prove_chain(factors, relu):
    combined = np.float32(1)
    for factor in factors:
        combined = np.float32(combined * np.float32(factor))
    if not np.isfinite(combined) or combined <= 0:
        raise ValueError("Combined readout scale is not positive finite FP32")
    # The pinned functional model rounds through signed int64 before clipping.
    # Keep that conversion defined for every INT32 input as well as proving the
    # mathematical output. Larger scales require a separate model/RTL contract.
    if combined > np.float32(1 << 31):
        raise ValueError("Readout scale exceeds the qualified INT64 conversion bound")

    def source(values):
        result = values.astype("float32")
        for factor in factors:
            result = (result * np.float32(factor)).astype("float32")
        if relu:
            result = np.maximum(result, np.float32(0))
        return np.clip(np.rint(result), -128, 127)

    def hardware(values):
        result = np.clip(np.rint(values.astype("float32") * combined), -128, 127)
        return np.maximum(result, 0) if relu else result

    source_thresholds, hardware_thresholds = (
        _thresholds(source, relu=relu),
        _thresholds(hardware, relu=relu),
    )
    if not np.array_equal(source_thresholds, hardware_thresholds):
        raise ValueError(
            "Source and single-scale readout have different integer rounding thresholds"
        )
    return {
        "proof": "equal monotone output-code thresholds over every signed INT32 input",
        "input_domain": [-(1 << 31), (1 << 31) - 1],
        "transitions_checked": len(source_thresholds),
        "thresholds_sha256": hashlib.sha256(source_thresholds.astype("<i8").tobytes()).hexdigest(),
        "scale_bits": int(np.array(combined).view("uint32")),
        "relu": bool(relu),
    }


def prove_readout(expression, accumulator):
    """Prove a guarded FP32 RNE/clamp/cast exactly equals native scalar readout."""
    if isinstance(expression, tir.Call) and expression.op == tvm.ir.Op.get("tir.if_then_else"):
        condition, zero, expression = expression.args
        if not (
            isinstance(condition, tir.Call)
            and condition.op == tvm.ir.Op.get("tir.isnan")
            and isinstance(zero, tir.IntImm)
            and int(zero) == 0
        ):
            raise ValueError("Readout has an unsupported conditional")
    else:
        condition = None
    if not isinstance(expression, tir.Cast) or expression.dtype != "int8":
        raise ValueError("Readout requires an INT8 output cast")
    high = _constant_pair(expression.value, tir.Min)
    low = _constant_pair(high[0], tir.Max) if high else None
    if high is None or low is None or high[1] != 127 or low[1] not in (-128, 0):
        raise ValueError("Readout requires exact signed-byte saturation")
    rounded = low[0]
    if not isinstance(rounded, tir.Call) or rounded.op != tvm.ir.Op.get("tir.nearbyint"):
        raise ValueError("Readout requires source round-to-nearest-even")
    if condition is not None and not tvm.ir.structural_equal(condition.args[0], rounded.args[0]):
        raise ValueError("Readout NaN guard does not cover the scaled value")
    factors, relu = _readout_chain(rounded.args[0], accumulator)
    relu = relu or low[1] == 0
    return dict(_prove_chain(factors, relu))


def _register_readout_intrinsic(expression, accumulator, rows, cols, proof):
    from .gemmini_elementwise import _canonical_description
    from .gemmini_intrin import ACC, register_intrinsic_once

    def value(buffer, i, j):
        def replace(node):
            if isinstance(node, tir.BufferLoad) and node.buffer.same_as(accumulator.buffer):
                return buffer[i, j]
            return None

        return tir.stmt_functor.ir_transform(
            tir.Evaluate(expression), None, replace, ["tir.BufferLoad"]
        ).value

    @T.prim_func
    def desc(a: T.handle, b: T.handle):
        A = T.match_buffer(
            a, (rows, cols), "int32", scope=ACC, strides=[T.int64(), T.int64(1)], offset_factor=1
        )
        B = T.match_buffer(
            b, (rows, cols), "int8", strides=[T.int64(), T.int64(1)], offset_factor=1
        )
        with T.block("root"):
            T.reads(A[0:rows, 0:cols])
            T.writes(B[0:rows, 0:cols])
            for i, j in T.grid(rows, cols):
                with T.block("scaled_readout"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B[vi, vj] = value(A, vi, vj)

    scale_bits, relu = proof["scale_bits"], int(proof["relu"])

    @T.prim_func
    def impl(a: T.handle, b: T.handle):
        A = T.match_buffer(
            a, (rows, cols), "int32", scope=ACC, strides=[T.int64(), T.int64(1)], offset_factor=1
        )
        B = T.match_buffer(
            b, (rows, cols), "int8", strides=[T.int64(), T.int64(1)], offset_factor=1
        )
        with T.block("root"):
            T.reads(A[0:rows, 0:cols])
            T.writes(B[0:rows, 0:cols])
            T.evaluate(
                T.call_extern(
                    "void",
                    "tvm_gemmini_dma_scaled_out",
                    A.access_ptr("r"),
                    B.access_ptr("w"),
                    T.uint32(rows),
                    T.uint32(cols),
                    A.strides[0],
                    B.strides[0],
                    T.uint32(scale_bits),
                    T.int32(relu),
                )
            )

    canonical = _canonical_description(desc)
    key = hashlib.sha256(tvm.ir.save_json(canonical).encode()).hexdigest()[:24]
    name = f"gemmini.readout.{rows}x{cols}.{key}"
    register_intrinsic_once(name, canonical, impl)
    return name


def fuse_readout(sch, write, rows, cols):
    """Fold a proved matrix epilogue into the ACC -> INT8 store intrinsic."""
    candidate = sch.copy()
    selected = candidate.get_block(sch.get(write).name_hint)
    try:
        while candidate.get(selected).body.buffer.dtype != "int8":
            consumers = candidate.get_consumers(selected)
            if len(consumers) != 1:
                return sch, write, None
            candidate.reverse_compute_inline(consumers[0])
        node = candidate.get(selected)
        loads = []
        tir.stmt_functor.post_order_visit(
            node.body.value,
            lambda value: loads.append(value) if isinstance(value, tir.BufferLoad) else None,
        )
        if not loads or any(load.buffer.scope() != "local.gemmini_acc" for load in loads):
            return sch, write, None
        proof = prove_readout(node.body.value, loads[0])
        intrinsic = _register_readout_intrinsic(node.body.value, loads[0], rows, cols, proof)
        candidate.annotate(candidate.get_block("root"), "gemmini.readout_fused", 1)
        candidate.annotate(candidate.get_block("root"), "gemmini.readout_proof",
                           json.dumps(proof, sort_keys=True))
        return candidate, selected, intrinsic
    except (ValueError, tir.schedule.ScheduleError):
        return sch, write, None


def make_readout_schedule(func, *, capabilities=None):
    """Schedule a matrix readout primitive for independent numeric qualification.

    Graph compilation fuses this primitive into contraction writeback. This
    helper also admits direct INT32 accumulator-load/readout experiments, so
    INT32 extremes and every rounding transition can be tested without relying
    on a matrix product to generate those values.
    """
    from .gemmini_bias import register_bias_intrinsic
    from .gemmini_intrin import ACC
    from .gemmini_tensorir import lower_tensorir_schedule

    blocks = []
    tir.stmt_functor.post_order_visit(
        func.body,
        lambda node: blocks.append(node)
        if isinstance(node, tir.Block) and isinstance(node.body, tir.BufferStore)
        else None,
    )
    if len(blocks) != 1 or len(blocks[0].iter_vars) != 2:
        raise ValueError("Readout primitive requires one matrix pointwise block")
    node = blocks[0]
    loads = []
    tir.stmt_functor.post_order_visit(
        node.body.value,
        lambda value: loads.append(value) if isinstance(value, tir.BufferLoad) else None,
    )
    if not loads or any(
        load.dtype != "int32"
        or not tvm.ir.structural_equal(load.indices, node.body.indices)
        or not load.buffer.same_as(loads[0].buffer)
        for load in loads
    ):
        raise ValueError("Readout primitive needs one compact INT32 matrix input")
    proof = prove_readout(node.body.value, loads[0])
    rows, cols = [
        max(factor for factor in range(1, 17) if int(axis.dom.extent) % factor == 0)
        for axis in node.iter_vars
    ]
    sch = tir.Schedule(func, debug_mask="all")
    block = sch.get_block(node.name_hint)
    i, j = sch.get_loops(block)
    io, ii = sch.split(i, [None, rows])
    jo, ji = sch.split(j, [None, cols])
    sch.reorder(io, jo, ii, ji)
    load = sch.cache_read(block, 0, ACC)
    sch.compute_at(load, jo, preserve_unit_loops=True)
    sch.storage_align(load, 0, axis=-2, factor=16, offset=0)
    readout_loads = []
    tir.stmt_functor.post_order_visit(
        sch.get(block).body.value,
        lambda value: readout_loads.append(value) if isinstance(value, tir.BufferLoad) else None,
    )
    store_intrinsic = _register_readout_intrinsic(
        sch.get(block).body.value, readout_loads[0], rows, cols, proof
    )
    sch.tensorize(
        sch.blockize(sch.get_loops(load)[-2]), register_bias_intrinsic(rows, cols, "matrix")
    )
    sch.tensorize(sch.blockize(sch.get_loops(block)[-2]), store_intrinsic)
    return lower_tensorir_schedule(
        func,
        sch,
        capabilities=capabilities,
        workload_metadata={
            "operation": "scaled_readout",
            "scaled_readout": True,
            "readout_proof": proof,
        },
    )
