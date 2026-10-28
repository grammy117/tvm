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
"""Exact copies, integer widening/addition and INT8 ReLU with DMA intrinsics.

The reference programs remain mathematical pointwise blocks. Scheduling caches
their output in real SRAM scopes, then tensorizes rectangular regions. No
whole-operation C kernel or CPU arithmetic is hidden inside the intrinsic.
"""

from functools import lru_cache
import hashlib
import struct

import tvm
from tvm import tir
from tvm.script import tir as T

from .gemmini_intrin import ACC, SPAD
from .gemmini_regions import (
    block_rv,
    copy_selection_loads,
    inline_index_constants,
    partition_copy_regions,
)


def _strides(rank):
    if rank == 0:
        return None
    return [tir.Var("row_stride", "int64"), tir.IntImm("int64", 1)][-rank:]


def _region(buffer):
    return tir.BufferRegion(
        buffer, [tvm.ir.Range.from_min_extent(tir.const(0, dim.dtype), dim) for dim in buffer.shape]
    )


def _read(buffer, indices):
    return tir.BufferLoad(buffer, list(indices) if len(buffer.shape) else [])


def _dma_stride(buffer, factor):
    return buffer.strides[0] * factor if len(buffer.shape) else tir.IntImm("int64", 0)


def _canonical_description(func):
    """Fold identity casts and match blockize's removal of extent-one axes."""
    func = tir.transform.Simplify()(tvm.IRModule({"main": func}))["main"]

    def rewrite(node):
        if isinstance(node, tir.BlockRealize):
            block = node.block
            keep = [
                index
                for index, axis in enumerate(block.iter_vars)
                if not (isinstance(axis.dom.extent, tir.IntImm) and int(axis.dom.extent) == 1)
            ]
            if len(keep) != len(block.iter_vars):
                block = tir.Block(
                    [block.iter_vars[index] for index in keep],
                    block.reads,
                    block.writes,
                    block.name_hint,
                    block.body,
                    block.init,
                    block.alloc_buffers,
                    block.match_buffers,
                    block.annotations,
                )
                return tir.BlockRealize(
                    [node.iter_values[index] for index in keep], node.predicate, block
                )
        return None

    return func.with_body(tir.stmt_functor.ir_transform(func.body, None, rewrite))


def _copy_pair(
    rows,
    cols,
    rank,
    source_dtype,
    output_dtype,
    source_scope,
    output_scope,
    symbol,
    source_scalar=False,
    source_rank=None,
):
    shape = (cols,) if rank == 1 else (rows, cols)
    source_shape = () if source_scalar else shape[-source_rank:] if source_rank else shape
    factor = cols if rank == 1 else 1
    alignment = (
        {"buffer_dim_align": [[0, rank - 2, 16, 0]]}
        if output_scope != "global" and rank > 1
        else {}
    )
    if rank == 1:

        @T.prim_func
        def desc(a: T.handle, b: T.handle):
            A = T.match_buffer(
                a,
                source_shape,
                source_dtype,
                scope=source_scope,
                strides=_strides(len(source_shape)),
                offset_factor=1,
            )
            B = T.match_buffer(
                b, shape, output_dtype, scope=output_scope, strides=_strides(rank), offset_factor=1
            )
            with T.block("root"):
                T.reads(_region(A))
                T.writes(B[0:cols])
                for j in T.serial(cols):
                    with T.block("copy"):
                        vj = T.axis.spatial(cols, j)
                        B[vj] = T.Cast(output_dtype, _read(A, (vj,)))
    else:

        @T.prim_func
        def desc(a: T.handle, b: T.handle):
            A = T.match_buffer(
                a,
                source_shape,
                source_dtype,
                scope=source_scope,
                strides=_strides(len(source_shape)),
                offset_factor=1,
            )
            B = T.match_buffer(
                b, shape, output_dtype, scope=output_scope, strides=_strides(rank), offset_factor=1
            )
            with T.block("root"):
                T.reads(_region(A))
                T.writes(B[0:rows, 0:cols])
                for i, j in T.grid(rows, cols):
                    with T.block("copy"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        B[vi, vj] = T.Cast(
                            output_dtype, _read(A, (vj,) if source_rank == 1 else (vi, vj))
                        )

    @T.prim_func
    def impl(a: T.handle, b: T.handle):
        A = T.match_buffer(
            a,
            source_shape,
            source_dtype,
            scope=source_scope,
            strides=_strides(len(source_shape)),
            offset_factor=1,
        )
        B = T.match_buffer(
            b, shape, output_dtype, scope=output_scope, strides=_strides(rank), offset_factor=1
        )
        with T.block("root"):
            T.block_attr(alignment)
            T.reads(_region(A))
            T.writes(_region(B))
            T.evaluate(
                T.call_extern(
                    "void",
                    symbol,
                    A.access_ptr("r"),
                    B.access_ptr("w"),
                    T.uint32(rows),
                    T.uint32(cols),
                    T.int64(0) if source_rank == 1 and rank == 2 else _dma_stride(A, factor),
                    _dma_stride(B, factor),
                )
            )

    return desc, impl


def register_elementwise_intrinsics(kind, rows, cols, rank=2, source_rank=None):
    """Describe primitive rectangles independently of graph/operator names."""
    return _register_elementwise_intrinsics(kind, rows, cols, rank, source_rank)


@lru_cache(None)
def _register_elementwise_intrinsics(kind, rows, cols, rank, source_rank):
    if kind not in (
        "copy_i8",
        "copy_i32",
        "copy_f32",
        "widen_i8_i32",
        "fill_i8",
        "fill_i32",
        "fill_f32",
        "fill_widen_i8_i32",
    ) or rank not in (1, 2):
        raise ValueError("Unsupported Gemmini elementwise intrinsic")
    if any(type(value) is not int or not 1 <= value <= 16 for value in (rows, cols)) or (
        rank == 1 and rows != 1
    ):
        raise ValueError("Invalid Gemmini DMA rectangle")
    if source_rank is not None and (
        source_rank != 1
        or rank != 2
        or kind not in ("copy_i8", "copy_i32", "copy_f32", "widen_i8_i32")
    ):
        raise ValueError("Rank-changing DMA supports a vector source and matrix tile")
    fill = kind.startswith("fill_")
    if fill and cols != 1:
        raise ValueError("Scalar broadcasts require one-column DMA")
    scope = SPAD if kind in ("copy_i8", "fill_i8") else ACC
    dtype = (
        "float32"
        if kind in ("copy_f32", "fill_f32")
        else "int8"
        if kind in ("copy_i8", "fill_i8")
        else "int32"
    )
    source_dtype = (
        "int8" if kind in ("copy_i8", "widen_i8_i32", "fill_i8", "fill_widen_i8_i32") else dtype
    )
    source_symbol = {
        "copy_i8": "tvm_gemmini_dma_a",
        "copy_i32": "tvm_gemmini_dma_acc_write",
        "copy_f32": "tvm_gemmini_dma_acc_write",
        "widen_i8_i32": "tvm_gemmini_dma_widen",
        "fill_i8": "tvm_gemmini_dma_a",
        "fill_i32": "tvm_gemmini_dma_acc_write",
        "fill_f32": "tvm_gemmini_dma_acc_write",
        "fill_widen_i8_i32": "tvm_gemmini_dma_widen",
    }.get(kind)
    op = _copy_pair(
        rows, cols, rank, source_dtype, dtype, "global", scope, source_symbol,
        source_scalar=fill, source_rank=source_rank,
    )
    store = _copy_pair(
        rows,
        cols,
        rank,
        dtype,
        dtype,
        scope,
        "global",
        "tvm_gemmini_dma_spad_out" if scope == SPAD else "tvm_gemmini_dma_out",
    )
    result = {"scope": scope}
    for name, (desc, impl) in (("op", op), ("store", store)):
        key = f"gemmini.tir.{kind}.{rank}d.{rows}x{cols}.{name}"
        if source_rank is not None:
            key += f".source{source_rank}d"
        from .gemmini_intrin import register_intrinsic_once
        register_intrinsic_once(key, _canonical_description(desc), impl)
        result[name] = key
    return result


def _relu_reference(value, zero_first):
    zero = tir.const(0, "int8")
    return tir.max(zero, value) if zero_first else tir.max(value, zero)


@lru_cache(None)
def register_relu_intrinsic(rows, cols, rank, zero_first=False):
    """An exact tile operation; its private accumulator is always INT32."""
    if (
        rank not in (1, 2)
        or any(type(value) is not int or not 1 <= value <= 16 for value in (rows, cols))
        or (rank == 1 and rows != 1)
    ):
        raise ValueError("Invalid Gemmini ReLU tile")
    shape = (cols,) if rank == 1 else (rows, cols)
    factor = cols if rank == 1 else 1

    if rank == 1:

        @T.prim_func
        def desc(a: T.handle, b: T.handle):
            A = T.match_buffer(a, shape, "int8", strides=_strides(rank), offset_factor=1)
            B = T.match_buffer(b, shape, "int8", strides=_strides(rank), offset_factor=1)
            with T.block("root"):
                T.reads(_region(A))
                T.writes(_region(B))
                for j in T.serial(cols):
                    with T.block("relu"):
                        vj = T.axis.spatial(cols, j)
                        B[vj] = _relu_reference(A[vj], zero_first)
    else:

        @T.prim_func
        def desc(a: T.handle, b: T.handle):
            A = T.match_buffer(a, shape, "int8", strides=_strides(rank), offset_factor=1)
            B = T.match_buffer(b, shape, "int8", strides=_strides(rank), offset_factor=1)
            with T.block("root"):
                T.reads(_region(A))
                T.writes(_region(B))
                for i, j in T.grid(rows, cols):
                    with T.block("relu"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        B[vi, vj] = _relu_reference(A[vi, vj], zero_first)

    @T.prim_func
    def impl(a: T.handle, b: T.handle):
        A = T.match_buffer(a, shape, "int8", strides=_strides(rank), offset_factor=1)
        B = T.match_buffer(b, shape, "int8", strides=_strides(rank), offset_factor=1)
        with T.block("root"):
            T.reads(_region(A))
            T.writes(_region(B))
            with T.block("relu_readout"):
                T.reads(_region(A))
                T.writes(_region(B))
                C = T.alloc_buffer((rows, 16), "int32", scope=ACC)
                T.evaluate(
                    T.call_extern(
                        "void",
                        "tvm_gemmini_dma_widen",
                        A.access_ptr("r"),
                        C.access_ptr("w"),
                        T.uint32(rows),
                        T.uint32(cols),
                        _dma_stride(A, factor),
                        T.int32(0),
                    )
                )
                T.evaluate(
                    T.call_extern(
                        "void",
                        "tvm_gemmini_dma_relu_out",
                        C.access_ptr("r"),
                        B.access_ptr("w"),
                        T.uint32(rows),
                        T.uint32(cols),
                        T.int64(16),
                        _dma_stride(B, factor),
                    )
                )

    name = f"gemmini.relu_i8.{rank}d.{rows}x{cols}" + (".zero_first" if zero_first else "")
    from .gemmini_intrin import register_intrinsic_once
    register_intrinsic_once(name, _canonical_description(desc), impl)
    return name


@lru_cache(None)
def register_literal_intrinsic(rows, cols, rank, dtype, value_bits):
    """Materialize one typed host scalar, then broadcast it through primitive DMA."""
    if dtype not in ("int8", "int32", "float32"):
        raise ValueError("Unsupported constant DMA dtype")
    value = tir.const(struct.unpack("!d" if dtype == "float32" else "!q", value_bits)[0], dtype)
    if (
        rank not in (1, 2)
        or any(type(extent) is not int or not 1 <= extent <= 16 for extent in (rows, cols))
        or (rank == 1 and rows != 1)
        or not isinstance(value, (tir.IntImm, tir.FloatImm))
    ):
        raise ValueError("Invalid constant DMA tile or literal")
    if cols != 1:
        raise ValueError("Constant DMA fill requires an admitted dtype and one column")
    # Keep the implementation literal as an explicit bit pattern. In v0.19,
    # Tensorize copies its implementation through JSON, whose FloatImm loader
    # cannot read NaN/Inf. A typed reinterpret also preserves signed zero.
    impl_value = (
        tir.reinterpret(
            "float32", tir.IntImm("uint32", struct.unpack("!I", struct.pack("!f", value.value))[0])
        )
        if dtype == "float32"
        else value
    )
    scope = SPAD if dtype == "int8" else ACC
    shape = (cols,) if rank == 1 else (rows, cols)
    if rank == 1:

        @T.prim_func
        def desc(b: T.handle):
            B = T.match_buffer(
                b, shape, dtype, scope=scope, strides=_strides(rank), offset_factor=1
            )
            with T.block("root"):
                T.reads()
                T.writes(_region(B))
                for j in T.serial(cols):
                    with T.block("fill"):
                        vj = T.axis.spatial(cols, j)
                        B[vj] = value
    else:

        @T.prim_func
        def desc(b: T.handle):
            B = T.match_buffer(
                b, shape, dtype, scope=scope, strides=_strides(rank), offset_factor=1
            )
            with T.block("root"):
                T.reads()
                T.writes(_region(B))
                for i, j in T.grid(rows, cols):
                    with T.block("fill"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        B[vi, vj] = value

    @T.prim_func
    def impl(b: T.handle):
        B = T.match_buffer(b, shape, dtype, scope=scope, strides=_strides(rank), offset_factor=1)
        with T.block("root"):
            T.reads()
            T.writes(_region(B))
            with T.block("constant_source"):
                T.reads()
                T.writes(_region(B))
                scalar = T.alloc_buffer((), dtype)
                scalar[()] = impl_value
                T.evaluate(
                    T.call_extern(
                        "void",
                        "tvm_gemmini_dma_a" if dtype == "int8" else "tvm_gemmini_dma_acc_write",
                        scalar.access_ptr("r"),
                        B.access_ptr("w"),
                        T.uint32(rows),
                        T.uint32(cols),
                        T.int64(0),
                        T.int32(0),
                    )
                )

    key = hashlib.sha256(dtype.encode() + value_bits).hexdigest()
    name = f"gemmini.constant.{dtype}.{rank}d.{rows}x{cols}.{key}"
    from .gemmini_intrin import register_intrinsic_once
    register_intrinsic_once(name, _canonical_description(desc), impl)
    return name


def elementwise_blocks(func):
    """Find exact copies, widening, bounded byte affine and INT8 ReLU blocks."""
    from .gemmini_bytes import BYTE_DTYPES
    from .gemmini_affine import affine_match

    matches = []

    def visit(node):
        if (
            not isinstance(node, tir.Block)
            or node.init is not None
            or not isinstance(node.body, tir.BufferStore)
            or not node.iter_vars
            or any(
                axis.iter_type != 0
                or not isinstance(axis.dom.extent, tir.IntImm)
                or int(axis.dom.extent) <= 0
                or not isinstance(axis.dom.min, tir.IntImm)
                or int(axis.dom.min) != 0
                for axis in node.iter_vars
            )
        ):
            return
        axes = [axis.var for axis in node.iter_vars]
        output, value = node.body.buffer, node.body.value
        if len(output.shape) > 5:
            return
        analyzer = tvm.arith.Analyzer()
        if len(node.body.indices) != len(axes) or any(
            not isinstance(offset := analyzer.simplify(index - axis), tir.IntImm)
            or int(offset) < 0
            or not analyzer.can_prove(offset + iterator.dom.extent <= extent)
            for index, axis, iterator, extent in zip(
                node.body.indices, axes, node.iter_vars, output.shape
            )
        ):
            return
        if (
            isinstance(value, tir.BufferLoad)
            and value.dtype == output.dtype
            and output.dtype in ("int8", "int32", "float32", *BYTE_DTYPES)
        ):
            kind, loads = (
                {"int8": "copy_i8", "int32": "copy_i32", "float32": "copy_f32"}.get(output.dtype, "bytes_" + output.dtype),
                [value],
            )
        elif (
            output.dtype in ("int8", "int32", "float32", *BYTE_DTYPES)
            and (selected := copy_selection_loads(value, output.dtype)) is not None
        ):
            kind, loads = "select_copy", selected
        elif isinstance(value, (tir.IntImm, tir.FloatImm)) and output.dtype in (
            "int8",
            "int32",
            "float32",
        ):
            kind, loads = "constant_" + output.dtype, []
        elif (
            isinstance(value, tir.Call)
            and value.op == tvm.ir.Op.get("tir.if_then_else")
            and isinstance(value.args[1], tir.BufferLoad)
            and isinstance(value.args[2], (tir.IntImm, tir.FloatImm))
            and value.args[1].dtype == output.dtype
            and output.dtype in ("int8", "int32", "float32", *BYTE_DTYPES)
        ):
            kind, loads = "padding", [value.args[1]]
        elif isinstance(value, tir.Max) and output.dtype == "int8":
            pairs = ((value.a, value.b), (value.b, value.a))
            reads = [
                load
                for load, zero in pairs
                if isinstance(load, tir.BufferLoad)
                and load.dtype == "int8"
                and isinstance(zero, tir.IntImm)
                and int(zero) == 0
            ]
            if len(reads) != 1:
                return
            kind, loads = "relu_i8", reads
        elif (
            isinstance(value, tir.Cast)
            and value.dtype == "int32"
            and isinstance(value.value, tir.BufferLoad)
            and value.value.dtype == "int8"
        ):
            kind, loads = "widen_i8_i32", [value.value]
        elif affine_match(node) is not None:
            kind, loads = "affine_i8", []
        else:
            return
        # Tensorize proves the access map below. Copies may use sliced,
        # repeated or permuted indices. Scalar DMA is an explicit experiment.
        if any(len(load.buffer.shape) > 5 or load.buffer.data.same_as(output.data) for load in loads):
            return
        scalar_inputs = [index for index, load in enumerate(loads) if not load.buffer.shape]
        if len(scalar_inputs) > 1:
            return
        if scalar_inputs:
            if kind.startswith("bytes_"):
                pass
            elif kind in ("copy_i8", "copy_i32", "copy_f32", "widen_i8_i32"):
                kind = {
                    "copy_i8": "fill_i8",
                    "copy_i32": "fill_i32",
                    "copy_f32": "fill_f32",
                    "widen_i8_i32": "fill_widen_i8_i32",
                }[kind]
            else:
                return
        matches.append((node, kind))

    tir.stmt_functor.post_order_visit(func.body, visit)
    return matches


def schedule_elementwise(
    func, *, tensorize=True, mark_tensorize=False, scalar=False
):
    """Let Tensorize prove native rectangular access maps without padded DMA."""
    if tensorize and mark_tensorize:
        raise ValueError("Choose immediate tensorization or MetaSchedule annotations")
    if not elementwise_blocks(func):
        raise ValueError("Expected supported exact pointwise blocks")
    failure = None
    # Scalar DMA is an explicit experimental schedule, never an automatic
    # placement fallback after rectangular tensorization fails.
    for use_scalar in [scalar]:
        try:
            sch = tir.Schedule(func, debug_mask="all")
            inline_index_constants(sch)
            selections = [
                node for node, kind in elementwise_blocks(sch.mod["main"]) if kind == "select_copy"
            ]
            if selections:
                partition_copy_regions(sch, selections)
            for node, kind in elementwise_blocks(sch.mod["main"]):
                if kind == "padding":
                    block = block_rv(sch, node)
                    loop = sch.get_loops(block)[0]
                    if not sch.can_decompose_padding(block, loop):
                        raise ValueError("TVM cannot prove a constant-padding decomposition")
                    sch.decompose_padding(block, loop)
            regions = elementwise_blocks(sch.mod["main"])
            metadata = []
            for original, kind in regions:
                if kind in ("padding", "select_copy"):
                    raise ValueError("An unscheduled copy selection remains after decomposition")
                block = block_rv(sch, original)
                predicates = []
                current = sch.get(block)
                tir.stmt_functor.post_order_visit(
                    sch.mod["main"].body,
                    lambda stmt: predicates.append(stmt.predicate)
                    if isinstance(stmt, tir.BlockRealize) and stmt.block.same_as(current)
                    else None,
                )
                if any(not isinstance(predicate, tir.IntImm) or not int(predicate)
                       for predicate in predicates):
                    raise ValueError("A predicated copy region cannot become rectangular DMA")
                if len(regions) > 1:
                    # Each region has one writer within this standard local
                    # block scope, even when the external output has many writers.
                    sch.blockize(sch.get_loops(block)[0])
                node = sch.get(block)
                rank = min(2, len(node.iter_vars))
                dimensions = [int(axis.dom.extent) for axis in node.iter_vars[-rank:]]
                factors = [
                    max(value for value in range(1, 17) if extent % value == 0)
                    for extent in dimensions
                ]
                if kind.startswith("bytes_"):
                    size = max(1, (tvm.DataType(node.body.buffer.dtype).bits + 7) // 8)
                    factors[-1] = max(value for value in range(1, 16 // size + 1)
                                      if dimensions[-1] % value == 0)
                    if any(not region.buffer.shape for region in node.reads):
                        factors[-1] = 1
                if kind.startswith(("fill_", "constant_")) or "_scalar_" in kind:
                    factors[-1] = 1
                if kind == "affine_i8":
                    factors[-1] = 1
                if use_scalar:
                    factors = [1] * rank
                metadata.append(
                    _schedule_elementwise_candidate(
                        sch, node, kind, rank, factors, tensorize or mark_tensorize
                    )
                )
            sch.annotate(sch.get_block("root"), "gemmini_regions_scheduled", 1)
            mathematical = sch.mod
            if tensorize:
                if not tvm.meta_schedule.postproc.RewriteTensorize().apply(sch):
                    raise ValueError("TVM Tensorize could not prove the selected DMA rectangles")
                # v0.19 RewriteTensorize logs a failed replacement but still
                # returns true. Apply the same target legality check as DSE
                # before accepting this candidate.
                from .gemmini_memory import lower_memory

                lower_memory(sch.mod)
            info = (
                metadata[0]
                if len(metadata) == 1
                else {"operation": "pointwise_regions", "regions": metadata}
            )
            return sch, mathematical, info
        except (ValueError, tir.schedule.ScheduleError) as error:
            failure = error
    raise failure


def _schedule_elementwise_candidate(sch, node, kind, rank, factors, mark_tensorize):
    rows, cols = factors if rank == 2 else (1, factors[0])
    if kind.startswith("bytes_") or kind == "affine_i8":
        intrinsics = None
    elif kind.startswith("constant_"):
        copy_kind = {"int8": "copy_i8", "int32": "copy_i32", "float32": "copy_f32"}[
            node.body.buffer.dtype
        ]
        intrinsics = dict(register_elementwise_intrinsics(copy_kind, rows, cols, rank))
        literal = node.body.value
        bits = struct.pack("!d" if literal.dtype == "float32" else "!q", literal.value)
        intrinsics["op"] = register_literal_intrinsic(rows, cols, rank, literal.dtype, bits)
    else:
        source = node.body.value.value if kind == "widen_i8_i32" else node.body.value
        source_rank = (
            1
            if kind in ("copy_i8", "copy_i32", "copy_f32", "widen_i8_i32")
            and isinstance(source, tir.BufferLoad)
            and len(source.buffer.shape) == 1
            and rank == 2
            else None
        )
        intrinsics = (
            None
            if kind == "relu_i8"
            else register_elementwise_intrinsics(kind, rows, cols, rank, source_rank)
        )
    block = block_rv(sch, node)
    # TOPI can conservatively infer an entire input region for a single indexed
    # load (for example, reflected indices). State the exact per-iteration
    # accesses through the public schedule API before placing/tensorizing caches.
    # Retain all index-buffer reads, and never narrow differing accesses to the
    # same buffer. v0.19's annotation wrapper requires equal block/buffer ranks.
    reads = []
    tir.stmt_functor.post_order_visit(
        node.body, lambda expr: reads.append(expr) if isinstance(expr, tir.BufferLoad) else None
    )
    analyzer = tvm.arith.Analyzer()
    for index, region in enumerate(node.reads):
        loads = [load for load in reads if load.buffer.same_as(region.buffer)]
        if not loads or len(region.buffer.shape) != len(node.iter_vars):
            continue
        indices = list(loads[0].indices)
        if any(not tvm.ir.structural_equal(load.indices, indices) for load in loads[1:]):
            continue
        if all(
            analyzer.can_prove(part.extent == 1) and analyzer.can_prove(part.min == point)
            for part, point in zip(region.region, indices)
        ):
            continue
        sch.annotate_buffer_access(block, index, "read", lambda *axes: indices)
    loops = sch.get_loops(block)[-rank:]
    split = [sch.split(loop, [None, factor]) for loop, factor in zip(loops, factors)]
    sch.reorder(*[pair[0] for pair in split], *[pair[1] for pair in split])
    if kind in ("relu_i8", "affine_i8") or kind.startswith("bytes_"):
        if mark_tensorize:
            tile = sch.blockize(sch.get_loops(block)[-rank])
            if kind == "relu_i8":
                intrinsic = register_relu_intrinsic(
                    rows, cols, rank, isinstance(node.body.value.a, tir.IntImm)
                )
            elif kind == "affine_i8":
                from .gemmini_affine import register_affine_intrinsic
                intrinsic = register_affine_intrinsic(node, rows, rank)
            else:
                from .gemmini_bytes import register_byte_copy
                source_rank = len(node.body.value.buffer.shape)
                intrinsic = register_byte_copy(rows, cols, rank, node.body.buffer.dtype,
                                               source_rank if source_rank < rank else None)
            sch.annotate(tile, "meta_schedule.auto_tensorize", intrinsic)
        metadata = {"operation": kind, "tile_rows": rows, "tile_cols": cols}
        if kind == "affine_i8":
            from .gemmini_affine import affine_match
            match = affine_match(node)
            metadata.update(accumulator_partial_abs_bound=match["accumulator_partial_abs_bound"],
                            pe_partial_abs_bound=match["pe_partial_abs_bound"],
                            coefficients=[c for _,c,_ in match["terms"]], offset=match["offset"],
                            operand_scale_bits=[struct.unpack("I",struct.pack("f",s))[0] for _,_,s in match["terms"]],
                            per_operand_rounding=match["per_operand_rounding"],
                            scaled_readout=match["readout"] is not None,
                            readout_proof=match["readout"])
        return metadata
    outer = split[-1][0]
    write = sch.cache_write(block, 0, intrinsics["scope"])
    sch.reverse_compute_at(write, outer, preserve_unit_loops=True)
    if rank == 2:
        sch.storage_align(block, 0, axis=-2, factor=16, offset=0)
    for target, key in ((block, "op"), (write, "store")):
        loop = sch.get_loops(target)[-rank]
        if mark_tensorize:
            tiled = sch.blockize(loop)
            if key == "op" and rank == 2:
                sch.storage_align(tiled, 0, axis=-2, factor=16, offset=0)
            sch.annotate(tiled, "meta_schedule.auto_tensorize", intrinsics[key])
    return {"operation": kind, "tile_rows": rows, "tile_cols": cols}


def make_elementwise_schedule(func, *, capabilities=None, scalar=False):
    """Finalize native DMA scheduling through the common scope/ABI lowering."""
    from .gemmini_tensorir import lower_tensorir_schedule

    sch, mathematical, metadata = schedule_elementwise(func, scalar=scalar)
    return lower_tensorir_schedule(
        func, sch, mathematical=mathematical, workload_metadata=metadata, capabilities=capabilities
    )
