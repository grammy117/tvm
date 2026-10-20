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
"""Exact integer copies, widening and addition with native DMA intrinsics.

The reference programs remain mathematical pointwise blocks. Scheduling caches
their output in real SRAM scopes, then tensorizes rectangular regions. No
whole-operation C kernel or CPU arithmetic is hidden inside the intrinsic.
"""

from functools import lru_cache

import tvm
from tvm import tir
from tvm.script import tir as T

from .gemmini_intrin import ACC, SPAD


def _strides(rank):
    if rank == 0:
        return None
    return [tir.Var("row_stride", "int64"), tir.IntImm("int64", 1)][-rank:]


def _region(buffer):
    return tir.BufferRegion(
        buffer, [tvm.ir.Range.from_min_extent(tir.const(0, dim.dtype), dim) for dim in buffer.shape]
    )


def _add_reference(left, right, wrapping):
    if wrapping:
        return tir.Cast("int32", tir.Cast("int64", left) + tir.Cast("int64", right))
    return left + right


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
):
    shape = (cols,) if rank == 1 else (rows, cols)
    source_shape = () if source_scalar else shape
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
                        B[vi, vj] = T.Cast(output_dtype, _read(A, (vi, vj)))

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
                    _dma_stride(A, factor),
                    _dma_stride(B, factor),
                )
            )

    return desc, impl


@lru_cache(None)
def register_elementwise_intrinsics(kind, rows, cols, rank=2):
    """Describe primitive rectangles independently of graph/operator names."""
    if kind not in (
        "copy_i8",
        "copy_i32",
        "widen_i8_i32",
        "add_i32",
        "add_i32_wrapping",
        "fill_i8",
        "fill_i32",
        "fill_widen_i8_i32",
        "add_i32_scalar_a",
        "add_i32_scalar_b",
        "add_i32_wrapping_scalar_a",
        "add_i32_wrapping_scalar_b",
    ) or rank not in (1, 2):
        raise ValueError("Unsupported Gemmini elementwise intrinsic")
    if any(type(value) is not int or not 1 <= value <= 16 for value in (rows, cols)) or (
        rank == 1 and rows != 1
    ):
        raise ValueError("Invalid Gemmini DMA rectangle")
    scalar_a, scalar_b = kind.endswith("_scalar_a"), kind.endswith("_scalar_b")
    fill = kind.startswith("fill_")
    if (fill or scalar_a or scalar_b) and cols != 1:
        raise ValueError("Scalar broadcasts require one-column DMA")
    scope = SPAD if kind in ("copy_i8", "fill_i8") else ACC
    dtype = "int8" if kind in ("copy_i8", "fill_i8") else "int32"
    source_dtype = (
        "int8" if kind in ("copy_i8", "widen_i8_i32", "fill_i8", "fill_widen_i8_i32") else "int32"
    )
    source_symbol = {
        "copy_i8": "tvm_gemmini_dma_a",
        "copy_i32": "tvm_gemmini_dma_acc_write",
        "widen_i8_i32": "tvm_gemmini_dma_widen",
        "fill_i8": "tvm_gemmini_dma_a",
        "fill_i32": "tvm_gemmini_dma_acc_write",
        "fill_widen_i8_i32": "tvm_gemmini_dma_widen",
    }.get(kind)
    shape = (cols,) if rank == 1 else (rows, cols)
    factor = cols if rank == 1 else 1
    a_shape, b_shape = () if scalar_a else shape, () if scalar_b else shape
    wrapping = kind.startswith("add_i32_wrapping")
    if kind.startswith("add_i32"):
        if rank == 1:

            @T.prim_func
            def desc(a: T.handle, b: T.handle, c: T.handle):
                A = T.match_buffer(
                    a, a_shape, "int32", strides=_strides(len(a_shape)), offset_factor=1
                )
                B = T.match_buffer(
                    b, b_shape, "int32", strides=_strides(len(b_shape)), offset_factor=1
                )
                C = T.match_buffer(
                    c, shape, "int32", scope=ACC, strides=_strides(rank), offset_factor=1
                )
                with T.block("root"):
                    T.reads(_region(A), _region(B))
                    T.writes(C[0:cols])
                    for j in T.serial(cols):
                        with T.block("add"):
                            vj = T.axis.spatial(cols, j)
                            C[vj] = _add_reference(_read(A, (vj,)), _read(B, (vj,)), wrapping)
        else:

            @T.prim_func
            def desc(a: T.handle, b: T.handle, c: T.handle):
                A = T.match_buffer(
                    a, a_shape, "int32", strides=_strides(len(a_shape)), offset_factor=1
                )
                B = T.match_buffer(
                    b, b_shape, "int32", strides=_strides(len(b_shape)), offset_factor=1
                )
                C = T.match_buffer(
                    c, shape, "int32", scope=ACC, strides=_strides(rank), offset_factor=1
                )
                with T.block("root"):
                    T.reads(_region(A), _region(B))
                    T.writes(C[0:rows, 0:cols])
                    for i, j in T.grid(rows, cols):
                        with T.block("add"):
                            vi, vj = T.axis.remap("SS", [i, j])
                            C[vi, vj] = _add_reference(
                                _read(A, (vi, vj)), _read(B, (vi, vj)), wrapping
                            )

        @T.prim_func
        def impl(a: T.handle, b: T.handle, c: T.handle):
            A = T.match_buffer(a, a_shape, "int32", strides=_strides(len(a_shape)), offset_factor=1)
            B = T.match_buffer(b, b_shape, "int32", strides=_strides(len(b_shape)), offset_factor=1)
            C = T.match_buffer(
                c, shape, "int32", scope=ACC, strides=_strides(rank), offset_factor=1
            )
            with T.block("root"):
                T.reads(_region(A), _region(B))
                T.writes(_region(C))
                T.evaluate(
                    T.call_extern(
                        "void",
                        "tvm_gemmini_dma_acc_write",
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
                        "tvm_gemmini_dma_acc_add",
                        B.access_ptr("r"),
                        C.access_ptr("rw"),
                        T.uint32(rows),
                        T.uint32(cols),
                        _dma_stride(B, factor),
                        T.int32(0),
                    )
                )

        op = desc, impl
    else:
        op = _copy_pair(
            rows,
            cols,
            rank,
            source_dtype,
            dtype,
            "global",
            scope,
            source_symbol,
            source_scalar=fill,
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
        tir.TensorIntrin.register(key, _canonical_description(desc), impl)
        result[name] = key
    return result


def elementwise_blocks(func):
    """Find exact direct integer copy, widening and INT32 addition blocks."""
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
        if not tvm.ir.structural_equal(node.body.indices, axes):
            return
        if (
            isinstance(value, tir.BufferLoad)
            and value.dtype == output.dtype
            and output.dtype in ("int8", "int32")
        ):
            kind, loads = "copy_i8" if output.dtype == "int8" else "copy_i32", [value]
        elif (
            isinstance(value, tir.Cast)
            and value.dtype == "int32"
            and isinstance(value.value, tir.BufferLoad)
            and value.value.dtype == "int8"
        ):
            kind, loads = "widen_i8_i32", [value.value]
        elif (
            isinstance(value, tir.Add)
            and value.dtype == "int32"
            and isinstance(value.a, tir.BufferLoad)
            and isinstance(value.b, tir.BufferLoad)
        ):
            kind, loads = "add_i32", [value.a, value.b]
        elif (
            isinstance(value, tir.Cast)
            and value.dtype == "int32"
            and isinstance(value.value, tir.Add)
            and value.value.dtype == "int64"
            and all(
                isinstance(expr, tir.Cast)
                and expr.dtype == "int64"
                and isinstance(expr.value, tir.BufferLoad)
                and expr.value.dtype == "int32"
                for expr in (value.value.a, value.value.b)
            )
        ):
            kind, loads = "add_i32_wrapping", [value.value.a.value, value.value.b.value]
        else:
            return
        # Tensorize proves the access map below. Copies may use sliced,
        # repeated or permuted indices, with scalar DMA as a legal fallback.
        if any(load.buffer.data.same_as(output.data) for load in loads):
            return
        scalar_inputs = [index for index, load in enumerate(loads) if not load.buffer.shape]
        if len(scalar_inputs) > 1:
            return
        if scalar_inputs:
            if kind in ("copy_i8", "copy_i32", "widen_i8_i32"):
                kind = {
                    "copy_i8": "fill_i8",
                    "copy_i32": "fill_i32",
                    "widen_i8_i32": "fill_widen_i8_i32",
                }[kind]
            else:
                kind += "_scalar_a" if scalar_inputs[0] == 0 else "_scalar_b"
        matches.append((node, kind))

    tir.stmt_functor.post_order_visit(func.body, visit)
    return matches


def schedule_elementwise(func, *, tensorize=True, mark_tensorize=False, scalar=False):
    """Let Tensorize prove native rectangular access maps without padded DMA."""
    if tensorize and mark_tensorize:
        raise ValueError("Choose immediate tensorization or MetaSchedule annotations")
    matches = elementwise_blocks(func)
    if len(matches) != 1:
        raise ValueError("Expected one supported exact integer pointwise block")
    node, kind = matches[0]
    rank = min(2, len(node.iter_vars))
    dimensions = [int(axis.dom.extent) for axis in node.iter_vars[-rank:]]
    # Divisor tiles keep every DMA inside the original buffer. Prime tails can
    # use smaller rectangles; no CPU packing or speculative padding is hidden.
    factors = [max(value for value in range(1, 17) if extent % value == 0) for extent in dimensions]
    if kind.startswith("fill_") or "_scalar_" in kind:
        factors[-1] = 1
    choices = [[1] * rank] if scalar else [factors]
    if tensorize and factors != [1] * rank and not scalar:
        choices.append([1] * rank)
    failure = None
    for selected in choices:
        try:
            return _schedule_elementwise_candidate(
                func, node, kind, rank, selected, tensorize, mark_tensorize
            )
        except tir.schedule.ScheduleError as error:
            failure = error
    raise failure


def _schedule_elementwise_candidate(func, node, kind, rank, factors, tensorize, mark_tensorize):
    rows, cols = factors if rank == 2 else (1, factors[0])
    intrinsics = register_elementwise_intrinsics(kind, rows, cols, rank)
    sch = tir.Schedule(func, debug_mask="all")
    block = sch.get_block(node.name_hint)
    loops = sch.get_loops(block)[-rank:]
    split = [sch.split(loop, [None, factor]) for loop, factor in zip(loops, factors)]
    sch.reorder(*[pair[0] for pair in split], *[pair[1] for pair in split])
    outer = split[-1][0]
    write = sch.cache_write(block, 0, intrinsics["scope"])
    sch.reverse_compute_at(write, outer, preserve_unit_loops=True)
    if rank == 2:
        sch.storage_align(block, 0, axis=-2, factor=16, offset=0)
    mathematical = sch.mod
    for target, key in ((block, "op"), (write, "store")):
        loop = sch.get_loops(target)[-rank]
        if tensorize:
            sch.tensorize(loop, intrinsics[key])
        elif mark_tensorize:
            tiled = sch.blockize(loop)
            sch.annotate(tiled, "meta_schedule.auto_tensorize", intrinsics[key])
    return sch, mathematical, {"operation": kind, "tile_rows": rows, "tile_cols": cols}


def make_elementwise_schedule(func, *, capabilities=None):
    """Finalize native DMA scheduling through the common scope/ABI lowering."""
    from .gemmini_tensorir import lower_tensorir_schedule

    sch, mathematical, metadata = schedule_elementwise(func)
    return lower_tensorir_schedule(
        func, sch, mathematical=mathematical, workload_metadata=metadata, capabilities=capabilities
    )
