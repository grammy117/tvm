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
"""Exact signed-INT8 sums, scheduled as array reductions against ones.

The workload remains a sum. Standard index maps, reindexing, caches and
reduction decomposition expose its DMA and array tiles to tensorization.
Only a scalar one is constructed by the CPU; every reduction runs on the array.
"""

from functools import lru_cache
import math

import tvm
from tvm import tir
from tvm.script import tir as T

from .gemmini_elementwise import _canonical_description
from .gemmini_intrin import ACC, SPAD, register_intrinsics
from .gemmini_regions import block_rv


def _signed_byte_source(value, producers=None):
    """Find pure signed widening from an INT8 load, including internal casts."""
    inline, visited = [], set()
    while value.dtype != "int8":
        if (
            isinstance(value, tir.Cast)
            and value.dtype in ("int32", "int64")
            and value.value.dtype in ("int8", "int32", "int64")
            and tvm.DataType(value.value.dtype).bits <= tvm.DataType(value.dtype).bits
        ):
            value = value.value
        elif isinstance(value, tir.BufferLoad) and producers is not None:
            candidates = producers.get(value.buffer.data, [])
            if len(candidates) != 1 or candidates[0] in visited:
                return None, ()
            producer = candidates[0]
            if producer.init is not None or any(
                axis.iter_type != tir.IterVar.DataPar for axis in producer.iter_vars
            ):
                return None, ()
            inline.append(producer)
            visited.add(producer)
            value = producer.body.value
        else:
            return None, ()
    return (value, tuple(inline)) if isinstance(value, tir.BufferLoad) else (None, ())


def reduction_blocks(func):
    """Find bounded INT32/INT64 sums of signed INT8 values, including fused casts."""
    producers, blocks = {}, []

    def collect(node):
        if isinstance(node, tir.Block) and isinstance(node.body, tir.BufferStore):
            blocks.append(node)
            producers.setdefault(node.body.buffer.data, []).append(node)

    tir.stmt_functor.post_order_visit(func.body, collect)
    result = []
    equal = tvm.ir.structural_equal
    for block in blocks:
        if not isinstance(block.init, tir.BufferStore) or len(block.writes) != 1:
            continue
        store, init = block.body, block.init
        value = store.value
        if (
            store.buffer.dtype not in ("int32", "int64")
            or not init.buffer.same_as(store.buffer)
            or not equal(init.indices, store.indices)
            or not isinstance(init.value, tir.IntImm)
            or int(init.value) != 0
            or not isinstance(value, tir.Add)
            or not isinstance(value.a, tir.BufferLoad)
            or not value.a.buffer.same_as(store.buffer)
            or not equal(value.a.indices, store.indices)
        ):
            continue
        source, inline = _signed_byte_source(value.b, producers)
        if source is None:
            continue
        if source.buffer.data.same_as(store.buffer.data):
            continue
        axes = list(block.iter_vars)
        if not axes or any(
            axis.iter_type not in (tir.IterVar.DataPar, tir.IterVar.CommReduce)
            or not isinstance(axis.dom.min, tir.IntImm)
            or int(axis.dom.min) != 0
            or not isinstance(axis.dom.extent, tir.IntImm)
            or int(axis.dom.extent) <= 0
            for axis in axes
        ):
            continue
        spatial = [i for i, axis in enumerate(axes) if axis.iter_type == tir.IterVar.DataPar]
        reduction = [i for i, axis in enumerate(axes) if axis.iter_type == tir.IterVar.CommReduce]
        depth = math.prod(int(axes[i].dom.extent) for i in reduction)
        # Every prefix is in [-128*K,127*K], hence exactly representable in INT32.
        if not reduction or depth > (1 << 24):
            continue
        if any(
            not isinstance(index, tir.Var)
            and not (isinstance(index, tir.IntImm) and int(index) == 0)
            for index in store.indices
        ):
            continue
        used = set()
        for index in store.indices:
            tir.stmt_functor.post_order_visit(
                index, lambda node: used.add(node) if isinstance(node, tir.Var) else None
            )
        required = {axes[i].var for i in spatial if int(axes[i].dom.extent) > 1}
        if not required <= used <= {axes[i].var for i in spatial}:
            continue
        rows = math.prod(int(axes[i].dom.extent) for i in spatial)
        index_dtype = axes[0].var.dtype
        if index_dtype not in ("int32", "int64") or any(
            axis.var.dtype != index_dtype for axis in axes
        ):
            continue
        if max(rows, depth) > (1 << (tvm.DataType(index_dtype).bits - 1)) - 1:
            # Fusing axes must not overflow the original index arithmetic.
            continue
        result.append((block, (rows, depth), (spatial, reduction), inline))
    return result


def _sum_update(previous, value, wide, source_casts):
    for dtype in reversed(source_casts):
        value = tir.Cast(dtype, value)
    previous = tir.Cast("int64", previous) if wide else previous
    total = previous + value
    return tir.Cast("int32", total) if wide else total


def reduction_metadata(match):
    node, (rows, depth), _, _ = match
    metadata = {"operation": "sum_i8_i32", "rows": rows, "depth": depth}
    if node.body.buffer.dtype == "int64":
        metadata.update(
            output_dtype="int64",
            accumulator_dtype="int32",
            prefix_min=-128 * depth,
            prefix_max=127 * depth,
            exact_accumulator_narrowing=True,
        )
    return metadata


@lru_cache(None)
def register_reduction_intrinsics(rows, depth, wide=False, source_casts=None):
    """A sum reference and an implementation using the existing primitive ABI."""
    if not 1 <= rows <= 16 or not 1 <= depth <= 16:
        raise ValueError("Gemmini sum tiles require dimensions in [1,16]")
    source_casts = source_casts or (("int64",) if wide else ("int32",))

    @T.prim_func
    def init_desc(c: T.handle):
        C = T.match_buffer(
            c, (rows, 1), "int32", scope=ACC, strides=[T.int64(), T.int64(1)], offset_factor=1
        )
        with T.block("root"):
            T.reads()
            T.writes(C[0:rows, 0:1])
            for i in T.serial(rows):
                with T.block("init"):
                    vi = T.axis.spatial(rows, i)
                    C[vi, 0] = T.int32(0)

    @T.prim_func
    def update_desc(a: T.handle, c: T.handle):
        A = T.match_buffer(
            a, (rows, depth), "int8", scope=SPAD, strides=[T.int64(), T.int64(1)], offset_factor=1
        )
        C = T.match_buffer(
            c, (rows, 1), "int32", scope=ACC, strides=[T.int64(), T.int64(1)], offset_factor=1
        )
        with T.block("root"):
            T.reads(C[0:rows, 0:1], A[0:rows, 0:depth])
            T.writes(C[0:rows, 0:1])
            for i, k in T.grid(rows, depth):
                with T.block("sum"):
                    vi, vk = T.axis.remap("SR", [i, k])
                    C[vi, 0] = _sum_update(C[vi, 0], A[vi, vk], wide, source_casts)

    @T.prim_func
    def update_impl(a: T.handle, c: T.handle):
        A = T.match_buffer(
            a, (rows, depth), "int8", scope=SPAD, strides=[T.int64(), T.int64(1)], offset_factor=1
        )
        C = T.match_buffer(
            c, (rows, 1), "int32", scope=ACC, strides=[T.int64(), T.int64(1)], offset_factor=1
        )
        with T.block("root"):
            T.reads(C[0:rows, 0:1], A[0:rows, 0:depth])
            T.writes(C[0:rows, 0:1])
            # Tensorize substitutes the implementation root's body. Keep the
            # owned buffers in a nested block so their allocations are retained.
            with T.block("ones_product"):
                T.reads(C[0:rows, 0:1], A[0:rows, 0:depth])
                T.writes(C[0:rows, 0:1])
                # Describe the full physical row: only column zero is loaded
                # and consumed, but each hardware row occupies sixteen bytes.
                B = T.alloc_buffer((depth, 16), "int8", scope=SPAD)
                one = T.alloc_buffer((), "int8")
                one[()] = T.int8(1)
                T.evaluate(
                    T.call_extern(
                        "void",
                        "tvm_gemmini_dma_b",
                        one.access_ptr("r"),
                        B.access_ptr("w"),
                        T.uint32(depth),
                        T.uint32(1),
                        T.int64(0),
                        T.int32(1),
                    )
                )
                T.evaluate(
                    T.call_extern(
                        "void",
                        "tvm_gemmini_compute",
                        A.access_ptr("r"),
                        B.access_ptr("r"),
                        C.access_ptr("rw"),
                        T.uint32(rows),
                        T.uint32(1),
                        T.uint32(depth),
                        T.int32(1),
                    )
                )

    names = register_intrinsics(rows, 1, depth)
    pairs = {
        key: (tir.TensorIntrin.get(names[key]).desc, tir.TensorIntrin.get(names[key]).impl)
        for key in ("init", "load_a", "store")
    }
    pairs["init"] = (init_desc, pairs["init"][1])
    pairs["compute"] = (update_desc, update_impl)
    result = {}
    for kind, (desc, impl) in pairs.items():
        name = f"gemmini.sum.{kind}.{rows}x{depth}" + (".wide" if wide else "")
        name += "." + "_".join(source_casts)
        from .gemmini_intrin import register_intrinsic_once
        register_intrinsic_once(name, _canonical_description(desc), impl)
        result[kind] = name
    return result


def schedule_reduction(func, *, tensorize=True, mark_tensorize=False, scalar=False):
    """Normalize and place bounded sum tiles through ordinary schedule APIs."""
    if tensorize and mark_tensorize:
        raise ValueError("Choose immediate tensorization or MetaSchedule annotations")
    matches = reduction_blocks(func)
    if len(matches) != 1:
        raise ValueError("Expected one bounded integer sum of signed INT8 values")
    node, (rows, depth), (spatial, reduction), inline = matches[0]
    sch = tir.Schedule(func, debug_mask="all")
    block = block_rv(sch, node)
    inline_blocks = [block_rv(sch, producer) for producer in inline]
    for producer in inline_blocks:
        sch.compute_inline(producer)
    wide = node.body.buffer.dtype == "int64"
    extents = [int(axis.dom.extent) for axis in node.iter_vars]
    dtype = node.iter_vars[0].var.dtype

    def flatten(indices, group):
        value = tir.const(0, dtype)
        for index in group:
            value = value * extents[index] + indices[index]
        return value

    sch.transform_block_layout(
        block,
        tir.IndexMap.from_func(
            lambda *i: (flatten(i, spatial), flatten(i, reduction)),
            ndim=len(extents),
            index_dtype=dtype,
        ),
    )
    sch.reindex_cache_write(
        block,
        0,
        "global",
        tir.IndexMap.from_func(
            lambda i, k: (i if rows > 1 else tir.const(0, dtype), tir.const(0, dtype)),
            index_dtype=dtype,
        ),
    )
    if wide:
        # The admission proof bounds every prefix by [-128*K, 127*K], with
        # K <= 2**24. Narrow only the internal reduction allocation; ordinary
        # cache writeback retains the required INT64 output conversion.
        sch.unsafe_set_dtype(block, 0, "int32")
    update = sch.get(block).body.value
    update = update.value if wide else update
    source_casts, value = [], update.b
    while isinstance(value, tir.Cast):
        source_casts.append(value.dtype)
        value = value.value
    source, _ = _signed_byte_source(update.b)
    if source is None:
        raise ValueError("The inlined sum no longer reads a pure widened INT8 value")
    source = source.buffer
    source_index = next(
        i for i, region in enumerate(sch.get(block).reads) if region.buffer.same_as(source)
    )
    gather = sch.reindex_cache_read(
        block,
        source_index,
        "global",
        tir.IndexMap.from_func(
            lambda i, k: (i if rows > 1 else tir.const(0, dtype), k), index_dtype=dtype
        ),
    )

    # Reindex makes arbitrary input layouts explicit. Placement bounds the
    # CPU gather to the currently consumed DMA tile; external inputs never pad.
    def tile(extent):
        return 1 if scalar else max(i for i in range(1, 17) if extent % i == 0)

    m, k = tile(rows), tile(depth)
    sch.annotate(sch.get_block("root"), "gemmini.sum_rows", m)
    sch.annotate(sch.get_block("root"), "gemmini.sum_chunk", k)
    i, r = sch.get_loops(block)
    io, ii = sch.split(i, [None, m])
    ko, ki = sch.split(r, [None, k])
    sch.reorder(io, ko, ii, ki)
    write = sch.cache_write(block, 0, ACC)
    sch.reverse_compute_at(write, io, preserve_unit_loops=True)
    load = sch.cache_read(block, 0, SPAD)
    sch.compute_at(load, ko, preserve_unit_loops=True)
    sch.storage_align(load, 0, axis=-2, factor=16, offset=0)
    sch.storage_align(block, 0, axis=-2, factor=16, offset=0)
    sch.compute_at(gather, ko, preserve_unit_loops=True)
    for consumer in sch.get_consumers(write):
        sch.reverse_compute_at(consumer, io, preserve_unit_loops=True)
        for downstream in sch.get_consumers(consumer):
            sch.reverse_compute_at(downstream, io, preserve_unit_loops=True)
    init = sch.decompose_reduction(block, ko)
    mathematical = sch.mod
    intrinsics = register_reduction_intrinsics(m, k, wide, tuple(source_casts))
    for target, axes, kind in (
        (init, 1, "init"),
        (load, 2, "load_a"),
        (block, 2, "compute"),
        (write, 2, "store"),
    ):
        loop = sch.get_loops(target)[-axes]
        if tensorize or mark_tensorize:
            tiled = sch.blockize(loop)
            if kind != "store":
                sch.storage_align(tiled, 0, axis=-2, factor=16, offset=0)
        if tensorize:
            sch.tensorize(tiled, intrinsics[kind])
        elif mark_tensorize:
            sch.annotate(tiled, "meta_schedule.auto_tensorize", intrinsics[kind])
    return sch, mathematical, reduction_metadata(matches[0])


def make_reduction_schedule(func, *, capabilities=None):
    """Finalize a sum schedule with the shared resource/ownership lowering."""
    from .gemmini_tensorir import lower_tensorir_schedule

    sch, mathematical, metadata = schedule_reduction(func)
    return lower_tensorir_schedule(
        func, sch, mathematical=mathematical, workload_metadata=metadata, capabilities=capabilities
    )
