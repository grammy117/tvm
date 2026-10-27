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
"""Values-only byte max windows using accumulator pooled readout.

Static index gathering is an ordinary visible TensorIR producer. The maximum
itself executes in the store controller. Physical pooling has no padded lanes;
any source padding remains explicit in the mathematical input producer.
"""

import math
from functools import lru_cache

import tvm
from tvm import tir
from tvm.script import tir as T

from .gemmini_elementwise import _canonical_description
from .gemmini_intrin import ACC, register_intrinsic_once
from .gemmini_regions import block_rv


def pool_blocks(func):
    result = []

    def visit(node):
        if (
            not isinstance(node, tir.Block)
            or not isinstance(node.body, tir.BufferStore)
            or not isinstance(node.init, tir.BufferStore)
        ):
            return
        store, init = node.body, node.init
        if (
            store.buffer.dtype != "int8"
            or not isinstance(store.value, tir.Max)
            or not isinstance(init.value, tir.IntImm)
            or int(init.value) != -128
        ):
            return
        if not init.buffer.same_as(store.buffer) or not tvm.ir.structural_equal(
            init.indices, store.indices
        ):
            return
        left, right = store.value.a, store.value.b
        if (
            not isinstance(left, tir.BufferLoad)
            or not left.buffer.same_as(store.buffer)
            or not tvm.ir.structural_equal(left.indices, store.indices)
            or not isinstance(right, tir.BufferLoad)
            or right.dtype != "int8"
            or right.buffer.data.same_as(store.buffer.data)
        ):
            return
        if any(
            not isinstance(axis.dom.extent, tir.IntImm)
            or int(axis.dom.extent) <= 0
            or not isinstance(axis.dom.min, tir.IntImm)
            or int(axis.dom.min) != 0
            or axis.iter_type not in (tir.IterVar.DataPar, tir.IterVar.CommReduce)
            for axis in node.iter_vars
        ):
            return
        spatial = [
            i for i, axis in enumerate(node.iter_vars) if axis.iter_type == tir.IterVar.DataPar
        ]
        reduction = [
            i for i, axis in enumerate(node.iter_vars) if axis.iter_type == tir.IterVar.CommReduce
        ]
        depth = math.prod(int(node.iter_vars[i].dom.extent) for i in reduction)
        if spatial and depth in (4, 9):
            result.append((node, spatial, reduction, math.isqrt(depth)))

    tir.stmt_functor.post_order_visit(func.body, visit)
    return result


@lru_cache(None)
def register_pool_intrinsic(size):
    if size not in (2, 3):
        raise ValueError("Byte max windows require 2x2 or 3x3")
    depth = size * size

    @T.prim_func
    def desc(a: T.handle, b: T.handle):
        A = T.match_buffer(a, (1, depth), "int8", strides=[T.int64(), T.int64(1)], offset_factor=1)
        B = T.match_buffer(b, (1, 1), "int8", strides=[T.int64(), T.int64(1)], offset_factor=1)
        with T.block("root"):
            T.reads(A[0:1, 0:depth])
            T.writes(B[0:1, 0:1])
            with T.init():
                with T.block("max_init"):
                    T.reads()
                    T.writes(B[0:1, 0:1])
                    B[0, 0] = T.int8(-128)
            for i, k in T.grid(1, depth):
                with T.block("max"):
                    vi, vk = T.axis.remap("SR", [i, k])
                    B[vi, 0] = T.max(B[vi, 0], A[vi, vk])

    @T.prim_func
    def impl(a: T.handle, b: T.handle):
        A = T.match_buffer(a, (1, depth), "int8", strides=[T.int64(), T.int64(1)], offset_factor=1)
        B = T.match_buffer(b, (1, 1), "int8", strides=[T.int64(), T.int64(1)], offset_factor=1)
        with T.block("root"):
            T.reads(A[0:1, 0:depth])
            T.writes(B[0:1, 0:1])
            with T.block("pooled_readout"):
                T.reads(A[0:1, 0:depth])
                T.writes(B[0:1, 0:1])
                C = T.alloc_buffer((depth, 16), "int32", scope=ACC)
                T.evaluate(
                    T.call_extern(
                        "void",
                        "tvm_gemmini_dma_widen",
                        A.access_ptr("r"),
                        C.access_ptr("w"),
                        T.uint32(depth),
                        T.uint32(1),
                        T.int64(1),
                        T.int32(0),
                    )
                )
                T.evaluate(
                    T.call_extern(
                        "void",
                        "tvm_gemmini_dma_pool_out",
                        C.access_ptr("r"),
                        B.access_ptr("w"),
                        T.uint32(size),
                    )
                )

    name = f"gemmini.pool.max_i8.{size}x{size}"
    register_intrinsic_once(name, _canonical_description(desc), impl)
    return name


def schedule_pool(func, *, tensorize=True, mark_tensorize=False, scalar=False):
    del scalar
    if tensorize and mark_tensorize:
        raise ValueError("Choose immediate tensorization or annotations")
    matches = pool_blocks(func)
    if len(matches) != 1:
        raise ValueError("Expected one values-only byte max window")
    node, spatial, reduction, size = matches[0]
    extents = [int(axis.dom.extent) for axis in node.iter_vars]
    rows = math.prod(extents[i] for i in spatial)
    dtype = node.iter_vars[0].var.dtype

    def flatten(indices, group):
        value = tir.const(0, dtype)
        for index in group:
            value = value * extents[index] + indices[index]
        return value

    sch = tir.Schedule(func, debug_mask="all")
    block = block_rv(sch, node)
    sch.transform_block_layout(
        block,
        tir.IndexMap.from_func(
            lambda *indices: (flatten(indices, spatial), flatten(indices, reduction)),
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
    current = sch.get(block)
    source = current.body.value.b.buffer
    source_index = next(
        i for i, region in enumerate(current.reads) if region.buffer.same_as(source)
    )
    gather = sch.reindex_cache_read(
        block,
        source_index,
        "global",
        tir.IndexMap.from_func(lambda i, k: (i, k), index_dtype=dtype),
    )
    i, k = sch.get_loops(block)
    io, ii = sch.split(i, [None, 1])
    sch.reorder(io, ii, k)
    sch.compute_at(gather, io, preserve_unit_loops=True)
    for consumer in sch.get_consumers(block):
        sch.reverse_compute_at(consumer, io, preserve_unit_loops=True)
    intrinsic = register_pool_intrinsic(size)
    tile = sch.blockize(sch.get_loops(block)[-2])
    mathematical = sch.mod
    if tensorize:
        sch.tensorize(tile, intrinsic)
        # Keep initialization outside Block.init after tensorization. Stock
        # postprocessors operate on ordinary statement/block schedule refs.
        sch.decompose_reduction(tile, io)
    elif mark_tensorize:
        sch.annotate(tile, "meta_schedule.auto_tensorize", intrinsic)
    sch.annotate(sch.get_block("root"), "gemmini.pool_scheduled", 1)
    return (
        sch,
        mathematical,
        {
            "operation": "max_pool_i8",
            "window": [size, size],
            "values_only": True,
            "physical_pool_padding": 0,
            "host_index_gather": True,
        },
    )


def make_pool_schedule(func, *, capabilities=None):
    from .gemmini_tensorir import lower_tensorir_schedule

    sch, mathematical, metadata = schedule_pool(func)
    return lower_tensorir_schedule(
        func, sch, mathematical=mathematical, workload_metadata=metadata, capabilities=capabilities
    )
