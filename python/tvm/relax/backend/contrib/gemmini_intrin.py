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
"""Tensor intrinsics for explicit Gemmini scratchpad and accumulator scopes.

Descriptions are ordinary mathematical copies and reductions. Implementations
carry local access pointers until target memory lowering assigns physical rows.
No graph names, generated block names or text parsing select an implementation.
"""

from functools import lru_cache
from threading import Lock

from tvm import tir
from tvm.script import tir as T

SPAD = "local.gemmini_spad"
ACC = "local.gemmini_acc"
_REGISTRATION_LOCK = Lock()


def register_intrinsic_once(name, description, implementation):
    """Make registry insertion atomic for concurrent workload preparation."""
    with _REGISTRATION_LOCK:
        if tir.TensorIntrin.get(name, allow_missing=True) is None:
            tir.TensorIntrin.register(name, description, implementation)


def _update_reference(current, a, b, wrapping):
    product = a.astype("int32") * b.astype("int32")
    if wrapping:
        return (current.astype("int64") + product.astype("int64")).astype("int32")
    return current + product


@lru_cache(None)
def register_intrinsics(rows=16, cols=16, depth=16, wrapping=False):
    """Register a shape family; dimensions describe hardware rectangles."""
    if any(type(x) is not int or not 1 <= x <= 16 for x in (rows, cols, depth)):
        raise ValueError("Gemmini primitive rectangles require dimensions in [1,16]")

    def copy_intrin(height, width, source_scope, destination_scope, dtype, symbol):
        alignment = {"buffer_dim_align": [[0, 0, 16, 0]]} if destination_scope != "global" else {}

        @T.prim_func
        def desc(a: T.handle, b: T.handle):
            A = T.match_buffer(
                a,
                (height, width),
                dtype,
                scope=source_scope,
                strides=[T.int64(), T.int64(1)],
                offset_factor=1,
            )
            B = T.match_buffer(
                b,
                (height, width),
                dtype,
                scope=destination_scope,
                strides=[T.int64(), T.int64(1)],
                offset_factor=1,
            )
            with T.block("root"):
                T.reads(A[0:height, 0:width])
                T.writes(B[0:height, 0:width])
                for i, j in T.grid(height, width):
                    with T.block("copy"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj]

        @T.prim_func
        def impl(a: T.handle, b: T.handle):
            A = T.match_buffer(
                a,
                (height, width),
                dtype,
                scope=source_scope,
                strides=[T.int64(), T.int64(1)],
                offset_factor=1,
            )
            B = T.match_buffer(
                b,
                (height, width),
                dtype,
                scope=destination_scope,
                strides=[T.int64(), T.int64(1)],
                offset_factor=1,
            )
            with T.block("root"):
                T.block_attr(alignment)
                T.reads(A[0:height, 0:width])
                T.writes(B[0:height, 0:width])
                T.evaluate(
                    T.call_extern(
                        "void",
                        symbol,
                        A.access_ptr("r"),
                        B.access_ptr("w"),
                        T.uint32(height),
                        T.uint32(width),
                        A.strides[0],
                        B.strides[0],
                    )
                )

        return desc, impl

    @T.prim_func
    def init_desc(c: T.handle):
        C = T.match_buffer(
            c, (rows, cols), "int32", scope=ACC, strides=[T.int64(), T.int64(1)], offset_factor=1
        )
        with T.block("root"):
            T.reads()
            T.writes(C[0:rows, 0:cols])
            for i, j in T.grid(rows, cols):
                with T.block("init"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    C[vi, vj] = T.int32(0)

    @T.prim_func
    def init_impl(c: T.handle):
        C = T.match_buffer(
            c, (rows, cols), "int32", scope=ACC, strides=[T.int64(), T.int64(1)], offset_factor=1
        )
        with T.block("root"):
            T.block_attr({"buffer_dim_align": [[0, 0, 16, 0]]})
            T.reads()
            T.writes(C[0:rows, 0:cols])
            T.evaluate(
                T.call_extern(
                    "void",
                    "tvm_gemmini_zero_acc",
                    C.access_ptr("w"),
                    T.uint32(rows),
                    T.uint32(cols),
                )
            )

    @T.prim_func
    def update_desc(a: T.handle, b: T.handle, c: T.handle):
        A = T.match_buffer(
            a, (rows, depth), "int8", scope=SPAD, strides=[T.int64(), T.int64(1)], offset_factor=1
        )
        B = T.match_buffer(
            b, (depth, cols), "int8", scope=SPAD, strides=[T.int64(), T.int64(1)], offset_factor=1
        )
        C = T.match_buffer(
            c, (rows, cols), "int32", scope=ACC, strides=[T.int64(), T.int64(1)], offset_factor=1
        )
        with T.block("root"):
            T.reads(C[0:rows, 0:cols], A[0:rows, 0:depth], B[0:depth, 0:cols])
            T.writes(C[0:rows, 0:cols])
            for i, j, k in T.grid(rows, cols, depth):
                with T.block("update"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    C[vi, vj] = _update_reference(C[vi, vj], A[vi, vk], B[vk, vj], wrapping)

    @T.prim_func
    def update_impl(a: T.handle, b: T.handle, c: T.handle):
        A = T.match_buffer(
            a, (rows, depth), "int8", scope=SPAD, strides=[T.int64(), T.int64(1)], offset_factor=1
        )
        B = T.match_buffer(
            b, (depth, cols), "int8", scope=SPAD, strides=[T.int64(), T.int64(1)], offset_factor=1
        )
        C = T.match_buffer(
            c, (rows, cols), "int32", scope=ACC, strides=[T.int64(), T.int64(1)], offset_factor=1
        )
        with T.block("root"):
            T.block_attr({"buffer_dim_align": [[0, 0, 16, 0]]})
            T.reads(C[0:rows, 0:cols], A[0:rows, 0:depth], B[0:depth, 0:cols])
            T.writes(C[0:rows, 0:cols])
            T.evaluate(
                T.call_extern(
                    "void",
                    "tvm_gemmini_compute",
                    A.access_ptr("r"),
                    B.access_ptr("r"),
                    C.access_ptr("rw"),
                    T.uint32(rows),
                    T.uint32(cols),
                    T.uint32(depth),
                    T.int32(1),
                )
            )

    @T.prim_func
    def update_ws_impl(a: T.handle, b: T.handle, c: T.handle):
        A = T.match_buffer(
            a, (rows, depth), "int8", scope=SPAD, strides=[T.int64(), T.int64(1)], offset_factor=1
        )
        B = T.match_buffer(
            b, (depth, cols), "int8", scope=SPAD, strides=[T.int64(), T.int64(1)], offset_factor=1
        )
        C = T.match_buffer(
            c, (rows, cols), "int32", scope=ACC, strides=[T.int64(), T.int64(1)], offset_factor=1
        )
        with T.block("root"):
            T.block_attr({"buffer_dim_align": [[0, 0, 16, 0]]})
            T.reads(C[0:rows, 0:cols], A[0:rows, 0:depth], B[0:depth, 0:cols])
            T.writes(C[0:rows, 0:cols])
            T.evaluate(
                T.call_extern(
                    "void",
                    "tvm_gemmini_compute_ws",
                    A.access_ptr("r"),
                    B.access_ptr("r"),
                    C.access_ptr("rw"),
                    T.uint32(rows),
                    T.uint32(cols),
                    T.uint32(depth),
                    T.int32(1),
                )
            )

    pairs = {
        "load_a": copy_intrin(rows, depth, "global", SPAD, "int8", "tvm_gemmini_dma_a"),
        "load_b": copy_intrin(depth, cols, "global", SPAD, "int8", "tvm_gemmini_dma_b"),
        "init": (init_desc, init_impl),
        "compute": (update_desc, update_impl),
        # Same semantics in the weight-stationary dataflow; the adapter keeps
        # weights in the array across consecutive calls naming the same B tile.
        "compute_ws": (update_desc, update_ws_impl),
        "store": copy_intrin(rows, cols, ACC, "global", "int32", "tvm_gemmini_dma_out"),
    }
    result = {}
    from .gemmini_elementwise import _canonical_description

    for kind, (desc, impl) in pairs.items():
        name = f"gemmini.tir.{kind}.{rows}x{cols}x{depth}" + (".wrapping" if wrapping else "")
        # functools caches distinguish omitted defaults and explicit keyword
        # arguments; the global intrinsic registry does not. Registration must
        # remain idempotent for both call spellings.
        register_intrinsic_once(name, _canonical_description(desc), impl)
        result[kind] = name
    return result
