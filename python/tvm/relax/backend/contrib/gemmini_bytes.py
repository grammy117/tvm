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
"""Opaque byte copies using the existing unscaled scratchpad DMA ABI.

The storage dtype never selects arithmetic.  Widths and strides are converted
to bytes; the scratchpad remains INT8 with the normal 16-byte row geometry.
"""

from functools import lru_cache

from tvm import DataType, tir
from tvm.script import tir as T

from .gemmini_elementwise import _canonical_description, _dma_stride, _read, _region, _strides
from .gemmini_intrin import SPAD, register_intrinsic_once


BYTE_DTYPES = ("bool", "int16", "float16", "bfloat16", "int64", "float64")


def _byte_ptr(buffer, mode, size):
    # Express the actual DMA ABI in bytes. In particular, BF16 compute
    # legalization must not promote an opaque pointer's offset to FP32 words.
    pointer = buffer.access_ptr(mode)
    annotation, data, offset, extent, mask = pointer.args
    del annotation
    return tir.Call(
        pointer.dtype,
        pointer.op,
        [tir.type_annotation("uint8"), data, offset * size, extent * size, mask],
    )


@lru_cache(None)
def register_byte_copy(rows, cols, rank, dtype, source_rank=None):
    """Copy one legal rectangle without interpreting its little-endian words."""
    size = max(1, (DataType(dtype).bits + 7) // 8)
    if (
        dtype not in BYTE_DTYPES
        or rank not in (1, 2)
        or not 1 <= rows <= 16
        or not 1 <= cols * size <= 16
    ):
        raise ValueError("Unsupported opaque DMA rectangle")
    if rank == 1 and rows != 1:
        raise ValueError("A vector DMA tile has one physical row")
    shape = (cols,) if rank == 1 else (rows, cols)
    source_shape = shape if source_rank is None else shape[-source_rank:] if source_rank else ()
    factor = cols if rank == 1 else 1

    def indices(i, j):
        return () if not source_shape else (j,) if len(source_shape) == 1 else (i, j)

    if rank == 1:

        @T.prim_func
        def desc(a: T.handle, b: T.handle):
            A = T.match_buffer(
                a, source_shape, dtype, strides=_strides(len(source_shape)), offset_factor=1
            )
            B = T.match_buffer(b, shape, dtype, strides=_strides(rank), offset_factor=1)
            with T.block("root"):
                T.reads(_region(A))
                T.writes(_region(B))
                for j in T.serial(cols):
                    with T.block("copy"):
                        vj = T.axis.spatial(cols, j)
                        B[vj] = _read(A, indices(0, vj))
    else:

        @T.prim_func
        def desc(a: T.handle, b: T.handle):
            A = T.match_buffer(
                a, source_shape, dtype, strides=_strides(len(source_shape)), offset_factor=1
            )
            B = T.match_buffer(b, shape, dtype, strides=_strides(rank), offset_factor=1)
            with T.block("root"):
                T.reads(_region(A))
                T.writes(_region(B))
                for i, j in T.grid(rows, cols):
                    with T.block("copy"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        B[vi, vj] = _read(A, indices(vi, vj))

    @T.prim_func
    def impl(a: T.handle, b: T.handle):
        A = T.match_buffer(
            a, source_shape, dtype, strides=_strides(len(source_shape)), offset_factor=1
        )
        B = T.match_buffer(b, shape, dtype, strides=_strides(rank), offset_factor=1)
        with T.block("root"):
            T.reads(_region(A))
            T.writes(_region(B))
            with T.block("byte_copy"):
                T.reads(_region(A))
                T.writes(_region(B))
                C = T.alloc_buffer((rows, 16), "int8", scope=SPAD)
                T.evaluate(
                    T.call_extern(
                        "void",
                        "tvm_gemmini_dma_a",
                        _byte_ptr(A, "r", size),
                        C.access_ptr("w"),
                        T.uint32(rows),
                        T.uint32(cols * size),
                        T.int64(0)
                        if source_rank in (0, 1) and rank == 2
                        else _dma_stride(A, factor) * size,
                        T.int32(0),
                    )
                )
                T.evaluate(
                    T.call_extern(
                        "void",
                        "tvm_gemmini_dma_spad_out",
                        C.access_ptr("r"),
                        _byte_ptr(B, "w", size),
                        T.uint32(rows),
                        T.uint32(cols * size),
                        T.int64(16),
                        _dma_stride(B, factor) * size,
                    )
                )

    name = f"gemmini.bytes.{dtype}.{rank}d.{rows}x{cols}.source{source_rank}"
    register_intrinsic_once(name, _canonical_description(desc), impl)
    return name
