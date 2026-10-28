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
"""TensorIntrinsics for accumulator initialization explicitly present in TensorIR.

Bias addition is never moved into a reduction by this module. The schedule
matches an existing initialization block and emits its unscaled INT32 DMA.
"""

from functools import lru_cache

import tvm
from tvm import tir
from tvm.script import tir as T

from .gemmini_elementwise import _canonical_description, _copy_pair, _read, _region, _strides
from .gemmini_intrin import ACC


def _bias_kind(node, cols):
    value = node.body.value
    if not isinstance(value, tir.BufferLoad) or len(node.iter_vars) < 2:
        return None
    i, j = [axis.var for axis in node.iter_vars[-2:]]

    def depends(expressions):
        variables = set()
        for index in expressions:
            tir.stmt_functor.post_order_visit(
                index, lambda n: variables.add(n) if isinstance(n, tir.Var) else None
            )
        return i in variables or j in variables

    if (
        value.indices
        and tvm.ir.structural_equal(value.indices[-1], j)
        and not depends(value.indices[:-1])
    ):
        return "vector"
    if (
        len(value.indices) >= 2
        and tvm.ir.structural_equal(value.indices[-2:], [i, j])
        and not depends(value.indices[:-2])
    ):
        return "matrix"
    if cols == 1 and not depends(value.indices):
        return "scalar"
    for axis, index in enumerate(value.indices):
        if (
            tvm.ir.structural_equal(index, j)
            and not depends(value.indices[:axis] + value.indices[axis + 1 :])
            and all(
                isinstance(size, tir.IntImm) and int(size) == 1
                for size in value.buffer.shape[axis + 1 :]
            )
        ):
            return ("channel_vector", len(value.indices), axis)
    return None


@lru_cache(None)
def register_bias_intrinsic(rows, cols, kind):
    channel = isinstance(kind, tuple) and len(kind) == 3 and kind[0] == "channel_vector"
    if (
        any(type(value) is not int or not 1 <= value <= 16 for value in (rows, cols))
        or (not channel and kind not in ("matrix", "vector", "scalar"))
        or (kind == "scalar" and cols != 1)
    ):
        raise ValueError("Invalid Gemmini bias DMA rectangle")
    if kind == "matrix":
        desc, impl = _copy_pair(
            rows, cols, 2, "int32", "int32", "global", ACC, "tvm_gemmini_dma_acc_write"
        )
    else:
        if channel:
            _, rank, axis = kind
            shape = tuple(cols if index == axis else 1 for index in range(rank))
            dstrides = [
                tir.Var(f"stride_{index}", "int64") if index < axis else tir.IntImm("int64", 1)
                for index in range(rank)
            ]

            def indices(column):
                return tuple(
                    column if index == axis else tir.const(0, column.dtype) for index in range(rank)
                )
        else:
            shape = (cols,) if kind == "vector" else ()
            dstrides = _strides(len(shape))

            def indices(column):
                return (column,) if shape else ()

        @T.prim_func
        def desc(d: T.handle, c: T.handle):
            D = T.match_buffer(
                d,
                shape,
                "int32",
                strides=dstrides,
                offset_factor=1,
            )
            C = T.match_buffer(
                c,
                (rows, cols),
                "int32",
                scope=ACC,
                strides=[T.int64(), T.int64(1)],
                offset_factor=1,
            )
            with T.block("root"):
                T.reads(_region(D))
                T.writes(C[0:rows, 0:cols])
                for i, j in T.grid(rows, cols):
                    with T.block("bias"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        C[vi, vj] = _read(D, indices(vj))

        @T.prim_func
        def impl(d: T.handle, c: T.handle):
            D = T.match_buffer(
                d,
                shape,
                "int32",
                strides=dstrides,
                offset_factor=1,
            )
            C = T.match_buffer(
                c,
                (rows, cols),
                "int32",
                scope=ACC,
                strides=[T.int64(), T.int64(1)],
                offset_factor=1,
            )
            with T.block("root"):
                T.reads(_region(D))
                T.writes(C[0:rows, 0:cols])
                T.evaluate(
                    T.call_extern(
                        "void",
                        "tvm_gemmini_dma_acc_write",
                        D.access_ptr("r"),
                        C.access_ptr("w"),
                        T.uint32(rows),
                        T.uint32(cols),
                        T.int64(0),
                        T.int32(0),
                    )
                )

    suffix = f"channel_r{kind[1]}_a{kind[2]}" if channel else kind
    name = f"gemmini.bias.{suffix}.{rows}x{cols}"
    from .gemmini_intrin import register_intrinsic_once
    register_intrinsic_once(name, _canonical_description(desc), impl)
    return name


def schedule_bias_init(sch, init, outer, rows, cols):
    """Inline native layouts; place any remaining mathematical bias gather."""
    candidate = sch.copy()
    for producer in candidate.get_producers(init):
        try:
            candidate.compute_inline(producer)
        except tir.schedule.ScheduleError:
            pass
    kind = _bias_kind(candidate.get(init), cols)
    if kind is not None:
        return candidate, register_bias_intrinsic(rows, cols, kind)
    copy = sch.cache_read(init, 0, "global")
    for producer in sch.get_producers(copy):
        sch.compute_inline(producer)
    sch.compute_at(copy, outer, preserve_unit_loops=True)
    if _bias_kind(sch.get(init), cols) != "matrix":
        raise ValueError("Bias layout did not form a rectangular initialization tile")
    return sch, register_bias_intrinsic(rows, cols, "matrix")
