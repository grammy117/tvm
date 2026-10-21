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
"""Lower scoped Gemmini buffers to row allocations after TVM compaction.

Host padding buffers can be reused by subsequent CPU iterations. DMA wrappers
must complete transfers involving those buffers before returning; transfers of
immutable external inputs can remain queued until the operation's final wait.
"""

import math

import tvm
from tvm import tir

from .gemmini_intrin import ACC, SPAD


@tir.transform.prim_func_pass(opt_level=0)
def _normalize_scalar_match_offsets(func, _mod, _ctx):
    """Use v0.19's INT32 zero offset for scalar intrinsic pointer bindings.

    Buffer.ElemOffset([]) returns INT32 zero even in an INT64-indexed graph.
    Tensorize can promote a scalar match-buffer offset to INT64 instead. Bind
    an INT32 offset and cast its pointer use, preserving both ABI and address.
    This bridge only accepts zero-offset scalar sources and pointer-only uses.
    """

    def rewrite(node):
        if not isinstance(node, tir.Block):
            return None
        matches, substitutions, scalar_data = [], {}, set()
        for match in node.match_buffers:
            source, target = match.source.buffer, match.buffer
            if not source.shape and not target.shape and target.elem_offset.dtype != "int32":
                if not isinstance(source.elem_offset, tir.IntImm) or int(source.elem_offset) != 0:
                    raise ValueError("Scalar intrinsic source must have zero element offset")
                offset = tir.Var("scalar_elem_offset", "int32")
                if not isinstance(target.elem_offset, tir.Var):
                    raise ValueError("Scalar intrinsic offset must be a bindable variable")
                substitutions[target.elem_offset] = tir.Cast(target.elem_offset.dtype, offset)
                scalar_data.add(target.data)
                view = tir.decl_buffer(
                    (),
                    target.dtype,
                    name=target.name,
                    data=target.data,
                    elem_offset=offset,
                    scope=target.scope(),
                    data_alignment=target.data_alignment,
                    offset_factor=target.offset_factor,
                )
                match = tir.MatchBufferRegion(view, match.source)
            matches.append(match)
        if substitutions:

            def check_pointer_use(expr):
                if (
                    isinstance(expr, (tir.BufferLoad, tir.BufferStore))
                    and expr.buffer.data in scalar_data
                ):
                    raise ValueError("Scalar intrinsic view must be used through an access pointer")

            tir.stmt_functor.post_order_visit(node.body, check_pointer_use)
            body = tir.stmt_functor.substitute(node.body, substitutions)
            return tir.Block(
                node.iter_vars,
                node.reads,
                node.writes,
                node.name_hint,
                body,
                node.init,
                node.alloc_buffers,
                matches,
                node.annotations,
            )
        return None

    return func.with_body(tir.stmt_functor.ir_transform(func.body, None, rewrite))


def lower_memory(mod, *, spad_rows=16384, acc_rows=1024):
    """Assign bounded physical rows; reject remaining CPU accesses to SRAM."""
    mod = tvm.transform.Sequential(
        [
            tir.transform.LowerInitBlock(),
            tir.transform.PlanAndUpdateBufferAllocationLocation(),
            tir.transform.ConvertBlocksToOpaque(),
            tir.transform.CompactBufferAllocation(),
            _normalize_scalar_match_offsets,
            tir.transform.LowerMatchBuffer(),
            tir.transform.Simplify(),
        ]
    )(mod)
    func = mod["main"]
    allocations, host_private = {}, set()
    next_row = {SPAD: 0, ACC: 0}
    analyzer = tvm.arith.Analyzer()

    def collect(node):
        if not isinstance(node, tir.Block):
            return
        for buffer in node.alloc_buffers:
            scope = buffer.scope()
            if scope not in next_row:
                host_private.add(buffer.data)
                continue
            if not all(isinstance(value, tir.IntImm) for value in buffer.shape):
                raise ValueError("Gemmini SRAM allocations must be statically bounded")
            strides = list(buffer.strides)
            if len(buffer.shape) >= 2:
                row_stride = strides[-2] if strides else buffer.shape[-1]
                column_stride = strides[-1] if strides else tir.IntImm("int32", 1)
                if (
                    not analyzer.can_prove(buffer.shape[-2] == 1)
                    and not analyzer.can_prove(row_stride == 16)
                ) or not analyzer.can_prove(column_stride == 1):
                    raise ValueError(
                        f"Gemmini SRAM buffer {buffer.name} requires a 16-element physical row stride "
                        f"(row={row_stride}, column={column_stride})"
                    )
            span = (
                sum((int(size) - 1) * int(stride) for size, stride in zip(buffer.shape, strides))
                + 1
                if strides
                else math.prod(int(value) for value in buffer.shape)
            )
            rows = ((span + 255) // 256) * 16
            allocations[buffer.data] = (scope, next_row[scope], rows)
            next_row[scope] += rows

    tir.stmt_functor.post_order_visit(func.body, collect)
    if next_row[SPAD] > spad_rows or next_row[ACC] > acc_rows:
        raise ValueError("TVM's scheduled buffers exceed Gemmini SRAM capacity")

    host_reads, host_writes = set(), set()

    def cpu_access(node):
        if isinstance(node, tir.BufferLoad) and node.buffer.data not in allocations:
            host_reads.add(node.buffer.data)
        if isinstance(node, tir.BufferStore) and node.buffer.data not in allocations:
            host_writes.add(node.buffer.data)

    tir.stmt_functor.post_order_visit(func.body, cpu_access)

    dma_writes, dma_reads = {}, set()

    def dma_access(node):
        if not isinstance(node, tir.Call) or node.op != tvm.ir.Op.get("tir.call_extern"):
            return
        symbol = node.args[0].value
        output = symbol in (
            "tvm_gemmini_dma_out",
            "tvm_gemmini_dma_spad_out",
            "tvm_gemmini_dma_relu_out",
        )
        if not output and symbol not in (
            "tvm_gemmini_dma_a",
            "tvm_gemmini_dma_b",
            "tvm_gemmini_dma_acc_write",
            "tvm_gemmini_dma_acc_add",
            "tvm_gemmini_dma_widen",
        ):
            return
        pointer = node.args[2 if output else 1]
        if isinstance(pointer, tir.Call) and pointer.op == tvm.ir.Op.get("tir.tvm_access_ptr"):
            data = pointer.args[1]
            if output:
                dma_writes[data] = dma_writes.get(data, 0) + 1
            else:
                dma_reads.add(data)

    tir.stmt_functor.post_order_visit(func.body, dma_access)

    def host_wait(expr, *, output=False):
        return (
            isinstance(expr, tir.Call)
            and expr.op == tvm.ir.Op.get("tir.tvm_access_ptr")
            and (
                expr.args[1] in host_private
                or expr.args[1] in host_writes
                or (
                    output
                    and (
                        expr.args[1] in host_reads
                        or expr.args[1] in dma_reads
                        or dma_writes.get(expr.args[1], 0) > 1
                    )
                )
            )
        )

    def rewrite(node):
        if isinstance(node, (tir.BufferLoad, tir.BufferStore)) and node.buffer.data in allocations:
            raise ValueError("CPU access to Gemmini SRAM remains after tensorization")
        if isinstance(node, tir.Call) and node.op == tvm.ir.Op.get("tir.tvm_access_ptr"):
            data, offset = node.args[1:3]
            if data in allocations:
                if not analyzer.can_prove(offset % 16 == 0):
                    raise ValueError("Gemmini tile access is not aligned to a physical row")
                return tir.Cast("uint32", allocations[data][1] + offset // 16)
        if isinstance(node, tir.Call) and node.op == tvm.ir.Op.get("tir.call_extern"):
            symbol = node.args[0].value
            args = list(node.args[1:])
            if symbol in (
                "tvm_gemmini_dma_a",
                "tvm_gemmini_dma_b",
                "tvm_gemmini_dma_acc_write",
                "tvm_gemmini_dma_acc_add",
                "tvm_gemmini_dma_widen",
            ):
                src, row, height, width, stride, _ = args
                return tir.call_extern(
                    "void",
                    symbol,
                    src,
                    row,
                    height,
                    width,
                    stride,
                    tir.IntImm("int32", int(host_wait(src))),
                )
            if symbol in (
                "tvm_gemmini_dma_out",
                "tvm_gemmini_dma_spad_out",
                "tvm_gemmini_dma_relu_out",
            ):
                row, dst, height, width, _, stride = args
                return tir.call_extern(
                    "void",
                    symbol,
                    dst,
                    row,
                    height,
                    width,
                    stride,
                    tir.IntImm("int32", int(host_wait(dst, output=True))),
                )
        if isinstance(node, tir.Block):
            reads = [region for region in node.reads if region.buffer.data not in allocations]
            writes = [region for region in node.writes if region.buffer.data not in allocations]
            buffers = [buffer for buffer in node.alloc_buffers if buffer.data not in allocations]
            annotations = {
                key: value for key, value in node.annotations.items() if key != "buffer_dim_align"
            }
            return tir.Block(
                node.iter_vars,
                reads,
                writes,
                node.name_hint,
                node.body,
                node.init,
                buffers,
                node.match_buffers,
                annotations,
            )
        return None

    body = tir.stmt_functor.ir_transform(func.body, None, rewrite)
    return tvm.IRModule({"main": func.with_body(body)}), {
        "spad_rows": next_row[SPAD],
        "acc_rows": next_row[ACC],
        "allocations": [
            {"scope": scope, "row": row, "rows": rows} for scope, row, rows in allocations.values()
        ],
        "host_private_buffers": len(host_private),
    }
