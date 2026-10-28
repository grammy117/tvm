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

The fixed lowering after scheduling expands software pipelines, assigns rows
with reuse across sibling scopes, and inserts completion waits for host-memory
hazards. Explicit initialization and DMA commands are preserved: the former
zero-init elimination and consecutive-load coalescing peepholes were removed.
Gemmini's reorder buffer orders commands that touch the same SRAM rows, so
reusing rows across scopes needs no fence; host memory has no such tracking.
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


_EXTERN = "tir.call_extern"
_ACCESS_PTR = "tir.tvm_access_ptr"
# Pre-lowering argument position of the host pointer in each DMA intrinsic.
_DMA_READS = {
    "tvm_gemmini_dma_a": 0,
    "tvm_gemmini_dma_b": 0,
    "tvm_gemmini_dma_a_blocks": 0,
    "tvm_gemmini_dma_b_blocks": 0,
    "tvm_gemmini_dma_acc_write": 0,
    "tvm_gemmini_dma_acc_add": 0,
    "tvm_gemmini_dma_widen": 0,
    "tvm_gemmini_dma_scaled_in": 0,
}
_DMA_WRITES = {
    "tvm_gemmini_dma_out": 1,
    "tvm_gemmini_dma_spad_out": 1,
    "tvm_gemmini_dma_relu_out": 1,
    "tvm_gemmini_dma_scaled_out": 1,
    "tvm_gemmini_dma_pool_out": 1,
}


def _is_op(node, name):
    return isinstance(node, tir.Call) and node.op.same_as(tvm.ir.Op.get(name))


def _extern(node):
    """Return (symbol, args) for a call_extern, else None."""
    if _is_op(node, _EXTERN) and isinstance(node.args[0], tir.StringImm):
        return node.args[0].value, list(node.args[1:])
    return None


def _pointer_data(expr):
    return expr.args[1] if _is_op(expr, _ACCESS_PTR) else None


def _allocate(stmt, *, spad_rows, acc_rows):
    """Stack-allocate SRAM rows: sibling scopes reuse rows; nesting stacks them."""
    allocations, peak, host_private = {}, {SPAD: 0, ACC: 0}, set()
    analyzer = tvm.arith.Analyzer()

    def rows_of(buffer):
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
            sum((int(size) - 1) * int(stride) for size, stride in zip(buffer.shape, strides)) + 1
            if strides
            else math.prod(int(value) for value in buffer.shape)
        )
        return ((span + 255) // 256) * 16

    def walk(node, base):
        if isinstance(node, tir.BlockRealize):
            walk(node.block, base)
        elif isinstance(node, tir.Block):
            base = dict(base)
            for buffer in node.alloc_buffers:
                scope = buffer.scope()
                if scope not in base:
                    host_private.add(buffer.data)
                    continue
                rows = rows_of(buffer)
                allocations[buffer.data] = (scope, base[scope], rows)
                base[scope] += rows
                peak[scope] = max(peak[scope], base[scope])
            walk(node.body, base)
        elif isinstance(node, tir.SeqStmt):
            for child in node.seq:
                walk(child, base)
        elif isinstance(node, (tir.For, tir.LetStmt, tir.AttrStmt)):
            walk(node.body, base)
        elif isinstance(node, tir.IfThenElse):
            walk(node.then_case, base)
            if node.else_case is not None:
                walk(node.else_case, base)
        elif isinstance(node, tir.Allocate):
            walk(node.body, base)

    walk(stmt, {SPAD: 0, ACC: 0})
    if peak[SPAD] > spad_rows or peak[ACC] > acc_rows:
        raise ValueError("TVM's scheduled buffers exceed Gemmini SRAM capacity")
    return allocations, peak, host_private


def _top_level(stmt):
    """The function's top-level statement sequence below allocation-only blocks."""
    while True:
        if isinstance(stmt, tir.BlockRealize):
            stmt = stmt.block.body
        elif isinstance(stmt, tir.SeqStmt):
            return list(stmt.seq)
        else:
            return [stmt]


def _host_hazards(stmt, allocations):
    """Classify host accesses per top-level statement.

    Returns (waits, fences): ``waits`` holds host buffers whose DMA must complete
    before the next host access in the same top-level statement (a loop-carried
    or in-iteration hazard); ``fences`` lists top-level statements that must be
    preceded by one completion wait because they access host memory touched by
    earlier asynchronous DMA.
    """
    statements = _top_level(stmt)
    access = []
    for child in statements:
        record = {"cpu_read": set(), "cpu_write": set(), "dma_read": set(), "dma_write": set()}

        def visit(node, record=record):
            if isinstance(node, tir.BufferLoad) and node.buffer.data not in allocations:
                record["cpu_read"].add(node.buffer.data)
            elif isinstance(node, tir.BufferStore) and node.buffer.data not in allocations:
                record["cpu_write"].add(node.buffer.data)
            else:
                call = _extern(node)
                if call is None:
                    return
                symbol, args = call
                if symbol in _DMA_READS:
                    data = _pointer_data(args[_DMA_READS[symbol]])
                    if data is not None:
                        record["dma_read"].add(data)
                elif symbol in _DMA_WRITES:
                    data = _pointer_data(args[_DMA_WRITES[symbol]])
                    if data is not None:
                        record["dma_write"].add(data)

        tir.stmt_functor.post_order_visit(child, visit)
        access.append(record)
    waits, fences = set(), set()
    for index, record in enumerate(access):
        # Within one top-level statement, any host access that can follow the
        # DMA (including in a later loop iteration) needs the DMA complete.
        waits |= record["dma_read"] & record["cpu_write"]
        waits |= record["dma_write"] & (record["cpu_read"] | record["cpu_write"] | record["dma_read"])
        for later in range(index + 1, len(access)):
            other = access[later]
            if (record["dma_read"] & other["cpu_write"]) or (
                record["dma_write"] & (other["cpu_read"] | other["cpu_write"] | other["dma_read"] | other["dma_write"])
            ):
                # Host destinations have no accelerator address scoreboard.
                # Two separate regions can overlap (e.g. padding fill followed
                # by the interior copy), so order their store groups too.
                fences.add(later)
    return waits, fences, statements


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
    mod = tvm.transform.Sequential(
        [tir.transform.InjectSoftwarePipeline(), tir.transform.Simplify()]
    )(tvm.IRModule({"main": func}))
    func = mod["main"]
    allocations, peak, host_private = _allocate(func.body, spad_rows=spad_rows, acc_rows=acc_rows)
    waits, fences, statements = _host_hazards(func.body, allocations)
    analyzer = tvm.arith.Analyzer()

    def wait_flag(pointer):
        data = _pointer_data(pointer)
        return tir.IntImm("int32", int(data is not None and data in waits))

    def rewrite(node):
        if isinstance(node, (tir.BufferLoad, tir.BufferStore)) and node.buffer.data in allocations:
            raise ValueError("CPU access to Gemmini SRAM remains after tensorization")
        if _is_op(node, _ACCESS_PTR):
            data, offset = node.args[1:3]
            if data in allocations:
                if not analyzer.can_prove(offset % 16 == 0):
                    raise ValueError("Gemmini tile access is not aligned to a physical row")
                return tir.Cast("uint32", allocations[data][1] + offset // 16)
        call = _extern(node)
        if call is not None:
            symbol, args = call
            if symbol in (
                "tvm_gemmini_dma_a",
                "tvm_gemmini_dma_b",
                "tvm_gemmini_dma_acc_write",
                "tvm_gemmini_dma_acc_add",
                "tvm_gemmini_dma_widen",
            ):
                src, row, height, width, stride, _ = args
                return tir.call_extern("void", symbol, src, row, height, width, stride, wait_flag(src))
            if symbol in ("tvm_gemmini_dma_a_blocks", "tvm_gemmini_dma_b_blocks"):
                src, row, height, width, stride, block_rows = args
                return tir.call_extern(
                    "void", symbol, src, row, height, width, stride, block_rows, wait_flag(src)
                )
            if symbol == "tvm_gemmini_dma_scaled_out":
                row, dst, height, width, _, stride, scale, relu = args
                return tir.call_extern(
                    "void", symbol, dst, row, height, width, stride, scale, relu, wait_flag(dst)
                )
            if symbol == "tvm_gemmini_dma_scaled_in":
                src, row, height, width, stride, scale, accumulate, _ = args
                return tir.call_extern("void", symbol, src, row, height, width, stride,
                                       scale, accumulate, wait_flag(src))
            if symbol == "tvm_gemmini_dma_pool_out":
                row, dst, size = args
                return tir.call_extern("void", symbol, dst, row, size, wait_flag(dst))
            if symbol in ("tvm_gemmini_dma_out", "tvm_gemmini_dma_spad_out", "tvm_gemmini_dma_relu_out"):
                row, dst, height, width, _, stride = args
                return tir.call_extern("void", symbol, dst, row, height, width, stride, wait_flag(dst))
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

    if fences:
        # One completion wait before each later statement that touches host
        # memory used by earlier queued DMA.
        fence = tir.Evaluate(tir.call_extern("void", "tvm_gemmini_end"))
        marked = [statements[index] for index in fences]

        def insert(node):
            if isinstance(node, tir.SeqStmt):
                children = []
                for child in node.seq:
                    if any(child.same_as(other) for other in marked):
                        children.append(fence)
                    children.append(child)
                if len(children) != len(node.seq):
                    return tir.SeqStmt(children)
            return None

        body = tir.stmt_functor.ir_transform(func.body, None, insert, ["tir.SeqStmt"])
        func = func.with_body(body)
    body = tir.stmt_functor.ir_transform(func.body, None, rewrite)
    return tvm.IRModule({"main": func.with_body(body)}), {
        "spad_rows": peak[SPAD],
        "acc_rows": peak[ACC],
        "allocations": [
            {"scope": scope, "row": row, "rows": rows} for scope, row, rows in allocations.values()
        ],
        "host_private_buffers": len(host_private),
        "completion_fences": len(fences),
        "waited_host_buffers": len(waits),
    }
