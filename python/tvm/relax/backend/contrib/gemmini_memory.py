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

The fixed lowering after scheduling: software pipelines are expanded before
rows are assigned, an accumulator zero fill folds into the first reduction
command, consecutive tile loads coalesce into multi-block DMA, sibling scopes
reuse rows, and completion waits are inserted only for host-memory hazards.
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
_MAX_BLOCKS = 4  # MAX_BLOCK_LEN for INT8 scratchpad rows of the pinned profile


def _is_op(node, name):
    return isinstance(node, tir.Call) and node.op.same_as(tvm.ir.Op.get(name))


def _extern(node):
    """Return (symbol, args) for a call_extern, else None."""
    if _is_op(node, _EXTERN) and isinstance(node.args[0], tir.StringImm):
        return node.args[0].value, list(node.args[1:])
    return None


def _pointer_data(expr):
    return expr.args[1] if _is_op(expr, _ACCESS_PTR) else None


def _unwrap(stmt):
    """Look through opaque wrapper blocks that allocate nothing."""
    while True:
        if isinstance(stmt, tir.BlockRealize) and tir.analysis.expr_deep_equal(
            stmt.predicate, tir.const(True, "bool")
        ):
            block = stmt.block
            if block.alloc_buffers or block.match_buffers:
                return stmt
            stmt = block.body
        else:
            return stmt


def _single_call(stmt):
    stmt = _unwrap(stmt)
    if isinstance(stmt, tir.Evaluate):
        return _extern(stmt.value)
    return None


def _loop_nest(stmt):
    """Split a perfect nest into (loops, leaf) through allocation-free wrappers."""
    loops = []
    stmt = _unwrap(stmt)
    while isinstance(stmt, tir.For):
        loops.append(stmt)
        stmt = _unwrap(stmt.body)
    return loops, stmt


def _calls(stmt):
    found = []

    def visit(node):
        call = _extern(node)
        if call is not None:
            found.append((node, call))

    tir.stmt_functor.post_order_visit(stmt, visit)
    return found


def _fuse_accumulator_init(stmt):
    """Fold a zero-filled accumulator group into the first reduction compute.

    Matches ``[zero nest, reduction loop]`` siblings where the zero nest writes
    exactly the accumulator rectangles that the reduction's single compute call
    site covers on each spatial iteration, and nothing else in the reduction
    touches that accumulator. The compute then overwrites (accumulate=0) on the
    first reduction iteration, which the hardware supports natively.
    """
    analyzer = tvm.arith.Analyzer()

    def zero_nest(child):
        loops, leaf = _loop_nest(child)
        call = _single_call(leaf) if not isinstance(leaf, tir.For) else None
        if call is None or call[0] != "tvm_gemmini_zero_acc":
            return None
        return loops, call[1]

    def fuse(zero, reduction):
        zero_loops, (zero_ptr, zero_rows, zero_cols) = zero
        if not isinstance(reduction, tir.For):
            return None
        data = _pointer_data(zero_ptr)
        sites = [(node, call) for node, call in _calls(reduction) if any(
            _pointer_data(arg) is not None and _pointer_data(arg).same_as(data) for arg in call[1]
        )]
        if len(sites) != 1 or sites[0][1][0] not in ("tvm_gemmini_compute", "tvm_gemmini_compute_ws"):
            return None
        site, (symbol, args) = sites[0]
        if not (
            analyzer.can_prove_equal(args[3], zero_rows) and analyzer.can_prove_equal(args[4], zero_cols)
        ):
            return None
        # Loops enclosing the compute inside the reduction nest.
        enclosing = []

        def find(node, path):
            node = _unwrap(node) if isinstance(node, tir.BlockRealize) else node
            if isinstance(node, tir.For):
                return find(node.body, path + [node])
            if isinstance(node, tir.SeqStmt):
                return any(find(child, path) for child in node.seq)
            if isinstance(node, tir.BlockRealize):
                return find(node.block.body, path)
            if isinstance(node, (tir.LetStmt, tir.AttrStmt)):
                return find(node.body, path)
            if isinstance(node, tir.IfThenElse):
                return False
            if isinstance(node, tir.Evaluate) and node.value.same_as(site):
                enclosing.extend(path)
                return True
            return False

        if not find(reduction, []):
            return None
        offset = args[2].args[2]
        used = []
        tir.stmt_functor.post_order_visit(
            offset, lambda n: used.append(n) if isinstance(n, tir.Var) else None
        )

        def uses(loop):
            return any(var.same_as(loop.loop_var) for var in used)

        def unit(loop):
            return isinstance(loop.extent, tir.IntImm) and int(loop.extent) == 1

        # Extent-one loops carry no iteration and may be simplified out of
        # offsets; only iterating loops define the rectangle sets.
        zero_loops = [loop for loop in zero_loops if not unit(loop)]
        spatial = [loop for loop in enclosing if uses(loop) and not unit(loop)]
        temporal = [loop for loop in enclosing if not uses(loop) and not unit(loop)]
        if not temporal or len(spatial) != len(zero_loops):
            return None
        mapping = {}
        for z, c in zip(zero_loops, spatial):
            if not (
                analyzer.can_prove_equal(z.min, c.min) and analyzer.can_prove_equal(z.extent, c.extent)
            ):
                return None
            mapping[z.loop_var] = c.loop_var
        zero_offset = tir.stmt_functor.substitute(zero_ptr.args[2], mapping)
        if not analyzer.can_prove_equal(zero_offset, offset):
            return None
        first = tvm.tir.const(True, "bool")
        for loop in temporal:
            first = tir.And(first, loop.loop_var == loop.min)
        accumulate = tir.Select(first, tir.IntImm("int32", 0), tir.IntImm("int32", 1))
        replaced = tir.call_extern("void", symbol, *args[:6], accumulate)

        def rewrite(node):
            return tir.Evaluate(replaced) if isinstance(node, tir.Evaluate) and node.value.same_as(site) else None

        return tir.stmt_functor.ir_transform(reduction, None, rewrite, ["tir.Evaluate"])

    def visit(node):
        if not isinstance(node, tir.SeqStmt):
            return None
        children, changed, index = list(node.seq), False, 0
        result = []
        while index < len(children):
            zero = zero_nest(children[index]) if index + 1 < len(children) else None
            fused = fuse(zero, children[index + 1]) if zero is not None else None
            if fused is not None:
                result.append(fused)
                index += 2
                changed = True
            else:
                result.append(children[index])
                index += 1
        return tir.SeqStmt(result) if changed else None

    return tir.stmt_functor.ir_transform(stmt, None, visit, ["tir.SeqStmt"])


def _coalesce_block_loads(stmt):
    """Merge a loop of single-tile operand loads into multi-block DMA commands.

    A loop whose only statement loads one DIM-column tile, with the host source
    advancing by DIM elements and the scratchpad destination by whole rows per
    iteration, becomes a loop over groups of up to MAX_BLOCK_LEN tiles.
    """
    analyzer = tvm.arith.Analyzer()
    names = {"tvm_gemmini_dma_a": "tvm_gemmini_dma_a_blocks", "tvm_gemmini_dma_b": "tvm_gemmini_dma_b_blocks"}

    def visit(node):
        if not isinstance(node, tir.For) or node.kind != tir.ForKind.SERIAL:
            return None
        call = _single_call(node.body)
        if call is None or call[0] not in names or not isinstance(node.extent, tir.IntImm):
            return None
        symbol, (source, destination, height, width, source_stride, _) = call
        if not (_is_op(source, _ACCESS_PTR) and _is_op(destination, _ACCESS_PTR)):
            return None
        if not (isinstance(width, tir.IntImm) and int(width) == 16):
            return None
        var, extent = node.loop_var, int(node.extent)
        blocks = next((b for b in range(_MAX_BLOCKS, 1, -1) if extent % b == 0), None)
        if blocks is None:
            return None

        def step(expr):
            shifted = tir.stmt_functor.substitute(expr, {var: var + 1})
            return analyzer.simplify(shifted - expr)

        source_step, destination_step = step(source.args[2]), step(destination.args[2])
        if not (
            isinstance(source_step, tir.IntImm)
            and int(source_step) == 16
            and isinstance(destination_step, tir.IntImm)
            and int(destination_step) % 16 == 0
        ):
            return None
        outer = tir.Var(var.name + "_blocks", var.dtype)
        first = {var: outer * blocks + node.min}

        def pointer(ptr, extent_value):
            return tir.call_intrin(
                ptr.dtype,
                _ACCESS_PTR,
                ptr.args[0],
                ptr.args[1],
                tir.stmt_functor.substitute(ptr.args[2], first),
                extent_value,
                ptr.args[4],
            )

        source_extent = (height - 1) * source_stride + 16 * blocks
        destination_extent = destination_step * blocks
        merged = tir.call_extern(
            "void",
            names[symbol],
            pointer(source, source_extent),
            pointer(destination, destination_extent),
            height,
            tir.IntImm(width.dtype, 16 * blocks),
            source_stride,
            tir.IntImm("int32", int(destination_step) // 16),
        )
        return tir.For(
            outer,
            tir.IntImm(node.extent.dtype, 0),
            tir.IntImm(node.extent.dtype, extent // blocks),
            node.kind,
            tir.Evaluate(merged),
        )

    return tir.stmt_functor.ir_transform(stmt, None, visit, ["tir.For"])


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
    func = func.with_body(_coalesce_block_loads(_fuse_accumulator_init(func.body)))
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
