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
"""Structural scheduling of mathematical TensorIR for Gemmini.

Unlike gemmini_schedule (the retained legacy route), this module starts with
ordinary legalized tensor functions and records transformations of those functions.
"""

import math
import json
from dataclasses import dataclass

import tvm
from tvm import tir

from .gemmini_intrin import ACC, SPAD, register_intrinsics


def _contraction_update(node):
    """Read an ordinary or explicitly modulo-32 product update structurally."""
    value = node.body.value
    wrapped = False
    if isinstance(value, tir.Cast) and value.dtype == "int32":
        add = value.value
        if (
            not isinstance(add, tir.Add)
            or add.dtype != "int64"
            or not all(isinstance(x, tir.Cast) and x.dtype == "int64" for x in (add.a, add.b))
        ):
            return None
        current, product, wrapped = add.a.value, add.b.value, True
    elif isinstance(value, tir.Add):
        current, product = value.a, value.b
    else:
        return None
    if not isinstance(current, tir.BufferLoad) or not isinstance(product, tir.Mul):
        return None
    return current, product, wrapped


def _supported_contraction_init(node):
    value = node.init.value
    return (isinstance(value, tir.IntImm) and int(value) == 0) or (
        isinstance(value, tir.BufferLoad) and value.dtype == "int32"
    )


def _normalizable_contractions(func):
    """Identify exact contractions whose access maps can form matrix caches.

    Spatial axes used by both operands are batches; axes used by only one
    operand form the two matrix dimensions. All reduction axes must occur in
    both operands. TVM's index-map and reindex primitives prove and perform the
    layout transformation, including convolution's sliding-window accesses.
    """
    matches = []

    def variables(expressions):
        result = set()
        for expr in expressions:
            tir.stmt_functor.post_order_visit(
                expr, lambda node: result.add(node) if isinstance(node, tir.Var) else None
            )
        return result

    def visit(node):
        if not isinstance(node, tir.Block) or node.init is None or len(node.writes) != 1:
            return
        if not isinstance(node.body, tir.BufferStore) or not isinstance(node.init, tir.BufferStore):
            return
        output = node.body.buffer
        update = _contraction_update(node)
        if output.dtype != "int32" or not _supported_contraction_init(node) or update is None:
            return
        current, product, _ = update
        equal = tvm.ir.structural_equal
        if (
            not current.buffer.same_as(output)
            or not node.init.buffer.same_as(output)
            or not equal(current.indices, node.body.indices)
            or not equal(node.init.indices, node.body.indices)
        ):
            return
        loads = []
        for expr in (product.a, product.b):
            if (
                not isinstance(expr, tir.Cast)
                or expr.dtype != "int32"
                or not isinstance(expr.value, tir.BufferLoad)
                or expr.value.dtype != "int8"
            ):
                return
            loads.append(expr.value)
        axes = list(node.iter_vars)
        if any(
            axis.iter_type not in (0, 2)
            or not isinstance(axis.dom.min, tir.IntImm)
            or int(axis.dom.min) != 0
            or not isinstance(axis.dom.extent, tir.IntImm)
            or int(axis.dom.extent) <= 0
            for axis in axes
        ):
            return
        spatial = [index for index, axis in enumerate(axes) if axis.iter_type == 0]
        reduction = [index for index, axis in enumerate(axes) if axis.iter_type == 2]
        if not reduction or len(node.body.indices) != len(spatial):
            return
        # Reindexing the output must preserve every original spatial element.
        if set(node.body.indices) != {axes[index].var for index in spatial}:
            return
        left, right = [variables(load.indices) for load in loads]
        if any(axes[index].var not in left or axes[index].var not in right for index in reduction):
            return
        batch = [index for index in spatial if axes[index].var in left & right]
        rows = [index for index in spatial if axes[index].var in left - right]
        cols = [index for index in spatial if axes[index].var in right - left]
        if not rows or not cols or len(batch + rows + cols) != len(spatial):
            return
        shape = tuple(
            math.prod(int(axes[index].dom.extent) for index in group)
            for group in (rows, cols, reduction)
        )
        if shape[2] > 131071:
            return
        matches.append((node, shape, (batch, rows, cols, reduction)))

    tir.stmt_functor.post_order_visit(func.body, visit)
    return matches


def _normalize_contraction(sch):
    """Use standard layout schedules, retaining the original workload trace."""
    matches = contraction_block(sch.mod["main"])
    if len(matches) == 1:
        return matches[0]
    matches = _normalizable_contractions(sch.mod["main"])
    if len(matches) != 1:
        raise ValueError("Expected one structurally supported exact INT8/INT32 contraction")
    node, shape, (batch, rows, cols, reduction) = matches[0]
    block = sch.get_block(node.name_hint)
    extents = [int(axis.dom.extent) for axis in node.iter_vars]

    def matrix_map(*indices):
        def flatten(group):
            value = indices[group[0]]
            for index in group[1:]:
                value = value * extents[index] + indices[index]
            return value

        return tuple(indices[index] for index in batch) + tuple(
            flatten(group) for group in (rows, cols, reduction)
        )

    index_map = tir.IndexMap.from_func(
        matrix_map, ndim=len(extents), index_dtype=node.iter_vars[0].var.dtype
    )
    sch.transform_block_layout(block, index_map)
    sch.reindex(block, ("write", 0))
    product = _contraction_update(sch.get(block))[1]
    a_buffer, b_buffer = product.a.value.buffer, product.b.value.buffer
    a_copy = sch.reindex(block, a_buffer)
    b_copy = sch.reindex(block, b_buffer)
    b_rank = len(sch.get(b_copy).writes[0].buffer.shape)
    transpose = tir.IndexMap.from_func(
        lambda *indices: (*indices[:-2], indices[-1], indices[-2]),
        ndim=b_rank,
        index_dtype=node.iter_vars[0].var.dtype,
    )
    sch.transform_layout(b_copy, ("write", 0), transpose)
    # Inline upstream index/padding expressions into the copies. compute_at
    # below then materializes only the matrix region needed by each DMA.
    for copy in (a_copy, b_copy):
        for producer in sch.get_producers(copy):
            try:
                sch.compute_inline(producer)
            except tir.schedule.ScheduleError:
                pass
    normalized = contraction_block(sch.mod["main"])
    if len(normalized) != 1:
        raise ValueError("TVM layout normalization did not form a canonical matrix contraction")
    return normalized[0][0], shape


def contraction_block(func):
    """Find an exact signed INT8 x INT8 -> INT32 matrix reduction by structure."""
    matches = []

    def visit(node):
        if not isinstance(node, tir.Block) or node.init is None:
            return
        if len(node.iter_vars) < 3 or len(node.writes) != 1:
            return
        if [axis.iter_type for axis in node.iter_vars] != [0] * (len(node.iter_vars) - 1) + [2]:
            return
        if not isinstance(node.body, tir.BufferStore) or not isinstance(node.init, tir.BufferStore):
            return
        output = node.body.buffer
        if output.dtype != "int32" or len(output.shape) != len(node.iter_vars) - 1:
            return
        if not _supported_contraction_init(node):
            return
        update = _contraction_update(node)
        if update is None:
            return
        current, product, _ = update
        if not current.buffer.same_as(output):
            return
        if (
            not node.init.buffer.same_as(output)
            or not tvm.ir.structural_equal(current.indices, node.body.indices)
            or not tvm.ir.structural_equal(node.init.indices, node.body.indices)
        ):
            return
        operands = []
        for expr in (product.a, product.b):
            if (
                not isinstance(expr, tir.Cast)
                or expr.dtype != "int32"
                or not isinstance(expr.value, tir.BufferLoad)
            ):
                return
            load = expr.value
            if load.buffer.dtype != "int8" or len(load.buffer.shape) < 2:
                return
            operands.append(load)
        batch = [axis.var for axis in node.iter_vars[:-3]]
        i, j, k = [axis.var for axis in node.iter_vars[-3:]]
        expected = ((i, k), (k, j))
        equal = tvm.ir.structural_equal
        if not all(equal(x, y) for x, y in zip(node.body.indices, (*batch, i, j))):
            return
        if not all(
            all(equal(x, y) for x, y in zip(load.indices[-2:], indices))
            for load, indices in zip(operands, expected)
        ):
            return
        for load in operands:
            if len(load.indices) > len(batch) + 2:
                return
            for index, axis in zip(reversed(load.indices[:-2]), reversed(batch)):
                if not equal(index, axis) and not (
                    isinstance(index, tir.IntImm) and int(index) == 0
                ):
                    return
        if not all(isinstance(axis.dom.extent, tir.IntImm) for axis in node.iter_vars):
            return
        m, n, depth = [int(axis.dom.extent) for axis in node.iter_vars[-3:]]
        if min(m, n, depth) <= 0 or depth > 131071:
            return
        matches.append((node, (m, n, depth)))

    tir.stmt_functor.post_order_visit(func.body, visit)
    return matches


def schedule_contraction(
    func,
    *,
    order="ij",
    load_order="ab",
    reduction_chunk=16,
    row_tiles=1,
    col_tiles=1,
    tensorize=True,
    mark_tensorize=False,
    sample_groups=False,
):
    """Schedule a canonical matrix reduction using ordinary TensorIR primitives.

    Padding, cache placement and reduction initialization remain mathematical
    operations until tensorization. Physical SRAM allocation is a separate pass.
    """
    sch = tir.Schedule(func, debug_mask="all")
    node, (m, n, depth) = _normalize_contraction(sch)
    biased = isinstance(node.init.value, tir.BufferLoad)
    wrapping = _contraction_update(node)[2]
    if order not in ("ij", "ji"):
        raise ValueError("Gemmini traversal order must be ij or ji")
    if load_order not in ("ab", "ba"):
        raise ValueError("Gemmini DMA order must be ab or ba")
    if type(reduction_chunk) is not int or not 2 <= reduction_chunk <= 16:
        raise ValueError("Gemmini reduction chunks require an integer in [2,16]")
    if any(type(value) is not int or value not in (1, 2, 4) for value in (row_tiles, col_tiles)):
        raise ValueError("Gemmini matrix cache groups require 1, 2 or 4 tiles")
    if tensorize and mark_tensorize:
        raise ValueError("Choose immediate tensorization or MetaSchedule annotations")
    if biased:
        choices = lambda extent, groups: [x for x in range(1, 17) if extent % (x * groups) == 0]
        if not choices(m, row_tiles) or not choices(n, col_tiles):
            raise ValueError("Bias cache groups must divide the mathematical output")
        rows, cols = max(choices(m, row_tiles)), max(choices(n, col_tiles))
        chunk = max(x for x in range(1, reduction_chunk + 1) if depth % x == 0)
    else:
        rows, cols = (max(2, min(16, value)) for value in (m, n))
        chunk = max(2, min(reduction_chunk, depth))
    row_factor, col_factor = row_tiles, col_tiles
    if sample_groups:
        if biased:
            raise ValueError("Sampled cache groups currently require zero-initialized contractions")

        # Divisor groups retain one padding layout across all decisions. This
        # makes replay/resampling sound: only actual split factors change.
        def sample(extent, tile, maximum):
            count = (extent + tile - 1) // tile
            choices = [value for value in (1, 2, 4) if value <= maximum and count % value == 0]
            return sch.sample_categorical(choices, [1 / len(choices)] * len(choices))

        row_factor, col_factor = sample(m, rows, row_tiles), sample(n, cols, col_tiles)
    for key, value in {
        "order": order,
        "load_order": load_order,
        "reduction_chunk": chunk,
        "row_tiles": row_factor,
        "col_tiles": col_factor,
    }.items():
        sch.annotate(sch.get_block("root"), "gemmini." + key, value)
    # Preserve nontrivial block axes through the pinned tensorization matcher.
    # Unit axes use standard mathematical zero-padding, never out-of-range DMA.
    block = sch.get_block(node.name_hint)
    padding_factors = (
        (rows, cols, chunk) if sample_groups else (rows * row_tiles, cols * col_tiles, chunk)
    )
    padded = any(value % factor for value, factor in zip((m, n, depth), padding_factors))
    if padded:
        if wrapping:
            raise ValueError("Explicit modular contractions require divisor tiles")
        sch.pad_einsum(block, [1] * (len(node.iter_vars) - 3) + list(padding_factors))
    i, j, k = sch.get_loops(block)[-3:]
    io, it, ii = sch.split(i, [None, row_factor, rows])
    jo, jt, ji = sch.split(j, [None, col_factor, cols])
    ko, ki = sch.split(k, [None, chunk])
    sch.reorder(*((io, jo) if order == "ij" else (jo, io)), ko, it, jt, ii, ji, ki)
    outer = jo if order == "ij" else io
    write = sch.cache_write(block, 0, ACC)
    sch.reverse_compute_at(write, outer, preserve_unit_loops=True)
    if biased:
        # The normalization's output copy can fold into the complete DMA
        # writeback block. The reduction itself remains a reduction block.
        for consumer in list(sch.get_consumers(write)):
            copy = sch.get(consumer)
            if (
                isinstance(copy.body, tir.BufferStore)
                and isinstance(copy.body.value, tir.BufferLoad)
                and tvm.ir.structural_equal(copy.body.indices, copy.body.value.indices)
                and tvm.ir.structural_equal(copy.body.buffer.shape, copy.body.value.buffer.shape)
            ):
                sch.reverse_compute_inline(consumer)
    product = _contraction_update(sch.get(block))[1]
    a_buffer, b_buffer = product.a.value.buffer, product.b.value.buffer
    read_index = lambda buffer: next(
        index for index, region in enumerate(sch.get(block).reads) if region.buffer.same_as(buffer)
    )
    load_a = sch.cache_read(block, read_index(a_buffer), SPAD)
    load_b = sch.cache_read(block, read_index(b_buffer), SPAD)
    for load in (load_a, load_b) if load_order == "ab" else (load_b, load_a):
        sch.compute_at(load, ko, preserve_unit_loops=True)
    # The array addresses each tile as consecutive 16-element SRAM rows. A
    # normal row-major multi-tile B/C cache has a wider row stride, so express
    # the hardware tile layout with standard index-map transformations.
    if sample_groups or col_tiles > 1:

        def tiled_columns(*indices):
            return (*indices[:-2], indices[-1] // cols, indices[-2], indices[-1] % cols)

        rank = len(sch.get(load_b).writes[0].buffer.shape)
        sch.transform_layout(
            load_b,
            ("write", 0),
            tir.IndexMap.from_func(
                tiled_columns, ndim=rank, index_dtype=node.iter_vars[0].var.dtype
            ),
        )
    if sample_groups or row_tiles > 1 or col_tiles > 1:

        def tiled_output(*indices):
            return (
                *indices[:-2],
                indices[-2] // rows,
                indices[-1] // cols,
                indices[-2] % rows,
                indices[-1] % cols,
            )

        rank = len(sch.get(block).writes[0].buffer.shape)
        sch.transform_layout(
            block,
            ("write", 0),
            tir.IndexMap.from_func(
                tiled_output, ndim=rank, index_dtype=node.iter_vars[0].var.dtype
            ),
        )
    a_row, a_k = sch.get_loops(load_a)[-2:]
    a_tile, a_inner = sch.split(a_row, [None, rows])
    sch.reorder(a_tile, a_inner, a_k)
    b_k, b_col = sch.get_loops(load_b)[-2:]
    b_tile, b_inner = sch.split(b_col, [None, cols])
    sch.reorder(b_tile, b_k, b_inner)
    c_row, c_col = sch.get_loops(write)[-2:]
    c_it, c_ii = sch.split(c_row, [None, rows])
    c_jt, c_ji = sch.split(c_col, [None, cols])
    sch.reorder(c_it, c_jt, c_ii, c_ji)
    for cache in (load_a, load_b, block):
        sch.storage_align(cache, 0, axis=-2, factor=16, offset=0)
    # Materialize only the needed padding regions near each DMA consumer.
    for load in (load_a, load_b):
        for producer in sch.get_producers(load):
            sch.compute_at(producer, ko, preserve_unit_loops=True)
            # A convolution reindex can feed a padding stage. Fold that copy
            # into the padding expression rather than retaining a full im2col.
            for upstream in sch.get_producers(producer):
                try:
                    sch.compute_inline(upstream)
                except tir.schedule.ScheduleError:
                    pass
    pending, placed = list(sch.get_consumers(write)), set()

    def inside_tile(producer):
        return any(sch.get(loop).same_as(sch.get(outer)) for loop in sch.get_loops(producer))

    while pending:
        consumer = pending.pop(0)
        name = sch.get(consumer).name_hint
        if name in placed:
            continue
        consumer_node = sch.get(consumer)
        if (
            isinstance(consumer_node.body, tir.BufferStore)
            and isinstance(consumer_node.body.value, tir.BufferLoad)
            and math.prod(int(x) for x in consumer_node.body.buffer.shape)
            < math.prod(int(x) for x in consumer_node.body.value.buffer.shape)
        ):
            # Keep an unpadding copy at root. Moving nested unpadding/layout
            # copies into a partial output tile breaks v0.19's region-cover
            # invariant. DMA completion before this CPU consumer is supplied
            # by the ordinary target memory/effect lowering.
            continue
        # A fused epilogue can have independent pointwise producers, such as
        # widening a narrow bias. Inline those ordinary expressions before
        # moving the consumer; keep non-inlineable CPU reductions outside.
        for producer in list(sch.get_producers(consumer)):
            if not inside_tile(producer):
                try:
                    sch.compute_inline(producer)
                except tir.schedule.ScheduleError:
                    pass
        if not all(inside_tile(producer) for producer in sch.get_producers(consumer)):
            continue
        sch.reverse_compute_at(consumer, outer, preserve_unit_loops=True)
        placed.add(name)
        pending.extend(sch.get_consumers(consumer))
    init = sch.decompose_reduction(block, ko)
    bias_intrinsic = None
    if biased:
        from .gemmini_bias import schedule_bias_init

        sch, bias_intrinsic = schedule_bias_init(sch, init, outer, rows, cols)
    from .gemmini_epilogue import fuse_readout
    sch, write, readout_intrinsic = fuse_readout(sch, write, rows, cols)
    mathematical = sch.mod
    if tensorize or mark_tensorize:
        intrin = register_intrinsics(rows, cols, chunk, wrapping=wrapping)
        if bias_intrinsic is not None:
            intrin = dict(intrin, init=bias_intrinsic)
        if readout_intrinsic is not None:
            intrin = dict(intrin, store=readout_intrinsic)
        for target, axes, kind in (
            (init, 2, "init"),
            (load_a, 2, "load_a"),
            (load_b, 2, "load_b"),
            (block, 3, "compute"),
            (write, 2, "store"),
        ):
            loop = sch.get_loops(target)[-axes]
            outer = sch.blockize(loop)
            if kind != "store":
                # Blockization changes the buffer rank seen by the outer
                # block. Keep physical row alignment on that block, where
                # Tensorize and allocation compaction consume the annotation.
                sch.storage_align(outer, 0, axis=-2, factor=16, offset=0)
            if tensorize:
                sch.tensorize(outer, intrin[kind])
            else:
                sch.annotate(outer, "meta_schedule.auto_tensorize", intrin[kind])
    return sch, mathematical


@dataclass(frozen=True)
class GemminiTensorIRSchedule:
    """The original workload, full transformation trace and deployable result."""

    workload: tvm.IRModule
    mathematical: tvm.IRModule
    scheduled_mod: tvm.IRModule
    trace: tir.schedule.Trace
    metadata: dict


def make_tensorir_schedule(
    func,
    *,
    order="ij",
    load_order="ab",
    reduction_chunk=16,
    row_tiles=1,
    col_tiles=1,
    capabilities=None,
):
    """Apply the structural rule and lower target scopes with admission checks."""
    sch, mathematical = schedule_contraction(
        func,
        order=order,
        load_order=load_order,
        reduction_chunk=reduction_chunk,
        row_tiles=row_tiles,
        col_tiles=col_tiles,
    )
    return lower_tensorir_schedule(
        func, sch, mathematical=mathematical, order=order, capabilities=capabilities
    )


def lower_tensorir_schedule(
    func, sch, *, mathematical=None, order="database", capabilities=None, workload_metadata=None
):
    """Finalize a TVM-selected schedule without replacing its scheduling choices."""
    from .gemmini_memory import lower_memory  # pylint: disable=import-outside-toplevel
    from .gemmini_target import GemminiCapabilities  # pylint: disable=import-outside-toplevel

    capabilities = capabilities or GemminiCapabilities()
    if any(param not in func.buffer_map for param in func.params):
        raise ValueError("Gemmini runtime ABI requires buffer parameters")
    mathematical = mathematical if mathematical is not None else tvm.IRModule({"main": func})
    mod, resources = lower_memory(
        sch.mod, spad_rows=capabilities.spad_rows, acc_rows=capabilities.acc_rows
    )
    device = mod["main"]
    buffers = [device.buffer_map[param] for param in device.params]
    if any(
        not all(isinstance(dim, tir.IntImm) and int(dim) > 0 for dim in buffer.shape)
        or buffer.strides
        for buffer in buffers
    ):
        raise ValueError("TensorIR admission requires positive static compact external buffers")
    sizes = [
        math.prod(int(dim) for dim in buffer.shape)
        * max(1, (tvm.DataType(buffer.dtype).bits + 7) // 8)
        for buffer in buffers
    ]
    if max(sizes) > (1 << 63) - 1:
        raise ValueError("Tensor buffer byte span exceeds the signed ABI")
    if any(
        (int(buffer.shape[-1]) if len(buffer.shape) else 1)
        * max(1, (tvm.DataType(buffer.dtype).bits + 7) // 8)
        > (1 << 32) - 1
        for buffer in buffers
    ):
        raise ValueError("External DMA row stride exceeds the Gemmini ABI")
    written = set()
    tir.stmt_functor.post_order_visit(
        func.body,
        lambda node: written.add(node.buffer.data) if isinstance(node, tir.BufferStore) else None,
    )
    # MetaSchedule may renew every Var while cloning a candidate. Ownership
    # comes from the original mathematical ABI's parameter positions, not
    # pointer identity shared accidentally with one scheduling path.
    original_buffers = [func.buffer_map[param] for param in func.params]
    if len(original_buffers) != len(buffers) or any(
        source.dtype != selected.dtype or not tvm.ir.structural_equal(source.shape, selected.shape)
        for source, selected in zip(original_buffers, buffers)
    ):
        raise ValueError("Scheduled function changed the mathematical tensor ABI")
    roles = [buffer.data in written for buffer in original_buffers]
    inputs = [(buffer, size) for buffer, size, output in zip(buffers, sizes, roles) if not output]
    outputs = [(buffer, size) for buffer, size, output in zip(buffers, sizes, roles) if output]
    if not outputs:
        raise ValueError("TensorIR admission could not establish input/output ownership")
    failure = tir.SeqStmt(
        [
            tir.Evaluate(
                tir.call_extern(
                    "void", "TVMAPISetLastError", tir.StringImm("Gemmini tensor buffers rejected")
                )
            ),
            tir.Evaluate(tir.tvm_throw_last_error()),
        ]
    )
    if not isinstance(device.body, tir.BlockRealize):
        raise ValueError("Gemmini lowering requires a TensorIR root block")
    realization = device.body
    root = realization.block
    body = tir.SeqStmt(
        [
            tir.Evaluate(
                tir.call_extern("void", "tvm_gemmini_begin", *[tir.IntImm("int64", 1)] * 3)
            ),
            root.body,
            tir.Evaluate(tir.call_extern("void", "tvm_gemmini_end")),
        ]
    )
    for output, output_bytes in outputs:
        checks = [
            tir.call_extern(
                "int32",
                "tvm_gemmini_validate_tensor_buffers",
                source.data,
                tir.IntImm("int64", source_bytes),
                source.data,
                tir.IntImm("int64", source_bytes),
                output.data,
                tir.IntImm("int64", output_bytes),
            )
            for source, source_bytes in inputs
        ]
        if not inputs:
            checks.append(
                tir.call_extern(
                    "int32",
                    "tvm_gemmini_validate_tensor_output",
                    output.data,
                    tir.IntImm("int64", output_bytes),
                )
            )
        for admission in checks:
            status = tir.Var("gemmini_admission", "int32")
            body = tir.LetStmt(status, admission, tir.IfThenElse(status != 0, failure, body))
    # Keep the ordinary TensorIR root for stock Relax analyses, including
    # RewriteDataflowReshape on single-input/single-output functions. Admission
    # and ownership still precede every device command within that root.
    root = tir.Block(
        root.iter_vars,
        root.reads,
        root.writes,
        root.name_hint,
        body,
        root.init,
        root.alloc_buffers,
        root.match_buffers,
        root.annotations,
    )
    device = device.with_body(
        tir.BlockRealize(realization.iter_values, realization.predicate, root)
    )
    if workload_metadata is None:
        matches = contraction_block(func) or _normalizable_contractions(func)
        shape = matches[0][1]
        workload_metadata = dict(zip(("m", "n", "k"), shape))
        if isinstance(matches[0][0].init.value, tir.BufferLoad):
            workload_metadata.update(operation="matmul_bias", bias_preloaded=True)
    cpu_reductions = []
    tir.stmt_functor.post_order_visit(
        sch.mod["main"].body,
        lambda node: cpu_reductions.append(node)
        if isinstance(node, tir.Block) and node.init is not None
        else None,
    )
    if cpu_reductions:
        workload_metadata = dict(workload_metadata, cpu_reduction_blocks=len(cpu_reductions))
    device = device.with_attr("gemmini.tensorir", 1)
    device = device.with_attr("gemmini.capability_id", capabilities.fingerprint)
    device = device.with_attr("op_pattern", 8)
    for key in ("m", "n", "k"):
        if key in workload_metadata:
            device = device.with_attr("gemmini." + key, workload_metadata[key])
    metadata = dict(
        resources,
        **workload_metadata,
        order=order,
        instruction_policy="primitive_no_fsm",
        performance_measured=False,
        capability_id=capabilities.fingerprint,
    )
    root = sch.mod["main"].body.block
    if root.annotations.get("gemmini.readout_fused", False):
        metadata["readout_fused"] = True
        metadata["readout_proof"] = json.loads(str(func.attrs["gemmini.readout_proof"]))
    metadata["schedule_parameters"] = {
        str(key)[8:]: int(value) if isinstance(value, tir.IntImm) else str(value)
        for key, value in root.annotations.items()
        if str(key).startswith("gemmini.")
    }
    return GemminiTensorIRSchedule(
        tvm.IRModule({"main": func}),
        mathematical,
        tvm.IRModule({"main": device}),
        sch.trace,
        metadata,
    )


@tvm.transform.module_pass(opt_level=0, name="ScheduleGemminiTensorIR")
class ScheduleGemminiTensorIR:
    """Schedule supported mathematical TIR; keep unsupported functions on CPU.

    Run after legalization and supported graph fusion. A target-specific schedule
    failure retains the original function and records its reason. Unexpected
    compiler failures propagate instead of silently changing placement.
    """

    def __init__(
        self,
        *,
        capabilities=None,
        order="ij",
        schedule_options=None,
        report=None,
        references=None,
        normalizations=None,
        offload_copies=False,
        movement_strategy="rectangular",
    ):
        from .gemmini_target import GemminiCapabilities  # pylint: disable=import-outside-toplevel

        self.capabilities = capabilities or GemminiCapabilities()
        self.order = order
        self.schedule_options = dict(schedule_options or {})
        if set(self.schedule_options) - {"load_order", "reduction_chunk", "row_tiles", "col_tiles"}:
            raise ValueError("Unknown Gemmini TensorIR schedule option")
        self.report = report if report is not None else []
        self.references = references if references is not None else {}
        self.normalizations = normalizations if normalizations is not None else {}
        if type(offload_copies) is not bool:
            raise ValueError("offload_copies must be a boolean")
        self.offload_copies = offload_copies
        if movement_strategy not in ("rectangular", "scalar"):
            raise ValueError(
                "Movement strategy must be rectangular or an explicit scalar experiment"
            )
        self.movement_strategy = movement_strategy

    def transform_module(self, mod, _ctx):
        from .gemmini_elementwise import elementwise_blocks, make_elementwise_schedule
        from .gemmini_reduction import reduction_blocks, reduction_metadata, make_reduction_schedule

        for gv, func in list(mod.functions_items()):
            if not isinstance(func, tir.PrimFunc) or (
                func.attrs and "gemmini.tensorir" in func.attrs
            ):
                continue
            matches = contraction_block(func) or _normalizable_contractions(func)
            reductions = reduction_blocks(func) if not matches else []
            pointwise = elementwise_blocks(func) if not matches and not reductions else []
            if not matches and not pointwise and not reductions:
                self.report.append(
                    {
                        "function": gv.name_hint,
                        "placement": "cpu",
                        "reason": "No supported exact integer contraction, bounded sum, copy, widening or addition",
                    }
                )
                continue
            try:
                database = tvm.meta_schedule.Database.current()
                target = self.capabilities.target()
                workload = tvm.IRModule({"main": func})
                record = (
                    database.query_tuning_record(workload, target, workload_name=gv.name_hint)
                    if database is not None
                    else None
                )
                # v0.19's default database query ignores its target argument.
                # Check the returned record before using the standard replay.
                if record is not None and (
                    record.target is None or not tvm.ir.structural_equal(record.target, target)
                ):
                    raise ValueError(
                        "MetaSchedule record belongs to a different Gemmini target profile"
                    )
                selected = (
                    database.query_schedule(workload, target, workload_name=gv.name_hint)
                    if record is not None
                    else None
                )
                if selected is not None:
                    bound_profile = selected.mod["main"].body.block.annotations.get(
                        "gemmini_measurement_profile"
                    )
                    if (
                        bound_profile is not None
                        and str(bound_profile) != self.capabilities.fingerprint
                    ):
                        raise ValueError(
                            "Selected cycle schedule belongs to another hardware profile"
                        )
                    native_calls = []
                    tir.stmt_functor.post_order_visit(
                        selected.mod["main"].body,
                        lambda node: native_calls.append(node)
                        if isinstance(node, tir.Call)
                        and node.op == tvm.ir.Op.get("tir.call_extern")
                        and node.args[0].value.startswith("tvm_gemmini_")
                        else None,
                    )
                    if not native_calls:
                        # The target rule can also expose an ordinary CPU
                        # candidate. Never label it native or insert an
                        # accelerator envelope merely because its database
                        # record carries this target's measurement identity.
                        mod.update_func(gv, selected.mod["main"])
                        self.report.append(
                            {
                                "function": gv.name_hint,
                                "placement": "cpu",
                                "origin": "TVM MetaSchedule database",
                                "reason": "Selected CPU schedule from matching tuning record",
                                "trace": selected.trace.as_json(),
                            }
                        )
                        continue
                if selected is not None:
                    result = lower_tensorir_schedule(
                        func,
                        selected,
                        capabilities=self.capabilities,
                        workload_metadata=(
                            {"operation": pointwise[0][1]}
                            if pointwise
                            else reduction_metadata(reductions[0])
                            if reductions
                            else None
                        ),
                    )
                    origin = "TVM MetaSchedule database"
                elif (
                    pointwise
                    and not self.offload_copies
                    and any(
                        not kind.startswith("add_i32") and kind != "relu_i8"
                        for _, kind in pointwise
                    )
                ):
                    self.report.append(
                        {
                            "function": gv.name_hint,
                            "placement": "cpu",
                            "reason": "Copy offload requires an explicit opt-in or a measured tuning record",
                        }
                    )
                    continue
                elif reductions:
                    result = make_reduction_schedule(func, capabilities=self.capabilities)
                    origin = "deterministic TensorIR reduction rule"
                elif pointwise:
                    result = make_elementwise_schedule(
                        func,
                        capabilities=self.capabilities,
                        scalar=self.movement_strategy == "scalar",
                    )
                    origin = (
                        "explicit scalar DMA experiment"
                        if self.movement_strategy == "scalar"
                        else "deterministic TensorIR DMA rule"
                    )
                else:
                    result = make_tensorir_schedule(
                        func,
                        capabilities=self.capabilities,
                        order=self.order,
                        **self.schedule_options,
                    )
                    origin = "deterministic TensorIR rule"
            except (ValueError, tir.schedule.ScheduleError) as error:
                if gv.name_hint in self.references:
                    mod.update_func(gv, self.references[gv.name_hint])
                self.report.append(
                    {"function": gv.name_hint, "placement": "cpu", "reason": str(error)}
                )
                continue
            mod.update_func(gv, result.scheduled_mod["main"])
            record = {
                "function": gv.name_hint,
                "placement": "gemmini",
                "metadata": result.metadata,
                "trace": result.trace.as_json(),
                "origin": origin,
            }
            if gv.name_hint in self.normalizations:
                record["mathematical_normalization"] = self.normalizations[gv.name_hint]
            self.report.append(record)
        return mod


def prepare_gemmini_tensorir_workloads(
    mod, *, optimize=True, bias_preload=True, prepack_weights=True
):
    """Expose the same mathematical workloads used by compile and DSE paths."""
    from tvm import relax  # pylint: disable=import-outside-toplevel

    if type(optimize) is not bool:
        raise ValueError("optimize must be a boolean")
    passes = [relax.transform.LegalizeOps(), relax.transform.AnnotateTIROpPattern()]
    if optimize:
        passes.extend(
            [
                relax.transform.FoldConstant(),
                relax.transform.CanonicalizeBindings(),
                relax.transform.FuseOps(),
                relax.transform.FuseTIR(),
                relax.transform.RewriteDataflowReshape(),
                relax.transform.DeadCodeElimination(),
            ]
        )
    mod = tvm.transform.Sequential(passes)(mod)
    from .gemmini_epilogue import SpecializeGemminiReadoutScales

    mod = SpecializeGemminiReadoutScales()(mod)
    if type(prepack_weights) is not bool:
        raise ValueError("prepack_weights must be a boolean")
    if prepack_weights:
        from .gemmini_layout import prepack_constants

        mod = prepack_constants(mod)
    from .gemmini_bias import CanonicalizeGemminiBias

    if type(bias_preload) is not bool:
        raise ValueError("bias_preload must be a boolean")
    references, normalizations = {}, {}
    if bias_preload:
        mod = CanonicalizeGemminiBias(references=references, report=normalizations)(mod)
    from .gemmini_epilogue import NormalizeGemminiReadout
    mod = NormalizeGemminiReadout()(mod)
    return mod, references, normalizations


def prepare_gemmini_tensorir_graph(
    mod,
    *,
    optimize=True,
    capabilities=None,
    order="ij",
    schedule_options=None,
    report=None,
    bias_preload=True,
    offload_copies=False,
    prepack_weights=True,
    movement_strategy="rectangular",
):
    """Use normal mathematical legalization/fusion before Gemmini scheduling."""
    from .gemmini import _inline_cpu_producers  # pylint: disable=import-outside-toplevel

    mod, references, normalizations = prepare_gemmini_tensorir_workloads(
        mod, optimize=optimize, bias_preload=bias_preload, prepack_weights=prepack_weights
    )
    mod = ScheduleGemminiTensorIR(
        capabilities=capabilities,
        order=order,
        schedule_options=schedule_options,
        report=report,
        references=references,
        normalizations=normalizations,
        offload_copies=offload_copies,
        movement_strategy=movement_strategy,
    )(mod)
    return _inline_cpu_producers(mod)
