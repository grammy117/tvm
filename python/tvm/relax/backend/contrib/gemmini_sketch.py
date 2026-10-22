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
"""Gemmini contraction sketches as ordinary traced TensorIR schedules.

A sketch fixes program structure: traversal order, the intrinsic rectangle,
operand-cache placement, dataflow and software pipelining. Tile sizes are
``sample_perfect_tile`` decisions, so the stock MetaSchedule mutators explore
operand reuse within a sketch and replayed traces stay valid for any decision.
Tensorization is only annotated (``meta_schedule.auto_tensorize``) and performed
by the stock RewriteTensorize postprocessor, unless a caller asks for the
immediate deterministic form used by untuned compilation.
"""

from dataclasses import dataclass, replace
import math

from tvm import tir

from .gemmini_intrin import ACC, SPAD, register_intrinsics
from .gemmini_target import GemminiCapabilities
from .gemmini_tensorir import (
    _contraction_update,
    _normalizable_contractions,
    _supported_contraction_init,
    contraction_block,
)

# Upper bound on tiles per operand-reuse group. Capacity is checked exactly by
# VerifyGemminiMemory; this only keeps sampling away from hopeless groups.
MAX_GROUP_TILES = 64


@dataclass(frozen=True)
class ContractionSketch:
    """Structural choices; every size inside the structure is sampled."""

    order: str = "ij"
    rows: int = 16
    cols: int = 16
    depth: int = 16
    dataflow: str = "os"
    hoist: bool = False
    pipeline: bool = False
    epilogue: str = "root"
    staging: str = "tile"
    tail: str = "none"

    def __post_init__(self):
        if self.order not in ("ij", "ji") or self.dataflow not in ("os", "ws"):
            raise ValueError("Unsupported Gemmini sketch order or dataflow")
        if self.tail not in ("none", "m", "n"):
            raise ValueError("Unsupported Gemmini tail partition")
        if self.epilogue not in ("root", "tile") or self.staging not in ("root", "tile"):
            raise ValueError("Unsupported Gemmini epilogue or staging placement")
        if any(type(x) is not int or not 1 <= x <= 16 for x in (self.rows, self.cols, self.depth)):
            raise ValueError("Gemmini rectangles require dimensions in [1,16]")

    @property
    def name(self):
        return (
            f"{self.order}.{self.rows}x{self.cols}x{self.depth}.{self.dataflow}"
            f".h{int(self.hoist)}.p{int(self.pipeline)}.e{self.epilogue}.s{self.staging}"
            f".t{self.tail}"
        )


@dataclass(frozen=True)
class ContractionInfo:
    """Shape facts read from a supported contraction block."""

    m: int
    n: int
    k: int
    biased: bool
    wrapping: bool
    producers: bool
    consumers: bool


def rectangle_options(extent, *, padded=False):
    """Intrinsic rectangle sizes for one axis.

    A full 16-wide rectangle is always preferred when it divides the extent.
    Otherwise the largest divisor up to 16 tiles the axis exactly; for the
    reduction axis a padded 16-deep rectangle is also offered, because zero
    products are the only way to keep full-depth commands on an odd K.
    """
    if extent % 16 == 0:
        return [16]
    options = [max(x for x in range(1, 17) if extent % x == 0)]
    if padded and extent > 16:
        options.append(16)
    return options


def contraction_info(func, block_name):
    """Return shape facts for a canonical or normalizable contraction, or None."""
    matches = [m for m in contraction_block(func) if m[0].name_hint == block_name]
    if matches:
        node, (m, n, k) = matches[0]
    else:
        matches = [m for m in _normalizable_contractions(func) if m[0].name_hint == block_name]
        if not matches:
            return None
        node, (m, n, k), _ = matches[0]
    consumers, producers = [], []
    written = node.body.buffer.data
    operands = {load.buffer.data for load in _operand_loads(node)}

    def visit(other):
        if not isinstance(other, tir.Block) or other.same_as(node) or other.name_hint == "root":
            return
        if any(region.buffer.data == written for region in other.reads):
            consumers.append(other)
        if any(region.buffer.data in operands for region in other.writes):
            producers.append(other)

    tir.stmt_functor.post_order_visit(func.body, visit)
    return ContractionInfo(
        m,
        n,
        k,
        biased=isinstance(node.init.value, tir.BufferLoad),
        wrapping=_contraction_update(node)[2],
        producers=bool(producers),
        consumers=bool(consumers),
    )


def _operand_loads(node):
    product = _contraction_update(node)[1]
    return [product.a.value, product.b.value]


def _rectangles(info):
    """(rows, cols, tail) choices: exact divisors, or 16 with a partitioned tail."""
    rows = rectangle_options(info.m)
    cols = rectangle_options(info.n)
    choices = [(rows[0], cols[0], "none")]
    if info.m % 16 and info.m > 16:
        choices.append((16, cols[0], "m"))
    if info.n % 16 and info.n > 16:
        choices.append((rows[0], 16, "n"))
    return choices


def sketch_space(info, *, dataflows=("os", "ws")):
    """Enumerate structurally distinct sketches for one contraction."""
    depths = rectangle_options(info.k, padded=not info.wrapping)
    sketches = []
    for r, c, tail in _rectangles(info):
        for d in depths:
            tiles_i, tiles_j = info.m // r, info.n // c
            tiles_k = math.ceil(info.k / d)
            # Traversal order only matters when both output axes have
            # several tiles; hoisting needs an outer loop to hoist out of.
            orders = ("ij", "ji") if tiles_i > 1 and tiles_j > 1 else ("ij",)
            for order in orders:
                outer_tiles = tiles_j if order == "ij" else tiles_i
                for hoist in (False, True) if outer_tiles > 1 else (False,):
                    for pipeline in (False, True) if tiles_k > 1 else (False,):
                        for dataflow in dataflows:
                            epilogues = ("root", "tile") if info.consumers and tail == "none" else ("root",)
                            for epilogue in epilogues:
                                padded = d != depths[0]
                                for staging in ("tile", "root") if info.producers or padded else ("tile",):
                                    sketches.append(
                                        ContractionSketch(
                                            order=order, rows=r, cols=c, depth=d,
                                            dataflow=dataflow, hoist=hoist, pipeline=pipeline,
                                            epilogue=epilogue, staging=staging, tail=tail,
                                        )
                                    )
    return sketches


def default_sketch(info):
    """The untuned structure: first rectangle, ij, OS, no hoist or pipeline."""
    return ContractionSketch(
        rows=rectangle_options(info.m)[0],
        cols=rectangle_options(info.n)[0],
        depth=rectangle_options(info.k, padded=not info.wrapping)[0],
    )


def _normalize_block(sch, block):
    """Bring one contraction to the canonical [batch..., i, j, k] matrix form."""
    func = sch.mod["main"]
    name = sch.get(block).name_hint
    canonical = [m for m in contraction_block(func) if m[0].name_hint == name]
    if canonical:
        return block, canonical[0][1]
    matches = [m for m in _normalizable_contractions(func) if m[0].name_hint == name]
    if len(matches) != 1:
        raise ValueError("Block is not a supported exact INT8/INT32 contraction")
    node, shape, (batch, rows, cols, reduction) = matches[0]
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

    dtype = node.iter_vars[0].var.dtype
    sch.transform_block_layout(block, tir.IndexMap.from_func(matrix_map, ndim=len(extents), index_dtype=dtype))
    sch.reindex(block, ("write", 0))
    product = _contraction_update(sch.get(block))[1]
    a_copy = sch.reindex(block, product.a.value.buffer)
    b_copy = sch.reindex(block, product.b.value.buffer)
    b_rank = len(sch.get(b_copy).writes[0].buffer.shape)
    sch.transform_layout(
        b_copy,
        ("write", 0),
        tir.IndexMap.from_func(
            lambda *idx: (*idx[:-2], idx[-1], idx[-2]), ndim=b_rank, index_dtype=dtype
        ),
    )
    for copy in (a_copy, b_copy):
        for producer in sch.get_producers(copy):
            try:
                sch.compute_inline(producer)
            except tir.schedule.ScheduleError:
                pass
    if not [m for m in contraction_block(sch.mod["main"]) if m[0].name_hint == name]:
        raise ValueError("TVM layout normalization did not form a canonical matrix contraction")
    return block, shape


def _tile_major(rows, cols, rank, dtype):
    """Place each rows x cols tile on consecutive physical rows."""
    return tir.IndexMap.from_func(
        lambda *idx: (*idx[:-2], idx[-2] // rows, idx[-1] // cols, idx[-2] % rows, idx[-1] % cols),
        ndim=rank,
        index_dtype=dtype,
    )


def _tile_copy(sch, copy, rows, cols):
    """Split a cache copy's matrix loops into tile loops and one tile body."""
    r, c = sch.get_loops(copy)[-2:]
    rt, ri = sch.split(r, [None, rows])
    ct, ci = sch.split(c, [None, cols])
    sch.reorder(rt, ct, ri, ci)
    return ri


def _place_epilogue(sch, write, outer):
    """Move pointwise CPU consumers of the output tile into the tile loop."""
    pending = [sch.get(block).name_hint for block in sch.get_consumers(write)]
    placed = set()

    def inside_tile(block):
        return any(sch.get(loop).same_as(sch.get(outer)) for loop in sch.get_loops(block))

    while pending:
        name = pending.pop(0)
        # A shared pointwise producer can also be queued as another consumer.
        # Inlining it removes its BlockRV; resolve queued names against the
        # current schedule instead of retaining invalid references.
        if name in placed or name not in _block_names(sch):
            continue
        consumer = sch.get_block(name)
        node = sch.get(consumer)
        if (
            isinstance(node.body, tir.BufferStore)
            and isinstance(node.body.value, tir.BufferLoad)
            and math.prod(int(x) for x in node.body.buffer.shape)
            < math.prod(int(x) for x in node.body.value.buffer.shape)
        ):
            continue
        for producer in list(sch.get_producers(consumer)):
            if not inside_tile(producer):
                try:
                    sch.compute_inline(producer)
                except tir.schedule.ScheduleError:
                    pass
        if not all(inside_tile(producer) for producer in sch.get_producers(consumer)):
            continue
        try:
            sch.reverse_compute_at(consumer, outer, preserve_unit_loops=True)
        except tir.schedule.ScheduleError:
            continue
        placed.add(name)
        pending.extend(sch.get(block).name_hint for block in sch.get_consumers(consumer))


def apply_contraction_sketch(
    sch, block, sketch, *, decisions=None, tensorize=False, capabilities=None
):
    """Transform ``block`` in ``sch`` into the given sketch and return the schedule.

    ``decisions`` optionally fixes the tile samples (``i``, ``j``, ``k`` as
    [outer, group] pairs; ``ti``/``tj``/``tk`` for a tail partition) for
    deterministic schedules. Bias and readout fusion may continue on a copy, so
    callers must use the returned schedule. Structure-violating inputs raise
    ``ValueError`` or ``tir.schedule.ScheduleError``.
    """
    capabilities = capabilities or GemminiCapabilities()
    decisions = decisions or {}
    block, (m, n, depth) = _normalize_block(sch, block)
    node = sch.get(block)
    if not _supported_contraction_init(node):
        raise ValueError("Unsupported contraction initialization")
    biased = isinstance(node.init.value, tir.BufferLoad)
    wrapping = _contraction_update(node)[2]
    rows, cols, chunk = sketch.rows, sketch.cols, sketch.depth
    if (m % rows and sketch.tail != "m") or (n % cols and sketch.tail != "n"):
        raise ValueError("Sketch rectangle must tile the output exactly")
    if chunk > capabilities.max_chunk:
        raise ValueError("Reduction chunk exceeds the PE output bound")
    rank = len(node.iter_vars)
    dtype = node.iter_vars[0].var.dtype
    if depth % chunk:
        if wrapping:
            raise ValueError("Explicit modular contractions require divisor tiles")
        sch.pad_einsum(block, [1] * (rank - 1) + [chunk])
    sch.annotate(sch.get_block("root"), "gemmini.sketch", sketch.name)
    name = sch.get(block).name_hint
    parts = [(block, rows, cols, "", "ij")]
    i, j, _ = sch.get_loops(block)[-3:]
    # Interior/tail split: the interior keeps full rectangles and the tail gets
    # a partial-rectangle intrinsic of its own, with no padding or staging.
    # Its size divides both the offset and remainder so transform_layout never
    # maps one rectangle across a tile-major buffer boundary.
    if sketch.tail == "m" and m % rows:
        interior, tail = _partition_tail(sch, name, i, m, rows)
        tail_rows = math.gcd(m - m % rows, m % rows)
        parts = [(interior, rows, cols, "", "ij"), (tail, tail_rows, cols, "t", "ij")]
    elif sketch.tail == "n" and n % cols:
        sch.reorder(j, i)
        interior, tail = _partition_tail(sch, name, j, n, cols)
        tail_cols = math.gcd(n - n % cols, n % cols)
        parts = [(interior, rows, cols, "", "ji"), (tail, rows, tail_cols, "t", "ji")]
    for part, part_rows, part_cols, tag, loop_order in parts:
        sch = _schedule_part(
            sch,
            part,
            sketch,
            part_rows,
            part_cols,
            chunk,
            {key[len(tag):]: value for key, value in decisions.items() if key.startswith(tag) and len(key) == len(tag) + 1},
            biased=biased,
            wrapping=wrapping,
            dtype=dtype,
            tensorize=tensorize,
            single=len(parts) == 1,
            loop_order=loop_order,
        )
    return sch


def _block_names(sch):
    names = []
    tir.stmt_functor.post_order_visit(
        sch.mod["main"].body,
        lambda node: names.append(node.name_hint) if isinstance(node, tir.Block) else None,
    )
    return names


def _partition_tail(sch, name, loop, extent, size):
    """Split ``loop`` into whole ``size`` tiles and one partial tile.

    The tail block keeps the interior's offset as a constant in its binding;
    blockize places that multiple of the tile size in the outer subspace.
    """
    before = set(_block_names(sch))
    sch.loop_partition(loop, [extent - extent % size])
    interior, tail = [
        item
        for item in _block_names(sch)
        if item not in before
        and item.startswith(name + "_")
        and not item.endswith("_partition")
        and not item.endswith("_common")
    ]
    return sch.get_block(interior), sch.get_block(tail)


def _schedule_part(
    sch, block, sketch, rows, cols, chunk, decisions, *, biased, wrapping, dtype, tensorize, single, loop_order
):
    """Tile, cache and tensorize one contraction block of a sketch."""
    loops = sch.get_loops(block)[-3:]
    i, j, k = (loops[0], loops[1], loops[2]) if loop_order == "ij" else (loops[1], loops[0], loops[2])
    m, n, depth = (int(sch.get(loop).extent) for loop in (i, j, k))
    tiles = {"i": m // rows, "j": n // cols, "k": math.ceil(depth / chunk)}
    it, i2 = sch.split(i, [None, rows])
    jt, j2 = sch.split(j, [None, cols])
    kt, k2 = sch.split(k, [None, chunk])
    factors = {}
    for key, loop in (("i", it), ("j", jt), ("k", kt)):
        factors[key] = sch.sample_perfect_tile(
            loop,
            n=2,
            max_innermost_factor=min(tiles[key], MAX_GROUP_TILES),
            decision=decisions.get(key),
        )
    i0, i1 = sch.split(it, factors["i"])
    j0, j1 = sch.split(jt, factors["j"])
    k0, k1 = sch.split(kt, factors["k"])
    outer = (i0, j0) if sketch.order == "ij" else (j0, i0)
    inner = (k1, i1, j1) if sketch.dataflow == "os" else (k1, j1, i1)
    sch.reorder(*outer, k0, *inner, i2, j2, k2)
    write = sch.cache_write(block, 0, ACC)
    sch.reverse_compute_at(write, outer[1], preserve_unit_loops=True)
    if biased and single:
        # The normalization's output copy folds into the DMA writeback block.
        for consumer in list(sch.get_consumers(write)):
            copy = sch.get(consumer)
            if (
                isinstance(copy.body, tir.BufferStore)
                and isinstance(copy.body.value, tir.BufferLoad)
                and tvm_equal(copy.body.indices, copy.body.value.indices)
                and tvm_equal(copy.body.buffer.shape, copy.body.value.buffer.shape)
            ):
                sch.reverse_compute_inline(consumer)
    a_load, b_load = _operand_loads(sch.get(block))
    reads = sch.get(block).reads

    def read_index(buffer):
        return next(index for index, region in enumerate(reads) if region.buffer.same_as(buffer))

    load_a = sch.cache_read(block, read_index(a_load.buffer), SPAD)
    load_b = sch.cache_read(block, read_index(b_load.buffer), SPAD)
    a_at = outer[0] if sketch.hoist and sketch.order == "ij" else k0
    b_at = outer[0] if sketch.hoist and sketch.order == "ji" else k0
    sch.compute_at(load_a, a_at, preserve_unit_loops=True)
    sch.compute_at(load_b, b_at, preserve_unit_loops=True)
    a_rank = len(sch.get(load_a).writes[0].buffer.shape)
    b_rank = len(sch.get(load_b).writes[0].buffer.shape)
    c_rank = len(sch.get(block).writes[0].buffer.shape)
    sch.transform_layout(load_a, ("write", 0), _tile_major(rows, chunk, a_rank, dtype))
    sch.transform_layout(load_b, ("write", 0), _tile_major(chunk, cols, b_rank, dtype))
    sch.transform_layout(block, ("write", 0), _tile_major(rows, cols, c_rank, dtype))
    for cache in (load_a, load_b, block):
        sch.storage_align(cache, 0, axis=-2, factor=16, offset=0)
    # Producers of the operand caches are padding or reindexing stages. Either
    # materialize them once at the root, or next to each consuming DMA.
    for load, location in ((load_a, a_at), (load_b, b_at)):
        for producer in sch.get_producers(load):
            for upstream in sch.get_producers(producer):
                try:
                    sch.compute_inline(upstream)
                except tir.schedule.ScheduleError:
                    pass
            if sketch.staging == "tile":
                sch.compute_at(producer, location, preserve_unit_loops=True)
    _tile_copy(sch, load_a, rows, chunk)
    _tile_copy(sch, load_b, chunk, cols)
    _tile_copy(sch, write, rows, cols)
    if sketch.epilogue == "tile" and single:
        _place_epilogue(sch, write, outer[1])
    init = sch.decompose_reduction(block, k0)
    bias_intrinsic = None
    if biased:
        from .gemmini_bias import schedule_bias_init

        sch, bias_intrinsic = schedule_bias_init(sch, init, outer[1], rows, cols)
    readout_intrinsic = None
    if single:
        from .gemmini_epilogue import fuse_readout

        sch, write, readout_intrinsic = fuse_readout(sch, write, rows, cols)
    if sketch.pipeline:
        _annotate_pipeline(sch, k0)
    intrinsics = register_intrinsics(rows, cols, chunk, wrapping=wrapping)
    if bias_intrinsic is not None:
        intrinsics = dict(intrinsics, init=bias_intrinsic)
    if readout_intrinsic is not None:
        intrinsics = dict(intrinsics, store=readout_intrinsic)
    if sketch.dataflow == "ws":
        intrinsics = dict(intrinsics, compute=intrinsics["compute_ws"])
    for target, axes, kind in (
        (init, 2, "init"),
        (load_a, 2, "load_a"),
        (load_b, 2, "load_b"),
        (block, 3, "compute"),
        (write, 2, "store"),
    ):
        tile = sch.blockize(sch.get_loops(target)[-axes], preserve_unit_iters=True)
        if kind != "store":
            # Blockization changes the buffer rank seen by the outer block.
            # Keep physical row alignment where tensorize and compaction read it.
            sch.storage_align(tile, 0, axis=-2, factor=16, offset=0)
        if tensorize:
            sch.tensorize(tile, intrinsics[kind])
        else:
            sch.annotate(tile, "meta_schedule.auto_tensorize", intrinsics[kind])
    return sch


def _annotate_pipeline(sch, loop):
    """Double-buffer operand caches: loads of tile k+1 issue before compute k."""
    body = sch.get(loop).body
    statements = list(body.seq) if isinstance(body, tir.SeqStmt) else [body]
    if len(statements) < 2:
        raise ValueError("Pipelining requires at least one operand load and a compute nest")
    # Integer immediates, as stock MetaSchedule records them, keep the
    # annotation replayable from a serialized trace.
    stages = [tir.IntImm("int32", 0)] * (len(statements) - 1) + [tir.IntImm("int32", 1)]
    sch.annotate(loop, "software_pipeline_stage", stages)
    sch.annotate(loop, "software_pipeline_order", [tir.IntImm("int32", i) for i in range(len(statements))])


def tvm_equal(a, b):
    import tvm  # pylint: disable=import-outside-toplevel

    return tvm.ir.structural_equal(a, b)


def with_rectangle(sketch, rows, cols, depth):
    return replace(sketch, rows=rows, cols=cols, depth=depth)


def default_schedule(func, *, capabilities=None, order="ij", chunk=None, row_tiles=1, col_tiles=1):
    """The untuned Gemmini schedule: default sketch, one tile per reuse group.

    ``order``, ``chunk`` and the group sizes select other points of the same
    sketch family for explicit experiments; tuning explores them by sampling.
    """
    names = []
    tir.stmt_functor.post_order_visit(
        func.body,
        lambda node: names.append(node.name_hint)
        if isinstance(node, tir.Block) and node.init is not None
        else None,
    )
    names = [name for name in names if contraction_info(func, name) is not None]
    if len(names) != 1:
        raise ValueError("Default Gemmini schedule requires exactly one supported contraction")
    info = contraction_info(func, names[0])
    sketch = default_sketch(info)
    if chunk is not None:
        depth = max(x for x in range(1, min(chunk, 16) + 1) if info.k % x == 0 or x == min(chunk, 16))
        sketch = replace(sketch, depth=depth)
    sketch = replace(sketch, order=order)
    tiles = {
        "i": info.m // sketch.rows,
        "j": info.n // sketch.cols,
        "k": math.ceil(info.k / sketch.depth),
    }
    groups = {"i": row_tiles, "j": col_tiles, "k": 1}
    if any(tiles[key] % groups[key] for key in tiles):
        raise ValueError("Reuse groups must divide the tile counts")
    sch = tir.Schedule(func)
    return apply_contraction_sketch(
        sch,
        sch.get_block(names[0]),
        sketch,
        decisions={key: [tiles[key] // groups[key], groups[key]] for key in tiles},
        tensorize=True,
        capabilities=capabilities,
    )
