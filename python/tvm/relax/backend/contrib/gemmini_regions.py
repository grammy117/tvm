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
"""Expose index-selected copies through replayable TensorIR transformations."""

import math

import tvm
from tvm import tir


def block_rv(sch, node):
    """Resolve structural identity even when generated blocks share a display name."""
    pending = [sch.get_block("root")]
    while pending:
        block = pending.pop()
        if sch.get(block).same_as(node):
            return block
        pending.extend(sch.get_child_blocks(block))
    raise ValueError("The selected copy block no longer belongs to this schedule")


def copy_selection_loads(value, dtype):
    """Recognize lazy selections of unmodified, equally typed buffer values."""
    if isinstance(value, tir.BufferLoad) and value.dtype == dtype:
        return [value]
    if isinstance(value, tir.Call) and value.op == tvm.ir.Op.get("tir.if_then_else"):
        left = copy_selection_loads(value.args[1], dtype)
        right = copy_selection_loads(value.args[2], dtype)
        if left is not None and right is not None:
            return left + right
    return None


def inline_index_constants(sch):
    """Inline internal index-only producers using the ordinary schedule API.

    TOPI strided_set expresses static begin/end/stride vectors as TE tensors.
    Their pure integer producers are mathematics, not runtime DMA operands.
    Only read-free, internally allocated buffers may be removed here.
    """
    root = sch.get(sch.get_block("root"))
    internal = {buffer.data for buffer in root.alloc_buffers}
    candidates = []

    def visit(node):
        if (
            not isinstance(node, tir.Block)
            or node.init is not None
            or not isinstance(node.body, tir.BufferStore)
            or node.body.buffer.data not in internal
            or node.reads
            or node.body.buffer.dtype not in ("int32", "int64")
        ):
            return
        axes = {axis.var for axis in node.iter_vars}
        unsafe = []

        def check(expr):
            if (
                isinstance(expr, tir.BufferLoad)
                or isinstance(expr, tir.Var)
                and expr not in axes
                or isinstance(expr, tir.Call)
                and expr.op != tvm.ir.Op.get("tir.if_then_else")
                or isinstance(expr, tir.FloatImm)
            ):
                unsafe.append(expr)

        tir.stmt_functor.post_order_visit(node.body.value, check)
        if not unsafe:
            candidates.append(block_rv(sch, node))

    tir.stmt_functor.post_order_visit(root.body, visit)
    for block in candidates:
        sch.compute_inline(block)


def _analyzer(node, axis=None, begin=None, end=None):
    analyzer = tvm.arith.Analyzer()
    for index, iterator in enumerate(node.iter_vars):
        domain = (
            tvm.ir.Range.from_min_extent(
                tir.const(begin, iterator.var.dtype), tir.const(end - begin, iterator.var.dtype)
            )
            if index == axis
            else iterator.dom
        )
        analyzer.bind(iterator.var, domain)
    return analyzer


def _predicate_cuts(predicate, node, limit):
    if isinstance(predicate, (tir.And, tir.Or)):
        cuts = _predicate_cuts(predicate.a, node, limit)
        for axis, points in _predicate_cuts(predicate.b, node, limit).items():
            cuts.setdefault(axis, set()).update(points)
        return cuts
    if isinstance(predicate, tir.Not):
        return _predicate_cuts(predicate.a, node, limit)
    analyzer = _analyzer(node)
    if isinstance(analyzer.simplify(predicate), tir.IntImm):
        return {}
    variables, reads = set(), []
    tir.stmt_functor.post_order_visit(
        predicate,
        lambda expr: variables.add(expr)
        if isinstance(expr, tir.Var)
        else reads.append(expr)
        if isinstance(expr, tir.BufferLoad)
        else None,
    )
    axes = [axis.var for axis in node.iter_vars]
    if reads or variables - set(axes):
        raise ValueError("Copy selection depends on runtime data rather than proven index bounds")
    dependencies = [i for i, axis in enumerate(axes) if axis in variables]
    if len(dependencies) != 1:
        cells = math.prod(int(node.iter_vars[i].dom.extent) for i in dependencies)
        if not dependencies or cells > limit:
            raise ValueError("Nonseparable copy predicate exceeds the region compilation budget")
        return {i: set(range(1, int(node.iter_vars[i].dom.extent))) for i in dependencies}
    axis = dependencies[0]
    extent = int(node.iter_vars[axis].dom.extent)
    points = None
    if isinstance(predicate, (tir.LT, tir.LE, tir.GT, tir.GE, tir.EQ, tir.NE)):
        coefficients = tvm.arith.detect_linear_equation(
            analyzer.simplify(predicate.a - predicate.b), axes
        )
        if coefficients and all(isinstance(x, tir.IntImm) for x in coefficients):
            slope, offset = int(coefficients[axis]), int(coefficients[-1])
            if slope:
                root = -offset // slope
                points = sorted({0, extent, *[x for x in (root, root + 1) if 0 < x < extent]})
    if points is not None:
        previous, result = None, set()
        for begin, end in zip(points, points[1:]):
            truth = _analyzer(node, axis, begin, end).simplify(predicate)
            if not isinstance(truth, tir.IntImm):
                raise ValueError("TVM could not prove a uniform copy predicate over its region")
            current = bool(int(truth))
            if previous is not None and current != previous:
                result.add(begin)
            previous = current
        return {axis: result} if result else {}
    if extent > 65536:
        raise ValueError("Nonlinear copy predicate exceeds the constant-analysis budget")
    previous, result = None, set()
    for index in range(extent):
        value = tir.stmt_functor.substitute(
            predicate, {axes[axis]: tir.const(index, axes[axis].dtype)}
        )
        truth = analyzer.simplify(value)
        if not isinstance(truth, tir.IntImm):
            raise ValueError("Copy predicate is not a proven function of static indices")
        current = bool(int(truth))
        if previous is not None and current != previous:
            result.add(index)
        if len(result) >= limit:
            raise ValueError("Copy selection exceeds the region compilation budget")
        previous = current
    return {axis: result} if result else {}


def _selection_cuts(value, node, limit):
    if isinstance(value, tir.BufferLoad):
        return {}
    cuts = _predicate_cuts(value.args[0], node, limit)
    for branch in value.args[1:]:
        for axis, points in _selection_cuts(branch, node, limit).items():
            cuts.setdefault(axis, set()).update(points)
    return cuts


def _canonical_copy_region(sch, block):
    # Write caching and inlining reconstruct zero-based copy loops. Blockization
    # simplifies the selector under those bounds; the second inlining refreshes
    # read regions after unreachable branches disappear. Both host caches are
    # eliminated, and every transformation is part of the normal schedule trace.
    copied = sch.cache_write(block, 0, "global")
    sch.compute_inline(block)
    sch.blockize(sch.get_loops(copied)[0], preserve_unit_iters=False)
    output = sch.cache_write(copied, 0, "global")
    sch.compute_inline(copied)
    return output


def partition_copy_regions(sch, nodes, max_regions=128):
    """Split only proven index regions; leave placement and tensorization to TVM."""
    pending = [block_rv(sch, node) for node in nodes]
    count = len(pending)
    while pending:
        block = pending.pop()
        node = sch.get(block)
        if isinstance(node.body.value, tir.BufferLoad):
            continue
        cuts = _selection_cuts(node.body.value, node, max_regions)
        if not cuts:
            rewritten = _canonical_copy_region(sch, block)
            if not isinstance(sch.get(rewritten).body.value, tir.BufferLoad):
                raise ValueError("Copy selector remains after constant-region simplification")
            continue
        axis, boundaries = next(iter(sorted(cuts.items())))
        lengths = [
            right - left for left, right in zip([0, *sorted(boundaries)], sorted(boundaries))
        ]
        count += len(lengths)
        if count > max_regions:
            raise ValueError("Copy selection exceeds the region compilation budget")
        loops = sch.get_loops(block)
        if len(loops) != len(node.iter_vars):
            raise ValueError("Copy partition requires complete local mathematical loops")
        chosen = loops[axis]
        sch.reorder(chosen, *[loop for i, loop in enumerate(loops) if i != axis])
        for partition in sch.loop_partition(chosen, lengths):
            children = sch.get_child_blocks(partition)
            if len(children) != 1:
                raise ValueError("Copy partition must contain one mathematical block")
            pending.append(_canonical_copy_region(sch, children[0]))
