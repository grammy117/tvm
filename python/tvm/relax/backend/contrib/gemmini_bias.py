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
"""Proof-bound integer bias normalization after ordinary Relax/TIR fusion.

This is a mathematical pass, separate from the replayable scheduling trace.
An unshared bounded product followed by addition becomes a bias-initialized
modulo-32 reduction and a copy. No tile, address or command is chosen here.
"""

from functools import lru_cache

import tvm
from tvm import tir
from tvm.script import tir as T

from .gemmini_elementwise import _canonical_description, _copy_pair, _read, _region, _strides
from .gemmini_intrin import ACC


def _point(buffer, indices):
    return tir.BufferRegion(
        buffer, [tvm.ir.Range.from_min_extent(i, tir.const(1, i.dtype)) for i in indices]
    )


def _addition(value):
    if isinstance(value, tir.Add) and value.dtype == "int32":
        return value.a, value.b, False
    if isinstance(value, tir.Cast) and value.dtype == "int32":
        value = value.value
        if (
            isinstance(value, tir.Add)
            and value.dtype == "int64"
            and all(isinstance(x, tir.Cast) and x.dtype == "int64" for x in (value.a, value.b))
        ):
            return value.a.value, value.b.value, True
    return None


def _broadcast_statement(statement, target, bias, product_load, buffer):
    """Clone an untiled pointwise definition with fresh loop and block variables."""

    def clone(node, variables):
        substitute = lambda expr: tir.stmt_functor.substitute(expr, variables)
        if isinstance(node, tir.For) and node.thread_binding is None:
            variable = tir.Var(node.loop_var.name + "_bias", node.loop_var.dtype)
            nested = dict(variables)
            nested[node.loop_var] = variable
            return tir.For(
                variable,
                substitute(node.min),
                substitute(node.extent),
                node.kind,
                clone(node.body, nested),
                annotations=node.annotations,
            )
        if not isinstance(node, tir.BlockRealize) or not node.block.same_as(target):
            raise ValueError("Bias broadcast requires an ordinary pointwise loop nest")
        if not tvm.arith.Analyzer().can_prove(node.predicate):
            raise ValueError("Bias broadcast requires an unconditional pointwise definition")
        axes, mapping = [], {}
        for axis in target.iter_vars:
            variable = tir.Var(axis.var.name + "_bias", axis.var.dtype)
            mapping[axis.var] = variable
            axes.append(tir.IterVar(axis.dom, variable, axis.iter_type, axis.thread_tag))
        value = tir.stmt_functor.substitute(bias, mapping)
        indices = [tir.stmt_functor.substitute(i, mapping) for i in product_load.indices]
        block = tir.Block(
            axes,
            [_point(value.buffer, value.indices)],
            [_point(buffer, indices)],
            buffer.name,
            tir.BufferStore(buffer, value, indices),
        )
        return tir.BlockRealize(
            [substitute(i) for i in node.iter_values], substitute(node.predicate), block
        )

    return clone(statement, {})


def fold_bias(func):
    """Preserve the original defined domain; never introduce signed-overflow UB."""
    from .gemmini_tensorir import contraction_block, _normalizable_contractions, _contraction_update

    matches = contraction_block(func) or _normalizable_contractions(func)
    if len(matches) != 1 or not isinstance(func.body, tir.BlockRealize):
        return func, None
    product = matches[0][0]
    if not isinstance(product.init.value, tir.IntImm) or int(product.init.value) != 0:
        return func, None
    blocks, loads, written = [], [], set()

    def visit(node):
        if isinstance(node, tir.Block):
            blocks.append(node)
        if isinstance(node, tir.BufferLoad) and node.buffer.same_as(product.body.buffer):
            loads.append(node)
        if isinstance(node, tir.BufferStore):
            written.add(node.buffer.data)

    tir.stmt_functor.post_order_visit(func.body, visit)
    parameters = {b.data for b in func.buffer_map.values()}
    candidates = []
    for block in blocks:
        if (
            block.init is not None
            or not isinstance(block.body, tir.BufferStore)
            or block.body.buffer.dtype != "int32"
        ):
            continue
        addition = _addition(block.body.value)
        if addition is None:
            continue
        left, right, wrapping = addition
        for value, bias in ((left, right), (right, left)):
            if (
                isinstance(value, tir.BufferLoad)
                and value.buffer.same_as(product.body.buffer)
                and isinstance(bias, tir.BufferLoad)
                and bias.dtype == "int32"
                and bias.buffer.data in parameters
                and bias.buffer.data not in written
            ):
                candidates.append((block, value, bias, wrapping))
    if len(candidates) != 1:
        return func, None
    epilogue, value, bias, wrapping = candidates[0]
    refused = lambda reason: (func, {"applied": False, "reason": reason})
    if product.body.buffer.data in parameters or len(loads) != 2:
        return refused("The unbiased product is returned, shared or read elsewhere")
    equal = tvm.ir.structural_equal
    if not equal(epilogue.body.buffer.shape, product.body.buffer.shape) or not equal(
        epilogue.body.indices, value.indices
    ):
        return refused("Bias addition changes the product layout or broadcasts its output")
    if any(axis.iter_type != tir.IterVar.DataPar for axis in epilogue.iter_vars):
        return refused("Bias addition is not pointwise")
    for axis in list(product.iter_vars) + list(epilogue.iter_vars):
        if (
            not isinstance(axis.dom.min, tir.IntImm)
            or int(axis.dom.min) != 0
            or not isinstance(axis.dom.extent, tir.IntImm)
            or int(axis.dom.extent) <= 0
        ):
            return refused("Bias normalization requires positive static zero-based axes")
    substitutions = {}
    for source, destination, size in zip(value.indices, product.body.indices, value.buffer.shape):
        if isinstance(source, tir.Var):
            if source in substitutions:
                return refused("The product access is not a bijective pointwise map")
            axis = next((axis for axis in epilogue.iter_vars if axis.var.same_as(source)), None)
            if axis is None or not equal(axis.dom.extent, size):
                return refused("The bias consumer does not cover the original product")
            substitutions[source] = destination
        elif not (isinstance(source, tir.IntImm) and int(source) == 0 and int(size) == 1):
            return refused("The product access is not a bijective pointwise map")
    for axis in epilogue.iter_vars:
        if axis.var not in substitutions:
            if int(axis.dom.extent) != 1:
                return refused("The bias consumer broadcasts the product")
            substitutions[axis.var] = tir.const(0, axis.var.dtype)
    escaped = []

    def pointer_use(node):
        if isinstance(node, tir.Call):
            for arg in node.args:
                tir.stmt_functor.post_order_visit(
                    arg,
                    lambda n: escaped.append(n)
                    if isinstance(n, tir.Var) and n.same_as(product.body.buffer.data)
                    else None,
                )

    tir.stmt_functor.post_order_visit(func.body, pointer_use)
    if escaped:
        return refused("The unbiased product escapes through a call")
    root = func.body.block
    statements = list(root.body.seq) if isinstance(root.body, tir.SeqStmt) else [root.body]

    def containing(target):
        positions = []
        for index, statement in enumerate(statements):
            found = []
            tir.stmt_functor.post_order_visit(
                statement,
                lambda n: found.append(n)
                if isinstance(n, tir.Block) and n.same_as(target)
                else None,
            )
            if found:
                positions.append(index)
        return positions

    producers, consumers = containing(product), containing(epilogue)
    if len(producers) != 1 or len(consumers) != 1 or producers[0] >= consumers[0]:
        return refused("Product and bias must have separate ordered loop nests")
    names = {b.name_hint for b in blocks} | {b.name for b in root.alloc_buffers}
    name = "gemmini_bias_broadcast"
    while name in names:
        name += "_"
    # CPU alternatives use the same mathematical function. Empty scope is not
    # the CPU global/stack allocation contract in LowerTVMBuiltin and can cause
    # even a tiny temporary to request TVM's runtime workspace allocator.
    broadcast = tir.decl_buffer(product.body.buffer.shape, "int32", name=name, scope="global")
    try:
        definition = _broadcast_statement(
            statements[consumers[0]], epilogue, bias, value, broadcast
        )
    except ValueError as error:
        return refused(str(error))
    current, multiplication, _ = _contraction_update(product)
    update = (current.astype("int64") + multiplication.astype("int64")).astype("int32")
    initialization = tir.BufferLoad(broadcast, product.body.indices)

    def rewrite(node):
        if isinstance(node, tir.Block) and node.same_as(product):
            return tir.Block(
                node.iter_vars,
                list(node.reads) + [_point(broadcast, product.body.indices)],
                node.writes,
                node.name_hint,
                tir.BufferStore(node.body.buffer, update, node.body.indices),
                tir.BufferStore(node.init.buffer, initialization, node.init.indices),
                node.alloc_buffers,
                node.match_buffers,
                node.annotations,
            )
        if isinstance(node, tir.Block) and node.same_as(epilogue):
            return tir.Block(
                node.iter_vars,
                [_point(value.buffer, value.indices)],
                node.writes,
                node.name_hint,
                tir.BufferStore(node.body.buffer, value, node.body.indices),
                alloc_buffers=node.alloc_buffers,
                match_buffers=node.match_buffers,
                annotations=node.annotations,
            )
        return None

    body = tir.stmt_functor.ir_transform(root.body, None, rewrite)
    statements = list(body.seq) if isinstance(body, tir.SeqStmt) else [body]
    statements.insert(producers[0], definition)
    root = tir.Block(
        root.iter_vars,
        root.reads,
        root.writes,
        root.name_hint,
        tir.SeqStmt(statements),
        root.init,
        list(root.alloc_buffers) + [broadcast],
        root.match_buffers,
        root.annotations,
    )
    normalized = func.with_body(tir.BlockRealize(func.body.iter_values, func.body.predicate, root))
    normalized = normalized.with_attr("gemmini.bias_preload", 1)
    return normalized, {
        "applied": True,
        "pass": "CanonicalizeGemminiBias",
        "proof": "An unshared bounded product plus bias equals a bias-initialized modulo-32 reduction on the original defined domain",
        "source_addition": "explicit_modulo32" if wrapping else "defined_no_overflow",
        "before_ir": tvm.ir.save_json(tvm.IRModule({"main": func})),
        "after_ir": tvm.ir.save_json(tvm.IRModule({"main": normalized})),
    }


@tvm.transform.module_pass(opt_level=0, name="CanonicalizeGemminiBias")
class CanonicalizeGemminiBias:
    def __init__(self, *, references=None, report=None):
        self.references = references if references is not None else {}
        self.report = report if report is not None else {}

    def transform_module(self, mod, _ctx):
        for gv, func in list(mod.functions_items()):
            if not isinstance(func, tir.PrimFunc):
                continue
            aligned = _align_bias_layout(func)
            normalized, note = fold_bias(aligned)
            if note is None or not note.get("applied"):
                normalized = func
            elif not aligned.same_as(func):
                note["before_ir"] = tvm.ir.save_json(tvm.IRModule({"main": func}))
                note["layout_normalized"] = True
            if note is not None:
                self.report[gv.name_hint] = note
            if not normalized.same_as(func):
                self.references[gv.name_hint] = func
                mod.update_func(gv, normalized)
        return mod


def _align_bias_layout(func):
    """Use scheduling APIs to match bias addition after a bijective output view."""
    from .gemmini_tensorir import contraction_block

    matches = contraction_block(func)
    if len(matches) != 1:
        return func
    sch = tir.Schedule(func, debug_mask="all")
    product = sch.get_block(matches[0][0].name_hint)
    parameters = {buffer.data for buffer in func.buffer_map.values()}
    try:
        while True:
            consumers = sch.get_consumers(product)
            if len(consumers) != 1:
                return func
            consumer = consumers[0]
            node = sch.get(consumer)
            if (
                isinstance(node.body, tir.BufferStore)
                and isinstance(node.body.value, tir.BufferLoad)
                and node.body.buffer.data not in parameters
            ):
                sch.compute_inline(consumer)
                continue
            break
        addition = _addition(node.body.value) if isinstance(node.body, tir.BufferStore) else None
        if addition is None:
            return func
        output = sch.get(product).body.buffer
        load = next(
            (
                value
                for value in addition[:2]
                if isinstance(value, tir.BufferLoad) and value.buffer.same_as(output)
            ),
            None,
        )
        if load is None:
            return func
        if tvm.ir.structural_equal(
            node.body.buffer.shape, output.shape
        ) and tvm.ir.structural_equal(node.body.indices, load.indices):
            return sch.mod["main"]
        index_map = tir.IndexMap([axis.var for axis in node.iter_vars], load.indices, None)
        sch.transform_block_layout(consumer, index_map)
        sch.reindex(consumer, ("write", 0))
        return sch.mod["main"]
    except (ValueError, tir.schedule.ScheduleError):
        return func


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
