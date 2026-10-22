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
"""Extract constant matrix packing through TVM's layout-preprocessing passes."""

from tvm import relax, tir


def prepack_constants(mod):
    """Pack immutable right operands once, preserving graph/function ownership.

    AttachAttrLayoutFreeBuffers identifies constants from the calling Relax IR.
    TensorIR reindex/padding states the desired layout. The stock preprocessing
    splitter and constant folder evaluate it; no model-specific weight generator
    or Python copy of a convolution implementation participates.

    RewriteLayout's index-map inference does not support the conditional padded
    convolution maps in this TVM revision. The schedule supplies its ordinary
    layout-rewrite annotation explicitly instead of patching that inference.
    """
    from .gemmini_sketch import _normalize_block
    from .gemmini_tensorir import (
        _contraction_update,
        _normalizable_contractions,
        contraction_block,
    )

    constants = []
    for func in mod.functions.values():
        if isinstance(func, relax.Function):
            relax.analysis.post_order_visit(
                func,
                lambda node: constants.append(node) if isinstance(node, relax.Constant) else None,
            )
    if not constants:
        return mod

    mod = relax.transform.AttachAttrLayoutFreeBuffers()(mod)
    changed = False
    for gv, func in list(mod.functions.items()):
        if not isinstance(func, tir.PrimFunc):
            continue
        free = set(int(index) for index in func.attrs.get("layout_free_buffers", []))
        matches = contraction_block(func) or _normalizable_contractions(func)
        if len(matches) != 1 or not free:
            continue
        product = _contraction_update(matches[0][0])[1]
        right = product.b.value.buffer
        if not any(func.buffer_map[func.params[index]].same_as(right) for index in free):
            continue
        # Already-compact matrix weights need no runtime reindex; padding is
        # nevertheless useful for tails and is also folded here.
        sch = tir.Schedule(func, debug_mask="all")
        try:
            block, (m, n, k) = _normalize_block(sch, sch.get_block(matches[0][0].name_hint))
            node = sch.get(block)
            if k > 131056:  # Padding must retain the existing prefix admission.
                continue
            # Only the right operand needs a compile-time layout. Preserve
            # logical rows and narrow column extents instead of introducing
            # extra output copies/padded lanes for small-channel convolutions.
            sch.pad_einsum(block, [1] * (len(node.iter_vars) - 3) + [1, max(2, min(n, 16)), 16])
            product = _contraction_update(sch.get(block))[1]
            packed = product.b.value.buffer
            producers = [
                p for p in sch.get_producers(block) if sch.get(p).writes[0].buffer.same_as(packed)
            ]
            if len(producers) != 1:
                continue
            packing = producers[0]
            for producer in list(sch.get_producers(packing)):
                sch.compute_inline(producer)
            reads = sch.get(packing).reads
            if len(reads) != 1 or not reads[0].buffer.same_as(right):
                continue
            sch.annotate(packing, "meta_schedule.layout_rewrite_preproc", True)
        except (ValueError, tir.schedule.ScheduleError):
            continue
        mod.update_func(gv, sch.mod["main"])
        changed = True
    if changed:
        mod = relax.transform.SplitLayoutRewritePreproc()(mod)
        mod = relax.transform.FoldConstant()(mod)
        mod = relax.transform.DeadCodeElimination()(mod)
    return mod
