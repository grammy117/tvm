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
"""Gemmini design space for stock MetaSchedule.

The space is a ``SpaceGeneratorUnion`` of two ordinary ``PostOrderApply``
generators over the same prepared function:

* Gemmini placement: ``GemminiScheduleRule`` turns supported blocks into
  sketches with sampled tile decisions (see ``gemmini_sketch``); stock rules
  schedule the remaining CPU blocks of the same function.
* CPU placement: the stock LLVM rules, configured for the bound single-core
  RV64GC target (no parallel launch, no vector unit).

Search, mutation, cost model and database are TVM's. The only Gemmini
postprocessor is a legality check, like ``VerifyGPUCode`` for GPUs.
"""

import tvm
from tvm import meta_schedule as ms, tir
from tvm.meta_schedule.utils import derived_object

from .gemmini_memory import lower_memory
from .gemmini_sketch import apply_contraction_sketch, contraction_info, sketch_space
from .gemmini_target import GemminiCapabilities
from .gemmini_elementwise import elementwise_blocks, schedule_elementwise
from .gemmini_reduction import reduction_blocks, schedule_reduction


def _supported(func):
    """Blocks the Gemmini rule can place, by name."""
    contractions = [
        block.name_hint
        for block in _blocks(func)
        if block.init is not None and contraction_info(func, block.name_hint) is not None
    ]
    if contractions:
        return "contraction", contractions
    if len(reduction_blocks(func)) == 1:
        return "reduction", [reduction_blocks(func)[0][0].name_hint]
    pointwise = elementwise_blocks(func)
    if pointwise:
        return "elementwise", [pointwise[0][0].name_hint]
    return None, []


def _blocks(func):
    found = []
    tir.stmt_functor.post_order_visit(
        func.body, lambda node: found.append(node) if isinstance(node, tir.Block) else None
    )
    return found


@derived_object
class GemminiScheduleRule(ms.schedule_rule.PyScheduleRule):
    """Expand supported blocks into Gemmini sketches on the given schedule."""

    def __init__(self, capabilities=None, dataflows=("os", "ws")):
        self.capabilities = capabilities or GemminiCapabilities()
        self.dataflows = tuple(dataflows)
        self.target = None

    def __str__(self):
        return "GemminiScheduleRule"

    def _initialize_with_tune_context(self, context):
        self.target = context.target
        if self.target is None or "gemmini" not in self.target.keys:
            raise ValueError("Gemmini schedule rules require an explicitly selected Gemmini target key")
        if self.target.model != "gemmini-" + self.capabilities.fingerprint:
            raise ValueError("MetaSchedule target does not bind the selected Gemmini capabilities")

    def apply(self, sch, block):
        func = sch.mod["main"]
        kind, names = _supported(func)
        if (kind != "contraction"
                and "gemmini.sketch" in func.body.block.annotations):
            # Decomposing a contraction removes its reduction init. Do not
            # then reinterpret its padding/reindexing producers as independent
            # copy kernels: their predicates belong to the contraction sketch.
            # Stock rules still schedule the remaining host blocks.
            return [sch]
        if kind is None:
            # This generator only describes Gemmini placements. The CPU
            # generator of the union covers functions without such blocks.
            return []
        name = sch.get(block).name_hint
        if name not in names:
            return [sch]
        if kind == "contraction":
            info = contraction_info(func, name)
            results = []
            for sketch in sketch_space(info, dataflows=self.dataflows):
                try:
                    candidate = apply_contraction_sketch(
                        sch.copy(), block, sketch, capabilities=self.capabilities
                    )
                except (ValueError, tvm.TVMError):
                    # A sketch whose structure this function cannot take.
                    continue
                results.append(candidate)
            return results
        # Secondary operators keep their structural rules: each candidate is
        # a recorded trace of ordinary primitives replayed onto this schedule.
        schedule = schedule_reduction if kind == "reduction" else schedule_elementwise
        results, traces = [], set()
        for scalar in (False, True):
            try:
                transformed, _, _ = schedule(func, tensorize=False, mark_tensorize=True, scalar=scalar)
            except (ValueError, tvm.TVMError):
                continue
            if str(transformed.trace) in traces:
                continue
            traces.add(str(transformed.trace))
            candidate = sch.copy()
            transformed.trace.apply_to_schedule(candidate, remove_postproc=False)
            results.append(candidate)
        return results

    def clone(self):
        return GemminiScheduleRule(self.capabilities, self.dataflows)


@derived_object
class VerifyGemminiMemory(ms.postproc.PyPostproc):
    """Reject untensorized CPU SRAM accesses and over-capacity row plans."""

    def __init__(self, capabilities=None):
        self.capabilities = capabilities or GemminiCapabilities()

    def _initialize_with_tune_context(self, context):
        del context

    def apply(self, sch):
        try:
            lower_memory(
                sch.mod, spad_rows=self.capabilities.spad_rows, acc_rows=self.capabilities.acc_rows
            )
        except (ValueError, tvm.TVMError):
            return False
        return True

    def clone(self):
        return VerifyGemminiMemory(self.capabilities)

    def __str__(self):
        return "VerifyGemminiMemory"


# Stock LLVM rules for the bound core: one RV64GC hart without a vector unit,
# so parallel launch and vectorization stay disabled.
_UNROLL_STEPS = [0, 16, 64, 512]


def cpu_schedule_rules():
    return [
        ms.schedule_rule.ApplyCustomRule(),
        ms.schedule_rule.InlineConstantScalars(),
        ms.schedule_rule.AutoInline(
            into_producer=False,
            into_consumer=True,
            inline_const_tensor=True,
            disallow_if_then_else=True,
            require_injective=True,
            require_ordered=True,
            disallow_op=["tir.exp"],
        ),
        ms.schedule_rule.MultiLevelTiling(
            structure="SSRSRS",
            tile_binds=None,
            max_innermost_factor=64,
            vector_load_lens=None,
            reuse_read=None,
            reuse_write=ms.schedule_rule.ReuseType(req="may", levels=[1, 2], scope="global"),
        ),
        ms.schedule_rule.ParallelizeVectorizeUnroll(
            max_jobs_per_core=-1,
            max_vectorize_extent=-1,
            unroll_max_steps=_UNROLL_STEPS,
            unroll_explicit=True,
        ),
        ms.schedule_rule.RandomComputeLocation(),
    ]


def gemmini_schedule_rules(capabilities=None, dataflows=("os", "ws")):
    return [
        GemminiScheduleRule(capabilities, dataflows),
        ms.schedule_rule.ParallelizeVectorizeUnroll(
            max_jobs_per_core=-1,
            max_vectorize_extent=-1,
            unroll_max_steps=_UNROLL_STEPS,
            unroll_explicit=True,
        ),
    ]


def gemmini_postprocs(capabilities=None):
    """The stock CPU-tensorization postprocessors plus the Gemmini legality check."""
    return [
        ms.postproc.DisallowDynamicLoop(),
        ms.postproc.RewriteParallelVectorizeUnroll(),
        ms.postproc.RewriteReductionBlock(),
        ms.postproc.RewriteTensorize(),
        VerifyGemminiMemory(capabilities),
    ]


def gemmini_mutator_probs():
    return {
        ms.mutator.MutateTileSize(): 0.9,
        ms.mutator.MutateComputeLocation(): 0.05,
        ms.mutator.MutateUnroll(): 0.05,
    }


def gemmini_space_generator(capabilities=None, *, include_cpu=True, dataflows=("os", "ws")):
    """Gemmini and CPU placements of one function as a stock union of spaces."""
    capabilities = capabilities or GemminiCapabilities()
    generators = [
        ms.space_generator.PostOrderApply(
            sch_rules=gemmini_schedule_rules(capabilities, dataflows),
            postprocs=[],
            mutator_probs={},
        )
    ]
    if include_cpu:
        generators.append(
            ms.space_generator.PostOrderApply(
                sch_rules=cpu_schedule_rules(), postprocs=[], mutator_probs={}
            )
        )
    return ms.space_generator.SpaceGeneratorUnion(
        generators,
        sch_rules=[],
        postprocs=gemmini_postprocs(capabilities),
        mutator_probs=gemmini_mutator_probs(),
    )
