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
"""Gemmini schedule rules in standard MetaSchedule design-space generation.

Candidates remain mathematical programs until RewriteTensorize postprocessing.
No alternative source-text templates, custom search controller or synthetic
performance labels are needed. A deployment builder/runner must still lower
target memory and measure on the explicitly selected target.
"""

import itertools
import random

from tvm import meta_schedule as ms, tir
from tvm.meta_schedule.utils import derived_object

from .gemmini_memory import lower_memory
from .gemmini_target import GemminiCapabilities
from .gemmini_tensorir import _normalizable_contractions, contraction_block, schedule_contraction
from .gemmini_elementwise import elementwise_blocks, schedule_elementwise
from .gemmini_reduction import reduction_blocks, schedule_reduction


@derived_object
class GemminiScheduleRule(ms.schedule_rule.PyScheduleRule):
    """Explore traversal, DMA order and PE-safe reduction chunks with TVM rules."""

    def __init__(
        self,
        capabilities=None,
        orders=("ij", "ji"),
        load_orders=("ab",),
        chunks=(16, 8, 4),
        row_tiles=(1, 2, 4),
        col_tiles=(1, 2, 4),
        include_cpu=False,
        sample_groups=False,
    ):
        self.capabilities = capabilities or GemminiCapabilities()
        self.orders = tuple(orders)
        self.load_orders = tuple(load_orders)
        self.chunks = tuple(chunks)
        self.row_tiles = tuple(row_tiles)
        self.col_tiles = tuple(col_tiles)
        if type(include_cpu) is not bool:
            raise ValueError("include_cpu must be a boolean")
        self.include_cpu = include_cpu
        self.sample_groups = sample_groups
        self.target = None

    def _alternatives(self, sch, candidates):
        """Expose CPU execution as an explicit measured placement alternative."""
        return [sch] + candidates if self.include_cpu else candidates

    def __str__(self):
        return "GemminiScheduleRule"

    def _initialize_with_tune_context(self, context):
        self.target = context.target
        if self.target is None or "gemmini" not in self.target.keys:
            raise ValueError(
                "Gemmini schedule rules require an explicitly selected Gemmini target key"
            )
        if self.target.model != "gemmini-" + self.capabilities.fingerprint:
            raise ValueError("MetaSchedule target does not bind the selected Gemmini capabilities")

    def apply(self, sch, block):
        func = sch.mod["main"]
        if sch.get(sch.get_block("root")).annotations.get("gemmini_regions_scheduled", False):
            return [sch]
        matches = contraction_block(func) or _normalizable_contractions(func)
        if not matches:
            reductions = reduction_blocks(func)
            if reductions and (
                len(reductions) != 1 or not sch.get(block).same_as(reductions[0][0])
            ):
                return [sch]
            if len(reductions) == 1 and sch.get(block).same_as(reductions[0][0]):
                candidates, traces = [], set()
                for scalar in (False, True):
                    transformed, _, _ = schedule_reduction(
                        func, tensorize=False, mark_tensorize=True, scalar=scalar
                    )
                    if str(transformed.trace) not in traces:
                        traces.add(str(transformed.trace))
                        candidate = sch.copy()
                        transformed.trace.apply_to_schedule(candidate, remove_postproc=False)
                        candidates.append(candidate)
                return self._alternatives(sch, candidates)
            pointwise = elementwise_blocks(func)
            if pointwise and sch.get(block).same_as(pointwise[0][0]):
                candidates, traces = [], set()
                for scalar in (False, True):
                    try:
                        transformed, _, _ = schedule_elementwise(
                            func, tensorize=False, mark_tensorize=True, scalar=scalar
                        )
                    except (ValueError, tir.schedule.ScheduleError):
                        continue
                    if str(transformed.trace) not in traces:
                        traces.add(str(transformed.trace))
                        candidate = sch.copy()
                        transformed.trace.apply_to_schedule(candidate, remove_postproc=False)
                        candidates.append(candidate)
                return self._alternatives(sch, candidates) if candidates else [sch]
            return [sch]
        if len(matches) != 1 or not sch.get(block).same_as(matches[0][0]):
            return [sch]
        results = []
        for order, load_order, chunk, rows, cols in itertools.product(
            self.orders,
            self.load_orders,
            self.chunks,
            (max(self.row_tiles),) if self.sample_groups else self.row_tiles,
            (max(self.col_tiles),) if self.sample_groups else self.col_tiles,
        ):
            m, n, depth = matches[0][1]
            # Groups beyond the mathematical tile count only add padded
            # work and staging. Keep unit groups for small/tail matrices.
            if not self.sample_groups and (rows > (m + 15) // 16 or cols > (n + 15) // 16):
                continue
            unit_traversal = (
                (m <= 16 or n <= 16) if self.sample_groups else (m <= 16 * rows or n <= 16 * cols)
            )
            if order != self.orders[0] and unit_traversal:
                continue
            # Chunks collapse to the same effective depth for short products.
            effective = max(2, min(chunk, depth))
            if any(
                max(2, min(prior, depth)) == effective
                for prior in self.chunks[: self.chunks.index(chunk)]
            ):
                continue
            candidate = sch.copy()
            try:
                transformed, _ = schedule_contraction(
                    func,
                    order=order,
                    load_order=load_order,
                    reduction_chunk=chunk,
                    tensorize=False,
                    mark_tensorize=True,
                    row_tiles=rows,
                    col_tiles=cols,
                    sample_groups=self.sample_groups,
                )
            except (ValueError, tir.schedule.ScheduleError):
                continue
            transformed.trace.apply_to_schedule(candidate, remove_postproc=False)
            results.append(candidate)
        return self._alternatives(sch, results)

    def clone(self):
        return GemminiScheduleRule(
            self.capabilities,
            self.orders,
            self.load_orders,
            self.chunks,
            self.row_tiles,
            self.col_tiles,
            self.include_cpu,
            self.sample_groups,
        )


@derived_object
class MutateGemminiCategorical(ms.mutator.PyMutator):
    """Mutate real categorical split factors through ordinary trace replay."""

    def _initialize_with_tune_context(self, context):
        del context

    def apply(self, trace, seed):
        rng = random.Random(int(seed))
        samples = [
            (index, inst)
            for index, inst in enumerate(trace.insts)
            if inst.kind.name == "SampleCategorical" and len(inst.attrs[0]) > 1
        ]
        if not samples:
            return None
        index, inst = rng.choice(samples)
        old = int(dict(trace.as_json()[1])[index])
        choices = [i for i in range(len(inst.attrs[0])) if i != old]
        return trace.with_decision(inst, rng.choice(choices), remove_postproc=True)

    def clone(self):
        return MutateGemminiCategorical()

    def __str__(self):
        return "MutateGemminiCategorical"


@derived_object
class VerifyGemminiMemory(ms.postproc.PyPostproc):
    """Reject untensorized CPU SRAM accesses and excessive physical allocations."""

    def __init__(self, capabilities=None):
        self.capabilities = capabilities or GemminiCapabilities()

    def _initialize_with_tune_context(self, context):
        del context

    def apply(self, sch):
        try:
            lower_memory(
                sch.mod, spad_rows=self.capabilities.spad_rows, acc_rows=self.capabilities.acc_rows
            )
        except ValueError:
            return False
        return True

    def clone(self):
        return VerifyGemminiMemory(self.capabilities)

    def __str__(self):
        return "VerifyGemminiMemory"


def gemmini_space_generator(capabilities=None, **rule_options):
    """Return a normal PostOrderApply generator with target-specific rules."""
    capabilities = capabilities or GemminiCapabilities()
    return ms.space_generator.PostOrderApply(
        sch_rules=[GemminiScheduleRule(capabilities, **rule_options)],
        postprocs=[ms.postproc.RewriteTensorize(), VerifyGemminiMemory(capabilities)],
        mutator_probs={MutateGemminiCategorical(): 1.0}
        if rule_options.get("sample_groups")
        else {},
    )
