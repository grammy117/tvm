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

from tvm import meta_schedule as ms
from tvm.meta_schedule.utils import derived_object

from .gemmini_memory import lower_memory
from .gemmini_target import GemminiCapabilities
from .gemmini_tensorir import _normalizable_contractions, contraction_block, schedule_contraction
from .gemmini_elementwise import elementwise_blocks, schedule_elementwise


@derived_object
class GemminiScheduleRule(ms.schedule_rule.PyScheduleRule):
    """Explore traversal, DMA order and PE-safe reduction chunks with TVM rules."""

    def __init__(
        self,
        capabilities=None,
        orders=("ij", "ji"),
        load_orders=("ab", "ba"),
        chunks=(8, 16),
        row_tiles=(1, 2, 4),
        col_tiles=(1, 2, 4),
    ):
        self.capabilities = capabilities or GemminiCapabilities()
        self.orders = tuple(orders)
        self.load_orders = tuple(load_orders)
        self.chunks = tuple(chunks)
        self.row_tiles = tuple(row_tiles)
        self.col_tiles = tuple(col_tiles)
        self.target = None

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
        matches = contraction_block(func) or _normalizable_contractions(func)
        if not matches:
            pointwise = elementwise_blocks(func)
            if len(pointwise) == 1 and sch.get(block).same_as(pointwise[0][0]):
                candidates, traces = [], set()
                for scalar in (False, True):
                    candidate = sch.copy()
                    transformed, _, _ = schedule_elementwise(
                        func, tensorize=False, mark_tensorize=True, scalar=scalar
                    )
                    if str(transformed.trace) not in traces:
                        traces.add(str(transformed.trace))
                        transformed.trace.apply_to_schedule(candidate, remove_postproc=False)
                        candidates.append(candidate)
                return candidates
            return [sch]
        if len(matches) != 1 or not sch.get(block).same_as(matches[0][0]):
            return [sch]
        results = []
        for order, load_order, chunk, rows, cols in itertools.product(
            self.orders, self.load_orders, self.chunks, self.row_tiles, self.col_tiles
        ):
            candidate = sch.copy()
            transformed, _ = schedule_contraction(
                func,
                order=order,
                load_order=load_order,
                reduction_chunk=chunk,
                tensorize=False,
                mark_tensorize=True,
                row_tiles=rows,
                col_tiles=cols,
            )
            transformed.trace.apply_to_schedule(candidate, remove_postproc=False)
            results.append(candidate)
        return results

    def clone(self):
        return GemminiScheduleRule(
            self.capabilities,
            self.orders,
            self.load_orders,
            self.chunks,
            self.row_tiles,
            self.col_tiles,
        )


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


def gemmini_space_generator(capabilities=None):
    """Return a normal PostOrderApply generator with target-specific rules."""
    capabilities = capabilities or GemminiCapabilities()
    return ms.space_generator.PostOrderApply(
        sch_rules=[GemminiScheduleRule(capabilities)],
        postprocs=[ms.postproc.RewriteTensorize(), VerifyGemminiMemory(capabilities)],
        mutator_probs={},
    )
