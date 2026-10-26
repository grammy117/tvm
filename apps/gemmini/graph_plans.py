# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Bounded graph preparation choices recorded using stock Relax tuning traces.

These choices alter fusion and constant packing before inner TensorIR search.
They do not create cross-function SRAM residency. No model name is consulted.
Measure a complete connected graph to choose between plans; operator minima
cannot account for the different host boundaries and CPU glue.
"""

from dataclasses import dataclass
import hashlib

import tvm
from tvm.relax.backend.contrib.gemmini_tensorir import prepare_gemmini_tensorir_workloads
from tvm.relax.transform.tuning_api import Choice, Knob, Trace


def module_digest(mod):
    # save_json traverses object maps in allocation-dependent order. TVMScript
    # sorts module functions and includes constant bytes with show_meta=True.
    return hashlib.sha256(mod.script(show_meta=True).encode()).hexdigest()


@tvm.register_func("relax.ext.gemmini.prepare_graph_plan", override=True)
def _prepare_choice(mod, fusion_level, prepack_weights, bias_preload, eliminate_common_subexpr=False):
    return prepare_gemmini_tensorir_workloads(
        mod, fusion_level=int(fusion_level), prepack_weights=bool(prepack_weights),
        bias_preload=bool(bias_preload), eliminate_common_subexpr=bool(eliminate_common_subexpr),
    )[0]


@dataclass(frozen=True)
class GraphPlan:
    """Compiler preparation settings, independent of workload labels."""

    fusion_level: int = 2
    prepack_weights: bool = True
    bias_preload: bool = True
    eliminate_common_subexpr: bool = False

    def __post_init__(self):
        if type(self.fusion_level) is not int or self.fusion_level not in (-1, 0, 1, 2, 3):
            raise ValueError("Graph fusion level must be -1 or an integer from 0 to 3")
        if type(self.prepack_weights) is not bool or type(self.bias_preload) is not bool:
            raise ValueError("Graph packing and bias settings must be booleans")
        if type(self.eliminate_common_subexpr) is not bool:
            raise ValueError("Graph CSE setting must be a boolean")

    @property
    def effective_fusion_level(self):
        return int(tvm.transform.PassContext.current().opt_level) if self.fusion_level == -1 else self.fusion_level

    @property
    def name(self):
        return (f"fuse{self.effective_fusion_level}-pack{int(self.prepack_weights)}"
                + ("-bias0" if not self.bias_preload else "")
                + ("-cse1" if self.eliminate_common_subexpr else ""))

    @property
    def preparation_options(self):
        # Resolve PassContext now so the saved decision is replayable elsewhere.
        return {"fusion_level": self.effective_fusion_level,
                "prepack_weights": self.prepack_weights, "bias_preload": self.bias_preload,
                "eliminate_common_subexpr": self.eliminate_common_subexpr}

    @classmethod
    def from_args(cls, args):
        return cls(fusion_level=getattr(args, "graph_fusion_level", -1),
                   prepack_weights=getattr(args, "prepack_weights", True),
                   bias_preload=getattr(args, "bias_preload", True),
                   eliminate_common_subexpr=getattr(args, "graph_cse", False))

    @classmethod
    def from_name(cls, name):
        """Parse a canonical compiler configuration, without a workload roster."""
        fields = name.split("-")
        if (len(fields) < 2 or fields[0] not in {f"fuse{i}" for i in range(4)}
                or fields[1] not in ("pack0", "pack1")
                or any(f not in ("bias0", "cse1") for f in fields[2:])):
            raise ValueError("Expected fuse{0,1,2,3}-pack{0,1}[-bias0][-cse1]")
        plan = cls(int(fields[0][4:]), fields[1] == "pack1", "bias0" not in fields,
                   "cse1" in fields)
        if plan.name != name:
            raise ValueError("Graph plan name must be canonical with no duplicate options")
        return plan

    def prepare(self, mod):
        return prepare_gemmini_tensorir_workloads(mod, **self.preparation_options)

    def trace(self, mod):
        choices = {p.name: p.choice() for p in plans(include_cse=True,
                   fusion_levels=(0, 1, 2, 3), bias_preload_choices=(True, False))}
        choices[self.name] = self.choice()
        return Trace(mod, [Knob("GemminiGraphPreparation", choices)], [self.name])

    def choice(self):
        return Choice("relax.ext.gemmini.prepare_graph_plan",
                      [self.effective_fusion_level, int(self.prepack_weights), int(self.bias_preload),
                       int(self.eliminate_common_subexpr)])

    def identity(self, mod, prepared, *, trace=None):
        trace = self.trace(mod) if trace is None else trace
        if not tvm.ir.structural_equal(trace.out_mod, prepared):
            raise ValueError("Graph preparation trace does not replay the compiled plan")
        return {"schema": "gemmini_graph_plan_v1", "name": self.name,
                "preparation_options": self.preparation_options,
                "ir_identity_encoding": "TVMScript(show_meta=True), including constant metadata",
                "semantic_ir_sha256": module_digest(mod),
                "prepared_ir_sha256": module_digest(prepared),
                "trace_format": "TVM Relax tuning_api.Trace; input module stored separately",
                "trace": trace.as_json(include_irmod=False),
                "cross_function_sram_residency": False,
                "selection_scope": "whole connected graph, not summed operator minima"}

    def extract(self, mod, capabilities):
        # Share the campaign's exclusion/weight accounting instead of keeping
        # a second implementation that could diverge from the inner search.
        from tune_models import extract_inventory

        return extract_inventory(mod, capabilities, **self.preparation_options)


def plans(*, include_cse=False, fusion_levels=(2, 0), prepack_choices=(True, False),
          bias_preload_choices=(True,)):
    """Generic Cartesian alternatives; the default experiment budget uses four."""
    result = tuple(GraphPlan(level, packed, bias, eliminate_common_subexpr=cse)
                 for cse in ((False, True) if include_cse else (False,))
                 for level in fusion_levels for packed in prepack_choices
                 for bias in bias_preload_choices)
    if not result or len({p.name for p in result}) != len(result):
        raise ValueError("Graph alternatives must be nonempty and distinct")
    return result


def audit_plans(mod, alternatives=None):
    """Group identical prepared programs before allocating measurement budget.

    Equality is structural, including constant bytes. Hashes only shortlist
    comparisons; they never establish equivalence. Reprepare a representative
    when comparing rather than retaining every large model variant in memory.
    This says nothing about which *different* program will execute fastest.
    """
    alternatives = tuple(alternatives) if alternatives is not None else plans(
        include_cse=True, fusion_levels=(0, 1, 2, 3),
        bias_preload_choices=(True, False))
    if not alternatives or len({p.name for p in alternatives}) != len(alternatives):
        raise ValueError("Audit requires nonempty, distinct graph plans")
    # Prefer the existing compiler default when several choices are equivalent.
    alternatives = tuple(sorted(alternatives, key=lambda p: (p != GraphPlan(), p.name)))
    groups, buckets, rows = [], {}, []
    for plan in alternatives:
        prepared = plan.prepare(mod)[0]
        structural_hash = int(tvm.ir.structural_hash(prepared))
        group_index = None
        for candidate in buckets.get(structural_hash, []):
            representative = GraphPlan.from_name(groups[candidate]["representative"])
            other = representative.prepare(mod)[0]
            equal = tvm.ir.structural_equal(prepared, other)
            del other
            if equal:
                group_index = candidate
                break
        if group_index is None:
            group_index = len(groups)
            groups.append({"representative": plan.name, "plans": []})
            buckets.setdefault(structural_hash, []).append(group_index)
        groups[group_index]["plans"].append(plan.name)
        rows.append({"plan": plan.name, "options": plan.preparation_options,
                     "group": group_index})
        del prepared
    fields = tuple(GraphPlan().preparation_options)
    effects = {}
    for field in fields:
        pairs = [(a, b) for index, a in enumerate(rows) for b in rows[index + 1:]
                 if a["options"][field] != b["options"][field]
                 and all(a["options"][key] == b["options"][key]
                         for key in fields if key != field)]
        changes = sum(a["group"] != b["group"] for a, b in pairs)
        effects[field] = {"compared_pairs": len(pairs), "changed_prepared_graph": changes,
                          "identical_prepared_graph": len(pairs) - changes}
    return {"schema": "gemmini_graph_plan_audit_v1", "configurations": len(rows),
            "distinct_prepared_graphs": len(groups), "groups": groups,
            "option_effects": effects,
            "equivalence": "TVM structural equality, including constants; hash collisions checked",
            "scope": "Preparation only; different graphs require connected timing and output checks",
            "performance_measured": False}


def replay(mod, document):
    """Replay standard serialized choices against the bound semantic program."""
    if document.get("schema") != "gemmini_graph_plan_v1" or module_digest(mod) != document["semantic_ir_sha256"]:
        raise ValueError("Graph plan input identity mismatch")
    record = document["trace"]
    trace = Trace(mod, [Knob.from_json(knob) for knob in record[0]], record[1])
    if module_digest(trace.out_mod) != document["prepared_ir_sha256"]:
        raise ValueError("Graph plan prepared identity mismatch")
    return trace
