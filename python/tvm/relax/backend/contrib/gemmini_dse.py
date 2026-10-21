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
"""Measured cycle search over ordinary Gemmini MetaSchedule candidates.

Scheduling, postprocessing, features and replay use TVM APIs. The external
measurement callback owns target build/run and numerical qualification. Cycles
remain cycles: they are never placed in TuningRecord.run_secs or labeled as
seconds using an invented target frequency. The winning trace is installed
through TVM's ScheduleFnDatabase, with timing provenance retained separately.
"""

from dataclasses import dataclass, field
import hashlib
import json
import math
import statistics
import time

import numpy as np
import tvm
from tvm import meta_schedule as ms, tir

from .gemmini_meta_schedule import gemmini_space_generator
from .gemmini_target import GemminiCapabilities


def knob_inventory():
    """Implemented search choices, separate from numerical/ISA constraints."""
    return {
        "placement": ["cpu", "gemmini"],
        "order": ["ij", "ji"],
        "load_order": ["ab", "ba"],
        "reduction_chunk": [4, 8, 16],
        "row_tiles": [1, 2, 4],
        "col_tiles": [1, 2, 4],
        "movement_strategy": ["rectangular", "scalar"],
        "sum_strategy": ["divisor", "scalar"],
        "graph_optimize": [False, True],
        "bias_preload": [False, True],
        "search": ["sweep", "ml"],
        "scope": {
            "function_search": "placement and legal schedule choices for one fixed mathematical PrimFunc",
            "outer_pipeline_options": ["graph_optimize", "bias_preload"],
            "outer_pipeline_search": "caller must prepare and measure each resulting whole graph",
            "numeric_recipe": "not searched; supplied source graph and correctness gates stay fixed",
        },
        "constraints": {
            "instruction_policy": "primitive_no_fsm",
            "dataflow": "OS",
            "numeric_recipe": "fixed_input_graph",
            "memory": "profile_bound_and_checked",
            "overlap": "not_implemented",
            "cross_call_sram_residency": "not_implemented",
        },
    }


def _native(func):
    calls = []
    tir.stmt_functor.post_order_visit(
        func.body,
        lambda node: calls.append(node)
        if isinstance(node, tir.Call)
        and node.op == tvm.ir.Op.get("tir.call_extern")
        and node.args[0].value.startswith("tvm_gemmini_")
        else None,
    )
    return bool(calls)


@dataclass
class Candidate:
    """A postprocessed legal TVM schedule and its pre-tensorization features."""

    id: str
    schedule: object
    placement: str
    parameters: dict
    features: np.ndarray = field(repr=False)


def generate_candidates(func, *, capabilities=None, rule_options=None):
    """Generate, validate and deduplicate actual framework schedules."""
    capabilities = capabilities or GemminiCapabilities()
    options = dict(rule_options or {})
    options.setdefault("include_cpu", True)
    options.setdefault("chunks", (4, 8, 16))
    context = ms.TuneContext(
        mod=func,
        target=capabilities.target(),
        space_generator=gemmini_space_generator(capabilities, **options),
        num_threads=1,
    )
    designs = context.generate_design_space()
    raw_features = ms.feature_extractor.PerStoreFeature().extract_from(
        context, [ms.search_strategy.MeasureCandidate(sch, []) for sch in designs]
    )
    candidates, seen = [], set()
    for design, feature in zip(designs, raw_features):
        sch = design.copy()
        sch.enter_postproc()
        if not all(post.apply(sch) for post in context.space_generator.postprocs):
            continue
        # Structural equality removes collapsed chunk/tail alternatives. Keep
        # the trace as the public candidate identity and replay contract.
        fingerprint = int(tvm.ir.structural_hash(sch.mod))
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        encoded = json.dumps(sch.trace.as_json(), sort_keys=True, separators=(",", ":"))
        name = hashlib.sha256(encoded.encode()).hexdigest()
        root = sch.mod["main"].body.block
        parameters = {
            str(key)[8:]: int(value) if isinstance(value, tir.IntImm) else str(value)
            for key, value in root.annotations.items()
            if str(key).startswith("gemmini.")
        }
        data = feature.numpy()
        if data.ndim != 2 or not data.shape[0] or not np.isfinite(data).all():
            raise ValueError("MetaSchedule produced invalid semantic features")
        vector = np.concatenate((data.mean(0), data.max(0), data.sum(0))).astype("float32")
        candidates.append(
            Candidate(
                name, sch, "gemmini" if _native(sch.mod["main"]) else "cpu", parameters, vector
            )
        )
    if not candidates:
        raise ValueError("No legal CPU or Gemmini schedules")
    return candidates


def _validate_measurement(record, *, profile, diagnostic):
    if record.get("metric") != "target_cycles":
        raise ValueError("Cycle DSE accepts explicit target_cycles, not host wall time")
    if record.get("engine") not in ("verilator", "gsim", "firesim", "device", "diagnostic"):
        raise ValueError("Functional Spike cannot provide a cycle timing label")
    if (record["engine"] == "diagnostic") != diagnostic:
        raise ValueError("Synthetic measurements require explicit diagnostic scope")
    if record.get("capability_fingerprint") != profile:
        raise ValueError("Measurement belongs to another hardware capability profile")
    if record.get("correctness") != "passed" or record.get("timed_out", False):
        raise ValueError("Incorrect or incomplete execution is not a performance sample")
    values = record.get("samples")
    if not values or any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("Target cycle samples must be positive integers")
    if not diagnostic and not record.get("receipt"):
        raise ValueError("Real measurements require an execution receipt")
    if not isinstance(record.get("measurement_profile"), str) or not record["measurement_profile"]:
        raise ValueError("Measurements must identify their timer, execution model and input scope")
    return float(statistics.median(values))


class CycleRanker:
    """XGBoost over standard TVM semantic features, predicting log cycles."""

    def __init__(self, seed=0):
        self.seed = seed
        self.model = None

    def fit(self, candidates, observations):
        import xgboost as xgb

        features = np.stack([candidates[index].features for index in observations])
        labels = np.log(np.array(list(observations.values()), dtype="float64"))
        self.model = xgb.train(
            {
                "objective": "reg:squarederror",
                "max_depth": 3,
                "eta": 0.15,
                "seed": self.seed,
                "nthread": 1,
                "tree_method": "hist",
            },
            xgb.DMatrix(features, label=labels),
            num_boost_round=32,
        )

    def predict(self, candidates):
        import xgboost as xgb

        if self.model is None:
            raise ValueError("The cycle ranker has no measured training data")
        return self.model.predict(xgb.DMatrix(np.stack([x.features for x in candidates])))


def search(
    func,
    measure,
    *,
    strategy="sweep",
    max_trials=16,
    warmup=4,
    time_budget_seconds=None,
    seed=0,
    capabilities=None,
    rule_options=None,
    diagnostic=False,
):
    """Measure legal candidates; select only a numerically checked observation.

    The callback receives Candidate and returns an independently checked timing
    record. A wall budget bounds experiment cost, not predicted target latency.
    Every measured candidate keeps its full record, including repeated samples.
    """
    if strategy not in ("sweep", "ml"):
        raise ValueError("Unknown search strategy")
    if type(max_trials) is not int or max_trials <= 0 or type(warmup) is not int or warmup < 2:
        raise ValueError("Search requires positive trials and at least two warmup samples")
    if time_budget_seconds is not None and (
        not math.isfinite(time_budget_seconds) or time_budget_seconds <= 0
    ):
        raise ValueError("Time budget must be finite and positive")
    capabilities = capabilities or GemminiCapabilities()
    start = time.monotonic()
    candidates = generate_candidates(func, capabilities=capabilities, rule_options=rule_options)
    rng = np.random.default_rng(seed)
    order = list(map(int, rng.permutation(len(candidates))))
    cpu = [i for i, candidate in enumerate(candidates) if candidate.placement == "cpu"]
    order = cpu + [i for i in order if i not in cpu]
    observations, records = {}, []
    measurement_profile = None
    ranker = CycleRanker(seed)
    while len(records) < min(max_trials, len(candidates)):
        if time_budget_seconds is not None and time.monotonic() - start >= time_budget_seconds:
            if not records:
                raise TimeoutError("Time budget exhausted before the first measurement")
            break
        remaining = [index for index in order if index not in observations]
        predicted = None
        if strategy == "ml" and len(observations) >= warmup:
            ranker.fit(candidates, observations)
            estimates = ranker.predict([candidates[index] for index in remaining])
            offset = int(np.argmin(estimates))
            index, predicted = remaining[offset], float(estimates[offset])
        else:
            index = remaining[0]
        candidate = candidates[index]
        record = measure(candidate)
        cycles = _validate_measurement(
            record, profile=capabilities.fingerprint, diagnostic=diagnostic
        )
        if measurement_profile is None:
            measurement_profile = record["measurement_profile"]
        if record["measurement_profile"] != measurement_profile:
            raise ValueError("Cannot rank samples from different timing models or input scopes")
        observations[index] = cycles
        records.append(
            {
                "candidate_id": candidate.id,
                "placement": candidate.placement,
                "parameters": candidate.parameters,
                "median_cycles": cycles,
                "predicted_log_cycles_before_measurement": predicted,
                "measurement": record,
            }
        )
    winner = min(observations, key=observations.get)
    return SearchResult(
        tvm.IRModule({"main": func}),
        capabilities,
        candidates[winner],
        records,
        {
            "strategy": strategy,
            "candidate_count": len(candidates),
            "trials": len(records),
            "wall_seconds": time.monotonic() - start,
            "seed": seed,
            "rule_options": dict(rule_options or {}),
            "metric": "target_cycles",
            "measurement_profile": measurement_profile,
            "budget_check": "between_trials; callback enforces its own build/run deadlines",
            "ml_model": "xgboost_log_cycles" if strategy == "ml" else None,
            "measurement_scope": "diagnostic" if diagnostic else "selected_target_model",
            "performance_qualified_for_firesim": False,
        },
    )


@dataclass
class SearchResult:
    workload: object
    capabilities: object
    winner: Candidate
    records: list
    summary: dict

    def database(self):
        """Install the checked winner without fabricating second-valued costs."""
        workload, trace = self.workload, self.winner.schedule.trace
        profile = self.capabilities.fingerprint

        def replay(sch):
            if not tvm.ir.structural_equal(sch.mod["main"], workload["main"]):
                return False
            trace.apply_to_schedule(sch, remove_postproc=False)
            sch.annotate(sch.get_block("root"), "gemmini_measurement_profile", profile)
            return True

        return ms.database.ScheduleFnDatabase(replay)

    def as_json(self):
        return {
            "schema": "gemmini_cycle_search_v1",
            "workload_json": tvm.ir.save_json(self.workload),
            "summary": self.summary,
            "capability_fingerprint": self.capabilities.fingerprint,
            "target": str(self.capabilities.target()),
            "winner": {
                "candidate_id": self.winner.id,
                "placement": self.winner.placement,
                "parameters": self.winner.parameters,
                "trace": self.winner.schedule.trace.as_json(),
            },
            "records": self.records,
        }


def load_result(document, func, *, capabilities=None, diagnostic=False):
    """Recreate a measured winner in a fresh process through current legal rules.

    Regenerating candidates also registers the shape-specific TensorIntrins
    needed to replay a saved standard trace. Workload, hardware, metric, scope
    and trace identity must agree; stale or unmeasured winners are rejected.
    """
    capabilities = capabilities or GemminiCapabilities()
    if document.get("schema") != "gemmini_cycle_search_v1":
        raise ValueError("Unrecognized cycle search record schema")
    if document["capability_fingerprint"] != capabilities.fingerprint:
        raise ValueError("Saved search belongs to another hardware profile")
    workload = tvm.ir.load_json(document["workload_json"])
    if not tvm.ir.structural_equal(workload["main"], func):
        raise ValueError("Saved search belongs to another mathematical workload")
    summary, records = document["summary"], document["records"]
    candidates = generate_candidates(
        func, capabilities=capabilities, rule_options=summary["rule_options"]
    )
    winner = next(
        (
            candidate
            for candidate in candidates
            if candidate.id == document["winner"]["candidate_id"]
        ),
        None,
    )
    if winner is None or winner.schedule.trace.as_json() != document["winner"]["trace"]:
        raise ValueError("Saved winner is absent from the current legal schedule space")
    costs = {}
    for row in records:
        cost = _validate_measurement(
            row["measurement"], profile=capabilities.fingerprint, diagnostic=diagnostic
        )
        if row["measurement"]["measurement_profile"] != summary["measurement_profile"]:
            raise ValueError("Saved search combines different measurement scopes")
        costs[row["candidate_id"]] = cost
    if not costs or winner.id not in costs or costs[winner.id] != min(costs.values()):
        raise ValueError("Saved winner must be a measured minimum")
    return SearchResult(workload, capabilities, winner, records, summary)
