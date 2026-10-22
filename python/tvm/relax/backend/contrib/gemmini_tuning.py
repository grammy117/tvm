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

"""Bounded Gemmini search using TVM MetaSchedule features and XGBModel.

Only nine primitive macro-tile schedules are proposed. Features are extracted
from executable semantic TIR before tensorization, not opaque external calls.
Training consumes caller-supplied timings with explicit provenance; this module
does not measure devices or treat Spike walltime/instruction counts as timings.
Predictions propose measurements; they are never evidence of a performance win.
"""

from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json

import numpy as np
import tvm
from tvm import tir
from tvm.meta_schedule import TuneContext
from tvm.meta_schedule.cost_model.xgb_model import XGBConfig, XGBModel
from tvm.meta_schedule.feature_extractor import PerStoreFeature
from tvm.meta_schedule.runner import RunnerResult
from tvm.meta_schedule.search_strategy import MeasureCandidate

from .gemmini_schedule import make_gemmini_matmul


def _digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _identity(workload_id, schedule, compiler_id, adapter_id):
    binding = {"workload_id": workload_id, "compiler_id": compiler_id, "adapter_id": adapter_id,
               "metadata": schedule.metadata, "lowering_steps": schedule.lowering_steps}
    return _digest("\n".join((json.dumps(binding, sort_keys=True), tvm.ir.save_json(schedule.semantic_mod),
                              tvm.ir.save_json(schedule.scheduled_mod), json.dumps(schedule.trace.as_json(), sort_keys=True))))


def _nonempty(value):
    return isinstance(value, str) and bool(value.strip())


@dataclass(frozen=True)
class GemminiCandidate:
    """Identity binds IR, trace, metadata and caller-supplied build identities."""

    candidate_id: str
    workload_id: str
    schedule: object
    compiler_id: str = None
    adapter_id: str = None

    def verify_identity(self):
        """Reject mutated nested IR or metadata before using an old identity."""
        if self.candidate_id != _identity(self.workload_id, self.schedule, self.compiler_id, self.adapter_id):
            raise ValueError("Candidate contents changed after identity construction")

    def measure_candidate(self):
        """Use semantic TIR for features; device TIR is retained for deployment."""
        self.verify_identity()
        return MeasureCandidate(tir.Schedule(self.schedule.semantic_mod), [])


@dataclass(frozen=True)
class TimingRecord:
    """Repeated seconds for an exact candidate on one identified platform.

    cycle_simulator_seconds requires a qualified cycle-accurate measurement
    converted to seconds by its producer. evidence identifies the timer, scope,
    frequency/conversion where relevant, and the measurement artifact.
    protocol_id identifies measurement scope, timer, warmup and frequency.
    compiler_id identifies the compiler build, target and compilation flags;
    adapter_id identifies the primitive C adapter build and configuration.
    These caller-supplied bindings enforce consistency, not authenticity.
    diagnostic_synthetic is exclusively an opted-in machinery test.
    """

    candidate_id: str
    workload_id: str
    platform_id: str
    source: str
    samples_seconds: tuple
    evidence: str
    protocol_id: str = None
    compiler_id: str = None
    adapter_id: str = None

    def __post_init__(self):
        object.__setattr__(self, "samples_seconds", tuple(self.samples_seconds))


@dataclass(frozen=True)
class RankedCandidate:
    """Higher MetaSchedule normalized score predicts a better candidate."""

    candidate: GemminiCandidate
    score: float
    origin: str


class BoundedGemminiSearch:
    """Enumerate legal macro tiles, gate correctness and update a TVM XGB model.

    The semantic gate executes exact CPU mathematics against NumPy. The separate
    device gate records a caller's candidate-specific target correctness result;
    it cannot be inferred from CPU execution. A session cannot mix measurement
    sources/platforms, including synthetic and qualified measurement labels.
    """

    def __init__(self, m, n, k, *, seed=0, allow_diagnostic=False, compiler_id=None, adapter_id=None):
        if type(seed) is not int or not 0 <= seed < (1 << 31) - 1:
            raise ValueError("seed must be an integer in [0, 2^31-2]")
        if type(allow_diagnostic) is not bool:
            raise ValueError("allow_diagnostic must be a boolean")
        if any(value is not None and not _nonempty(value) for value in (compiler_id, adapter_id)):
            raise ValueError("Compiler and adapter identities must be nonempty when supplied")
        self.compiler_id = compiler_id
        self.adapter_id = adapter_id
        self.shape = (m, n, k)
        self.seed = seed
        self.allow_diagnostic = allow_diagnostic
        self.workload_id = _digest(json.dumps({"shape": self.shape, "input": "int8", "output": "int32", "abi": "primitive_no_fsm"}, sort_keys=True))
        candidates = []
        for tile_i in (1, 2, 4):
            for tile_j in (1, 2, 4):
                schedule = make_gemmini_matmul(m, n, k, tile_i, tile_j)
                identity = _identity(self.workload_id, schedule, compiler_id, adapter_id)
                candidates.append(GemminiCandidate(identity, self.workload_id, schedule, compiler_id, adapter_id))
        self.candidates = tuple(candidates)
        self._by_id = {candidate.candidate_id: candidate for candidate in candidates}
        self.context = TuneContext(mod=candidates[0].schedule.semantic_mod, target="llvm", num_threads=1, rand_state=seed + 1)
        self.extractor = PerStoreFeature()
        self._semantic_checks = {}
        self._device_checks = {}
        self._records = ()
        self.model = None
        self.measurement_domain = None

    @property
    def semantic_checks(self):
        """Snapshots cannot inject or alter internal CPU correctness receipts."""
        return deepcopy(self._semantic_checks)

    @property
    def device_checks(self):
        return deepcopy(self._device_checks)

    @property
    def records(self):
        return self._records

    def _candidate(self, candidate_id):
        try:
            candidate = self._by_id[candidate_id]
        except KeyError as error:
            raise ValueError("Unknown candidate identity") from error
        candidate.verify_identity()
        if (candidate.compiler_id, candidate.adapter_id) != (self.compiler_id, self.adapter_id):
            raise ValueError("Search build identity changed after candidate construction")
        return candidate

    def schedule_for(self, candidate_id, *, require_device_check=False, platform_id=None):
        """Return the exact bound GemminiSchedule; this does not lower a graph.

        Proposals can be emitted before target validation. Requiring device
        checks enforces both correctness gates and the exact platform receipt.
        """
        candidate = self._candidate(candidate_id)
        if require_device_check:
            if not _nonempty(platform_id):
                raise ValueError("Applying a checked schedule requires a platform identity")
            self._require_gates(candidate_id, platform_id, diagnostic=False)
        return candidate.schedule

    def features(self):
        """Return TVM's per-store structural features for each semantic schedule."""
        measure_candidates = [self._candidate(candidate.candidate_id).measure_candidate() for candidate in self.candidates]
        return [feature.numpy() for feature in self.extractor.extract_from(self.context, measure_candidates)]

    def validate_semantics(self, candidate_id):
        """Run random, extremal and alternating inputs; raise on any mismatch."""
        candidate = self._candidate(candidate_id)
        m, n, k = self.shape
        rng = np.random.default_rng(self.seed)
        cases = [
            (rng.integers(-128, 128, (m, k), dtype=np.int8), rng.integers(-128, 128, (k, n), dtype=np.int8)),
            (np.full((m, k), -128, dtype=np.int8), np.full((k, n), -128, dtype=np.int8)),
            (np.resize(np.array([-128, 127], dtype=np.int8), (m, k)), np.resize(np.array([127, -128], dtype=np.int8), (k, n))),
        ]
        self._semantic_checks[candidate_id] = {"passed": False, "seed": self.seed, "cases": len(cases)}
        function = tvm.build(candidate.schedule.semantic_mod, target="llvm")
        for a, b in cases:
            c = tvm.nd.empty((m, n), dtype="int32")
            function(tvm.nd.array(a), tvm.nd.array(b), c)
            np.testing.assert_array_equal(c.numpy(), a.astype(np.int64) @ b.astype(np.int64))
        check = {"passed": True, "seed": self.seed, "cases": len(cases), "reference": "numpy_int64_matmul"}
        self._semantic_checks[candidate_id] = check
        return dict(check)

    def record_device_check(self, candidate_id, *, passed, evidence, platform_id):
        """Record an external target check, with its artifact or command identity."""
        self._candidate(candidate_id)
        if type(passed) is not bool or not isinstance(evidence, str) or not evidence.strip():
            raise ValueError("Device correctness requires a boolean result and evidence")
        if not _nonempty(platform_id):
            raise ValueError("Device correctness requires a platform identity")
        self._device_checks[candidate_id] = {"passed": passed, "evidence": evidence, "platform_id": platform_id,
                                             "compiler_id": self.compiler_id, "adapter_id": self.adapter_id}

    def _require_gates(self, candidate_id, platform_id, diagnostic):
        if not self._semantic_checks.get(candidate_id, {}).get("passed"):
            raise ValueError("Candidate has not passed its semantic correctness gate")
        if not diagnostic:
            check = self._device_checks.get(candidate_id, {})
            if not check.get("passed"):
                raise ValueError("Candidate has not passed its device correctness gate")
            if (check["platform_id"], check["compiler_id"], check["adapter_id"]) != (platform_id, self.compiler_id, self.adapter_id):
                raise ValueError("Device correctness platform/build identity does not match timing")

    def update(self, records):
        """Validate a complete batch before training on its measured repeats.

        Timings require at least two finite, positive seconds. MetaSchedule uses
        their median, normalizes within this workload and retrains cumulatively.
        A fresh model fits all accepted records; training or prediction failure
        leaves the prior model, ledger and measurement domain unchanged.
        Install the optional xgboost dependency to use learned updates.
        """
        records = list(records)
        if not records:
            return
        domain = self.measurement_domain
        accepted = []
        for record in records:
            self._candidate(record.candidate_id)
            if record.workload_id != self.workload_id:
                raise ValueError("Timing workload identity does not match this search")
            if not isinstance(record.platform_id, str) or not record.platform_id.strip():
                raise ValueError("Timings require a platform identity")
            if not isinstance(record.evidence, str) or not record.evidence.strip():
                raise ValueError("Timings require measurement evidence")
            if record.source not in ("hardware_seconds", "cycle_simulator_seconds", "diagnostic_synthetic"):
                raise ValueError("Unsupported timing source; Spike proxies are not qualified timings")
            diagnostic = record.source == "diagnostic_synthetic"
            if diagnostic and not self.allow_diagnostic:
                raise ValueError("Synthetic labels require explicit diagnostic opt-in")
            if not diagnostic and "spike" in (record.platform_id + record.evidence).lower():
                raise ValueError("Spike proxies are not qualified target timings")
            if not diagnostic and not all(_nonempty(value) for value in (self.compiler_id, self.adapter_id, record.protocol_id)):
                raise ValueError("Real timings require compiler, adapter and measurement protocol identities")
            if (record.compiler_id, record.adapter_id) != (self.compiler_id, self.adapter_id):
                raise ValueError("Timing compiler/adapter identities do not match this search")
            self._require_gates(record.candidate_id, record.platform_id, diagnostic)
            samples = np.asarray(record.samples_seconds, dtype=np.float64)
            if samples.ndim != 1 or len(samples) < 2 or not np.all(np.isfinite(samples) & (samples > 0)):
                raise ValueError("Timings require at least two finite positive repeats")
            if np.max(samples) > np.finfo(np.float32).max or np.min(samples) < np.finfo(np.float32).tiny:
                raise ValueError("Timings exceed the cost model's float32 range")
            current_domain = (record.platform_id, record.source, record.protocol_id, record.compiler_id, record.adapter_id)
            if domain is not None and domain != current_domain:
                raise ValueError("A search cannot mix timing platform, source, protocol or build identities")
            domain = current_domain
            accepted.append(replace(record, samples_seconds=tuple(float(value) for value in samples)))
        all_records = self._records + tuple(accepted)
        medians = np.array([np.median(record.samples_seconds) for record in all_records])
        if np.min(medians) / np.max(medians) < np.finfo(np.float32).tiny:
            raise ValueError("Timing ratios exceed the cost model's float32 range")
        candidates = [self._candidate(record.candidate_id).measure_candidate() for record in all_records]
        results = [RunnerResult(run_secs=list(record.samples_seconds), error_msg=None) for record in all_records]
        model = self._new_model()
        model.update(self.context, candidates, results)
        scores = np.asarray(model.predict(self.context, [self._candidate(candidate.candidate_id).measure_candidate() for candidate in self.candidates]))
        self._check_scores(scores)
        self.model, self._records, self.measurement_domain = model, all_records, domain

    def _new_model(self):
        try:
            import xgboost  # pylint: disable=import-outside-toplevel,unused-import
        except ImportError as error:
            raise RuntimeError("Gemmini learned search requires the optional xgboost package") from error
        return XGBModel(extractor=self.extractor, config=XGBConfig(max_depth=3, seed=self.seed, nthread=1, tree_method="hist"), num_warmup_samples=1, early_stopping_rounds=10, verbose_eval=0, adaptive_training=False)

    def _check_scores(self, scores):
        if scores.shape != (len(self.candidates),) or not np.all(np.isfinite(scores)):
            raise ValueError("Cost model returned invalid or nonfinite predictions")

    def rank(self):
        """Return reproducible proposals, labeled by training origin.

        Before training, candidate order is a seeded permutation, with equal
        scores. Model predictions are proposals, including unmeasured candidates.
        """
        if self.model is None:
            for candidate in self.candidates:
                self._candidate(candidate.candidate_id)
            order = np.random.default_rng(self.seed).permutation(len(self.candidates))
            return [RankedCandidate(self.candidates[i], 0.0, "untrained") for i in order]
        scores = np.asarray(self.model.predict(self.context, [self._candidate(candidate.candidate_id).measure_candidate() for candidate in self.candidates]))
        self._check_scores(scores)
        origin = "diagnostic" if self.measurement_domain[1] == "diagnostic_synthetic" else "measured_model"
        ranked = [RankedCandidate(candidate, float(score), origin) for candidate, score in zip(self.candidates, scores)]
        return sorted(ranked, key=lambda item: (-item.score, item.candidate.candidate_id))

    def manifest(self):
        """Identity/provenance snapshot; caller evidence is not authenticated."""
        for candidate in self.candidates:
            self._candidate(candidate.candidate_id)
        return {
            "schema": "gemmini_bounded_search_v1", "shape": list(self.shape), "seed": self.seed,
            "workload_id": self.workload_id, "compiler_id": self.compiler_id, "adapter_id": self.adapter_id, "feature_extractor": "TVM MetaSchedule PerStoreFeature on semantic TIR",
            "measurement_domain": self.measurement_domain,
            "candidates": [{"candidate_id": candidate.candidate_id, "metadata": dict(candidate.schedule.metadata),
                            "semantic_check": self.semantic_checks.get(candidate.candidate_id),
                            "device_check": self.device_checks.get(candidate.candidate_id)} for candidate in self.candidates],
            "timings": [dict(record.__dict__, samples_seconds=list(record.samples_seconds)) for record in self.records],
        }


def collect_gemmini_workloads(mod):
    """Inventory semantic contractions; eligibility is not execution coverage."""
    from tvm import relax
    from .gemmini import DecomposeGemminiLinear, _eligible
    from .gemmini_conv import DecomposeGemminiConv2D

    semantic = DecomposeGemminiLinear()(DecomposeGemminiConv2D()(mod))
    admitted, host = {}, []
    for gv, function in semantic.functions_items():
        if not isinstance(function, relax.Function):
            continue
        if function.attrs and ("Codegen" in function.attrs or "Composite" in function.attrs):
            host.append({"function": gv.name_hint, "reason": "externally owned function"})
            continue
        calls = []
        relax.analysis.post_order_visit(function.body, lambda node: calls.append(node) if isinstance(node, relax.Call) and isinstance(node.op, tvm.ir.Op) and node.op.name in ("relax.matmul", "relax.nn.conv2d") else None)
        for index, call in enumerate(calls):
            site = gv.name_hint + ":" + call.op.name.removeprefix("relax.") + ":" + str(index)
            shape = _eligible(call)
            if shape is None or max(shape[2], shape[1] * 4) > (1 << 32) - 1:
                host.append({"site": site, "reason": "unsupported static INT8/INT32 primitive contract"})
            else:
                admitted.setdefault(shape, []).append(site)
    return {"workloads": [{"shape": list(shape), "sites": sites} for shape, sites in sorted(admitted.items())], "host_contractions": host, "execution_qualified": False}


class GemminiModelSearch:
    """Budget candidate measurements across model contraction shapes.

    Requests bind build, adapter, workload and candidate identities. Callers execute
    them, establish correctness and supply TimingRecords; this class never measures
    a target or treats predictions as winners. Each shape retains its own cost model.
    """

    def __init__(self, inventory, *, budget, compiler_id, adapter_id, seed=0, allow_diagnostic=False):
        if type(budget) is not int or budget < 1:
            raise ValueError("Measurement budget must be a positive integer")
        if not all(_nonempty(value) for value in (compiler_id, adapter_id)):
            raise ValueError("Model search requires compiler and adapter identities")
        shapes = tuple(tuple(item["shape"]) for item in inventory["workloads"])
        if not shapes or len(set(shapes)) != len(shapes):
            raise ValueError("Model search requires distinct admitted workloads")
        self.inventory = deepcopy(inventory)
        self.budget, self.compiler_id, self.adapter_id = budget, compiler_id, adapter_id
        self.searches = {shape: BoundedGemminiSearch(*shape, seed=seed, allow_diagnostic=allow_diagnostic, compiler_id=compiler_id, adapter_id=adapter_id) for shape in shapes}
        self._by_workload = {search.workload_id: search for search in self.searches.values()}
        self._issued, self._requests, self._cursor = set(), [], 0
        self.measurement_domain = None

    def next_measurements(self, count=1):
        """Consume a finite budget in reproducible rounds without duplicate candidates."""
        if type(count) is not int or count < 1:
            raise ValueError("Requested measurement count must be positive")
        requests, searches = [], tuple(self.searches.values())
        issued, cursor = set(self._issued), self._cursor
        empty_round = 0
        while len(requests) < count and len(self._requests) + len(requests) < self.budget and empty_round < len(searches):
            search = searches[cursor % len(searches)]
            cursor += 1
            proposal = next((item for item in search.rank() if item.candidate.candidate_id not in issued), None)
            if proposal is None:
                empty_round += 1
                continue
            empty_round = 0
            candidate = proposal.candidate
            request = {"request_index": len(self._requests) + len(requests), "workload_id": search.workload_id, "candidate_id": candidate.candidate_id, "shape": list(search.shape), "tile_i": candidate.schedule.metadata["tile_i"], "tile_j": candidate.schedule.metadata["tile_j"], "origin": proposal.origin, "compiler_id": self.compiler_id, "adapter_id": self.adapter_id, "minimum_timing_repeats": 2, "performance_winner": False}
            issued.add(candidate.candidate_id)
            requests.append(request)
        self._issued, self._cursor = issued, cursor
        self._requests.extend(deepcopy(requests))
        return requests

    def search_for(self, workload_id):
        try:
            return self._by_workload[workload_id]
        except KeyError as error:
            raise ValueError("Unknown model workload identity") from error

    def ingest(self, record):
        """Update one requested candidate after existing correctness/provenance gates."""
        if not isinstance(record, TimingRecord) or record.candidate_id not in self._issued:
            raise ValueError("Timings must refer to a requested model candidate")
        search = self.search_for(record.workload_id)
        domain = (record.platform_id, record.source, record.protocol_id, record.compiler_id, record.adapter_id)
        if self.measurement_domain is not None and domain != self.measurement_domain:
            raise ValueError("Model search cannot mix measurement domains")
        if any(value.candidate_id == record.candidate_id for value in search.records):
            raise ValueError("Candidate already has a repeated-sample measurement")
        search.update([record])
        self.measurement_domain = domain

    def manifest(self):
        return {"schema": "gemmini_model_search_v1", "inventory": deepcopy(self.inventory), "budget": self.budget, "issued": len(self._requests), "remaining_budget": max(0, self.budget - len(self._requests)), "requests": deepcopy(self._requests), "measurement_domain": self.measurement_domain, "workloads": [search.manifest() for search in self.searches.values()], "timing_qualified": False, "performance_winner": False, "required_ablations": ["untuned", "graph_baseline", "graph_optimized"], "session_timing_scope": "complete declared session, including required host, packing and conversion work"}
