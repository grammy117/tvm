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
"""Standard MetaSchedule evolutionary search with externally measured cycles.

The v0.19 RunnerResult numeric field is named run_secs. Internally this adapter
passes cycle-valued costs to the framework's relative-cost XGB model. Those
numbers are never published as seconds or transferred to the replay database;
the durable result retains explicit target_cycles and measurement provenance.
No target frequency is assumed. Build and run callbacks own hard deadlines.
"""

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import tvm
from tvm import meta_schedule as ms, tir
from tvm.meta_schedule.utils import derived_object

from .gemmini_dse import (
    Candidate,
    SearchResult,
    TrialFailure,
    _native,
    _validate_measurement,
    generate_candidates,
)
from .gemmini_meta_schedule import gemmini_space_generator
from .gemmini_memory import lower_memory
from .gemmini_target import GemminiCapabilities


def candidate_from_schedule(sch):
    encoded = json.dumps(sch.trace.as_json(), sort_keys=True, separators=(",", ":"))
    attrs = sch.mod["main"].body.block.annotations
    parameters = {
        str(key)[8:]: int(value) if isinstance(value, tir.IntImm) else str(value)
        for key, value in attrs.items()
        if str(key).startswith("gemmini.")
    }
    return Candidate(
        hashlib.sha256(encoded.encode()).hexdigest(),
        sch,
        "gemmini" if _native(sch.mod["main"]) else "cpu",
        parameters,
        np.empty(0),
    )


def restore_candidate(func, trace_json, *, capabilities, rule_options):
    """Check a saved sampled trace against current framework sketch families."""
    options = dict(rule_options)
    # Register all currently legal intrinsic implementations before replay.
    context = ms.TuneContext(
        mod=func,
        target=capabilities.target(),
        space_generator=gemmini_space_generator(capabilities, **options),
        num_threads=1,
    )
    sketches = context.generate_design_space()
    sch = tir.Schedule(func)
    tir.schedule.Trace.apply_json_to_schedule(trace_json, sch)
    skeleton = sch.trace.simplified(remove_postproc=True).as_json()[0]
    legal = any(
        sketch.trace.simplified(remove_postproc=True).as_json()[0] == skeleton
        for sketch in sketches
    )
    if not legal:
        baseline_options = dict(options, sample_groups=False)
        baselines = generate_candidates(
            func, capabilities=capabilities, rule_options=baseline_options
        )
        legal = any(candidate.schedule.trace.as_json() == trace_json for candidate in baselines)
    if not legal:
        raise ValueError("Saved trace is not an instance of a current legal sketch")
    lower_memory(sch.mod, spad_rows=capabilities.spad_rows, acc_rows=capabilities.acc_rows)
    return candidate_from_schedule(sch)


@derived_object
class MathematicalFeatures(ms.feature_extractor.PyFeatureExtractor):
    """Use stock PerStoreFeature on the trace before tensorization postprocessing."""

    def extract_from(self, context, candidates):
        mathematical = []
        for candidate in candidates:
            sch = tir.Schedule(context.mod)
            candidate.sch.trace.apply_to_schedule(sch, remove_postproc=True)
            mathematical.append(ms.search_strategy.MeasureCandidate(sch, candidate.args_info))
        return ms.feature_extractor.PerStoreFeature().extract_from(context, mathematical)


@derived_object
class ObservedXGBModel(ms.cost_model.PyCostModel):
    """Instrument the framework model without changing its training or ranking."""

    def __init__(self, warmup):
        self.model = ms.cost_model.XGBModel(
            extractor=MathematicalFeatures(),
            num_warmup_samples=warmup,
            early_stopping_rounds=8,
            verbose_eval=0,
            adaptive_training=False,
            num_tuning_cores=1,
        )
        self.events = []

    def predict(self, context, candidates):
        predictions = self.model.predict(context, candidates)
        self.events.append(
            {
                "candidates": len(candidates),
                "distinct_predictions": len(np.unique(predictions)),
                "prediction_span": float(np.ptp(predictions)) if len(predictions) else 0,
                "measured_training_samples": self.model.data_size,
                "booster_trained": self.model.booster is not None,
            }
        )
        return predictions

    def update(self, context, candidates, results):
        self.model.update(context, candidates, results)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model.load(path)


@derived_object
class CompletedFuture(ms.runner.PyRunnerFuture):
    def __init__(self, result):
        self.value = result

    def done(self):
        return True

    def result(self):
        return self.value


@derived_object
class NativeBuilder(ms.builder.PyBuilder):
    def __init__(self, build, state):
        self.build_callback, self.state = build, state

    def build(self, build_inputs):
        results = []
        for item in build_inputs:
            try:
                artifact = self.build_callback(candidate_from_schedule(tir.Schedule(item.mod)))
                key = artifact["folder"]
                self.state["artifacts"][key] = artifact
                results.append(ms.builder.BuilderResult(key, None))
            except TrialFailure as error:
                results.append(ms.builder.BuilderResult(None, str(error)))
        return results


@derived_object
class NativeRunner(ms.runner.PyRunner):
    def __init__(self, run, state, capabilities, diagnostic):
        self.run_callback, self.state = run, state
        self.capabilities, self.diagnostic = capabilities, diagnostic

    def run(self, runner_inputs):
        results = []
        for item in runner_inputs:
            key = item.artifact_path
            try:
                measurement = self.run_callback(self.state["artifacts"][key])
                _validate_measurement(
                    measurement, profile=self.capabilities.fingerprint, diagnostic=self.diagnostic
                )
                if (
                    self.state["profile"] is not None
                    and measurement["measurement_profile"] != self.state["profile"]
                ):
                    raise ValueError("Cannot combine different timing/input profiles")
                self.state["profile"] = measurement["measurement_profile"]
                self.state["measurements"][key] = measurement
                result = ms.runner.RunnerResult([float(x) for x in measurement["samples"]], None)
            except TrialFailure as error:
                self.state["failures"][key] = {"error": str(error), "evidence": error.evidence}
                result = ms.runner.RunnerResult(None, str(error))
            results.append(CompletedFuture(result))
        return results


@derived_object
class RecordCycles(ms.measure_callback.PyMeasureCallback):
    def __init__(self, state, records, candidates, on_record):
        self.state, self.records, self.candidates, self.on_record = (
            state,
            records,
            candidates,
            on_record,
        )

    def apply(self, task_scheduler, task_id, measure_candidates, builder_results, runner_results):
        del task_scheduler, task_id
        for measured, built, ran in zip(measure_candidates, builder_results, runner_results):
            candidate = candidate_from_schedule(measured.sch)
            self.candidates[candidate.id] = candidate
            row = {
                "candidate_id": candidate.id,
                "placement": candidate.placement,
                "parameters": candidate.parameters,
                "trace": candidate.schedule.trace.as_json(),
                "selection": "TVM EvolutionarySearch/XGBModel",
            }
            key = built.artifact_path
            if built.error_msg is not None or ran.error_msg is not None:
                row.update(
                    status="failed",
                    error=str(built.error_msg or ran.error_msg),
                    evidence=self.state["failures"].get(key),
                )
            else:
                measurement = self.state["measurements"][key]
                row.update(
                    status="passed",
                    measurement=measurement,
                    median_cycles=float(np.median(measurement["samples"])),
                )
            self.records.append(row)
            if self.on_record is not None:
                self.on_record(row)


def tune(
    func,
    build,
    run,
    *,
    work_dir,
    max_trials=32,
    seed=0,
    capabilities=None,
    rule_options=None,
    warmup=8,
    on_record=None,
    diagnostic=False,
    resume_records=(),
):
    """Measure CPU/default baselines, then use TVM's complete tuning controller."""
    capabilities = capabilities or GemminiCapabilities()
    from .gemmini_dse import with_required_baselines

    options = with_required_baselines(rule_options)
    records, candidates = [], {}
    profile, started = None, time.monotonic()
    for row in resume_records:
        candidate = restore_candidate(
            func,
            row["trace"],
            capabilities=capabilities,
            rule_options=dict(options, sample_groups=True),
        )
        if candidate.id != row["candidate_id"]:
            raise ValueError("Resumed trace identity differs")
        candidates[candidate.id] = candidate
        if row["status"] == "passed":
            _validate_measurement(
                row["measurement"], profile=capabilities.fingerprint, diagnostic=diagnostic
            )
            observed_profile = row["measurement"]["measurement_profile"]
            if profile is not None and profile != observed_profile:
                raise ValueError("Resumed timing profiles disagree")
            profile = observed_profile
        records.append(row)
    # Baselines are mandatory and consume the same declared trial budget.
    legal = generate_candidates(func, capabilities=capabilities, rule_options=options)
    default = [
        c
        for c in legal
        if c.placement == "gemmini"
        and c.parameters.get("order") == "ij"
        and c.parameters.get("row_tiles") == c.parameters.get("col_tiles") == 1
    ]
    baselines = [c for c in legal if c.placement == "cpu"]
    if default:
        baselines.append(max(default, key=lambda c: c.parameters["reduction_chunk"]))
    for candidate in baselines[:max_trials]:
        if any(row["candidate_id"] == candidate.id for row in records):
            continue
        if len(records) >= max_trials:
            break
        candidates[candidate.id] = candidate
        row = {
            "candidate_id": candidate.id,
            "placement": candidate.placement,
            "parameters": candidate.parameters,
            "trace": candidate.schedule.trace.as_json(),
            "selection": "required_baseline",
        }
        try:
            measurement = run(build(candidate))
            cost = _validate_measurement(
                measurement, profile=capabilities.fingerprint, diagnostic=diagnostic
            )
            if profile is not None and profile != measurement["measurement_profile"]:
                raise ValueError("Baseline timing profiles disagree")
            profile = measurement["measurement_profile"]
            row.update(status="passed", measurement=measurement, median_cycles=cost)
        except TrialFailure as error:
            row.update(status="failed", error=str(error), evidence=error.evidence)
        records.append(row)
        if on_record is not None:
            on_record(row)
    state = {"artifacts": {}, "measurements": {}, "failures": {}, "profile": profile}
    remaining = max(0, max_trials - len(records))
    model = None
    if remaining:
        sampled = dict(options, sample_groups=True)
        context = ms.TuneContext(
            mod=func,
            target=capabilities.target(),
            space_generator=gemmini_space_generator(capabilities, **sampled),
            search_strategy=ms.search_strategy.EvolutionarySearch(
                population_size=32, init_min_unmeasured=8, genetic_num_iters=2, max_fail_count=3
            ),
            num_threads=1,
            rand_state=seed,
        )
        model = ObservedXGBModel(warmup)
        callback = RecordCycles(state, records, candidates, on_record)
        database = ms.database.MemoryDatabase()
        workload = database.commit_workload(tvm.IRModule({"main": func}))
        observed_candidates, observed_results = [], []
        for row in records:
            if row["status"] != "passed":
                continue
            candidate = candidates[row["candidate_id"]]
            costs = [float(x) for x in row["measurement"]["samples"]]
            database.commit_tuning_record(
                ms.database.TuningRecord(
                    candidate.schedule.trace, workload, costs, capabilities.target(), []
                )
            )
            observed_candidates.append(ms.search_strategy.MeasureCandidate(candidate.schedule, []))
            observed_results.append(ms.runner.RunnerResult(costs, None))
        if observed_candidates:
            model.update(context, observed_candidates, observed_results)
        ms.tune_tasks(
            tasks=[context],
            task_weights=[1.0],
            work_dir=str(Path(work_dir)),
            max_trials_global=remaining,
            num_trials_per_iter=1,
            builder=NativeBuilder(build, state),
            runner=NativeRunner(run, state, capabilities, diagnostic),
            database=database,
            cost_model=model,
            measure_callbacks=[
                ms.measure_callback.AddToDatabase(),
                ms.measure_callback.UpdateCostModel(),
                callback,
            ],
            task_scheduler="round-robin",
        )
    passed = [row for row in records if row["status"] == "passed"]
    if not passed:
        raise TrialFailure("No qualified baseline or evolutionary trial")
    best = min(passed, key=lambda row: row["median_cycles"])
    return SearchResult(
        tvm.IRModule({"main": func}),
        capabilities,
        candidates[best["candidate_id"]],
        records,
        {
            "strategy": "metaschedule",
            "rule_options": dict(options, sample_groups=True),
            "trials": len(records),
            "successful_trials": len(passed),
            "failed_trials": len(records) - len(passed),
            "metric": "target_cycles",
            "measurement_profile": state["profile"],
            "seed": seed,
            "wall_seconds": time.monotonic() - started,
            "ml_model": "TVM XGBModel relative-cost",
            "controller": "TVM tune_tasks/EvolutionarySearch",
            "internal_runner_cost_unit": "cycles, never seconds",
            "baseline_candidate_ids": [c.id for c in baselines],
            "baseline_complete": all(
                any(row["candidate_id"] == c.id and row["status"] == "passed" for row in records)
                for c in baselines
            ),
            "ranking_events": model.events if model is not None else [],
            "xgb_training_samples": model.model.data_size if model is not None else 0,
            "measurement_scope": "diagnostic" if diagnostic else "selected_target_model",
            "performance_qualified_for_firesim": False,
        },
    )
