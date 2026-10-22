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
"""Stock MetaSchedule tuning of Gemmini tasks with an external cycle runner.

Tasks come from ``ms.relax_integration.extract_tasks`` (call-count weights),
search is ``tune_tasks`` with ``EvolutionarySearch`` and ``XGBModel``, and the
records go to a ``JSONDatabase`` that ``relax.transform.MetaScheduleApplyDatabase``
replays at compile time. This module only supplies what MetaSchedule leaves to
the target: a builder and runner for the bare-metal simulators, a workload key
that lets them find each task's fixed inputs, and a feature view of candidates
before tensorization.

The runner reports target cycles in ``RunnerResult.run_secs``. MetaSchedule
uses that value only for ranking; it is never interpreted as seconds.
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import tvm
from tvm import meta_schedule as ms, tir
from tvm.meta_schedule.utils import derived_object

from .gemmini_meta_schedule import gemmini_space_generator
from .gemmini_target import GemminiCapabilities

WORKLOAD_ATTR = "gemmini.workload"


def workload_key(func):
    """A deterministic identity of a mathematical function's own structure."""
    if func.attrs and WORKLOAD_ATTR in func.attrs:
        func = func.without_attr(WORKLOAD_ATTR)
    return "%016x" % (tvm.ir.structural_hash(func) & 0xFFFFFFFFFFFFFFFF)


@tvm.transform.module_pass(opt_level=0, name="AnnotateGemminiWorkloads")
class AnnotateGemminiWorkloads:
    """Attach a structural workload key to every PrimFunc.

    The key is computed from the function itself, so extraction and the later
    ``MetaScheduleApplyDatabase`` see identical workloads, and a scheduled
    candidate still names the task whose inputs and oracle measure it.
    """

    def transform_module(self, mod, _ctx):
        for gv, func in list(mod.functions.items()):
            if isinstance(func, tir.PrimFunc):
                mod.update_func(gv, func.with_attr(WORKLOAD_ATTR, workload_key(func)))
        return mod


def candidate_workload(mod):
    func = mod["main"]
    if not func.attrs or WORKLOAD_ATTR not in func.attrs:
        raise ValueError("Candidate does not carry a Gemmini workload key")
    return str(func.attrs[WORKLOAD_ATTR])


def open_gemmini_database(path, prepared, *, capabilities=None):
    """Register workload-dependent intrinsic families, then open stock records.

    TVM validates Tensorize instructions while deserializing a JSONDatabase.
    Generate the same target sketches first, including bias/readout families,
    so a fresh compiler process has the registrations made during tuning.
    The database itself owns matching and replay.
    """
    capabilities = capabilities or GemminiCapabilities()
    for func in prepared.functions.values():
        if not isinstance(func, tir.PrimFunc):
            continue
        ms.TuneContext(
            mod=tvm.IRModule({"main": func.with_attr("global_symbol", "main")}),
            target=capabilities.target(), num_threads=1,
            space_generator=gemmini_space_generator(capabilities, include_cpu=False),
        ).generate_design_space()
    return ms.database.JSONDatabase(work_dir=str(path))


@derived_object
class MathematicalFeatures(ms.feature_extractor.PyFeatureExtractor):
    """Stock PerStoreFeature on each candidate's trace without postprocessing.

    After RewriteTensorize the Gemmini work is opaque extern calls with no
    buffer stores; before it, tiles, caches and copies remain visible loops.
    """

    def extract_from(self, context, candidates):
        mathematical = []
        for candidate in candidates:
            sch = tir.Schedule(context.mod)
            candidate.sch.trace.apply_to_schedule(sch, remove_postproc=True)
            mathematical.append(ms.search_strategy.MeasureCandidate(sch, candidate.args_info))
        return ms.feature_extractor.PerStoreFeature().extract_from(context, mathematical)


@derived_object
class GemminiBuilder(ms.builder.PyBuilder):
    """Build candidates concurrently through a caller-supplied build function.

    ``build(mod, workload_key) -> artifact path`` raises on refusal; the error
    text becomes the BuilderResult error.
    """

    def __init__(self, build, max_workers=4):
        self.build_callback = build
        self.max_workers = max_workers

    def build(self, build_inputs):
        def one(item):
            try:
                return ms.builder.BuilderResult(
                    str(self.build_callback(item.mod, candidate_workload(item.mod))), None
                )
            except Exception as error:  # pylint: disable=broad-except
                return ms.builder.BuilderResult(None, f"{type(error).__name__}: {error}")

        with ThreadPoolExecutor(self.max_workers) as pool:
            return list(pool.map(one, build_inputs))


@derived_object
class GemminiRunnerFuture(ms.runner.PyRunnerFuture):
    def __init__(self, future):
        self.future = future

    def done(self):
        return self.future.done()

    def result(self):
        try:
            cycles = self.future.result()
        except Exception as error:  # pylint: disable=broad-except
            return ms.runner.RunnerResult(None, f"{type(error).__name__}: {error}")
        return ms.runner.RunnerResult([float(cycles)], None)


@derived_object
class GemminiRunner(ms.runner.PyRunner):
    """Run candidates concurrently; ``run(artifact path) -> cycles`` or raise."""

    def __init__(self, run, max_workers=8):
        self.run_callback = run
        self.pool = ThreadPoolExecutor(max_workers)

    def run(self, runner_inputs):
        return [
            GemminiRunnerFuture(self.pool.submit(self.run_callback, item.artifact_path))
            for item in runner_inputs
        ]


@derived_object
class RecordMeasurements(ms.measure_callback.PyMeasureCallback):
    """Append every measured candidate to a JSON-lines campaign log."""

    def __init__(self, path, describe=None):
        self.path = str(path)
        self.describe = describe

    def apply(self, task_scheduler, task_id, measure_candidates, builder_results, runner_results):
        task = task_scheduler.tasks_[task_id].ctx
        with open(self.path, "a", encoding="utf-8") as stream:
            for measured, built, ran in zip(measure_candidates, builder_results, runner_results):
                row = {
                    "task": task.task_name,
                    "workload": candidate_workload(measured.sch.mod),
                    "sketch": str(
                        measured.sch.mod["main"].body.block.annotations.get("gemmini.sketch", "cpu")
                    ),
                    "trace": json.loads(tvm.ir.save_json(measured.sch.trace.as_json())),
                    "artifact": built.artifact_path,
                    "build_error": built.error_msg,
                    "run_error": ran.error_msg,
                    "cycles": float(ran.run_secs[0]) if ran.run_secs else None,
                }
                if self.describe is not None:
                    row.update(self.describe(built.artifact_path))
                stream.write(json.dumps(row) + "\n")


def tune_gemmini_tasks(
    tasks,
    *,
    work_dir,
    builder,
    runner,
    max_trials_global,
    num_trials_per_iter=16,
    capabilities=None,
    strategy="evolutionary",
    seed=0,
    log=None,
    describe=None,
    include_cpu=True,
    warmup=4,
):
    """Tune extracted tasks with stock ``tune_tasks`` and return the JSONDatabase."""
    capabilities = capabilities or GemminiCapabilities()
    target = capabilities.target()
    contexts, weights = [], []
    for index, task in enumerate(tasks):
        if strategy == "evolutionary":
            search = ms.search_strategy.EvolutionarySearch(
                population_size=64,
                init_measured_ratio=0.2,
                init_min_unmeasured=16,
                max_fail_count=8,
                genetic_num_iters=3,
                genetic_mutate_prob=0.85,
                genetic_max_fail_count=10,
                eps_greedy=0.1,
            )
        elif strategy == "replay":
            search = ms.search_strategy.ReplayTrace(max_fail_count=8)
        else:
            raise ValueError("Unknown MetaSchedule strategy")
        contexts.append(
            ms.TuneContext(
                mod=task.dispatched[0],
                target=target,
                space_generator=gemmini_space_generator(capabilities, include_cpu=include_cpu),
                search_strategy=search,
                task_name=task.task_name,
                rand_state=seed + index,
                num_threads=1,
            )
        )
        weights.append(float(task.weight))
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    if (Path(work_dir) / "database_tuning_record.json").exists():
        for context in contexts:
            context.generate_design_space()
    database = ms.database.JSONDatabase(work_dir=str(work_dir))
    callbacks = [ms.measure_callback.AddToDatabase(), ms.measure_callback.UpdateCostModel()]
    if log is not None:
        callbacks.append(RecordMeasurements(log, describe))
    cost_model = ms.cost_model.XGBModel(
        extractor=MathematicalFeatures(), adaptive_training=False, num_tuning_cores=1,
        num_warmup_samples=warmup,
    )
    model_path = Path(work_dir) / "cost_model.xgb"
    if model_path.exists():
        cost_model.load(str(model_path))
    ms.tune_tasks(
        tasks=contexts,
        task_weights=weights,
        work_dir=str(work_dir),
        max_trials_global=max_trials_global,
        num_trials_per_iter=num_trials_per_iter,
        builder=builder,
        runner=runner,
        database=database,
        cost_model=cost_model,
        measure_callbacks=callbacks,
        task_scheduler="gradient",
    )
    if cost_model.booster is not None:
        cost_model.save(str(model_path))
    (Path(work_dir) / "cost-model.json").write_text(json.dumps({
        "implementation": "TVM XGBModel", "warmup_samples": warmup,
        "training_samples": cost_model.data_size, "booster_trained": cost_model.booster is not None,
        "learned_predictions_enabled": cost_model.data_size >= warmup and cost_model.booster is not None,
    }, indent=2) + "\n")
    return database


def cycles_digest(values):
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
