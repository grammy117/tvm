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
"""Framework mutation, measured feedback, failure isolation and replay contracts."""

import itertools
import unittest
import numpy as np
import tvm
from tvm import tir, meta_schedule as ms
from tvm.relax.backend.contrib.gemmini_dse import TrialFailure, load_result
from tvm.relax.backend.contrib.gemmini_search import tune
from tvm.relax.backend.contrib.gemmini_target import GemminiCapabilities
from tvm.relax.backend.contrib.gemmini_tensorir import schedule_contraction, lower_tensorir_schedule

import test_backend_contrib_gemmini_tensorir as reference


class GemminiFrameworkSearchTests(unittest.TestCase):
    setUp = reference.GemminiTensorIRTests.setUp
    build = reference.GemminiTensorIRTests.build

    def test_framework_cloned_vars_preserve_buffer_ownership_by_abi_position(self):
        func = reference.workload(3, 4, 5)
        sch, _ = schedule_contraction(func)
        renewed = tvm.ir.load_json(tvm.ir.save_json(sch.mod))
        self.assertFalse(renewed["main"].params[0].same_as(func.params[0]))
        selected = tir.Schedule(renewed)
        lowered = lower_tensorir_schedule(func, selected)
        runtime, control = self.build(lowered.scheduled_mod)
        a = np.arange(15, dtype="int8").reshape(3, 5) - 7
        b = np.arange(20, dtype="int8").reshape(5, 4) - 11
        out = tvm.nd.empty((3, 4), "int32")
        runtime(tvm.nd.array(a), tvm.nd.array(b), out)
        np.testing.assert_array_equal(out.numpy(), a.astype("int64") @ b.astype("int64"))
        self.assertEqual(control.emulator_faults(), 0)

    def test_categorical_decisions_change_real_groups_and_preserve_native_math(self):
        func = reference.workload(32, 32, 32)
        sch, _ = schedule_contraction(
            func, row_tiles=2, col_tiles=2, sample_groups=True, tensorize=False, mark_tensorize=True
        )
        samples = [inst for inst in sch.trace.insts if inst.kind.name == "SampleCategorical"]
        self.assertEqual(len(samples), 2)
        observed = set()
        a = np.resize(np.array([-128, 127, -1, 0, 11], "int8"), (32, 32))
        b = np.resize(np.array([127, -128, 3, 0], "int8"), (32, 32))
        for row, col in itertools.product(range(2), repeat=2):
            trace = sch.trace.with_decision(samples[0], row, remove_postproc=True)
            trace = trace.with_decision(samples[1], col, remove_postproc=True)
            changed = tir.Schedule(func)
            trace.apply_to_schedule(changed, remove_postproc=True)
            attrs = changed.mod["main"].body.block.annotations
            observed.add((int(attrs["gemmini.row_tiles"]), int(attrs["gemmini.col_tiles"])))
            changed.enter_postproc()
            self.assertTrue(ms.postproc.RewriteTensorize().apply(changed))
            lowered = lower_tensorir_schedule(func, changed)
            runtime, control = self.build(lowered.scheduled_mod)
            output = tvm.nd.empty((32, 32), "int32")
            runtime(tvm.nd.array(a), tvm.nd.array(b), output)
            np.testing.assert_array_equal(output.numpy(), a.astype("int64") @ b.astype("int64"))
            self.assertEqual(control.emulator_faults(), 0)
        self.assertEqual(observed, {(1, 1), (1, 2), (2, 1), (2, 2)})

    def test_standard_controller_trains_continues_after_failure_and_replays(self):
        caps, artifacts, calls, journal = GemminiCapabilities(), {}, [], []

        def build(candidate):
            calls.append(candidate.id)
            if len(calls) == 3:
                raise TrialFailure("diagnostic build timeout")
            key = str(self.root / f"diagnostic-{len(calls)}")
            artifacts[key] = candidate
            return {"folder": key}

        def run(artifact):
            candidate = artifacts[artifact["folder"]]
            cost = (
                1000
                if candidate.placement == "cpu"
                else 200 // int(candidate.parameters.get("row_tiles", 1))
            )
            return {
                "metric": "target_cycles",
                "engine": "diagnostic",
                "capability_fingerprint": caps.fingerprint,
                "measurement_profile": "framework-unit-test-only",
                "correctness": "passed",
                "samples": [cost] * 3,
            }

        func = reference.workload(32, 32, 32)
        result = tune(
            func,
            build,
            run,
            work_dir=self.root / "search",
            max_trials=6,
            warmup=2,
            diagnostic=True,
            on_record=journal.append,
            rule_options={"chunks": (16,), "row_tiles": (1, 2), "col_tiles": (1, 2)},
        )
        self.assertEqual(result.summary["controller"], "TVM tune_tasks/EvolutionarySearch")
        self.assertEqual(result.summary["failed_trials"], 1)
        self.assertGreaterEqual(result.summary["successful_trials"], 4)
        self.assertEqual(len(journal), len(result.records))
        self.assertTrue(
            any(
                event["booster_trained"] and event["distinct_predictions"] > 1
                for event in result.summary["ranking_events"]
            )
        )
        restored = load_result(result.as_json(), func, diagnostic=True)
        self.assertEqual(restored.winner.id, result.winner.id)
        completed = len(calls)
        resumed = tune(
            func,
            build,
            run,
            work_dir=self.root / "resumed",
            max_trials=6,
            warmup=2,
            diagnostic=True,
            resume_records=result.records,
            rule_options={"chunks": (16,), "row_tiles": (1, 2), "col_tiles": (1, 2)},
        )
        self.assertEqual(len(calls), completed)
        self.assertEqual(resumed.winner.id, result.winner.id)
