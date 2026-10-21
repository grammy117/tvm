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
"""Placement and bias choices retain mathematical semantics and honest labels."""

import unittest
import json
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import tvm
from tvm import meta_schedule as ms, relax
from tvm.relax.backend.contrib.gemmini import prepare_gemmini_graph
from tvm.relax.backend.contrib.gemmini_meta_schedule import gemmini_space_generator
from tvm.relax.backend.contrib.gemmini_target import GemminiCapabilities
from tvm.relax.backend.contrib.gemmini_tensorir import ScheduleGemminiTensorIR

import test_backend_contrib_gemmini_tensorir as reference


class GemminiDSETests(unittest.TestCase):
    setUp = reference.GemminiTensorIRTests.setUp
    build = reference.GemminiTensorIRTests.build

    def test_ml_uses_observations_and_native_winner_reloads_in_a_fresh_process(self):
        from tvm.relax.backend.contrib.gemmini_dse import load_result, search

        caps = GemminiCapabilities()
        func = reference.workload(3, 4, 17)
        options = {
            "orders": ("ij", "ji"),
            "load_orders": ("ab", "ba"),
            "chunks": (4, 8),
            "row_tiles": (1,),
            "col_tiles": (1,),
        }

        def measured(candidate):
            return {
                "metric": "target_cycles",
                "engine": "diagnostic",
                "capability_fingerprint": caps.fingerprint,
                "measurement_profile": "synthetic-replay-test-only",
                "correctness": "passed",
                "samples": [1000 if candidate.placement == "cpu" else 100] * 3,
            }

        result = search(
            func,
            measured,
            strategy="ml",
            max_trials=3,
            warmup=2,
            rule_options=options,
            diagnostic=True,
        )
        self.assertIsNotNone(result.records[2]["predicted_log_cycles_before_measurement"])
        body = json.loads(json.dumps(result.as_json()))
        restored = load_result(body, func, diagnostic=True)
        self.assertEqual(restored.winner.id, result.winner.id)
        with self.assertRaisesRegex(ValueError, "Synthetic"):
            load_result(body, func)
        path = self.root / "saved-cycle-search.json"
        path.write_text(json.dumps(body))
        subprocess.run(
            [
                sys.executable,
                "-c",
                """
import json, sys, tvm
from tvm.relax.backend.contrib.gemmini_dse import load_result
from tvm.relax.backend.contrib.gemmini_tensorir import ScheduleGemminiTensorIR
body = json.load(open(sys.argv[1]))
mod = tvm.ir.load_json(body['workload_json'])
result = load_result(body, mod['main'], diagnostic=True)
report = []
with result.database():
    ScheduleGemminiTensorIR(report=report)(mod)
assert report[0]['placement'] == 'gemmini', report
""",
                str(path),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=45,
        )

    def test_cycle_search_rejects_spike_and_replays_cpu_winner_without_fake_seconds(self):
        from tvm.relax.backend.contrib.gemmini_dse import search

        caps = GemminiCapabilities()
        func = reference.workload(3, 4, 5)
        options = {
            "orders": ("ij",),
            "load_orders": ("ab",),
            "chunks": (4,),
            "row_tiles": (1,),
            "col_tiles": (1,),
        }

        def measured(candidate):
            return {
                "metric": "target_cycles",
                "engine": "diagnostic",
                "capability_fingerprint": caps.fingerprint,
                "measurement_profile": "synthetic-api-test-only",
                "correctness": "passed",
                "samples": [100 if candidate.placement == "cpu" else 200] * 3,
            }

        result = search(func, measured, max_trials=2, rule_options=options, diagnostic=True)
        self.assertEqual(result.winner.placement, "cpu")
        self.assertEqual(result.summary["measurement_scope"], "diagnostic")
        self.assertEqual(len(result.records), 2)
        self.assertIn(16, result.summary["rule_options"]["chunks"])
        with result.database() as database:
            record = database.query_tuning_record(
                tvm.IRModule({"main": func}), caps.target(), "main"
            )
            self.assertIsNone(record.run_secs)
            report = []
            prepared = ScheduleGemminiTensorIR(report=report)(tvm.IRModule({"main": func}))
        self.assertEqual(report[0]["placement"], "cpu")
        self.assertNotIn("gemmini.tensorir", prepared["main"].attrs)

        def spike(candidate):
            return dict(measured(candidate), engine="spike")

        with self.assertRaisesRegex(ValueError, "Spike"):
            search(func, spike, max_trials=1, rule_options=options)

    def test_cpu_alternative_replays_without_accelerator_calls_or_native_label(self):
        original = reference.workload(17, 19, 33)
        caps = GemminiCapabilities()
        context = ms.TuneContext(
            mod=original,
            target=caps.target(),
            space_generator=gemmini_space_generator(include_cpu=True),
            num_threads=1,
        )
        designs = context.generate_design_space()
        self.assertGreater(len(designs), 1)
        # Groups beyond ceil(M/16), ceil(N/16) only add padded work.
        for design in designs:
            attrs = design.get(design.get_block("root")).annotations
            self.assertLessEqual(int(attrs.get("gemmini.row_tiles", 1)), 2)
            self.assertLessEqual(int(attrs.get("gemmini.col_tiles", 1)), 2)
        cpu = next(
            design
            for design in designs
            if not any(inst.kind.name == "Split" for inst in design.trace.insts)
        )
        cpu.enter_postproc()
        self.assertTrue(all(p.apply(cpu) for p in context.space_generator.postprocs))
        database = ms.database.MemoryDatabase()
        workload = database.commit_workload(tvm.IRModule({"main": original}))
        # Diagnostic API-selection label, never a measured hardware cost.
        database.commit_tuning_record(
            ms.database.TuningRecord(cpu.trace, workload, [1.0], caps.target(), [])
        )
        report = []
        with database:
            result = ScheduleGemminiTensorIR(report=report)(tvm.IRModule({"main": original}))
        self.assertEqual(report[0]["placement"], "cpu")
        self.assertEqual(report[0]["origin"], "TVM MetaSchedule database")
        self.assertNotIn("gemmini.tensorir", result["main"].attrs)
        runtime, control = self.build(result)
        a = np.resize(np.array([-128, 127, -3, 0, 7], "int8"), (17, 33))
        b = np.resize(np.array([19, -128, 127, 0], "int8"), (33, 19))
        out = tvm.nd.empty((17, 19), "int32")
        runtime(tvm.nd.array(a), tvm.nd.array(b), out)
        np.testing.assert_array_equal(out.numpy(), a.astype("int64") @ b.astype("int64"))
        self.assertEqual(control.emulator_commands(), 0)

    def test_bias_preload_can_be_disabled_without_changing_results(self):
        for preload in (False, True):
            a = relax.Var("a", relax.TensorStructInfo((3, 5), "int8"))
            b = relax.Var("b", relax.TensorStructInfo((5, 4), "int8"))
            bias = relax.Var("bias", relax.TensorStructInfo((4,), "int32"))
            builder = relax.BlockBuilder()
            with builder.function("main", [a, b, bias]):
                with builder.dataflow():
                    product = builder.emit(relax.op.matmul(a, b, out_dtype="int32"))
                    output = builder.emit_output(relax.op.add(product, bias))
                builder.emit_func_output(output)
            report = []
            graph = prepare_gemmini_graph(builder.get(), bias_preload=preload, report=report)
            selected = [row for row in report if row["placement"] == "gemmini"]
            self.assertEqual(len(selected), 1, report)
            self.assertEqual(bool(selected[0]["metadata"].get("bias_preloaded")), preload)
            self.assertEqual("mathematical_normalization" in selected[0], preload)
            self.index += 1
            path = self.root / f"bias-policy-{self.index}.so"
            relax.build(graph, "llvm").export_library(
                str(path), addons=[str(self.emulator)], cc="cc", options=["-std=c11"]
            )
            vm = relax.VirtualMachine(tvm.runtime.load_module(str(path)), tvm.cpu())
            av = np.arange(15, dtype="int8").reshape(3, 5) - 7
            bv = np.arange(20, dtype="int8").reshape(5, 4) - 9
            dv = np.array([-19, 23, 0, 1], "int32")
            np.testing.assert_array_equal(
                vm["main"](tvm.nd.array(av), tvm.nd.array(bv), tvm.nd.array(dv)).numpy(),
                av.astype("int64") @ bv.astype("int64") + dv,
            )

    def test_default_is_measured_before_exploration_and_cannot_be_regressed(self):
        from tvm.relax.backend.contrib.gemmini_dse import search

        caps = GemminiCapabilities()
        seen = []

        def measured(candidate):
            seen.append(candidate)
            default = candidate.parameters.get("reduction_chunk") == 16
            return {
                "metric": "target_cycles",
                "engine": "diagnostic",
                "capability_fingerprint": caps.fingerprint,
                "measurement_profile": "baseline-test",
                "correctness": "passed",
                "samples": [10 if default else 100] * 3,
            }

        result = search(reference.workload(8, 8, 16), measured, max_trials=100, diagnostic=True)
        self.assertEqual(seen[0].placement, "cpu")
        self.assertEqual(
            seen[1].parameters,
            {
                "order": "ij",
                "load_order": "ab",
                "reduction_chunk": 16,
                "row_tiles": 1,
                "col_tiles": 1,
            },
        )
        self.assertEqual(result.winner.id, seen[1].id)
        self.assertTrue(result.summary["space_exhausted"])
        self.assertTrue(result.summary["baseline_complete"])
        self.assertEqual(len(seen), 4)

    def test_failed_trials_are_durable_and_resume_does_not_repeat_completed_work(self):
        from tvm.relax.backend.contrib.gemmini_dse import TrialFailure, search

        caps = GemminiCapabilities()
        journal = []
        calls = []

        def measured(candidate):
            calls.append(candidate.id)
            if len(calls) == 2:
                raise TrialFailure("bounded runner timeout", evidence={"receipt": "test-only"})
            return {
                "metric": "target_cycles",
                "engine": "diagnostic",
                "capability_fingerprint": caps.fingerprint,
                "measurement_profile": "failure-test",
                "correctness": "passed",
                "samples": [100] * 3,
            }

        func = reference.workload(8, 8, 16)
        first = search(func, measured, max_trials=3, on_record=journal.append, diagnostic=True)
        self.assertEqual(len(journal), 3)
        self.assertEqual(journal[1]["status"], "failed")
        self.assertEqual(first.summary["failed_trials"], 1)
        resumed = search(func, measured, max_trials=100, resume_records=journal, diagnostic=True)
        self.assertEqual(len(calls), 4)
        self.assertEqual(len(set(calls)), 4)
        self.assertEqual(resumed.summary["trials"], 4)

    def test_constant_predictions_are_labeled_exploration(self):
        from tvm.relax.backend.contrib.gemmini_dse import search

        caps = GemminiCapabilities()

        def measured(candidate):
            return {
                "metric": "target_cycles",
                "engine": "diagnostic",
                "capability_fingerprint": caps.fingerprint,
                "measurement_profile": "ties-test",
                "correctness": "passed",
                "samples": [100] * 3,
            }

        with (
            patch("tvm.relax.backend.contrib.gemmini_dse.CycleRanker.fit"),
            patch(
                "tvm.relax.backend.contrib.gemmini_dse.CycleRanker.predict",
                side_effect=lambda candidates: np.ones(len(candidates)),
            ),
        ):
            result = search(
                reference.workload(17, 19, 33),
                measured,
                strategy="ml",
                warmup=2,
                max_trials=3,
                diagnostic=True,
            )
        row = result.records[2]
        self.assertEqual(row["selection"], "ml_tie_exploration")
        self.assertEqual(row["ranking"]["distinct_predictions"], 1)
        self.assertGreater(row["ranking"]["best_tie_count"], 1)

    def test_copies_default_to_cpu_and_require_explicit_policy(self):
        x = relax.Var("x", relax.TensorStructInfo((2, 16), "float32"))
        builder = relax.BlockBuilder()
        with builder.function("main", [x]):
            with builder.dataflow():
                y = builder.emit_output(relax.op.permute_dims(x))
            builder.emit_func_output(y)
        report = []
        prepare_gemmini_graph(builder.get(), report=report)
        self.assertTrue(report)
        self.assertTrue(all(row["placement"] == "cpu" for row in report))
        self.assertIn("opt-in", report[0]["reason"])
