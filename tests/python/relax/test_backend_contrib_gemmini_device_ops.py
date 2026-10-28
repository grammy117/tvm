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
"""Real command execution, trace replay and refusal boundaries for device extensions."""

from pathlib import Path
import sys
import unittest

import numpy as np
import tvm
from tvm import te, tir

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "apps/gemmini"))
from verify_device_ops import fixtures, graph, quantize
from tvm.relax.backend.contrib.gemmini import prepare_gemmini_graph
from tvm.relax.backend.contrib.gemmini_elementwise import make_elementwise_schedule
from tvm.relax.backend.contrib.gemmini_tensorir import lower_tensorir_schedule

import test_backend_contrib_gemmini_tensorir as reference


class DeviceOpsTests(unittest.TestCase):
    setUp = reference.GemminiTensorIRTests.setUp
    build = reference.GemminiTensorIRTests.build

    def test_arithmetic_windows_and_dma_execute_real_primitives(self):
        for case in fixtures():
            name, mod, arrays, expected = case[:4]
            if name in ("dma_bfloat16", "group16_preserved"):
                # Its uint16 storage is qualified in the Spike guest; host NDArray
                # construction would perform a numerical dtype conversion. The
                # bound-constant group16 graph also uses the complete guest path.
                continue
            with self.subTest(case=name):
                report = []
                prepared = prepare_gemmini_graph(
                    mod, report=report, **(case[5] if len(case) > 5 else {})
                )
                device = len(case) < 5 or case[4] != "host"
                self.assertEqual(sum(r["placement"] == "gemmini" for r in report), int(device), report)
                func = next(f for f in prepared.functions.values() if isinstance(f, tir.PrimFunc))
                runtime, control = self.build(
                    tvm.IRModule({"main": func.with_attr("global_symbol", "main")})
                )
                values = [tvm.nd.array(array) for array in arrays]
                output = tvm.nd.empty(expected.shape, str(expected.dtype))
                runtime(*values, output)
                np.testing.assert_array_equal(output.numpy().view("uint8"), expected.view("uint8"))
                self.assertEqual(control.emulator_commands() > 0, device)
                self.assertEqual(control.emulator_faults(), 0)
                self.assertEqual(control.emulator_pending(), 0)
                for actual, expected_input in zip(values, arrays):
                    np.testing.assert_array_equal(
                        actual.numpy().view("uint8"), expected_input.view("uint8")
                    )

    def test_new_affine_and_byte_traces_replay(self):
        for name in ("subtract", "scalar_affine", "single_round", "per_operand_05", "dma_int64"):
            case = next(case for case in fixtures() if case[0] == name)
            func = next(f for f in case[1].functions.values() if isinstance(f, tir.PrimFunc))
            result = make_elementwise_schedule(func, scalar=name.startswith("dma_"))
            replay = tir.Schedule(func, debug_mask="all")
            result.trace.apply_to_schedule(replay, remove_postproc=False)
            actual = lower_tensorir_schedule(
                func, replay, workload_metadata={"operation": result.metadata["operation"]}
            )
            tvm.ir.assert_structural_equal(actual.scheduled_mod, result.scheduled_mod)

    def test_default_baselines_cover_secondary_families_and_replay(self):
        from tvm.relax.backend.contrib.gemmini_meta_schedule import default_device_schedule

        for name in ("subtract", "max_pool_2", "max_pool_3", "dma_int64_rectangular",
                     "window_average_4", "window_average_9"):
            with self.subTest(case=name):
                case = next(case for case in fixtures() if case[0] == name)
                func = next(f for f in case[1].functions.values() if isinstance(f, tir.PrimFunc))
                schedule = default_device_schedule(func)
                replay = tir.Schedule(func, debug_mask="all")
                schedule.trace.apply_to_schedule(replay, remove_postproc=False)
                tvm.ir.assert_structural_equal(schedule.mod, replay.mod)
                lowered = lower_tensorir_schedule(func, replay, workload_metadata={"operation": name})
                self.assertIn("tvm_gemmini_", lowered.scheduled_mod.script())

    def test_new_families_are_visible_to_stock_metaschedule(self):
        from tvm import meta_schedule as ms
        from tvm.relax.backend.contrib.gemmini_meta_schedule import gemmini_space_generator
        from tvm.relax.backend.contrib.gemmini_target import GemminiCapabilities

        for name in ("subtract", "max_pool_2", "max_pool_3", "dma_int64"):
            case = next(case for case in fixtures() if case[0] == name)
            func = next(f for f in case[1].functions.values() if isinstance(f, tir.PrimFunc))
            context = ms.TuneContext(
                mod=func.with_attr("global_symbol", "main"),
                target=GemminiCapabilities().target(),
                space_generator=gemmini_space_generator(),
                num_threads=1,
            )
            found = False
            for design in context.generate_design_space():
                design.enter_postproc()
                if all(postproc.apply(design) for postproc in context.space_generator.postprocs):
                    found |= "local.gemmini" in design.mod.script()
            self.assertTrue(found, name)

    def test_frozen_group_scales_keep_one_contraction_and_original_outputs(self):
        from test_backend_contrib_gemmini_views import GemminiViewTests

        case = next(case for case in fixtures() if case[0] == "group16_preserved")
        report = []
        prepared = prepare_gemmini_graph(case[1], report=report, **case[5])
        self.assertEqual(sum(row["placement"] == "gemmini" for row in report), case[6])
        _, outputs, control = GemminiViewTests.static_run(self, prepared, case[2])
        np.testing.assert_array_equal(outputs[0], case[3])
        self.assertGreater(control.emulator_commands(), 0)
        self.assertEqual(control.emulator_faults(), 0)

    def test_declared_source_rank_boundaries(self):
        from tvm.relax.backend.contrib.gemmini_meta_schedule import _supported

        for rank in (4, 5):
            batch = (1,) * (rank - 2)
            a = te.placeholder(batch + (2, 4), "int8", "a")
            b = te.placeholder(batch + (4, 3), "int8", "b")
            k = te.reduce_axis((0, 4), "k")
            c = te.compute(batch + (2, 3), lambda *i: te.sum(
                a[i[:-1] + (k,)].astype("int32")
                * b[i[:-2] + (k, i[-1])].astype("int32"), axis=k))
            self.assertEqual(_supported(te.create_prim_func([a, b, c]))[0],
                             "contraction" if rank == 4 else None)
        for shape in ((3,), (2, 3)):
            a = te.placeholder(shape + (4,), "int8", "pool_input")
            k = te.reduce_axis((0, 4), "k")
            c = te.compute(shape, lambda *i: te.max(a[i + (k,)], axis=k))
            self.assertEqual(_supported(te.create_prim_func([a, c]))[0],
                             "pool" if len(shape) == 2 else None)
        shape = (1, 1, 1, 1, 1, 3)
        a = te.placeholder(shape, "int8", "copy_input")
        c = te.compute(shape, lambda *i: a[i])
        self.assertEqual(_supported(te.create_prim_func([a, c])), (None, []))

    def test_unbounded_standalone_readout_has_no_affine_admission(self):
        from tvm.relax.backend.contrib.gemmini_meta_schedule import _supported, default_device_schedule
        case = next(case for case in fixtures() if case[0] == "standalone_readout")
        func = next(f for f in case[1].functions.values() if isinstance(f, tir.PrimFunc))
        self.assertEqual(_supported(func), (None, []))
        with self.assertRaises(ValueError):
            default_device_schedule(func)

    def test_affine_metadata_bounds_cover_every_prefix(self):
        for name in ("subtract", "scalar_affine", "affine_relu"):
            case = next(case for case in fixtures() if case[0] == name)
            func = next(f for f in case[1].functions.values() if isinstance(f, tir.PrimFunc))
            info = make_elementwise_schedule(func).metadata
            expected = {"subtract": (256, 128), "scalar_affine": (919, 896),
                        "affine_relu": (659, 384)}[name]
            self.assertEqual((info["accumulator_partial_abs_bound"], info["pe_partial_abs_bound"]), expected)

    def test_per_operand_rounding_refuses_undeclared_ranks(self):
        from tvm.relax.backend.contrib.gemmini_elementwise import elementwise_blocks

        for shape in ((5,), (2, 3, 5)):
            with self.subTest(shape=shape):
                a = te.placeholder(shape, "int8", "a")
                b = te.placeholder(shape, "int8", "b")
                c = te.compute(shape, lambda *i: quantize(
                    quantize(a[i], .5).astype("int32")
                    + quantize(b[i], .25).astype("int32"), .5))
                func = te.create_prim_func([a, b, c])
                self.assertEqual(elementwise_blocks(func), [])
                with self.assertRaises(ValueError):
                    make_elementwise_schedule(func)

    def test_full_domain_i32_add_has_no_device_schedule_or_intrinsic(self):
        from tvm.relax.backend.contrib.gemmini_elementwise import register_elementwise_intrinsics
        from tvm.relax.backend.contrib.gemmini_meta_schedule import _supported, default_device_schedule

        a = te.placeholder((3, 5), "int32", "a")
        b = te.placeholder((3, 5), "int32", "b")
        for wrapped in (False, True):
            c = te.compute((3, 5), lambda *i:
                (a[i].astype("int64") + b[i].astype("int64")).astype("int32")
                if wrapped else a[i] + b[i])
            func = te.create_prim_func([a, b, c])
            self.assertEqual(_supported(func), (None, []))
            with self.assertRaises(ValueError):
                default_device_schedule(func)
            with self.assertRaises(ValueError):
                register_elementwise_intrinsics("add_i32_wrapping" if wrapped else "add_i32", 3, 5)

    def test_unsupported_numerics_remain_cpu(self):
        for kind in ("float_add", "wide_coefficient", "wrong_rounding", "scaled_operand_over_one"):
            with self.subTest(kind=kind):
                a = te.placeholder((3, 5), "float32" if kind == "float_add" else "int8", "a")
                if kind == "float_add":
                    c = te.compute((3, 5), lambda i, j: a[i, j] + 1.5)
                elif kind == "wide_coefficient":
                    c = te.compute((3, 5), lambda i, j: a[i, j].astype("int32") * 128)
                elif kind == "wrong_rounding":
                    c = te.compute(
                        (3, 5),
                        lambda i, j: te.min(
                            te.max(tir.round(a[i, j].astype("float32") * 0.5), -128.0), 127.0
                        ).astype("int8"),
                    )
                else:
                    c = te.compute(
                        (3, 5),
                        lambda i, j: quantize(
                            quantize(a[i, j], 2.0).astype("int32") + a[i, j].astype("int32"), 1.0
                        ),
                    )
                report = []
                prepare_gemmini_graph(graph(te.create_prim_func([a, c]), 1), report=report)
                self.assertFalse(any(r["placement"] == "gemmini" for r in report), report)


if __name__ == "__main__":
    unittest.main()
