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
"""Gemmini sketches, the MetaSchedule design space and the fixed lowering.

Numerics run against the queued-command emulator: every command is delayed to
the next fence, so a missing completion wait or a wrong accumulator overwrite
shows up as a fault or a wrong result.
"""

import ctypes
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import tvm
from tvm import meta_schedule as ms, relax, te, tir
from tvm.relax.backend.contrib.gemmini_meta_schedule import (
    GemminiScheduleRule,
    gemmini_space_generator,
)
from tvm.relax.backend.contrib.gemmini_search import (
    AnnotateGemminiWorkloads,
    WORKLOAD_ATTR,
    workload_key,
)
from tvm.relax.backend.contrib.gemmini_sketch import (
    ContractionSketch,
    apply_contraction_sketch,
    contraction_info,
    default_schedule,
    sketch_space,
)
from tvm.relax.backend.contrib.gemmini_target import GemminiCapabilities
from tvm.relax.backend.contrib.gemmini_tensorir import (
    lower_tensorir_schedule,
    prepare_gemmini_tensorir_graph,
)

from test_backend_contrib_gemmini_tensorir import EMULATOR


def matmul(m, k, n):
    a = te.placeholder((m, k), "int8", "a")
    b = te.placeholder((k, n), "int8", "b")
    r = te.reduce_axis((0, k), "r")
    c = te.compute(
        (m, n),
        lambda i, j: te.sum(a[i, r].astype("int32") * b[r, j].astype("int32"), axis=r),
        name="product",
    )
    return te.create_prim_func([a, b, c])


class EmulatedCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gemmini-sketch-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "emulator.c").write_text(EMULATOR)
        self.index = 0

    def run_lowered(self, func, lowered):
        self.index += 1
        path = self.root / f"kernel-{self.index}.so"
        tvm.build(lowered, "llvm").export_library(
            str(path), addons=[str(self.root / "emulator.c")], cc="cc", options=["-std=c11"]
        )
        runtime, control = tvm.runtime.load_module(str(path)), ctypes.CDLL(str(path))
        (m, k), n = [int(x) for x in func.buffer_map[func.params[0]].shape], int(
            func.buffer_map[func.params[1]].shape[1]
        )
        rng = np.random.default_rng(self.index)
        a = rng.integers(-128, 128, (m, k), dtype="int8")
        b = rng.integers(-128, 128, (k, n), dtype="int8")
        out = tvm.nd.array(np.full((m, n), 7, "int32"))
        control.emulator_reset()
        runtime(tvm.nd.array(a), tvm.nd.array(b), out)
        np.testing.assert_array_equal(out.numpy(), a.astype("int64") @ b.astype("int64"))
        self.assertEqual(control.emulator_faults(), 0)
        return control.emulator_commands(), control.emulator_fences()

    def check(self, shape, sketch, decisions):
        func = matmul(*shape)
        sch = tir.Schedule(func, debug_mask="all")
        sch = apply_contraction_sketch(
            sch, sch.get_block("product"), sketch, decisions=decisions, tensorize=True
        )
        lowered = lower_tensorir_schedule(func, sch)
        return (*self.run_lowered(func, lowered.scheduled_mod), lowered.metadata)


class SketchTests(EmulatedCase):
    def test_partitioned_tails_use_rectangles_without_host_staging(self):
        for shape, sketch, decisions in (
            ((50, 32, 32), ContractionSketch(tail="m"),
             {"i": [3, 1], "j": [2, 1], "k": [2, 1],
              "ti": [1, 1], "tj": [2, 1], "tk": [2, 1]}),
            ((32, 32, 19), ContractionSketch(tail="n"),
             {"i": [2, 1], "j": [1, 1], "k": [2, 1],
              "ti": [2, 1], "tj": [3, 1], "tk": [2, 1]}),
        ):
            _, fences, metadata = self.check(shape, sketch, decisions)
            self.assertEqual(metadata["host_private_buffers"], 0)
            self.assertEqual(metadata["waited_host_buffers"], 0)
            # Separate output regions are conservatively ordered once; no
            # per-tile waits or host padding are needed for their edge tiles.
            self.assertLessEqual(fences, 2)

    def test_round_even_clip_is_fused_and_numerically_exact(self):
        a = te.placeholder((3, 5), "int8", "a")
        b = te.placeholder((5, 4), "int8", "b")
        k = te.reduce_axis((0, 5), "k")
        product = te.compute((3, 4), lambda i, j: te.sum(
            a[i, k].astype("int32") * b[k, j].astype("int32"), axis=k), name="product")
        output = te.compute((3, 4), lambda i, j: tir.min(tir.max(
            tir.nearbyint(product[i, j].astype("float32") / tir.const(16.0, "float32")),
            tir.const(0.0, "float32")), tir.const(127.0, "float32")).astype("int8"), name="readout")
        func = te.create_prim_func([a, b, output])
        lowered = lower_tensorir_schedule(func, default_schedule(func)).scheduled_mod
        self.assertIn("tvm_gemmini_dma_scaled_out", lowered.script())
        path = self.root / "readout.so"
        tvm.build(lowered, "llvm").export_library(str(path), addons=[str(self.root / "emulator.c")],
                                                  cc="cc", options=["-std=c11"])
        runtime = tvm.runtime.load_module(str(path))
        av = np.resize(np.array([-128, 127, -1, 0, 16], "int8"), (3, 5))
        bv = np.resize(np.array([127, -128, 1, 2, 0, 7, -3], "int8"), (5, 4))
        out = tvm.nd.empty((3, 4), "int8")
        runtime(tvm.nd.array(av), tvm.nd.array(bv), out)
        expected = np.clip(np.rint((av.astype("int32") @ bv.astype("int32")).astype("float32") / 16), 0, 127).astype("int8")
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_rule_keeps_prior_schedule_operations(self):
        func = matmul(32, 32, 32)
        sch = tir.Schedule(func)
        block = sch.get_block("product")
        sch.annotate(sch.get_block("root"), "test.caller_schedule", 7)
        supplied = str(sch.trace)
        candidates = GemminiScheduleRule().apply(sch, block)
        self.assertTrue(candidates)
        for candidate in candidates:
            self.assertEqual(int(candidate.mod["main"].body.block.annotations["test.caller_schedule"]), 7)
            self.assertIn('"test.caller_schedule"', str(candidate.trace))
        self.assertEqual(str(sch.trace), supplied)

    def test_sketch_structures_are_exact(self):
        cases = [
            ((64, 64, 48), ContractionSketch(), {"i": [2, 2], "j": [3, 1], "k": [2, 2]}),
            ((64, 64, 48), ContractionSketch(order="ji", hoist=True), {"i": [1, 4], "j": [3, 1], "k": [4, 1]}),
            ((1, 64, 48), ContractionSketch(rows=1), {"i": [1, 1], "j": [1, 3], "k": [1, 4]}),
            ((50, 32, 32), ContractionSketch(rows=10), {"i": [5, 1], "j": [1, 2], "k": [1, 2]}),
            ((48, 36, 32), ContractionSketch(staging="root"), {"i": [3, 1], "j": [1, 2], "k": [3, 1]}),
            ((48, 36, 32), ContractionSketch(depth=12), {"i": [3, 1], "j": [1, 2], "k": [1, 3]}),
            ((64, 64, 48), ContractionSketch(dataflow="ws"), {"i": [1, 4], "j": [3, 1], "k": [2, 2]}),
            ((64, 64, 48), ContractionSketch(pipeline=True), {"i": [2, 2], "j": [3, 1], "k": [4, 1]}),
        ]
        for shape, sketch, decisions in cases:
            with self.subTest(shape=shape, sketch=sketch.name):
                self.check(shape, sketch, decisions)

    def test_double_buffered_pipeline_doubles_operand_rows(self):
        decisions = {"i": [2, 2], "j": [3, 1], "k": [4, 1]}
        _, _, plain = self.check((64, 64, 48), ContractionSketch(), decisions)
        _, _, piped = self.check((64, 64, 48), ContractionSketch(pipeline=True), decisions)
        self.assertGreater(piped["spad_rows"], plain["spad_rows"])

    def test_full_sram_capacity_is_usable_and_overflow_is_rejected(self):
        decisions = {"i": [1, 8], "j": [1, 8], "k": [1, 64]}
        _, _, metadata = self.check((128, 1024, 128), ContractionSketch(), decisions)
        self.assertEqual(metadata["spad_rows"], 1024 * 16)
        self.assertEqual(metadata["acc_rows"], 64 * 16)
        func = matmul(144, 1024, 128)
        sch = tir.Schedule(func, debug_mask="all")
        sch = apply_contraction_sketch(
            sch, sch.get_block("product"), ContractionSketch(),
            decisions={"i": [1, 9], "j": [1, 8], "k": [1, 64]}, tensorize=True)
        with self.assertRaisesRegex(ValueError, "SRAM capacity"):
            lower_tensorir_schedule(func, sch)

    def test_root_staging_needs_no_per_dma_wait(self):
        decisions = {"i": [3, 1], "j": [1, 2], "k": [3, 1]}
        _, fences, metadata = self.check((48, 36, 32), ContractionSketch(staging="root"), decisions)
        self.assertEqual(metadata["waited_host_buffers"], 0)
        self.assertEqual(fences, 1)
        _, fences, metadata = self.check((48, 36, 32), ContractionSketch(staging="tile"), decisions)
        self.assertGreater(metadata["waited_host_buffers"], 0)
        self.assertGreater(fences, 1)

    def test_lowering_preserves_explicit_init_and_individual_dma(self):
        func = matmul(64, 128, 64)
        sch = tir.Schedule(func, debug_mask="all")
        sch = apply_contraction_sketch(
            sch,
            sch.get_block("product"),
            ContractionSketch(),
            decisions={"i": [1, 4], "j": [1, 4], "k": [2, 4]},
            tensorize=True,
        )
        text = lower_tensorir_schedule(func, sch).scheduled_mod.script()
        self.assertIn("tvm_gemmini_zero_acc", text)
        self.assertIn("tvm_gemmini_dma_a", text)
        self.assertIn("tvm_gemmini_dma_b", text)
        self.assertNotIn("tvm_gemmini_dma_a_blocks", text)
        self.assertNotIn("tvm_gemmini_dma_b_blocks", text)
        self.run_lowered(func, lower_tensorir_schedule(func, sch).scheduled_mod)

    def test_traces_replay_from_json(self):
        func = matmul(64, 64, 48)
        sch = tir.Schedule(func, debug_mask="all")
        sch = apply_contraction_sketch(
            sch, sch.get_block("product"), ContractionSketch(pipeline=True), decisions={"i": [2, 2], "j": [3, 1], "k": [4, 1]}
        )
        replay = tir.Schedule(func, debug_mask="all")
        tir.schedule.Trace.apply_json_to_schedule(sch.trace.as_json(), replay)
        tvm.ir.assert_structural_equal(sch.mod, replay.mod)
        kinds = {inst.kind.name for inst in sch.trace.insts}
        self.assertTrue({"SamplePerfectTile", "CacheRead", "CacheWrite", "Blockize", "Annotate"} <= kinds)

    def test_sketch_space_prunes_redundant_structure(self):
        decode = contraction_info(matmul(1, 2048, 256), "product")
        prefill = contraction_info(matmul(64, 2048, 256), "product")
        orders = {sketch.order for sketch in sketch_space(decode)}
        self.assertEqual(orders, {"ij"})
        self.assertEqual({sketch.order for sketch in sketch_space(prefill)}, {"ij", "ji"})
        odd = contraction_info(matmul(50, 36, 32), "product")
        self.assertIn(16, {sketch.depth for sketch in sketch_space(odd)})
        self.assertIn(12, {sketch.depth for sketch in sketch_space(odd)})
        self.assertIn("m", {sketch.tail for sketch in sketch_space(odd)})

    def test_default_schedule_uses_reuse_and_preserves_explicit_groups(self):
        sch = default_schedule(matmul(64, 64, 48))
        names = [inst.kind.name for inst in sch.trace.insts]
        self.assertIn("SamplePerfectTile", names)
        decisions = [
            [int(x) for x in sch.trace.get_decision(inst)]
            for inst in sch.trace.insts
            if inst.kind.name == "SamplePerfectTile"
        ]
        self.assertTrue(any(decision[1] > 1 for decision in decisions))
        func = matmul(64, 64, 48)
        old = default_schedule(func, row_tiles=1, col_tiles=1)
        commands, _ = self.run_lowered(func, lower_tensorir_schedule(func, sch).scheduled_mod)
        old_commands, _ = self.run_lowered(func, lower_tensorir_schedule(func, old).scheduled_mod)
        self.assertLess(commands, old_commands)

    def test_default_reuse_respects_small_sram_and_decoding_shape(self):
        for shape in ((64, 64, 48), (1, 64, 256), (7, 19, 11)):
            with self.subTest(shape=shape):
                caps = GemminiCapabilities(spad_rows=64, acc_rows=32)
                func = matmul(*shape)
                sch = default_schedule(func, capabilities=caps)
                lowered = lower_tensorir_schedule(func, sch, capabilities=caps)
                self.assertLessEqual(lowered.metadata["acc_rows"], caps.acc_rows)
                self.run_lowered(func, lowered.scheduled_mod)
                replay = tir.Schedule(func)
                sch.trace.apply_to_schedule(replay, remove_postproc=False)
                tvm.ir.assert_structural_equal(replay.mod, sch.mod)


class DesignSpaceTests(EmulatedCase):
    def test_postorder_rule_preserves_convolution_padding_predicates(self):
        data = te.placeholder((7, 7, 16), "int8", "data")
        weight = te.placeholder((144, 16), "int8", "weight")

        def im2col(i, k):
            h, w = i // 7 + k // 48 - 1, i % 7 + (k // 16) % 3 - 1
            return te.if_then_else(tir.all(h >= 0, h < 7, w >= 0, w < 7),
                                   data[h, w, k % 16], tir.const(0, "int8"))

        padded = te.compute((49, 144), im2col, name="padded")
        from tvm.relax.backend.contrib.gemmini_elementwise import make_elementwise_schedule

        with self.assertRaisesRegex(ValueError, "predicated copy"):
            make_elementwise_schedule(te.create_prim_func([data, padded]))
        k = te.reduce_axis((0, 144), "k")
        product = te.compute((49, 16), lambda i, j: te.sum(
            padded[i, k].astype("int32") * weight[k, j].astype("int32"), axis=k),
            name="product")
        func = te.create_prim_func([data, weight, product])
        context = ms.TuneContext(
            mod=tvm.IRModule({"main": func}), target=GemminiCapabilities().target(),
            space_generator=gemmini_space_generator(include_cpu=False, dataflows=("os",)),
            num_threads=1, rand_state=1)
        spaces = context.generate_design_space()
        rng = np.random.default_rng(0)
        a, b = (rng.integers(-7, 8, shape, dtype="int8") for shape in ((7, 7, 16), (144, 16)))
        expected = tvm.nd.empty((49, 16), "int32")
        tvm.build(func, "llvm")(tvm.nd.array(a), tvm.nd.array(b), expected)
        checked = 0
        for sch in spaces:
            name = str(sch.mod["main"].body.block.annotations.get("gemmini.sketch", ""))
            if ".sroot." not in name:
                continue
            sch.enter_postproc()
            if not all(post.apply(sch) for post in context.space_generator.postprocs):
                continue
            lowered = lower_tensorir_schedule(func, sch).scheduled_mod
            # A rectangular DMA must not replace the predicated im2col read.
            self.assertNotIn("tvm_gemmini_dma_spad_out", lowered.script())
            path = self.root / f"padding-{checked}.so"
            tvm.build(lowered, "llvm").export_library(
                str(path), addons=[str(self.root / "emulator.c")], cc="cc", options=["-std=c11"])
            actual = tvm.nd.empty((49, 16), "int32")
            tvm.runtime.load_module(str(path))(tvm.nd.array(a), tvm.nd.array(b), actual)
            np.testing.assert_array_equal(actual.numpy(), expected.numpy())
            checked += 1
            if checked == 3:
                break
        self.assertEqual(checked, 3)

    def test_shared_convolution_epilogue_survives_inlining_and_trace_replay(self):
        from tvm.relax.backend.contrib.gemmini_tensorir import prepare_gemmini_tensorir_workloads

        data = relax.Var("data", relax.TensorStructInfo((1, 7, 7, 16), "int8"))
        weight = relax.Var("weight", relax.TensorStructInfo((16, 3, 3, 16), "int8"))
        bias = relax.Var("bias", relax.TensorStructInfo((1, 1, 1, 16), "int32"))
        bb = relax.BlockBuilder()
        with bb.function("main", [data, weight, bias]):
            with bb.dataflow():
                product = bb.emit(relax.op.nn.conv2d(
                    data, weight, padding=1, data_layout="NHWC", kernel_layout="OHWI",
                    out_dtype="int32"))
                wide = bb.emit(relax.op.astype(bb.emit(relax.op.add(product, bias)), "int64"))
                magnitude = bb.emit(relax.op.abs(wide))
                rounded = bb.emit(relax.op.floor_divide(
                    bb.emit(relax.op.add(magnitude, relax.const(32, "int64"))),
                    relax.const(64, "int64")))
                signed = bb.emit(relax.op.where(
                    bb.emit(relax.op.less(wide, relax.const(0, "int64"))),
                    bb.emit(relax.op.negative(rounded)), rounded))
                output = bb.emit_output(relax.op.astype(relax.op.clip(signed, -127, 127), "int8"))
            bb.emit_func_output(output)
        prepared, _, _ = prepare_gemmini_tensorir_workloads(bb.get())
        task = ms.relax_integration.extract_tasks(prepared, GemminiCapabilities().target())[0]
        func = task.dispatched[0]["main"]
        sch = tir.Schedule(task.dispatched[0], seed=1)
        sch = apply_contraction_sketch(
            sch, sch.get_block(contraction_names(func)[0]),
            ContractionSketch(rows=7, epilogue="tile"), tensorize=True)
        rng = np.random.default_rng(0)
        arrays = [rng.integers(-5, 6, tuple(int(x) for x in func.buffer_map[p].shape),
                               dtype=func.buffer_map[p].dtype) for p in func.params]
        reference = [tvm.nd.array(a) for a in arrays]
        tvm.build(task.dispatched[0], "llvm")(*reference)
        for seed in (1, 5, 17):
            replay = tir.Schedule(task.dispatched[0], seed=seed)
            tir.schedule.Trace(sch.trace.insts, {}).apply_to_schedule(replay, remove_postproc=False)
            lowered = lower_tensorir_schedule(func, replay).scheduled_mod
            path = self.root / f"conv-{seed}.so"
            tvm.build(lowered, "llvm").export_library(
                str(path), addons=[str(self.root / "emulator.c")], cc="cc", options=["-std=c11"])
            actual = [tvm.nd.array(a) for a in arrays]
            tvm.runtime.load_module(str(path))(*actual)
            np.testing.assert_array_equal(actual[-1].numpy(), reference[-1].numpy())

    def test_union_space_random_points_are_exact(self):
        caps = GemminiCapabilities()
        rng = random.Random(0)
        for shape in ((64, 64, 48), (1, 256, 64), (50, 32, 32)):
            func = matmul(*shape)
            context = ms.TuneContext(
                mod=tvm.IRModule({"main": func}),
                target=caps.target(),
                space_generator=gemmini_space_generator(caps),
                search_strategy=ms.search_strategy.ReplayTrace(),
                num_threads=1,
                rand_state=1,
            )
            spaces = context.generate_design_space()
            sketches = {
                str(space.mod["main"].body.block.annotations.get("gemmini.sketch", "cpu"))
                for space in spaces
            }
            self.assertIn("cpu", sketches)
            self.assertGreater(len(sketches), 2)
            for space in rng.sample(list(spaces), min(6, len(spaces))):
                sch = tir.Schedule(tvm.IRModule({"main": func}), seed=rng.randint(0, 1 << 30))
                trace = space.trace.simplified(remove_postproc=True)
                tir.schedule.Trace(trace.insts, {}).apply_to_schedule(sch, remove_postproc=False)
                if not all(post.apply(sch) for post in context.space_generator.postprocs):
                    continue
                if "tvm_gemmini" not in sch.mod.script():
                    continue
                self.run_lowered(func, lower_tensorir_schedule(func, sch).scheduled_mod)

    def test_rule_requires_bound_capabilities(self):
        rule = GemminiScheduleRule()
        with self.assertRaisesRegex(Exception, "target key"):
            ms.TuneContext(
                mod=tvm.IRModule({"main": matmul(16, 16, 16)}),
                target=tvm.target.Target("llvm"),
                space_generator=ms.space_generator.PostOrderApply(
                    sch_rules=[rule], postprocs=[], mutator_probs={}
                ),
            ).generate_design_space()


class DatabaseTests(EmulatedCase):
    def test_workload_key_is_structural_and_idempotent(self):
        func = matmul(32, 32, 32)
        key = workload_key(func)
        annotated = func.with_attr(WORKLOAD_ATTR, key)
        self.assertEqual(workload_key(annotated), key)
        self.assertNotEqual(workload_key(matmul(32, 32, 48)), key)

    def test_database_schedule_is_applied_by_stock_pass(self):
        bb = relax.BlockBuilder()
        x = relax.Var("x", relax.TensorStructInfo((64, 64), "int8"))
        weight = relax.const(np.ones((64, 48), "int8"))
        with bb.function("main", [x]):
            with bb.dataflow():
                out = bb.emit_output(relax.op.matmul(x, weight, out_dtype="int32"))
            bb.emit_func_output(out)
        mod = bb.get()
        caps = GemminiCapabilities()
        from tvm.relax.backend.contrib.gemmini_tensorir import prepare_gemmini_tensorir_workloads

        prepared, _, _ = prepare_gemmini_tensorir_workloads(mod)
        tasks = ms.relax_integration.extract_tasks(prepared, caps.target())
        self.assertEqual(len(tasks), 1)
        workload = tasks[0].dispatched[0]
        sch = tir.Schedule(workload)
        sch = apply_contraction_sketch(
            sch,
            sch.get_block(contraction_names(workload["main"])[0]),
            ContractionSketch(),
            decisions={"i": [1, 4], "j": [1, 3], "k": [1, 4]},
        )
        for post in (ms.postproc.RewriteTensorize(),):
            post.apply(sch)
        with tempfile.TemporaryDirectory() as folder:
            database = ms.database.JSONDatabase(work_dir=folder)
            record = ms.database.TuningRecord(
                sch.trace, database.commit_workload(workload), [100.0], caps.target(), []
            )
            database.commit_tuning_record(record)
            report = []
            compiled = prepare_gemmini_tensorir_graph(mod, database=database, report=report)
        origins = {row.get("origin") for row in report if row["placement"] == "gemmini"}
        self.assertEqual(origins, {"TVM MetaSchedule database"})
        texts = [
            func.script() for func in compiled.functions.values() if isinstance(func, tir.PrimFunc)
        ]
        self.assertTrue(any("tvm_gemmini_dma_a" in text for text in texts))
        self.assertFalse(any("tvm_gemmini_dma_a_blocks" in text for text in texts))


def contraction_names(func):
    names = []
    tir.stmt_functor.post_order_visit(
        func.body,
        lambda node: names.append(node.name_hint)
        if isinstance(node, tir.Block) and node.init is not None
        else None,
    )
    return [name for name in names if contraction_info(func, name) is not None]


if __name__ == "__main__":
    unittest.main()
