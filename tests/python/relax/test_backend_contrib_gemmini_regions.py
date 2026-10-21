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
"""Region scheduling and DMA completion for overlapping mathematical writes."""

import unittest

import numpy as np
import tvm
from tvm import relax, tir, meta_schedule as ms
from tvm.relax.backend.contrib.gemmini import prepare_gemmini_graph
from tvm.relax.backend.contrib.gemmini_elementwise import make_elementwise_schedule
from tvm.relax.backend.contrib.gemmini_meta_schedule import gemmini_space_generator
from tvm.relax.backend.contrib.gemmini_target import GemminiCapabilities
from tvm.relax.backend.contrib.gemmini_tensorir import lower_tensorir_schedule
from tvm.tir.schedule.testing import verify_trace_roundtrip

import test_backend_contrib_gemmini_tensorir as reference


def padding_graph(shape, dtype, widths, value):
    source = relax.Var("source", relax.TensorStructInfo(shape, dtype))
    builder = relax.BlockBuilder()
    with builder.function("main", [source]):
        with builder.dataflow():
            output = builder.emit_output(
                relax.op.nn.pad(source, tuple(x for pair in widths for x in pair), value)
            )
        builder.emit_func_output(output)
    return builder.get()


class GemminiRegionTests(unittest.TestCase):
    setUp = reference.GemminiTensorIRTests.setUp
    build = reference.GemminiTensorIRTests.build

    def test_constant_padding_preserves_dtype_bits_offsets_and_trace(self):
        for shape, widths in (
            ((4,), ((1, 2),)),
            ((3, 4), ((1, 1), (2, 1))),
            ((1, 2, 3), ((0, 0), (1, 2), (2, 0))),
            ((3, 4), ((-1, 1), (2, -1))),
        ):
            for dtype, value in (
                ("int8", -128),
                ("int8", 127),
                ("int32", -2147483648),
                ("int32", 2147483647),
                ("float32", -3.5),
                ("float32", -0.0),
                ("float32", 0.0),
                ("float32", float("nan")),
                ("float32", float("-nan")),
                ("float32", float("inf")),
                ("float32", float("-inf")),
            ):
                with self.subTest(shape=shape, dtype=dtype, value=value):
                    values = (
                        np.resize(
                            np.array(
                                [0, 0x80000000, 0x7FC12345, 0x7F812345, 1, 0xFF800000], "uint32"
                            ),
                            shape,
                        ).view("float32")
                        if dtype == "float32"
                        else (np.arange(np.prod(shape)).reshape(shape) * 17 - 128).astype(dtype)
                    )
                    expected = np.pad(
                        values, [(max(a, 0), max(b, 0)) for a, b in widths], constant_values=value
                    )
                    expected = expected[
                        tuple(
                            slice(max(-a, 0), extent - max(-b, 0))
                            for (a, b), extent in zip(widths, expected.shape)
                        )
                    ].copy()
                    graph = padding_graph(shape, dtype, widths, value)
                    report = []
                    prepared = prepare_gemmini_graph(graph, report=report)
                    self.assertEqual(sum(r["placement"] == "gemmini" for r in report), 1, report)
                    device = next(
                        f for f in prepared.functions.values() if isinstance(f, tir.PrimFunc)
                    )
                    runtime, control = self.build(
                        tvm.IRModule({"main": device.with_attr("global_symbol", "main")})
                    )
                    source, actual = tvm.nd.array(values), tvm.nd.empty(expected.shape, dtype)
                    runtime(source, actual)
                    np.testing.assert_array_equal(
                        actual.numpy().view("uint8"), expected.view("uint8")
                    )
                    np.testing.assert_array_equal(
                        source.numpy().view("uint8"), values.view("uint8")
                    )
                    self.assertGreater(control.emulator_commands(), 0)
                    self.assertEqual(control.emulator_faults(), 0)
                    self.assertEqual(control.emulator_pending(), 0)
                    legalized = relax.transform.LegalizeOps()(graph)
                    original = next(
                        f for f in legalized.functions.values() if isinstance(f, tir.PrimFunc)
                    )
                    saved = tvm.ir.save_json(original)
                    result = make_elementwise_schedule(original)
                    replay = tir.Schedule(original, debug_mask="all")
                    result.trace.apply_to_schedule(replay, remove_postproc=False)
                    verify_trace_roundtrip(replay, original)
                    self.assertEqual(tvm.ir.save_json(original), saved)
                    self.assertEqual(len(result.metadata["regions"]), 2)
                    calls = []
                    tir.stmt_functor.post_order_visit(
                        device.body,
                        lambda n: calls.append(n)
                        if isinstance(n, tir.Call) and n.op == tvm.ir.Op.get("tir.call_extern")
                        else None,
                    )
                    stores = [
                        c
                        for c in calls
                        if c.args[0].value in ("tvm_gemmini_dma_out", "tvm_gemmini_dma_spad_out")
                    ]
                    self.assertEqual(len(stores), 2)
                    self.assertTrue(all(int(c.args[-1]) == 1 for c in stores))

    def test_constant_padding_uses_standard_metaschedule_candidates(self):
        graph = padding_graph((3, 4), "float32", ((1, 1), (2, 1)), -0.0)
        legalized = relax.transform.LegalizeOps()(graph)
        original = next(f for f in legalized.functions.values() if isinstance(f, tir.PrimFunc))
        context = ms.TuneContext(
            mod=original.with_attr("global_symbol", "main"),
            target=GemminiCapabilities().target(),
            space_generator=gemmini_space_generator(),
            num_threads=1,
        )
        bits = np.resize(np.array([0x7F812345, 0x80000000, 1, 0xFF800000], "uint32"), (3, 4))
        values = bits.view("float32")
        expected = np.pad(values, ((1, 1), (2, 1)), constant_values=-0.0)
        legal = 0
        for design in context.generate_design_space():
            design.enter_postproc()
            if not all(p.apply(design) for p in context.space_generator.postprocs):
                continue
            result = lower_tensorir_schedule(
                context.mod["main"], design, workload_metadata={"operation": "pointwise_regions"}
            )
            runtime, control = self.build(result.scheduled_mod)
            actual = tvm.nd.empty(expected.shape, "float32")
            runtime(tvm.nd.array(values), actual)
            np.testing.assert_array_equal(actual.numpy().view("uint32"), expected.view("uint32"))
            self.assertEqual(control.emulator_faults(), 0)
            legal += 1
        self.assertGreaterEqual(legal, 1)

    def test_overlapping_region_writes_require_completion(self):
        graph = relax.transform.LegalizeOps()(
            padding_graph((3, 4), "int32", ((1, 1), (2, 0)), -123)
        )
        func = next(f for f in graph.functions.values() if isinstance(f, tir.PrimFunc))
        result = make_elementwise_schedule(func.with_attr("global_symbol", "main"))
        values = np.arange(12, dtype="int32").reshape(3, 4)
        expected = np.pad(values, ((1, 1), (2, 0)), constant_values=-123)
        runtime, control = self.build(result.scheduled_mod)
        control.emulator_reverse_stores(1)
        output = tvm.nd.empty(expected.shape, "int32")
        runtime(tvm.nd.array(values), output)
        np.testing.assert_array_equal(output.numpy(), expected)
        self.assertEqual(control.emulator_faults(), 0)

        def remove_wait(node):
            if (
                isinstance(node, tir.Call)
                and node.op == tvm.ir.Op.get("tir.call_extern")
                and node.args[0].value == "tvm_gemmini_dma_out"
            ):
                return tir.call_extern(
                    "void", node.args[0].value, *node.args[1:-1], tir.IntImm("int32", 0)
                )
            return None

        device = result.scheduled_mod["main"]
        broken = device.with_body(tir.stmt_functor.ir_transform(device.body, None, remove_wait))
        runtime, control = self.build(tvm.IRModule({"main": broken}))
        control.emulator_reverse_stores(1)
        runtime(tvm.nd.array(values), output)
        self.assertFalse(np.array_equal(output.numpy(), expected))
