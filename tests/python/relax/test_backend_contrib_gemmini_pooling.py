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
"""Pooling arithmetic on CPU composed with independently scheduled DMA movement."""

import unittest

import numpy as np
import tvm
from tvm import relax, tir
from tvm.relax.backend.contrib.gemmini import prepare_gemmini_graph
from tvm.relax.backend.contrib.gemmini_elementwise import make_elementwise_schedule
from tvm.tir.schedule.testing import verify_trace_roundtrip

import test_backend_contrib_gemmini_tensorir as reference
import test_backend_contrib_gemmini_views as view_reference


class GemminiPoolingTests(unittest.TestCase):
    setUp = reference.GemminiTensorIRTests.setUp
    build = reference.GemminiTensorIRTests.build
    static_run = view_reference.GemminiViewTests.static_run

    def test_pooling_keeps_cpu_arithmetic_and_admits_only_legal_movement(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is unavailable")
        from tvm.relax.frontend.torch import from_exported_program

        for kind, dim in (
            ("regular", 1),
            ("regular", 2),
            ("regular", 3),
            ("crop", 2),
            ("override", 2),
            ("adaptive", 1),
            ("adaptive", 2),
            ("adaptive", 3),
            ("integer", 2),
        ):
            with self.subTest(kind=kind, dim=dim):

                class Pool(torch.nn.Module):
                    def forward(self, x):
                        if kind == "adaptive":
                            return getattr(torch.nn.functional, f"adaptive_avg_pool{dim}d")(
                                x, (2,) * dim
                            )
                        return (
                            getattr(torch.nn.functional, f"avg_pool{dim}d")(
                                x,
                                (2 if kind == "crop" else 3,) * dim,
                                (2,) * dim,
                                (1,) * dim,
                                ceil_mode=kind == "crop",
                                divisor_override=5 if kind == "override" else None,
                            )
                            if dim > 1
                            else torch.nn.functional.avg_pool1d(x, 3, 2, 1)
                        )

                shape = (1, 1) + (3 if kind == "crop" else 4,) * dim
                dtype = "int64" if kind == "integer" else "float32"
                values = (np.arange(np.prod(shape)).reshape(shape) % 19 - 10).astype(dtype)
                model = Pool()
                example = torch.from_numpy(values)
                graph = from_exported_program(
                    torch.export.export(model, (example,)), unwrap_unit_return_tuple=True
                )
                report = []
                prepared = prepare_gemmini_graph(graph, report=report)
                native = False  # Automatic payload-copy offload is disabled.
                self.assertEqual(
                    sum(r["placement"] == "gemmini" for r in report), int(native), report
                )
                if native:
                    selected = next(r for r in report if r["placement"] == "gemmini")
                    self.assertGreaterEqual(selected["metadata"]["cpu_reduction_blocks"], 1)
                _, outputs, emulator = self.static_run(prepared, [values])
                expected = model(example).numpy()
                np.testing.assert_allclose(outputs[0], expected, rtol=1e-6, atol=1e-7)
                self.assertEqual(outputs[0].dtype, expected.dtype)
                self.assertEqual(emulator.emulator_pending(), 0)
                if native:
                    self.assertGreater(emulator.emulator_commands(), 0)
                else:
                    self.assertEqual(emulator.emulator_commands(), 0)

    def test_trace_replay_and_cpu_reduction_require_dma_completion(self):
        x = relax.Var("x", relax.TensorStructInfo((1, 1, 3, 4), "float32"))
        builder = relax.BlockBuilder()
        with builder.function("main", [x]):
            with builder.dataflow():
                output = builder.emit_output(
                    relax.op.nn.avg_pool2d(x, (3, 3), (1, 2), 1, count_include_pad=True)
                )
            builder.emit_func_output(output)
        graph = relax.transform.LegalizeOps()(builder.get())
        original = next(
            f for f in graph.functions.values() if isinstance(f, tir.PrimFunc)
        ).with_attr("global_symbol", "main")
        result = make_elementwise_schedule(original)
        self.assertEqual(result.metadata["cpu_reduction_blocks"], 1)
        replay = tir.Schedule(original, debug_mask="all")
        result.trace.apply_to_schedule(replay, remove_postproc=False)
        verify_trace_roundtrip(replay, tvm.IRModule({"main": original}))
        values = (np.arange(12).reshape(1, 1, 3, 4) - 7).astype("float32")
        expected = np.empty((1, 1, 3, 2), "float32")
        for i in range(3):
            for j in range(2):
                expected[0, 0, i, j] = (
                    values[
                        0, 0, max(i - 1, 0) : min(i + 2, 3), max(j * 2 - 1, 0) : min(j * 2 + 2, 4)
                    ].sum()
                    / 9
                )
        runtime, control = self.build(result.scheduled_mod)
        output = tvm.nd.empty(expected.shape, "float32")
        runtime(tvm.nd.array(values), output)
        np.testing.assert_allclose(output.numpy(), expected, rtol=1e-6, atol=1e-7)
        removed = []

        def remove_wait(node):
            if (isinstance(node, tir.Call) and node.op == tvm.ir.Op.get("tir.call_extern")
                    and node.args[0].value == "tvm_gemmini_end"):
                removed.append(node)
                return tir.const(0, "int32")
            if (
                isinstance(node, tir.Call)
                and node.op == tvm.ir.Op.get("tir.call_extern")
                and node.args[0].value == "tvm_gemmini_dma_out"
                and int(node.args[-1])
            ):
                removed.append(node)
                return tir.call_extern(
                    "void", node.args[0].value, *node.args[1:-1], tir.IntImm("int32", 0)
                )
            return None

        device = result.scheduled_mod["main"]
        broken = device.with_body(tir.stmt_functor.ir_transform(device.body, None, remove_wait))
        self.assertTrue(removed)
        runtime, control = self.build(tvm.IRModule({"main": broken}))
        different = -values + 13
        for i in range(3):
            for j in range(2):
                expected[0, 0, i, j] = (
                    different[
                        0, 0, max(i - 1, 0) : min(i + 2, 3), max(j * 2 - 1, 0) : min(j * 2 + 2, 4)
                    ].sum()
                    / 9
                )
        runtime(tvm.nd.array(different), output)
        self.assertFalse(np.allclose(output.numpy(), expected, rtol=1e-6, atol=1e-7))
