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
"""Convolution import composed with structural scheduling and static deployment."""

import unittest

import numpy as np
import tvm
from tvm import te, tir
from tvm.relax.backend.contrib.gemmini import prepare_gemmini_graph
from tvm.relax.backend.contrib.gemmini_tensorir import make_tensorir_schedule
from tvm.tir.schedule.testing import verify_trace_roundtrip

import test_backend_contrib_gemmini_tensorir as reference
import test_backend_contrib_gemmini_views as view_reference
import test_frontend_from_exported_program_convolution as frontend


class GemminiConvolutionFrontendTests(unittest.TestCase):
    setUp = reference.GemminiTensorIRTests.setUp
    build = reference.GemminiTensorIRTests.build
    static_run = view_reference.GemminiViewTests.static_run

    def test_epilogue_placement_handles_independent_and_reduction_bias_producers(self):
        for reduced in (False, True):
            with self.subTest(reduced=reduced):
                a, b = te.placeholder((17, 11), "int8", "a"), te.placeholder((11, 19), "int8", "b")
                source = te.placeholder(
                    (2, 19) if reduced else (19,), "int32" if reduced else "int8", "bias"
                )
                k = te.reduce_axis((0, 11), "k")
                product = te.compute(
                    (17, 19),
                    lambda i, j: te.sum(a[i, k].astype("int32") * b[k, j].astype("int32"), k),
                    name="product",
                )
                if reduced:
                    axis = te.reduce_axis((0, 2), "bias_axis")
                    bias = te.compute(
                        (19,), lambda j: te.sum(source[axis, j], axis), name="bias_sum"
                    )
                else:
                    bias = te.compute((19,), lambda j: source[j].astype("int32"), name="bias_widen")
                added = te.compute((17, 19), lambda i, j: product[i, j] + bias[j], name="biased")
                output = te.compute(
                    (17, 19), lambda i, j: added[i, j].astype("int8"), name="narrow"
                )
                func = te.create_prim_func([a, b, source, output])
                original = tvm.ir.save_json(func)
                scheduled = make_tensorir_schedule(func)
                replay = tir.Schedule(func, debug_mask="all")
                scheduled.trace.apply_to_schedule(replay, remove_postproc=False)
                verify_trace_roundtrip(replay, tvm.IRModule({"main": func}))
                self.assertEqual(tvm.ir.save_json(func), original)
                if reduced:
                    self.assertGreaterEqual(scheduled.metadata["cpu_reduction_blocks"], 1)
                values = (np.arange(187).reshape(17, 11) * 37 - 128).astype("int8")
                weights = (np.arange(209).reshape(11, 19) * 29 - 128).astype("int8")
                initial = (
                    np.arange(38 if reduced else 19).reshape((2, 19) if reduced else (19,)) * 3 - 31
                ).astype("int32" if reduced else "int8")
                expected = (
                    values.astype("int64") @ weights.astype("int64")
                    + (initial.sum(axis=0) if reduced else initial)
                ).astype("int8")
                runtime, emulator = self.build(scheduled.scheduled_mod)
                actual = tvm.nd.empty(expected.shape, "int8")
                runtime(tvm.nd.array(values), tvm.nd.array(weights), tvm.nd.array(initial), actual)
                np.testing.assert_array_equal(actual.numpy(), expected)
                self.assertEqual(emulator.emulator_pending(), 0)
                self.assertEqual(emulator.emulator_faults(), 0)

    def test_direct_integer_convolutions_use_native_contractions(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is unavailable")
        from tvm.relax.frontend.torch import from_exported_program

        for dim in (1, 2, 3):
            with self.subTest(dim=dim):
                shape, w_shape = (1, 2) + (3,) * dim, (3, 2) + (2,) * dim
                values = (np.arange(np.prod(shape)).reshape(shape) * 37 - 128).astype("int8")
                weights = (np.arange(np.prod(w_shape)).reshape(w_shape) * 29 - 128).astype("int8")
                bias = np.array([-128, 127, 17], "int8")
                model = frontend.model_forward(dim, False, True)
                args = tuple(torch.from_numpy(array) for array in (values, weights, bias))
                graph = from_exported_program(
                    torch.export.export(model, args), unwrap_unit_return_tuple=True
                )
                report = []
                graph = prepare_gemmini_graph(graph, report=report)
                selected = [row for row in report if row["placement"] == "gemmini"]
                self.assertTrue(any("k" in row["metadata"] for row in selected), report)
                _, outputs, emulator = self.static_run(graph, [values, weights, bias])
                np.testing.assert_array_equal(outputs[0], model(*args).numpy())
                self.assertEqual(outputs[0].dtype, np.dtype("int8"))
                self.assertGreater(emulator.emulator_commands(), 0)
                self.assertEqual(emulator.emulator_pending(), 0)

    def test_full_width_and_floating_gradient_contracts_keep_cpu_arithmetic(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is unavailable")
        from tvm.relax.frontend.torch import from_exported_program

        for kind in (
            "integer",
            "transpose",
            "float64",
            "gradient2d",
            "gradient3d",
            "transpose_gradient",
        ):
            with self.subTest(kind=kind):
                gradient_case = "gradient" in kind
                dim = 3 if kind == "gradient3d" else 2
                dtype = (
                    "int64"
                    if kind in ("integer", "transpose")
                    else "float64"
                    if kind == "float64"
                    else "float32"
                )
                shape, w_shape = (1, 2) + (3,) * dim, (2, 2) + (2,) * dim
                values = (np.arange(np.prod(shape)).reshape(shape) % 13 - 7).astype(dtype)
                weights = (np.arange(np.prod(w_shape)).reshape(w_shape) % 7 - 4).astype(dtype)
                if dtype == "int64":
                    seed = np.array([-(1 << 63), (1 << 63) - 1, -17, 33], "int64")
                    values, weights = np.resize(seed, shape), np.resize(seed[::-1], w_shape)
                transpose = kind in ("transpose", "transpose_gradient")
                if gradient_case:
                    op = getattr(
                        torch.nn.functional, f"conv{'_transpose' if transpose else ''}{dim}d"
                    )
                    output = op(torch.from_numpy(values), torch.from_numpy(weights), padding=1)
                    gradient = (np.arange(output.numel()).reshape(output.shape) % 11 - 6).astype(
                        dtype
                    )
                    arrays = [gradient, values, weights]
                    model = frontend.model_backward(dim, transpose, 1, 1, 1)
                else:
                    arrays = [values, weights, np.array([-17, 23], dtype)]
                    model = frontend.model_forward(dim, transpose, True)
                args = tuple(torch.from_numpy(array) for array in arrays)
                expected = model(*args)
                expected = list(expected) if isinstance(expected, tuple) else [expected]
                graph = from_exported_program(
                    torch.export.export(model, args), unwrap_unit_return_tuple=True
                )
                report = []
                graph = prepare_gemmini_graph(graph, report=report)
                compute = []
                for func in graph.functions.values():
                    if isinstance(func, tir.PrimFunc):
                        tir.stmt_functor.post_order_visit(
                            func.body,
                            lambda node: compute.append(node)
                            if isinstance(node, tir.Call)
                            and node.op == tvm.ir.Op.get("tir.call_extern")
                            and node.args[0].value == "tvm_gemmini_compute"
                            else None,
                        )
                self.assertFalse(compute, report)
                _, outputs, emulator = self.static_run(graph, arrays)
                self.assertEqual(len(outputs), len(expected))
                for actual, reference in zip(outputs, expected):
                    reference = reference.numpy()
                    self.assertEqual(actual.dtype, reference.dtype)
                    if dtype == "int64":
                        np.testing.assert_array_equal(actual, reference)
                    else:
                        np.testing.assert_allclose(actual, reference, rtol=1e-6, atol=1e-6)
                self.assertEqual(emulator.emulator_pending(), 0)
                if dtype in ("int64", "float64"):
                    self.assertEqual(emulator.emulator_commands(), 0)
