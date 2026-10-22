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
"""ATen arithmetic through ordinary math, native scheduling and CPU retention."""

import unittest

import numpy as np
from tvm.relax.backend.contrib.gemmini import prepare_gemmini_graph

import test_backend_contrib_gemmini_tensorir as reference
import test_backend_contrib_gemmini_views as view_reference


class GemminiArithmeticTests(unittest.TestCase):
    setUp = reference.GemminiTensorIRTests.setUp
    static_run = view_reference.GemminiViewTests.static_run

    def test_native_contracts_and_cpu_numeric_fallback_keep_aten_results(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is unavailable")
        from tvm.relax.frontend.torch import from_exported_program

        for kind in (
            "mm",
            "bmm",
            "addmm",
            "add",
            "scalar_add",
            "subtract",
            "multiply",
            "divide",
            "alpha",
            "int64_add",
            "int64_mm",
        ):
            with self.subTest(kind=kind):

                class Arithmetic(torch.nn.Module):
                    def forward(self, x, y, bias):
                        if kind in ("mm", "int64_mm"):
                            return torch.mm(x, y)
                        if kind == "bmm":
                            return torch.bmm(x, y)
                        if kind == "addmm":
                            return torch.addmm(bias, x, y)
                        if kind in ("add", "int64_add"):
                            return torch.add(x, y)
                        if kind == "scalar_add":
                            return torch.ops.aten.add.Scalar(x, 17)
                        if kind == "subtract":
                            return torch.sub(x, y, alpha=3)
                        if kind == "multiply":
                            return torch.ops.aten.mul.Scalar(x, -3)
                        if kind == "divide":
                            return torch.ops.aten.div.Scalar(x, 3)
                        return torch.add(x, y, alpha=1.1)

                dtype = (
                    "int64"
                    if kind.startswith("int64")
                    else "float32"
                    if kind == "alpha"
                    else "int8"
                    if kind in ("mm", "bmm", "addmm")
                    else "int32"
                )
                matrix = kind in ("mm", "addmm", "int64_mm")
                shapes = (
                    ((2, 3), (3, 4), (4,))
                    if matrix
                    else (
                        ((2, 2, 3), (2, 3, 4), (4,)) if kind == "bmm" else ((3, 17), (3, 17), (17,))
                    )
                )
                seed = (
                    np.array([-3.75, 1.125, -0.0, 1, 0.25, -0.5], dtype)
                    if dtype == "float32"
                    else np.array([np.iinfo(dtype).min, np.iinfo(dtype).max, -1, 0, 1, 17], dtype)
                )
                values = [
                    np.resize(seed if i % 2 == 0 else seed[::-1], shape)
                    for i, shape in enumerate(shapes)
                ]
                model = Arithmetic()
                examples = [torch.from_numpy(value) for value in values]
                graph = from_exported_program(
                    torch.export.export(model, tuple(examples)), unwrap_unit_return_tuple=True
                )
                report = []
                prepared = prepare_gemmini_graph(graph, report=report)
                native = kind in ("mm", "bmm", "addmm", "add", "scalar_add")
                self.assertEqual(
                    sum(row["placement"] == "gemmini" for row in report), int(native), report
                )
                manifest, outputs, emulator = self.static_run(prepared, values)
                self.assertTrue(manifest["calls"])
                expected = model(*examples).numpy()
                self.assertEqual(outputs[0].dtype, expected.dtype)
                np.testing.assert_array_equal(
                    outputs[0].reshape(-1).view("uint8"), expected.reshape(-1).view("uint8")
                )
                self.assertEqual(emulator.emulator_pending(), 0)
                if native:
                    self.assertGreater(emulator.emulator_commands(), 0)
                else:
                    self.assertEqual(emulator.emulator_commands(), 0)
