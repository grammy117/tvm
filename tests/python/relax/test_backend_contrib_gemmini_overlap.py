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
"""Exact CPU overlap math retained through Gemmini preparation and static export."""

import unittest

import numpy as np
from tvm.relax.backend.contrib.gemmini import prepare_gemmini_graph

import test_backend_contrib_gemmini_tensorir as reference
import test_backend_contrib_gemmini_views as view_reference


class GemminiOverlapTests(unittest.TestCase):
    setUp = reference.GemminiTensorIRTests.setUp
    static_run = view_reference.GemminiViewTests.static_run

    def test_gradient_and_overlap_arithmetic_remains_cpu_work(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is unavailable")
        from tvm.relax.frontend.torch import from_exported_program

        for kind in ("average", "override", "adaptive", "col2im", "boolean", "integer"):
            with self.subTest(kind=kind):

                class Overlap(torch.nn.Module):
                    def forward(self, g, x):
                        if kind in ("col2im", "boolean"):
                            return torch.ops.aten.col2im.default(
                                g, [4, 5], [2, 3], [1, 1], [0, 0], [1, 1]
                            )
                        if kind == "adaptive":
                            return torch.ops.aten._adaptive_avg_pool2d_backward.default(g, x)
                        return torch.ops.aten.avg_pool2d_backward.default(
                            g,
                            x,
                            [3, 3],
                            [2, 2],
                            [1, 1],
                            True,
                            True,
                            5 if kind == "override" else 1 if kind == "integer" else None,
                        )

                dtype = "bool" if kind == "boolean" else "int64" if kind == "integer" else "float32"
                shape = (
                    (1, 6, 9)
                    if kind in ("col2im", "boolean")
                    else (1, 1, 2, 3)
                    if kind == "adaptive"
                    else (1, 1, 3, 4)
                )
                gradient = (np.arange(np.prod(shape)).reshape(shape) % 19 - 11).astype(dtype)
                if kind == "integer":
                    gradient = np.resize(
                        np.array([-(1 << 63), (1 << 63) - 1, -17, 33], "int64"), shape
                    )
                values = np.zeros((1, 1, 5, 6), dtype)
                model = Overlap()
                args = (torch.from_numpy(gradient), torch.from_numpy(values))
                graph = from_exported_program(
                    torch.export.export(model, args), unwrap_unit_return_tuple=True
                )
                report = []
                graph = prepare_gemmini_graph(graph, report=report)
                self.assertFalse(any(row["placement"] == "gemmini" for row in report), report)
                _, outputs, emulator = self.static_run(graph, [gradient, values])
                expected = model(*args).numpy()
                self.assertEqual(outputs[0].shape, expected.shape)
                self.assertEqual(outputs[0].dtype, expected.dtype)
                if dtype in ("bool", "int64"):
                    np.testing.assert_array_equal(outputs[0], expected)
                else:
                    np.testing.assert_allclose(outputs[0], expected, rtol=1e-6, atol=1e-7)
                self.assertEqual(emulator.emulator_commands(), 0)
                self.assertEqual(emulator.emulator_pending(), 0)
