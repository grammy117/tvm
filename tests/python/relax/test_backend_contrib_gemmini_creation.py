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
"""Standard constant folding and existing native literal DMA for creation ops."""

import unittest

import numpy as np

from tvm.relax.backend.contrib.gemmini import prepare_gemmini_graph
import test_backend_contrib_gemmini_tensorir as reference
import test_backend_contrib_gemmini_views as view_reference


class GemminiCreationTests(unittest.TestCase):
    setUp = reference.GemminiTensorIRTests.setUp
    static_run = view_reference.GemminiViewTests.static_run

    def test_full_and_full_like_fold_or_tensorize_without_new_kernels(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is unavailable")
        from tvm.relax.frontend.torch import from_exported_program

        for kind in ("full", "full_like"):
            for dtype, value in (("int8", -128), ("int32", -2147483648), ("float32", -0.0)):

                class Creation(torch.nn.Module):
                    def forward(self, x):
                        return (
                            torch.full((3, 17), value, dtype=getattr(torch, dtype))
                            if kind == "full"
                            else torch.full_like(x, value)
                        )

                model = Creation()
                values = np.arange(51).reshape(3, 17).astype(dtype)
                example = torch.from_numpy(values)
                graph = from_exported_program(
                    torch.export.export(model, (example,)), unwrap_unit_return_tuple=True
                )
                expected = model(example).numpy()
                for optimized in (False, True):
                    with self.subTest(kind=kind, dtype=dtype, optimized=optimized):
                        report = []
                        prepared = prepare_gemmini_graph(graph, optimize=optimized, report=report)
                        self.assertEqual(
                            sum(row["placement"] == "gemmini" for row in report),
                            0,
                            report,
                        )
                        manifest, outputs, emulator = self.static_run(prepared, [values])
                        self.assertEqual(len(outputs), 1)
                        np.testing.assert_array_equal(
                            outputs[0].view("uint8"), expected.view("uint8")
                        )
                        if optimized:
                            self.assertFalse(manifest["calls"])
                            self.assertEqual(emulator.emulator_commands(), 0)
                        else:
                            self.assertEqual(emulator.emulator_commands(), 0)
