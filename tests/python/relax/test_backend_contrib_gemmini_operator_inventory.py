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
"""Execute every requested ATen spelling through the actual deployment pipeline."""

from pathlib import Path
import sys
import unittest

import numpy as np
import tvm
from tvm import tir
from tvm.relax.backend.contrib.gemmini import prepare_gemmini_graph

import test_backend_contrib_gemmini_tensorir as reference
import test_backend_contrib_gemmini_views as view_reference

sys.path.insert(0, str(Path(tvm.__file__).resolve().parents[2] / "apps/gemmini"))
from operator_contracts import REQUESTED_OPERATORS, fixtures


class GemminiOperatorInventoryTests(unittest.TestCase):
    setUp = reference.GemminiTensorIRTests.setUp
    static_run = view_reference.GemminiViewTests.static_run

    def test_all_requested_spellings_preserve_reference_values_and_execution_kind(self):
        from tvm.relax.frontend.torch.exported_program_translator import ExportedProgramImporter

        mapping = ExportedProgramImporter().convert_map
        seen = []
        for case in fixtures():
            name, graph, arrays, expected, placement, options, selected_count, contract = case
            with self.subTest(name=name):
                op = contract["requested_operator"]
                seen.append(op)
                self.assertIn(contract["converter_key"], mapping)
                self.assertIn("aten." + contract["converter_key"], contract["source_ops"])
                report = []
                graph = prepare_gemmini_graph(graph, report=report, **options)
                selected = [row for row in report if row["placement"] == "gemmini"]
                self.assertEqual(len(selected), selected_count, report)
                computes = []
                for func in graph.functions.values():
                    if isinstance(func, tir.PrimFunc):
                        tir.stmt_functor.post_order_visit(
                            func.body,
                            lambda node: computes.append(node)
                            if isinstance(node, tir.Call)
                            and node.op == tvm.ir.Op.get("tir.call_extern")
                            and node.args[0].value == "tvm_gemmini_compute"
                            else None,
                        )
                self.assertEqual(bool(computes), contract["execution_kind"] == "array", report)
                _, outputs, emulator = self.static_run(graph, arrays)
                expected = expected if isinstance(expected, tuple) else (expected,)
                self.assertEqual(len(outputs), len(expected))
                for actual, original in zip(outputs, expected):
                    self.assertEqual(actual.dtype, original.dtype)
                    self.assertEqual(actual.shape, original.shape)
                    if original.dtype.kind in "iu":
                        np.testing.assert_array_equal(actual, original)
                    else:
                        np.testing.assert_allclose(actual, original, rtol=1e-6, atol=1e-6)
                self.assertEqual(emulator.emulator_pending(), 0)
                self.assertEqual(emulator.emulator_commands() > 0, placement != "host")
        self.assertEqual(tuple(seen), tuple(REQUESTED_OPERATORS))
        self.assertEqual(len(set(seen)), 51)
