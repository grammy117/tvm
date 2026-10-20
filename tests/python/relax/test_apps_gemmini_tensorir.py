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
"""Guest-result classification controls for bounded TensorIR verification."""

from pathlib import Path
import sys
import unittest

import tvm

sys.path.insert(0, str(Path(tvm.__file__).resolve().parents[2] / "apps/gemmini"))
from verify_tensorir import classify_output


class TensorIRResultTests(unittest.TestCase):
    def test_rtl_negative_decodes_whitespace_and_requires_exact_guest_code(self):
        observed = (
            "TENSORIR_NUMERICAL_FAILURE\n"
            "Assertion failed: *** FAILED *** (exit code =         22)\n"
        )
        self.assertTrue(classify_output("verilator", observed, 255, False, True))
        for text, code, timeout in (
            (observed.replace("22", "220"), 255, False),
            (observed, 0, False),
            (observed, 255, True),
            (observed.replace("TENSORIR_NUMERICAL_FAILURE", ""), 255, False),
        ):
            self.assertFalse(classify_output("verilator", text, code, timeout, True))

    def test_positive_requires_completion_and_negative_spike_requires_guest_22(self):
        marker = "TENSORIR_PASS cases=4 exact=1 guards=1 primitives=1\n"
        self.assertTrue(classify_output("spike", marker, 0, False))
        self.assertFalse(classify_output("verilator", marker, 0, False))
        self.assertTrue(classify_output("verilator", marker + "Verilog $finish", 0, False))
        self.assertFalse(classify_output("verilator", marker + "Verilog $finish", 0, True))
        self.assertFalse(classify_output("spike", "", 0, False))
        self.assertTrue(classify_output("spike", "TENSORIR_NUMERICAL_FAILURE", 22, False, True))
        self.assertFalse(classify_output("spike", "TENSORIR_NUMERICAL_FAILURE", 1, False, True))
        two_cases = marker.replace("cases=4", "cases=2")
        self.assertTrue(classify_output("spike", two_cases, 0, False, expected_cases=2))
        self.assertFalse(classify_output("spike", two_cases, 0, False))
        self.assertFalse(classify_output("spike", marker.replace("cases=4", "cases=40"), 0, False))


if __name__ == "__main__":
    unittest.main()
