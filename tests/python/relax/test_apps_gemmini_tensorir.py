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
import subprocess
import tempfile
import unittest

import numpy as np
import tvm

sys.path.insert(0, str(Path(tvm.__file__).resolve().parents[2] / "apps/gemmini"))
from verify_tensorir import classify_output, fixture_guest


class TensorIRResultTests(unittest.TestCase):
    def test_timing_parser_requires_complete_unique_positive_guest_intervals(self):
        from verify_tensorir import parse_cycle_samples

        text = "DSE_CYCLES stage=0 sample=0 cycles=17\nDSE_CYCLES stage=0 sample=1 cycles=19\n"
        self.assertEqual(parse_cycle_samples(text, case_count=1, repeats=2), [[17, 19]])
        for invalid in (text.splitlines()[0], text + text, text.replace("cycles=17", "cycles=0")):
            with self.assertRaises(ValueError):
                parse_cycle_samples(invalid, case_count=1, repeats=2)
        with self.assertRaises(ValueError):
            parse_cycle_samples(text, case_count=1, repeats=2, cycle_limit=18)

    def test_float_tolerance_and_timed_repeats_keep_wrong_oracle_detection(self):
        declarations, checks = fixture_guest(
            0, [np.array([3], "int32")], np.array([1.25], "float32"), 0, None,
            "host", timing_repeats=2, rtol=1e-5, atol=1e-6,
        )
        source = r"""
#include <stdint.h>
#include <stddef.h>
#include <string.h>
static unsigned computes,zeros,dmas,accumulates,acc_writes,relus,calls,samples;
static void puts_htif(const char* text) {(void)text;}
static uint64_t hash(const void* p,size_t n) {
  const unsigned char* b=p;uint64_t h=0;for(size_t i=0;i<n;++i)h=h*257+b[i];return h;
}
static uint64_t dse_cycles(void) {static uint64_t count;return ++count;}
static void dse_sample(unsigned a,unsigned b,uint64_t value) {(void)a;(void)b;if(value)++samples;}
int stage_0_model_run(const void*const* in,void*const* out,uint8_t* work,uint64_t size) {
  (void)in;(void)work;(void)size;++calls;((uint32_t*)out[0])[0]=0x3fa00001;return 0;
}
""" + "\n".join(declarations) + "\nint main(void){\n" + "\n".join(checks) + "\nreturn calls==3 && samples==2 ? 0 : 99;}\n"
        with tempfile.TemporaryDirectory(prefix="tensorir-cycle-check-") as directory:
            root = Path(directory)
            path = root / "guest.c"
            path.write_text(source)
            for bad, expected_code in ((0, 0), (1, 22)):
                binary = root / f"guest-{bad}"
                subprocess.run(
                    ["cc", "-std=c11", "-O2", f"-DBAD_ORACLE={bad}", path, "-o", binary],
                    check=True, capture_output=True,
                )
                self.assertEqual(subprocess.run([binary], capture_output=True).returncode, expected_code)

    def test_mixed_movement_rejects_array_arithmetic_even_with_correct_outputs(self):
        declarations, checks = fixture_guest(
            0, [np.array([3], "int32")], np.array([1.25], "float32"), 0, "copy_f32", "mixed"
        )
        source = (
            r"""
#include <stdint.h>
#include <stddef.h>
#include <string.h>
static unsigned computes,zeros,dmas,accumulates,acc_writes,relus;
static void puts_htif(const char* text) {(void)text;}
static uint64_t hash(const void* p,size_t n) {
  const unsigned char* b=p;uint64_t h=0;for(size_t i=0;i<n;++i)h=h*257+b[i];return h;
}
int stage_0_model_run(const void*const* in,void*const* out,uint8_t* work,uint64_t size) {
  (void)in;(void)work;(void)size;
  ++dmas;computes+=BAD_COMPUTE;((uint32_t*)out[0])[0]=0x3fa00000;return 0;
}
"""
            + "\n".join(declarations)
            + "\nint main(void){\n"
            + "\n".join(checks)
            + "\nreturn 0;}\n"
        )
        with tempfile.TemporaryDirectory(prefix="tensorir-mixed-check-") as directory:
            root = Path(directory)
            path = root / "guest.c"
            path.write_text(source)
            for compute, expected_code in ((0, 0), (1, 21)):
                binary = root / f"guest-{compute}"
                subprocess.run(
                    [
                        "cc",
                        "-std=c11",
                        "-O2",
                        "-DBAD_ORACLE=0",
                        f"-DBAD_COMPUTE={compute}",
                        path,
                        "-o",
                        binary,
                    ],
                    check=True,
                    capture_output=True,
                )
                result = subprocess.run([binary], capture_output=True)
                self.assertEqual(result.returncode, expected_code)

    def test_guest_checks_second_output_bits_and_guards(self):
        expected = (
            np.array([11, -7], "int32"),
            np.array([0x80000000, 0x7FC12345], "uint32").view("float32"),
        )
        declarations, checks = fixture_guest(0, [np.array([3], "int32")], expected, 0, None, "host")
        source = (
            r"""
#include <stdint.h>
#include <stddef.h>
#include <string.h>
static unsigned computes,zeros,dmas,accumulates,acc_writes,relus;
static void puts_htif(const char* text) { (void)text; }
static uint64_t hash(const void* p,size_t size) {
  const unsigned char* data=p;uint64_t value=0;
  for(size_t i=0;i<size;++i)value=value*257+data[i];return value;
}
int stage_0_model_run(const void*const*,void*const*,uint8_t*,uint64_t);
"""
            + "\n".join(declarations)
            + "\nint main(void){\n"
            + "\n".join(checks)
            + "\nreturn 0;}\n"
        )
        with tempfile.TemporaryDirectory(prefix="tensorir-guest-check-") as directory:
            root = Path(directory)
            path = root / "guest.c"
            path.write_text(source)
            stub_path = root / "stub.c"
            stub_path.write_text(r"""
#include <stdint.h>
int stage_0_model_run(const void*const* in,void*const* out,uint8_t* work,uint64_t size) {
  (void)in;(void)work;(void)size;
  ((int32_t*)out[0])[0]=11;((int32_t*)out[0])[1]=-7;
  ((uint32_t*)out[1])[0]=0x80000000;((uint32_t*)out[1])[1]=0x7FC12345^BAD_SECOND;
  if(BAD_GUARD)((volatile unsigned char*)out[1])[8]=0;
  return 0;
}
""")
            for oracle, second, guard, expected_code in (
                (0, 0, 0, 0),
                (0, 1, 0, 22),
                (1, 0, 0, 22),
                (0, 0, 1, 23),
            ):
                binary = root / f"guest-{oracle}-{second}-{guard}"
                subprocess.run(
                    [
                        "cc",
                        "-std=c11",
                        "-O2",
                        f"-DBAD_ORACLE={oracle}",
                        f"-DBAD_SECOND={second}",
                        f"-DBAD_GUARD={guard}",
                        path,
                        stub_path,
                        "-o",
                        binary,
                    ],
                    check=True,
                    capture_output=True,
                )
                result = subprocess.run([binary], capture_output=True)
                self.assertEqual(result.returncode, expected_code)

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
        tolerant = marker.replace("exact=1", "exact=0")
        self.assertTrue(classify_output("spike", tolerant, 0, False, exact=False))
        self.assertFalse(classify_output("spike", tolerant, 0, False))
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
