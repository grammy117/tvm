# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied. See the License for the
# specific language governing permissions and limitations
# under the License.
"""Compile the emitted C checker and test actual IEEE-754 edge behavior."""
import ast
import ctypes
import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.fixture(scope="module")
def checker(tmp_path_factory):
    compiler = shutil.which("cc") or shutil.which("gcc")
    if compiler is None:
        pytest.skip("Host C compiler is required for checker semantics")
    # The override permits testing a scratch proposal before touching live guests.
    source = Path(os.environ.get("GEMMINI_CHECKER_SOURCE", Path(__file__).resolve().parents[3] / "apps/gemmini/measure.py"))
    tree = ast.parse(source.read_text())
    runner = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "RUNNER" for target in node.targets))
    compare = runner[runner.index("static int compare(unsigned index){"):runner.index("\nint main(void){")]
    preamble = r"""
#include <stdint.h>
#include <stddef.h>
#include <string.h>
_Static_assert(sizeof(float)==4, "IEEE single precision required");
static uint8_t outputs_area[4] __attribute__((aligned(64)));
static uint8_t expected_area[4] __attribute__((aligned(64)));
static const uint8_t *dse_expected=expected_area;
static const uint64_t output_offsets[]={0},expected_offsets[]={0},output_bytes[]={4};
static uint8_t output_kinds[]={1};
static int bad_oracle;
static float check_rtol,check_atol;
#define BAD_ORACLE bad_oracle
#define RTOL check_rtol
#define ATOL check_atol
"""
    wrapper = r"""
int check_bits(uint32_t actual,uint32_t expected,float rtol,float atol,int wrong,int kind){
  memcpy(outputs_area,&actual,4);memcpy(expected_area,&expected,4);
  check_rtol=rtol;check_atol=atol;bad_oracle=wrong;output_kinds[0]=(uint8_t)kind;
  return compare(0);
}
"""
    root = tmp_path_factory.mktemp("finite-checker")
    (root / "checker.c").write_text(preamble + compare + wrapper)
    subprocess.run([compiler, "-std=c11", "-O2", "-fPIC", "-shared", "-fno-fast-math",
                    "-ffp-contract=off", str(root / "checker.c"), "-o", str(root / "checker.so")],
                   check=True, capture_output=True)
    lib = ctypes.CDLL(str(root / "checker.so"))
    fn = lib.check_bits
    fn.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_int]
    fn.restype = ctypes.c_int
    return lambda actual, expected, rtol=1e-5, atol=2e-6, wrong=0, kind=1: fn(actual, expected, rtol, atol, wrong, kind)


@pytest.mark.parametrize("bits", [0, 0x80000000, 1, 0x007fffff, 0x00800000,
                                  0x3f800000, 0xbf800000, 0x7f7fffff, 0xff7fffff])
def test_equal_finite_values_pass(checker, bits):
    assert checker(bits, bits) == 0


@pytest.mark.parametrize("actual,expected", [(0, 0x80000000), (0x80000000, 0)])
def test_signed_zero_uses_equivalent_tolerance_semantics(checker, actual, expected):
    assert checker(actual, expected, rtol=0, atol=0) == 0


@pytest.mark.parametrize("actual,expected", [
    (0x7f800000, 0x7f7fffff),  # finite expected upper bound can overflow
    (0xff800000, 0xff7fffff),  # finite expected lower bound can overflow
    (0x7f800000, 0x7f800000), (0xff800000, 0xff800000),
    (0x7fc00001, 0x7fc00001), (0x7f800001, 0x7f800001),
    (0x7fc00001, 0x3f800000), (0x3f800000, 0x7fc00001),
    (0x3f800000, 0x7f800000),
])
def test_nan_and_infinity_never_pass(checker, actual, expected):
    assert checker(actual, expected) == 1


@pytest.mark.parametrize("actual,expected,rtol,atol,answer", [
    (0x3f800001, 0x3f800000, 1e-5, 2e-6, 0),
    (0x3f8147ae, 0x3f800000, 1e-5, 2e-6, 1),
    (1, 0, 0, 0, 1),
    (1, 0, 0, 1e-7, 0),
    (0x3f800000, 0x3f800000, -1, 0, 1),
])
def test_finite_mismatch_retains_tolerance_fallback(checker, actual, expected, rtol, atol, answer):
    assert checker(actual, expected, rtol=rtol, atol=atol) == answer


@pytest.mark.parametrize("bits", [0, 0x80000000, 1, 0x3f800000, 0x7f7fffff])
def test_wrong_oracle_mutation_precedes_fast_equality(checker, bits):
    assert checker(bits, bits, wrong=0) == 0
    assert checker(bits, bits, wrong=1) == 1


def test_integer_byte_comparison_unchanged(checker):
    assert checker(0x12345678, 0x12345678, kind=0) == 0
    assert checker(0x12345678, 0x12345678, wrong=1, kind=0) == 1
