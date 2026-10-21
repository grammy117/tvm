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
"""Bounded integer RNE agrees at every half tie and saturation/NaN boundary."""

import numpy as np
import pytest
import tvm
from tvm import te
from tvm.relax.frontend.torch._quantization import quantize_per_tensor
from tvm.tir.transform.rounding import lower_bounded_rne


@pytest.mark.parametrize("zero", [-128, -1, 0, 1, 127])
def test_hardware_integer_rne_preserves_clamped_quantization(zero):
    ties = np.arange(-520, 521, dtype="float32") / 2
    values = np.concatenate(
        [
            ties,
            np.nextafter(ties, np.float32(-np.inf)),
            np.nextafter(ties, np.float32(np.inf)),
            np.array([np.nan, np.inf, -np.inf, 1e30, -1e30], "float32"),
        ]
    )
    x = te.placeholder(values.shape, "float32", "x")
    y = quantize_per_tensor(x, 1.0, zero, -128, 127, "int8")
    source = te.create_prim_func([x, y])
    lowered, count = lower_bounded_rne(source)
    assert count == 1
    actual = tvm.nd.empty(values.shape, "int8")
    tvm.build(lowered, target="llvm")(tvm.nd.array(values), actual)
    with np.errstate(invalid="ignore"):
        expected = np.clip(np.rint(values) + zero, -128, 127).astype("int8")
    expected[np.isnan(values)] = 0
    np.testing.assert_array_equal(actual.numpy(), expected)
    machine = tvm.build(
        lowered,
        target="llvm -mtriple=riscv64-unknown-elf -mcpu=generic-rv64 -mattr=+m,+a,+f,+d,+c -mabi=lp64d",
    )
    assembly = machine.get_source("asm")
    assert "fcvt.l.s" in assembly or "fcvt.w.s" in assembly
    assert "rintf" not in assembly
