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
"""Readout equality needs exact rounding proofs, independent of fixture values."""

import numpy as np
import pytest
import tvm
from tvm import tir
from tvm.relax.backend.contrib.gemmini_epilogue import prove_readout


def expression(factors, relu=True):
    accumulator = tir.decl_buffer((1,), "int32", scope="local.gemmini_acc")[0]
    value = accumulator.astype("float32")
    if relu:
        value = tir.max(value, tir.const(0, "float32"))
    for factor in factors:
        value = value * tir.const(factor, "float32")
    result = tir.min(
        tir.max(tir.nearbyint(value), tir.const(-128, "float32")), tir.const(127, "float32")
    ).astype("int8")
    return tir.if_then_else(tir.isnan(value), tir.const(0, "int8"), result), accumulator


@pytest.mark.parametrize("factors", [(1.0,), (0.0031129,), (0.125, 0.123456)])
@pytest.mark.parametrize("relu", [True, False])
def test_exact_readout_threshold_proof_covers_full_int32_domain(factors, relu):
    expr, accumulator = expression(factors, relu)
    proof = prove_readout(expr, accumulator)
    assert proof["input_domain"] == [-(1 << 31), (1 << 31) - 1]
    assert proof["transitions_checked"] == (127 if relu else 255)
    assert len(proof["thresholds_sha256"]) == 64


def test_positive_double_rounding_counterexample_is_rejected():
    expr, accumulator = expression((0.1, 0.1))
    assert np.rint(np.float32(np.float32(50) * np.float32(0.1)) * np.float32(0.1)) == 0
    assert np.rint(np.float32(50) * np.float32(np.float32(0.1) * np.float32(0.1))) == 1
    with pytest.raises(ValueError, match="different integer rounding thresholds"):
        prove_readout(expr, accumulator)


def test_readout_scale_keeps_functional_model_integer_conversion_defined():
    expr, accumulator = expression((float(1 << 32),))
    with pytest.raises(ValueError, match="INT64 conversion bound"):
        prove_readout(expr, accumulator)


def test_nonmonotone_and_source_rounding_variants_are_refused():
    expr, accumulator = expression((-0.1,))
    with pytest.raises(ValueError, match="positive finite"):
        prove_readout(expr, accumulator)
    expr, accumulator = expression((1.0,))

    def change(node):
        if isinstance(node, tir.Call) and node.op == tvm.ir.Op.get("tir.nearbyint"):
            return tir.round(node.args[0])
        return None

    other = tir.stmt_functor.ir_transform(tir.Evaluate(expr), None, change, ["tir.Call"]).value
    with pytest.raises(ValueError, match="round-to-nearest-even"):
        prove_readout(other, accumulator)


@pytest.mark.parametrize("shift", [0, 1, 4, 16, 30])
def test_clip_round_even_power_of_two_division_proves_relu_readout(shift):
    accumulator = tir.decl_buffer((1,), "int32", scope="local.gemmini_acc")[0]
    value = tir.Div(accumulator.astype("float32"), tir.const(float(1 << shift), "float32"))
    result = tir.min(tir.max(tir.nearbyint(value), tir.const(0, "float32")),
                     tir.const(127, "float32")).astype("int8")
    proof = prove_readout(result, accumulator)
    assert proof["relu"]
    assert proof["transitions_checked"] == 127
