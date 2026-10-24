# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Independent references reject unknown semantics and preserve arithmetic boundaries."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import tvm
from tvm import relax

sys.path.insert(0, str(Path(tvm.__file__).resolve().parents[2] / "apps/gemmini"))
from workload_reference import evaluate, integer_conv2d, integer_matmul


def graph(operation, shape=(2, 3), dtype="float32"):
    x = relax.Var("x", relax.TensorStructInfo(shape, dtype))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        with bb.dataflow():
            out = bb.emit_output(operation(x))
        bb.emit_func_output(out)
    return bb.get()


def test_integer_dot_uses_wide_accumulation_and_refuses_output_overflow():
    x = np.full((2, 4), 127, dtype=np.int8)
    w = np.full((4, 3), 127, dtype=np.int8)
    np.testing.assert_array_equal(integer_matmul(x, w, "int32"), np.full((2, 3), 64516))
    with pytest.raises(ValueError, match="exceeds"):
        integer_matmul(x, w, "int8")


def test_relax_round_is_ties_away_before_integer_cast():
    mod = graph(lambda x: relax.op.astype(relax.op.round(x), "int8"))
    x = np.array([[-2.5, -1.5, -.5], [.5, 1.5, 2.5]], dtype="float32")
    out, reference = evaluate(mod, [x], retain_checkpoints=True)
    np.testing.assert_array_equal(out[0], [[-3, -2, -1], [1, 2, 3]])
    assert reference.checkpoints


def test_sigmoid_is_finite_at_large_positive_and_negative_inputs():
    mod = graph(relax.op.sigmoid)
    x = np.array([[-1000, -1, 0], [1, 10, 1000]], dtype="float32")
    out, _ = evaluate(mod, [x])
    assert out[0].dtype == np.float32
    np.testing.assert_allclose(out[0], [[0, .26894142137, .5], [.73105857863, .99995460213, 1]],
                               rtol=1e-7, atol=1e-8)


def test_unknown_semantics_and_incorrect_input_types_are_refused():
    mod = graph(relax.op.exp)
    with pytest.raises(ValueError, match="Unsupported reference operator"):
        evaluate(mod, [np.ones((2, 3), dtype="float32")])
    with pytest.raises(ValueError, match="type mismatch"):
        evaluate(mod, [np.ones((2, 3), dtype="float64")])


def test_nhwc_integer_conv_matches_independent_scalar_definition():
    x = np.arange(12, dtype=np.int8).reshape(1, 2, 3, 2) - 3
    w = np.array([[[[1, 2], [-3, 1]], [[2, 0], [-1, 2]]]], dtype=np.int8)
    attrs = SimpleNamespace(data_layout="NHWC", kernel_layout="OHWI", out_layout="NHWC",
                            groups=1, strides=(1, 1), dilation=(1, 1),
                            padding=(1, 0, 0, 1), out_dtype="int32")
    out = integer_conv2d(x, w, attrs)
    expected = np.zeros(out.shape, dtype=np.int32)
    for row in range(2):
        for col in range(3):
            total = 0
            for i in range(2):
                for j in range(2):
                    y, z = row + i - 1, col + j
                    if 0 <= y < 2 and 0 <= z < 3:
                        for c in range(2):
                            total += int(x[0, y, z, c]) * int(w[0, i, j, c])
            expected[0, row, col, 0] = total
    np.testing.assert_array_equal(out, expected)


def test_reference_matches_cpu_execution_for_quantized_connected_arithmetic():
    w = relax.const(np.array([[1, -2], [3, 1], [-1, 2]], dtype=np.int8))
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo((2, 3), "float32"))
    with bb.function("main", [x]):
        with bb.dataflow():
            rounded = bb.emit(relax.op.astype(relax.op.round(x), "int8"))
            dot = bb.emit(relax.op.matmul(rounded, w, out_dtype="int32"))
            value = bb.emit(relax.op.astype(dot, "float32"))
            out = bb.emit_output(relax.op.multiply(value, relax.op.sigmoid(value)))
        bb.emit_func_output(out)
    mod = bb.get()
    inputs = [np.array([[-2.5, 1.5, .5], [3.5, -1.5, -.5]], dtype="float32")]
    reference, _ = evaluate(mod, inputs)
    vm = relax.VirtualMachine(relax.build(mod, target="llvm"), tvm.cpu())
    actual = vm["main"](tvm.nd.array(inputs[0])).numpy()
    np.testing.assert_allclose(actual, reference[0], rtol=1e-5, atol=2e-6)
