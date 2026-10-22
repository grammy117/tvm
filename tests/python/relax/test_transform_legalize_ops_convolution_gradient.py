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
"""Convolution transposes and adjoints against independent scatter oracles."""

import itertools

import numpy as np
import pytest
import tvm
from tvm import relax, topi
from tvm.relax.op._op_gradient import conv2d_grad


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("groups", [1, 2])
@pytest.mark.parametrize("extra", [0, 1])
def test_dilated_group_transpose_preserves_real_taps_and_output_padding(dim, dtype, groups, extra):
    shape = (1, 4) + tuple(3 + i % 2 for i in range(dim))
    weight_shape = (4, 2) + (2,) * dim
    values = ((np.arange(np.prod(shape)).reshape(shape) % 13 - 7) / 4).astype(dtype)
    weights = ((np.arange(np.prod(weight_shape)).reshape(weight_shape) % 7 - 4) / 4).astype(dtype)
    strides, dilation = (2,) * dim, (2,) * dim
    before, after, output_padding = (1,) * dim, (0,) * dim, (extra,) * dim
    output_shape = (
        shape[:1]
        + (2 * groups,)
        + tuple(
            (size - 1) * 2 - 1 + 2 * (kernel - 1) + 1 + extra
            for size, kernel in zip(shape[2:], weight_shape[2:])
        )
    )
    expected = np.zeros(output_shape, dtype)
    for channel in range(4):
        group = channel // (4 // groups)
        for source in np.ndindex(shape[2:]):
            for oc, tap in itertools.product(range(2), np.ndindex(weight_shape[2:])):
                target = tuple(index * 2 - 1 + offset * 2 for index, offset in zip(source, tap))
                if all(0 <= index < extent for index, extent in zip(target, output_shape[2:])):
                    expected[(0, group * 2 + oc) + target] += (
                        values[(0, channel) + source] * weights[(channel, oc) + tap]
                    )
    x = relax.Var("x", relax.TensorStructInfo(shape, dtype))
    w = relax.Var("w", relax.TensorStructInfo(weight_shape, dtype))
    bb = relax.BlockBuilder()
    with bb.function("main", [x, w]):
        with bb.dataflow():
            if dim < 3:
                result = getattr(relax.op.nn, f"conv{dim}d_transpose")(
                    x, w, strides, before + after, output_padding, dilation, groups, out_dtype=dtype
                )
            else:
                result = bb.emit_te(
                    topi.nn.group_conv_transpose_ncx,
                    x,
                    w,
                    strides,
                    before + after,
                    output_padding,
                    dilation,
                    groups,
                    dtype,
                )
            output = bb.emit_output(result)
        bb.emit_func_output(output)
    vm = relax.VirtualMachine(relax.build(bb.get(), "llvm"), tvm.cpu())
    actual = vm["main"](tvm.nd.array(values), tvm.nd.array(weights)).numpy()
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("groups", [1, 2])
@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("stride", [1, 2])
def test_conv2d_adjoint_includes_dilation_in_shape(dtype, groups, dilation, stride):
    values = ((np.arange(168).reshape(1, 4, 6, 7) % 13 - 7) / 4).astype(dtype)
    w_shape = (6, 4 // groups, 3, 2)
    weights = ((np.arange(np.prod(w_shape)).reshape(w_shape) % 7 - 4) / 4).astype(dtype)
    padding = (1, 1, 0, 2)
    output_shape = (1, 6) + tuple(
        (size + before + after - dilation * (kernel - 1) - 1) // stride + 1
        for size, kernel, before, after in zip((6, 7), (3, 2), padding[:2], padding[2:])
    )
    gradient = ((np.arange(np.prod(output_shape)).reshape(output_shape) % 11 - 6) / 4).astype(dtype)
    expected_x, expected_w = np.zeros_like(values), np.zeros_like(weights)
    for oc, y, x in np.ndindex(output_shape[1:]):
        group = oc // (6 // groups)
        for ic, ky, kx in np.ndindex(w_shape[1:]):
            iy, ix = y * stride - 1 + ky * dilation, x * stride - 1 + kx * dilation
            if 0 <= iy < 6 and 0 <= ix < 7:
                channel = group * (4 // groups) + ic
                expected_x[0, channel, iy, ix] += gradient[0, oc, y, x] * weights[oc, ic, ky, kx]
                expected_w[oc, ic, ky, kx] += gradient[0, oc, y, x] * values[0, channel, iy, ix]
    bb = relax.BlockBuilder()
    xv = relax.Var("x", relax.TensorStructInfo(values.shape, dtype))
    wv = relax.Var("w", relax.TensorStructInfo(weights.shape, dtype))
    gv = relax.Var("g", relax.TensorStructInfo(gradient.shape, dtype))
    with bb.function("main", [xv, wv, gv]):
        with bb.dataflow():
            original = bb.emit(
                relax.op.nn.conv2d(
                    xv,
                    wv,
                    strides=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    out_dtype=dtype,
                )
            )
            gx, gw = [
                bb.emit(value)
                for value in conv2d_grad(original, bb.lookup_binding(original), gv, bb)
            ]
            output = bb.emit_output(relax.Tuple([gx, gw]))
        bb.emit_func_output(output)
    vm = relax.VirtualMachine(relax.build(bb.get(), "llvm"), tvm.cpu())
    actual = vm["main"](tvm.nd.array(values), tvm.nd.array(weights), tvm.nd.array(gradient))
    np.testing.assert_array_equal(actual[0].numpy(), expected_x)
    np.testing.assert_array_equal(actual[1].numpy(), expected_w)


def test_dilated_transpose_uncovered_outputs_do_not_consume_nan_input():
    bb = relax.BlockBuilder()
    data = relax.Var("data", relax.TensorStructInfo((1, 1, 1), "float32"))
    weight = relax.Var("weight", relax.TensorStructInfo((1, 1, 2), "float32"))
    with bb.function("main", [data, weight]):
        with bb.dataflow():
            output = bb.emit_output(relax.op.nn.conv1d_transpose(data, weight, dilation=2))
        bb.emit_func_output(output)
    vm = relax.VirtualMachine(relax.build(bb.get(), "llvm"), tvm.cpu())
    actual = vm["main"](
        tvm.nd.array(np.full((1, 1, 1), np.nan, "float32")),
        tvm.nd.array(np.ones((1, 1, 2), "float32")),
    ).numpy()
    np.testing.assert_array_equal(actual, np.array([[[np.nan, 0, np.nan]]], "float32"))
