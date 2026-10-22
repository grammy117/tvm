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
"""Unscheduled convolution contracts absent from existing Relax operators."""

from tvm import te, tir

from ._numeric import add_uint64, multiply_uint64


def modular_convolution(data, weight, shape, strides, padding, dilation, groups, transpose):
    """ATen full-width integer convolution with defined modulo arithmetic."""
    dtype = data.dtype
    bits = 32 if dtype == "int32" else 64
    mask = tir.const((1 << bits) - 1, "uint64")
    in_group, out_group = data.shape[1] // groups, shape[1] // groups
    channel = te.reduce_axis((0, in_group), name="channel")
    taps = [te.reduce_axis((0, size), name=f"tap_{i}") for i, size in enumerate(weight.shape[2:])]
    reducer = te.comm_reducer(
        (lambda a, b: (a + b) & mask) if bits == 32 else add_uint64,
        lambda t: tir.const(0, t),
        name="modulo_convolution",
    )

    def value(n, c, *indices):
        group = tir.floordiv(c, out_group)
        ic = group * in_group + channel
        positions, checks = [], []
        for index, tap, stride, pad, dil, size in zip(
            indices, taps, strides, padding, dilation, data.shape[2:]
        ):
            position = index + pad - tap * dil if transpose else index * stride - pad + tap * dil
            if transpose:
                checks.append(tir.floormod(position, stride) == 0)
                position = tir.floordiv(position, stride)
            positions.append(position)
            checks.extend((position >= 0, position < size))
        left = data[(n, ic) + tuple(positions)]
        right = weight[
            ((ic, tir.floormod(c, out_group)) if transpose else (c, channel)) + tuple(taps)
        ]
        if bits == 32:
            product = (
                left.astype("uint32").astype("uint64") * right.astype("uint32").astype("uint64")
            ) & mask
        else:
            product = multiply_uint64(left.astype("uint64"), right.astype("uint64"))
        return reducer(
            tir.if_then_else(tir.all(*checks), product, tir.const(0, "uint64")), [channel] + taps
        )

    summed = te.compute(shape, value, name="modular_convolution")
    return te.compute(shape, lambda *i: summed[i].astype(dtype), name="signed_convolution")


def convolution_input_gradient(
    gradient, weight, shape, strides, padding, dilation, groups, transpose
):
    """Input adjoint of a convolution, including the transpose convention."""
    dim = len(shape) - 2
    out_group = gradient.shape[1] // groups
    in_group = shape[1] // groups
    channel = te.reduce_axis((0, out_group), name="output_channel")
    taps = [te.reduce_axis((0, size), name=f"tap_{i}") for i, size in enumerate(weight.shape[2:])]

    def value(n, c, *indices):
        group = tir.floordiv(c, in_group)
        oc = group * out_group + channel
        positions, checks = [], []
        for index, tap, stride, pad, dil, size in zip(
            indices, taps, strides, padding, dilation, gradient.shape[2:]
        ):
            if transpose:
                position = index * stride - pad + tap * dil
            else:
                offset = index + pad - tap * dil
                checks.append(tir.floormod(offset, stride) == 0)
                position = tir.floordiv(offset, stride)
            positions.append(position)
            checks.extend((position >= 0, position < size))
        w_index = (c, channel) if transpose else (oc, tir.floormod(c, in_group))
        product = gradient[(n, oc) + tuple(positions)] * weight[w_index + tuple(taps)]
        return te.sum(
            tir.if_then_else(tir.all(*checks), product, tir.const(0, gradient.dtype)),
            [channel] + taps,
        )

    return te.compute(shape, value, name=f"conv{dim}d_input_gradient")


def convolution_weight_gradient(
    data, gradient, weight_shape, strides, padding, dilation, groups, transpose
):
    """Weight adjoint as its batch/spatial contraction, without a schedule."""
    batch = te.reduce_axis((0, data.shape[0]), name="batch")
    sizes = data.shape[2:] if transpose else gradient.shape[2:]
    positions = [te.reduce_axis((0, size), name=f"position_{i}") for i, size in enumerate(sizes)]
    in_group, out_group = data.shape[1] // groups, gradient.shape[1] // groups

    def value(c, other, *taps):
        group = tir.floordiv(c, in_group if transpose else out_group)
        mapped, checks = [], []
        target_shape = gradient.shape[2:] if transpose else data.shape[2:]
        for index, tap, stride, pad, dil, size in zip(
            positions, taps, strides, padding, dilation, target_shape
        ):
            position = index * stride - pad + tap * dil
            mapped.append(position)
            checks.extend((position >= 0, position < size))
        if transpose:
            left = data[(batch, c) + tuple(positions)]
            right = gradient[(batch, group * out_group + other) + tuple(mapped)]
        else:
            left = data[(batch, group * in_group + other) + tuple(mapped)]
            right = gradient[(batch, c) + tuple(positions)]
        return te.sum(
            tir.if_then_else(tir.all(*checks), left * right, tir.const(0, data.dtype)),
            [batch] + positions,
        )

    return te.compute(weight_shape, value, name="convolution_weight_gradient")
