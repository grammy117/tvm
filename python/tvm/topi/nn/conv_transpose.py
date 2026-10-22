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
"""Grouped transposed-convolution mathematics with arbitrary spatial dilation."""

from tvm import te, tir


def group_conv_transpose_ncx(
    data, weight, strides, padding, output_padding, dilation, groups, out_dtype
):
    """Compute a grouped NCX/IOX transpose using only real input/kernel taps.

    ``X`` denotes one, two or three spatial axes. Padding is ordered as the
    leading axes followed by trailing axes. The guarded gather expresses the
    stride lattice directly, without introducing fictitious zero kernel taps.
    This is an unscheduled TE definition with no target-specific decisions.
    """
    dim = len(data.shape) - 2
    if dim not in (1, 2, 3) or len(weight.shape) != dim + 2:
        raise ValueError("Transposed convolution requires matching NCX and IOX ranks")
    if len(padding) != 2 * dim:
        raise ValueError("Transposed convolution requires leading and trailing padding")
    in_group = data.shape[1] // groups
    out_group = weight.shape[1]
    spatial = tuple(
        (size - 1) * stride - before - after + dil * (kernel - 1) + 1 + extra
        for size, kernel, stride, before, after, dil, extra in zip(
            data.shape[2:],
            weight.shape[2:],
            strides,
            padding[:dim],
            padding[dim:],
            dilation,
            output_padding,
        )
    )
    shape = (data.shape[0], out_group * groups) + spatial
    channel = te.reduce_axis((0, in_group), name="input_channel")
    taps = [te.reduce_axis((0, size), name=f"tap_{i}") for i, size in enumerate(weight.shape[2:])]

    def compute(n, c, *indices):
        coordinates = [
            index + before - tap * dil
            for index, before, tap, dil in zip(indices, padding[:dim], taps, dilation)
        ]
        positions = [tir.floordiv(index, stride) for index, stride in zip(coordinates, strides)]
        valid = tir.all(
            *[
                condition
                for index, stride, position, size in zip(
                    coordinates, strides, positions, data.shape[2:]
                )
                for condition in (tir.floormod(index, stride) == 0, position >= 0, position < size)
            ]
        )
        ic = tir.floordiv(c, out_group) * in_group + channel
        lhs = data[(n, ic) + tuple(positions)].astype(out_dtype)
        rhs = weight[(ic, tir.floormod(c, out_group)) + tuple(taps)].astype(out_dtype)
        return te.sum(tir.if_then_else(valid, lhs * rhs, tir.const(0, out_dtype)), [channel] + taps)

    return te.compute(shape, compute, name="group_conv_transpose", tag="group_conv_transpose_ncx")
