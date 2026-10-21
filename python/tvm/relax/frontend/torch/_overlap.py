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
"""Unscheduled window adjoints and overlap sums missing from Relax operators."""

from tvm import te, tir

from ._numeric import add_uint64


def positive_spatial(shape):
    if not all(isinstance(size, tir.IntImm) and int(size) > 0 for size in shape):
        raise NotImplementedError("Window adjoints require positive static spatial extents")
    return tuple(int(size) for size in shape)


def sum_values(value, axes, dtype):
    if dtype == "int64":
        reducer = te.comm_reducer(add_uint64, lambda t: tir.const(0, t), name="modulo_sum")
        return reducer(value.astype("uint64"), axis=axes)
    if dtype == "bool":
        reducer = te.comm_reducer(
            lambda a, b: tir.Or(a, b), lambda t: tir.const(False, t), name="any_overlap"
        )
        return reducer(value, axis=axes)
    return te.sum(value, axis=axes)


def divide(value, denominator):
    if value.dtype == "int64":
        denominator = tir.Cast("int64", denominator)
        return tir.if_then_else(
            tir.all(value == -(1 << 63), denominator == -1),
            value,
            tir.truncdiv(value, denominator),
        )
    # A reduced floating scalar's denominator and each division have its
    # actual dtype, including Half's intermediate rounding in the CPU adjoint.
    return value / tir.Cast(value.dtype, denominator)


def average_pool_backward(gradient, input_shape, kernel, strides, padding, include, divisor):
    height, width = positive_spatial(input_shape[-2:])
    out_h, out_w = positive_spatial(gradient.shape[-2:])

    def delta(*indices):
        start_h = indices[-2] * strides[0] - padding[0]
        start_w = indices[-1] * strides[1] - padding[1]
        end_h = tir.min(start_h + kernel[0], height + (padding[0] if include else 0))
        end_w = tir.min(start_w + kernel[1], width + (padding[1] if include else 0))
        if not include:
            start_h, start_w = tir.max(start_h, 0), tir.max(start_w, 0)
        denominator = (
            tir.const(divisor, "int64")
            if divisor is not None
            else (end_h - start_h) * (end_w - start_w)
        )
        return divide(gradient[indices], denominator)

    deltas = te.compute(gradient.shape, delta, name="pool_gradient_delta")
    axes = [
        te.reduce_axis((0, (k + stride - 1) // stride), name=f"window_{i}")
        for i, (k, stride) in enumerate(zip(kernel, strides))
    ]

    def gather(*indices):
        positions, conditions = [], []
        for index, pad, k, stride, extent, axis in zip(
            indices[-2:], padding, kernel, strides, (out_h, out_w), axes
        ):
            count = (k + stride - 1) // stride
            position = tir.floordiv(index + pad, stride) - count + 1 + axis
            positions.append(position)
            conditions.extend(
                (position >= 0, position < extent, position * stride + k > index + pad)
            )
        # At most ceil(kernel/stride) output windows cover one input point.
        # Absent windows contribute zero, including wholly uncovered inputs.
        value = tir.if_then_else(
            tir.all(*conditions),
            deltas[indices[:-2] + tuple(positions)],
            tir.const(0, gradient.dtype),
        )
        return sum_values(value, axes, gradient.dtype)

    summed = te.compute(input_shape, gather, name="pool_input_gradient")
    if gradient.dtype == "int64":
        return te.compute(
            input_shape, lambda *i: summed[i].astype("int64"), name="wrapped_gradient"
        )
    return summed


def adaptive_pool_backward(gradient, input_shape):
    height, width = positive_spatial(input_shape[-2:])
    out_h, out_w = positive_spatial(gradient.shape[-2:])

    def delta(*indices):
        value = gradient[indices]
        for index, size, out_size in zip(indices[-2:], (height, width), (out_h, out_w)):
            start = tir.floordiv(index * size, out_size)
            end = tir.floordiv((index + 1) * size + out_size - 1, out_size)
            value = divide(value, end - start)
        return value

    deltas = te.compute(gradient.shape, delta, name="adaptive_gradient_delta")

    def gather(*indices):
        axes = [
            te.reduce_axis(
                (
                    tir.floordiv(index * out_size, size),
                    tir.floordiv((index + 1) * out_size + size - 1, size),
                ),
                name=f"window_{i}",
            )
            for i, (index, size, out_size) in enumerate(
                zip(indices[-2:], (height, width), (out_h, out_w))
            )
        ]
        return te.sum(deltas[indices[:-2] + tuple(axes)], axis=axes)

    return te.compute(input_shape, gather, name="adaptive_input_gradient")


def col2im(data, output_size, kernel, dilation, padding, strides):
    height, width = output_size
    channels_col, columns = positive_spatial(data.shape[-2:])
    if channels_col % (kernel[0] * kernel[1]):
        raise ValueError("Column channels must be divisible by the kernel area")
    column_shape = tuple(
        (size + 2 * pad - dil * (k - 1) - 1) // stride + 1
        for size, pad, dil, k, stride in zip(output_size, padding, dilation, kernel, strides)
    )
    if min(column_shape) <= 0 or columns != column_shape[0] * column_shape[1]:
        raise ValueError("Column count does not match the declared image and window geometry")
    shape = tuple(data.shape[:-2]) + (channels_col // (kernel[0] * kernel[1]), height, width)
    axes = [te.reduce_axis((0, size), name=f"offset_{i}") for i, size in enumerate(kernel)]

    def gather(*indices):
        position = [
            index + pad - offset * dil
            for index, pad, offset, dil in zip(indices[-2:], padding, axes, dilation)
        ]
        column = [tir.floordiv(index, stride) for index, stride in zip(position, strides)]
        valid = tir.all(
            *[
                condition
                for index, stride, col, extent in zip(position, strides, column, column_shape)
                for condition in (tir.floormod(index, stride) == 0, col >= 0, col < extent)
            ]
        )
        row = (indices[-3] * kernel[0] + axes[0]) * kernel[1] + axes[1]
        value = data[indices[:-3] + (row, column[0] * column_shape[1] + column[1])]
        return sum_values(
            tir.if_then_else(valid, value, tir.const(0, data.dtype)), axes, data.dtype
        )

    return te.compute(shape, gather, name="overlap_sum")
