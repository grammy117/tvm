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
"""Unscheduled pooling mathematics for ATen contracts absent from Relax ops."""

import math

from tvm import te, tir

from ._numeric import add_uint64


def output_extent(size, kernel, stride, padding, ceil_mode):
    extent = (size + 2 * padding - kernel + (stride - 1 if ceil_mode else 0)) // stride + 1
    if ceil_mode and (extent - 1) * stride >= size + padding:
        extent -= 1
    return extent


def average_pool(data, kernel, strides, padding, ceil_mode, count_include_pad, divisor):
    """A window sum followed by one specified division, including integer wrap."""
    dim = len(kernel)
    sizes = data.shape[-dim:]
    if not all(isinstance(size, tir.IntImm) and int(size) > 0 for size in sizes):
        raise NotImplementedError(
            "Explicit average-pool divisors require positive static spatial extents"
        )
    sizes = tuple(int(size) for size in sizes)
    shape = tuple(data.shape[:-dim]) + tuple(
        output_extent(size, k, s, p, ceil_mode)
        for size, k, s, p in zip(sizes, kernel, strides, padding)
    )
    axes = [te.reduce_axis((0, k), name=f"window_{i}") for i, k in enumerate(kernel)]
    integer = data.dtype == "int64"
    reducer = (
        te.comm_reducer(add_uint64, lambda dtype: tir.const(0, dtype), name="modulo_sum")
        if integer
        else te.sum
    )

    def summed(*indices):
        spatial = [
            i * stride - pad + axis
            for i, stride, pad, axis in zip(indices[-dim:], strides, padding, axes)
        ]
        valid = tir.all(
            *[condition for i, size in zip(spatial, sizes) for condition in (i >= 0, i < size)]
        )
        value = data[indices[:-dim] + tuple(spatial)]
        value = value.astype("uint64") if integer else value
        return reducer(tir.if_then_else(valid, value, tir.const(0, value.dtype)), axis=axes)

    accumulated = te.compute(shape, summed, name="pool_sum")

    def divided(*indices):
        if divisor is None:
            counts = []
            for index, size, k, stride, pad in zip(indices[-dim:], sizes, kernel, strides, padding):
                start = index * stride - pad
                lower = start if count_include_pad else tir.max(start, 0)
                upper = tir.min(start + k, size + pad if count_include_pad else size)
                counts.append(tir.max(upper - lower, 1))
            denominator = math.prod(counts)
        else:
            denominator = tir.const(divisor, "int64")
        value = accumulated[indices].astype(data.dtype)
        if integer:
            # Avoid the undefined signed division case without changing any
            # defined ATen result; all other division truncates toward zero.
            result = tir.truncdiv(value, denominator)
            return tir.if_then_else(tir.all(value == -(1 << 63), denominator == -1), value, result)
        return value / tir.Cast(value.dtype, denominator)

    return te.compute(shape, divided, name="pool_average")
