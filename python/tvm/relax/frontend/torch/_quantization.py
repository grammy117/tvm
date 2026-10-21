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
"""Mathematical ATen Q/DQ definitions; no target selection or new quantization policy."""

from tvm import te, tir


def quantize_per_tensor(data, inverse_scale, zero_point, low, high, dtype):
    """Match ATen's FP32 multiply, RNE, zero-point add, clamp, then conversion."""

    def value(*indices):
        scaled = data[indices].astype("float32") * tir.const(inverse_scale, "float32")
        rounded = tir.nearbyint(scaled) + tir.const(zero_point, "float32")
        clamped = tir.min(tir.max(rounded, tir.const(low, "float32")), tir.const(high, "float32"))
        return tir.if_then_else(tir.isnan(scaled), tir.const(0, dtype), clamped.astype(dtype))

    return te.compute(data.shape, value, name="quantize_rne")


def dequantize_per_tensor(data, scale, zero_point):
    return te.compute(
        data.shape,
        lambda *indices: (data[indices].astype("float32") - tir.const(zero_point, "float32"))
        * tir.const(scale, "float32"),
        name="dequantize_affine",
    )


def dequantize_per_channel(data, scales, zeros, axis):
    def value(*indices):
        # The importer admits bounded frozen zero points, so this difference
        # is exact in INT32. ATen then promotes the difference to scale dtype.
        difference = data[indices].astype("int32") - zeros[indices[axis]].astype("int32")
        return (difference.astype(scales.dtype) * scales[indices[axis]]).astype("float32")

    return te.compute(data.shape, value, name="dequantize_channel_affine")
