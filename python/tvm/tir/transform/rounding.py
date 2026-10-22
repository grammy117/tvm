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
"""LLVM lowering of bounded round-to-integer quantization expressions."""

import tvm
from tvm import tir


def lower_bounded_rne(func):
    """Convert a byte-range FP32 RNE/clamp/cast with LLVM's integer rint.

    For integral byte bounds L,H and integral zero point Z, clipping x to
    [L-Z,H-Z] before RNE yields clamp(RNE(x)+Z,L,H). Interior integer addition
    is exact in FP32. Exterior values saturate identically. The original NaN
    guard is retained; unguarded casts are deliberately not rewritten. As with
    nearbyint, the caller must establish the RNE floating environment.
    """
    changed = [0]

    def pair(expr, cls):
        if not isinstance(expr, cls):
            return None
        for value, bound in ((expr.a, expr.b), (expr.b, expr.a)):
            if isinstance(bound, tir.FloatImm) and float(bound.value).is_integer():
                return value, int(bound.value)
        return None

    def rewrite(node):
        if not isinstance(node, tir.Call) or node.op != tvm.ir.Op.get("tir.if_then_else"):
            return None
        condition, nan_value, cast = node.args
        if not (
            isinstance(condition, tir.Call)
            and condition.op == tvm.ir.Op.get("tir.isnan")
            and isinstance(nan_value, tir.IntImm)
            and int(nan_value) == 0
            and isinstance(cast, tir.Cast)
            and cast.dtype == "int8"
        ):
            return None
        high_pair = pair(cast.value, tir.Min)
        low_pair = pair(high_pair[0], tir.Max) if high_pair else None
        if low_pair is None:
            return None
        rounded, low = low_pair
        high, zero = high_pair[1], 0
        if isinstance(rounded, tir.Add):
            addition = pair(rounded, tir.Add)
            if addition is None:
                return None
            rounded, zero = addition
        if not (
            isinstance(rounded, tir.Call)
            and rounded.op == tvm.ir.Op.get("tir.nearbyint")
            and rounded.dtype == "float32"
            and -128 <= low <= high <= 127
            and -128 <= zero <= 127
            and tvm.ir.structural_equal(condition.args[0], rounded.args[0])
        ):
            return None
        clipped = tir.min(
            tir.max(rounded.args[0], tir.const(low - zero, "float32")),
            tir.const(high - zero, "float32"),
        )
        integer = tir.call_llvm_pure_intrin("int32", "llvm.lrint", tir.const(2, "uint32"), clipped)
        result = (integer + tir.const(zero, "int32")).astype("int8")
        changed[0] += 1
        return tir.if_then_else(condition, nan_value, result)

    body = tir.stmt_functor.ir_transform(func.body, None, rewrite, ["tir.Call"])
    return func.with_body(body), changed[0]
