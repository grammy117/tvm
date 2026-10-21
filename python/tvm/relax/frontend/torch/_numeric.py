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
"""Mathematical integer wrapping and fused rounding for exported Torch operators.

These are unscheduled, target-independent computations except that strict fused
floating rounding uses TVM's LLVM intrinsic API. No device instructions, layouts
or scheduling choices belong here. Explicit bounded intermediate arithmetic
avoids depending on signed/unsigned overflow in ordinary TVM arithmetic.
"""

from tvm import te, tir


def _limbs(value):
    value = value.astype("uint64")
    return value & tir.const(0xFFFFFFFF, "uint64"), value >> 32


def add_uint64(lhs, rhs):
    """Modulo-64 addition: each intermediate sum is at most 2**33 - 1."""
    a0, a1 = _limbs(lhs)
    b0, b1 = _limbs(rhs)
    low = a0 + b0
    high = (a1 + b1 + (low >> 32)) & tir.const(0xFFFFFFFF, "uint64")
    return (high << 32) | (low & tir.const(0xFFFFFFFF, "uint64"))


def multiply_uint64(lhs, rhs):
    """Modulo-64 product with individually bounded 32-by-32 products."""
    a0, a1 = _limbs(lhs)
    b0, b1 = _limbs(rhs)
    mask = tir.const(0xFFFFFFFF, "uint64")
    low_product = a0 * b0
    cross = ((a0 * b1) & mask) + ((a1 * b0) & mask)
    high = ((low_product >> 32) + (cross & mask)) & mask
    return (high << 32) | (low_product & mask)


def binary_int64(lhs, rhs, operation, alpha=1):
    """Broadcasted, explicitly wrapping INT64 pointwise mathematics."""
    from tvm import topi

    # TOPI's broadcast operation derives and validates symbolic output shapes.
    shape = topi.add(lhs, rhs).shape

    def load(data, indices):
        indices = indices[len(shape) - len(data.shape) :]
        return data[
            tuple(
                0 if isinstance(size, tir.IntImm) and int(size) == 1 else i
                for size, i in zip(data.shape, indices)
            )
        ]

    def value(*indices):
        a = load(lhs, indices).astype("uint64")
        b = load(rhs, indices).astype("uint64")
        if operation == "multiply":
            result = multiply_uint64(a, b)
        else:
            scaled = multiply_uint64(b, tir.const(alpha % (1 << 64), "uint64"))
            if operation == "subtract":
                scaled = add_uint64(~scaled, tir.const(1, "uint64"))
            result = add_uint64(a, scaled)
        return result.astype("int64")

    return te.compute(shape, value, name="wrapping_" + operation)


def binary_narrow_integer(lhs, rhs, operation, alpha=1):
    """INT8/INT32 wrapping with a proven, nonoverflowing INT64 intermediate."""
    from tvm import topi

    shape = topi.add(lhs, rhs).shape

    def load(data, indices):
        indices = indices[len(shape) - len(data.shape) :]
        return data[
            tuple(
                0 if isinstance(size, tir.IntImm) and int(size) == 1 else i
                for size, i in zip(data.shape, indices)
            )
        ].astype("int64")

    def value(*indices):
        a, b = load(lhs, indices), load(rhs, indices)
        if operation == "multiply":
            result = a * b
        else:
            scaled = b * tir.const(alpha, "int64")
            result = a + scaled if operation == "add" else a - scaled
        return result.astype(lhs.dtype)

    return te.compute(shape, value, name="wrapping_" + operation)


def modular_matmul(lhs, rhs, shape):
    """Ordinary contraction with a modulo reducer, for INT32/INT64 outputs."""
    dtype = lhs.dtype
    bits = 32 if dtype == "int32" else 64
    mask = tir.const((1 << bits) - 1, "uint64")
    vector_a, vector_b = len(lhs.shape) == 1, len(rhs.shape) == 1
    batch_rank = max(len(lhs.shape), len(rhs.shape), 2) - 2
    axis = te.reduce_axis((0, lhs.shape[-1]), "k")

    def combine(a, b):
        return ((a + b) & mask) if bits == 32 else add_uint64(a, b)

    reducer = te.comm_reducer(combine, lambda dtype: tir.const(0, dtype), name="modulo_sum")

    def value(*indices):
        def batches(data):
            rank = max(len(data.shape) - 2, 0)
            return [
                0
                if isinstance(size, tir.IntImm) and int(size) == 1
                else indices[batch_rank - rank + i]
                for i, size in enumerate(data.shape[:rank])
            ]

        a_indices = [axis] if vector_a else batches(lhs) + [indices[batch_rank], axis]
        b_indices = [axis] if vector_b else batches(rhs) + [axis, indices[-1]]
        a, b = lhs[tuple(a_indices)], rhs[tuple(b_indices)]
        if bits == 32:
            product = (
                a.astype("uint32").astype("uint64") * b.astype("uint32").astype("uint64")
            ) & mask
        else:
            product = multiply_uint64(a.astype("uint64"), b.astype("uint64"))
        return reducer(product, axis=axis)

    accumulated = te.compute(shape, value, name="modular_matmul")
    return te.compute(
        shape, lambda *indices: accumulated[indices].astype(dtype), name="cast_result"
    )


def fused_alpha_add(lhs, rhs, alpha):
    """Preserve one fused round for nontrivial floating add/sub alpha."""
    from tvm import topi

    shape = topi.add(lhs, rhs).shape

    def load(data, indices):
        indices = indices[len(shape) - len(data.shape) :]
        return data[
            tuple(
                0 if isinstance(size, tir.IntImm) and int(size) == 1 else i
                for size, i in zip(data.shape, indices)
            )
        ]

    return te.compute(
        shape,
        lambda *indices: tir.call_llvm_pure_intrin(
            lhs.dtype,
            "llvm.fma",
            tir.const(3, "uint32"),
            load(rhs, indices),
            tir.const(alpha, lhs.dtype),
            load(lhs, indices),
        ),
        name="alpha_add",
    )
