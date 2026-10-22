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
import tvm
import tvm.script
from tvm.script import tir as T


def get_before():
    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def main(
            Aptr: T.handle("bfloat16"), Bptr: T.handle("bfloat16"), Dptr: T.handle("bfloat16")
        ):
            T.func_attr({"global_symbol": "main"})
            A = T.decl_buffer((100,), "bfloat16", data=Aptr)
            B = T.decl_buffer((100,), "bfloat16", data=Bptr)
            D = T.decl_buffer((100,), "bfloat16", data=Dptr)
            C = T.decl_buffer((100,), "bfloat16")
            for i in T.grid(100):
                C[i] = A[i] + B[i]
                D[i] = T.exp(C[i])

    return Before


def u16tof32(v):
    uint32_v = v.astype("uint32")
    uint32_v = uint32_v << tvm.tir.const(16, "uint32")
    return T.reinterpret("float32", uint32_v)


def bf16tof32(v):
    return u16tof32(T.reinterpret("uint16", v))


def f32tou16(v):
    uint32_v = T.reinterpret("uint32", v)
    rounding_bias = (uint32_v >> tvm.tir.const(16, "uint32")) & tvm.tir.const(1, "uint32")
    rounding_bias += tvm.tir.const(0x7FFF, "uint32")
    uint32_v = uint32_v + rounding_bias
    return (uint32_v >> tvm.tir.const(16, "uint32")).astype("uint16")


def f32tobf16(v):
    return T.reinterpret("bfloat16", f32tou16(v))


def get_after_compute_legalize():
    @tvm.script.ir_module
    class After:
        @T.prim_func
        def main(
            Aptr: T.handle("bfloat16"), Bptr: T.handle("bfloat16"), Dptr: T.handle("bfloat16")
        ):
            T.func_attr({"global_symbol": "main"})
            A = T.decl_buffer((100,), "bfloat16", data=Aptr)
            B = T.decl_buffer((100,), "bfloat16", data=Bptr)
            D = T.decl_buffer((100,), "bfloat16", data=Dptr)
            C = T.decl_buffer((100,), "float32")
            for i in T.grid(100):
                C[i] = bf16tof32(f32tobf16(bf16tof32(A[i]) + bf16tof32(B[i])))
                D[i] = f32tobf16(T.exp(C[i]))

    return After


def get_after_storage_legalize():
    @tvm.script.ir_module
    class After:
        @T.prim_func
        def main(Aptr: T.handle("uint16"), Bptr: T.handle("uint16"), Dptr: T.handle("uint16")):
            T.func_attr({"global_symbol": "main"})
            A = T.decl_buffer((100,), "uint16", data=Aptr)
            B = T.decl_buffer((100,), "uint16", data=Bptr)
            D = T.decl_buffer((100,), "uint16", data=Dptr)
            C = T.decl_buffer((100,), "float32")
            for i in T.grid(100):
                C[i] = u16tof32(f32tou16(u16tof32(A[i]) + u16tof32(B[i])))
                D[i] = f32tou16(T.exp(C[i]))

    return After


def test_bf16_compute_legalize():
    before = get_before()
    expected = get_after_compute_legalize()
    # run the transform twice to ensure we can afford to deal
    # with this repeative optimizations
    after = tvm.tir.transform.BF16ComputeLegalize()(before)
    after = tvm.tir.transform.BF16ComputeLegalize()(after)

    tvm.ir.assert_structural_equal(after, expected)


def test_bf16_storage_legalize():
    before = get_after_compute_legalize()
    after = tvm.tir.transform.BF16StorageLegalize()(before)
    expected = get_after_storage_legalize()
    tvm.ir.assert_structural_equal(after, expected)


def vectorize_loops(module):
    def vectorize(stmt):
        if isinstance(stmt, tvm.tir.For):
            return tvm.tir.For(stmt.loop_var, stmt.min, stmt.extent, tvm.tir.ForKind.VECTORIZED, stmt.body)
    body = tvm.tir.stmt_functor.ir_transform(module["main"].body, None, vectorize, ["tir.For"])
    return tvm.tir.transform.VectorizeLoop()(tvm.IRModule({"main": module["main"].with_body(body)}))


def test_bf16_copy_to_promoted_buffer():
    import ml_dtypes
    import numpy as np

    @T.prim_func
    def copy(A: T.Buffer((4,), "bfloat16"), B: T.Buffer((4,), "bfloat16")):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        workspace = T.allocate([4], "bfloat16", "global")
        C = T.decl_buffer((4,), "bfloat16", data=workspace)
        for i in range(4):
            C[i] = A[i]
        for i in range(4):
            B[i] = C[i]

    @T.prim_func
    def expected(A: T.Buffer((4,), "bfloat16"), B: T.Buffer((4,), "bfloat16")):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        workspace = T.allocate([4], "float32", "global")
        C = T.decl_buffer((4,), "float32", data=workspace)
        for i in range(4):
            C[i] = bf16tof32(A[i])
        for i in range(4):
            B[i] = f32tobf16(C[i])

    before = tvm.IRModule({"main": copy})
    words = np.array([15616, 48896, 16256, 49152], dtype=np.uint16)
    source = tvm.nd.array(words.view(ml_dtypes.bfloat16))
    for vectorized in (False, True):
        candidate, oracle = before, tvm.IRModule({"main": expected})
        if vectorized:
            candidate = vectorize_loops(candidate)
            oracle = vectorize_loops(oracle)
        after = tvm.tir.transform.BF16ComputeLegalize()(candidate)
        tvm.ir.assert_structural_equal(after, oracle)
        tvm.ir.assert_structural_equal(tvm.tir.transform.BF16ComputeLegalize()(after), after)
        result = tvm.nd.empty((4,), "bfloat16")
        tvm.build(candidate, "llvm")(source, result)
        np.testing.assert_array_equal(result.numpy(), words)


def test_bf16_cast_roundtrip_retains_rounding():
    import numpy as np

    @T.prim_func
    def roundtrip(A: T.Buffer((4,), "float32"), B: T.Buffer((4,), "float32")):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        workspace = T.allocate([4], "bfloat16", "global")
        C = T.decl_buffer((4,), "bfloat16", data=workspace)
        for i in range(4):
            C[i] = T.Cast("bfloat16", A[i])
        for i in range(4):
            B[i] = T.Cast("float32", C[i]) * T.float32(1000)

    values = np.array([.03126, -.501, 1.001, -2.005], dtype=np.float32)
    result = tvm.nd.empty((4,), "float32")
    tvm.build(roundtrip, "llvm")(tvm.nd.array(values), result)
    np.testing.assert_array_equal(result.numpy(), np.array([31.25, -500, 1000, -2000], dtype=np.float32))


def test_bf16_implicit_store_retains_rounding():
    import ml_dtypes
    import numpy as np

    @T.prim_func
    def arithmetic(A: T.Buffer((4,), "bfloat16"), B: T.Buffer((4,), "bfloat16")):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        workspace = T.allocate([4], "bfloat16", "global")
        C = T.decl_buffer((4,), "bfloat16", data=workspace)
        for i in range(4):
            C[i] = A[i] + T.Cast("bfloat16", T.float32(0.00390625))
        for i in range(4):
            B[i] = C[i] * T.Cast("bfloat16", T.float32(1000))

    source = tvm.nd.array(np.array([1, 1.0078125, -1, -1.0078125], dtype=ml_dtypes.bfloat16))
    expected = np.array([1000, 1016, -996, -1000], dtype=ml_dtypes.bfloat16).view(np.uint16)
    for vectorized in (False, True):
        module = tvm.IRModule({"main": arithmetic})
        if vectorized:
            module = vectorize_loops(module)
        legalized = tvm.tir.transform.BF16ComputeLegalize()(module)
        tvm.ir.assert_structural_equal(tvm.tir.transform.BF16ComputeLegalize()(legalized), legalized)
        result = tvm.nd.empty((4,), "bfloat16")
        tvm.build(module, "llvm")(source, result)
        np.testing.assert_array_equal(result.numpy(), expected)


def test_bf16_store_preserves_accumulator_dtype():
    import ml_dtypes
    import numpy as np

    @T.prim_func
    def bf16_accumulator(A: T.Buffer((4,), "bfloat16"), B: T.Buffer((1,), "bfloat16")):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        workspace = T.allocate([1], "bfloat16", "global")
        C = T.decl_buffer((1,), "bfloat16", data=workspace)
        C[0] = T.Cast("bfloat16", T.float32(1))
        for k in range(4):
            C[0] = C[0] + A[k]
        B[0] = C[0]

    @T.prim_func
    def fp32_accumulator(A: T.Buffer((4,), "bfloat16"), B: T.Buffer((1,), "bfloat16")):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        workspace = T.allocate([1], "float32", "global")
        C = T.decl_buffer((1,), "float32", data=workspace)
        C[0] = T.float32(1)
        for k in range(4):
            C[0] = C[0] + T.Cast("float32", A[k])
        B[0] = T.Cast("bfloat16", C[0])

    source = tvm.nd.array(np.full((4,), 0.00390625, dtype=ml_dtypes.bfloat16))
    for function, expected in ((bf16_accumulator, 1), (fp32_accumulator, 1.015625)):
        module = tvm.IRModule({"main": function})
        legalized = tvm.tir.transform.BF16ComputeLegalize()(module)
        tvm.ir.assert_structural_equal(tvm.tir.transform.BF16ComputeLegalize()(legalized), legalized)
        result = tvm.nd.empty((1,), "bfloat16")
        tvm.build(module, "llvm")(source, result)
        np.testing.assert_array_equal(result.numpy(), np.array([expected], dtype=ml_dtypes.bfloat16).view(np.uint16))


def test_bf16_storage_scope():
    def get_before():
        @tvm.script.ir_module
        class Before:
            @T.prim_func
            def main(
                Aptr: T.handle("bfloat16", storage_scope="shared"),
                Bptr: T.handle("bfloat16", storage_scope="local"),
                Dptr: T.handle("bfloat16"),
            ):
                T.func_attr({"global_symbol": "main"})
                A = T.decl_buffer((100,), "bfloat16", data=Aptr)
                B = T.decl_buffer((100,), "bfloat16", data=Bptr)
                D = T.decl_buffer((100,), "bfloat16", data=Dptr)
                C = T.decl_buffer((100,), "bfloat16")
                for i in T.grid(100):
                    C[i] = A[i] + B[i]
                    D[i] = T.exp(C[i])

        return Before

    def after_compute_legalize():
        @tvm.script.ir_module
        class After:
            @T.prim_func
            def main(
                Aptr: T.handle("bfloat16", storage_scope="shared"),
                Bptr: T.handle("bfloat16", storage_scope="local"),
                Dptr: T.handle("bfloat16"),
            ):
                T.func_attr({"global_symbol": "main"})
                A = T.decl_buffer((100,), "bfloat16", data=Aptr)
                B = T.decl_buffer((100,), "bfloat16", data=Bptr)
                D = T.decl_buffer((100,), "bfloat16", data=Dptr)
                C = T.decl_buffer((100,), "float32")
                for i in T.grid(100):
                    C[i] = bf16tof32(f32tobf16(bf16tof32(A[i]) + bf16tof32(B[i])))
                    D[i] = f32tobf16(T.exp(C[i]))

        return After

    def after_storage_legalize():
        @tvm.script.ir_module
        class After:
            @T.prim_func
            def main(
                Aptr: T.handle("uint16", storage_scope="shared"),
                Bptr: T.handle("uint16", storage_scope="local"),
                Dptr: T.handle("uint16"),
            ):
                T.func_attr({"global_symbol": "main"})
                A = T.decl_buffer((100,), "uint16", data=Aptr)
                B = T.decl_buffer((100,), "uint16", data=Bptr)
                D = T.decl_buffer((100,), "uint16", data=Dptr)
                C = T.decl_buffer((100,), "float32")
                for i in T.grid(100):
                    C[i] = u16tof32(f32tou16(u16tof32(A[i]) + u16tof32(B[i])))
                    D[i] = f32tou16(T.exp(C[i]))

        return After

    before = get_before()
    after_compute = tvm.tir.transform.BF16ComputeLegalize()(before)
    after_storage = tvm.tir.transform.BF16StorageLegalize()(after_compute)
    tvm.ir.assert_structural_equal(after_compute, after_compute_legalize())
    tvm.ir.assert_structural_equal(after_storage, after_storage_legalize())


if __name__ == "__main__":
    test_bf16_storage_legalize()
    test_bf16_storage_scope()
