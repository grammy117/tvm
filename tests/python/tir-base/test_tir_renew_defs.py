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
import tvm.testing
import numpy as np
from tvm.script import tir as T
from tvm.tir.buffer import Buffer
from tvm.tir.function import PrimFunc
from tvm.tir.stmt import Block


def _check_func_signature_remap(lhs: PrimFunc, rhs: PrimFunc):
    assert lhs != rhs
    for x, y in zip(lhs.params, rhs.params):
        assert x != y
        assert lhs.buffer_map[x] != rhs.buffer_map[y]


def _check_buffer_decl(lhs: Buffer, rhs: Buffer):
    assert lhs != rhs
    assert lhs.data != rhs.data


def _check_block_signature_remap(lhs: Block, rhs: Block):
    assert lhs != rhs
    for x, y in zip(lhs.iter_vars, rhs.iter_vars):
        assert x != y
        assert x.var != y.var
    for x, y in zip(lhs.alloc_buffers, rhs.alloc_buffers):
        _check_buffer_decl(x, y)
    for x, y in zip(lhs.match_buffers, rhs.match_buffers):
        assert x != y
        _check_buffer_decl(x.buffer, y.buffer)


def test_simple():
    @T.prim_func
    # Buffer A should be remapped
    def elementwise(A: T.Buffer((128, 128), "float32")):
        # Buffer B should be remapped
        B = T.alloc_buffer((128, 128), "float32")
        # i, j should be remapped
        for i, j in T.grid(128, 128):
            with T.block("B"):
                # vi, vj should be remapped
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(A[vi, vj])
                T.writes(B[vi, vj])
                B[vi, vj] = A[vi, vj] * 2.0

    f1 = elementwise
    f2 = tvm.tir.stmt_functor.renew_defs(f1)
    tvm.ir.assert_structural_equal(f1, f2)

    _check_func_signature_remap(f1, f2)
    # check root block
    _check_block_signature_remap(f1.body.block, f2.body.block)
    # check remap of i
    assert f1.body.block.body.loop_var != f2.body.block.body.loop_var
    # check remap of j
    assert f1.body.block.body.body.loop_var != f2.body.block.body.body.loop_var

    # check inner block
    def _get_block(f):
        return f.body.block.body.body.body.block

    _check_block_signature_remap(_get_block(f1), _get_block(f2))


def test_match_buffer():
    # well-formed checker complains about multiple definitions for variable A0_s1,
    # likely stemming from strides=[s, s]
    @T.prim_func(check_well_formed=False)
    # A and B should be remapped
    def func_match_buffer(A: T.Buffer((128, 128), "float32"), B: T.Buffer((128, 128), "float32")):
        with T.block("root"):
            s = T.int32()
            e = T.int32()
            # A0 should be remapped
            A0 = T.match_buffer(
                A[0:128, 0:128],
                shape=(128, 128),
                dtype="float32",
                # s and e should be remapped
                strides=[s, s],
                elem_offset=e,
            )
            for i, j in T.grid(128, 128):
                with T.block("B"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B[vi, vj] = A0[vi, vj] * 2.0

    f1 = func_match_buffer
    f2 = tvm.tir.stmt_functor.renew_defs(f1)
    tvm.ir.assert_structural_equal(f1, f2)

    _check_func_signature_remap(f1, f2)
    _check_block_signature_remap(f1.body.block, f2.body.block)
    assert f1.body.block.body.loop_var != f2.body.block.body.loop_var

    def _get_block(f):
        return f.body.block

    block1 = _get_block(f1)
    block2 = _get_block(f2)
    _check_block_signature_remap(block1, block2)

    matched_buffer1 = block1.match_buffers[0].buffer
    matched_buffer2 = block2.match_buffers[0].buffer
    # Stride var s should be remapped
    assert matched_buffer1.strides[0] != matched_buffer2.strides[0]
    assert matched_buffer1.strides[1] != matched_buffer2.strides[1]
    # s should be only remapped once
    assert matched_buffer1.strides[0] == matched_buffer1.strides[1]
    assert matched_buffer2.strides[0] == matched_buffer2.strides[1]
    # Element-offset var e should be remapped
    assert matched_buffer1.elem_offset != matched_buffer2.elem_offset


def test_undefined_buffer():
    @T.prim_func
    def access_alloc():
        # Buffer A should be remapped
        A_data = T.allocate([128], "float16", "global")
        A = T.Buffer(shape=[128], dtype="float16", data=A_data)
        # check if buffer var also get remapped
        T.evaluate(A.data)
        for i in range(128):
            A[i] = A[i] + T.float16(1.0)

    f1 = access_alloc
    f2 = tvm.tir.stmt_functor.renew_defs(f1)
    tvm.ir.assert_structural_equal(f1, f2)

    assert f1.body.buffer_var != f2.body.buffer_var

    def _get_buffer_store_buffer(f):
        return f.body.body[1].body.buffer

    _check_buffer_decl(_get_buffer_store_buffer(f1), _get_buffer_store_buffer(f2))


def test_symbolic_func():
    @T.prim_func
    def symbolic_func(a: T.handle, b: T.handle, n: T.int32):
        m = T.int32()
        A = T.match_buffer(a, (n, m))
        B = T.match_buffer(b, (n, m * 2))
        for i, j in T.grid(n, m):
            B[i, j * 2] = A[i, j]
            B[i, j * 2 + 1] = A[i, j]

    f1 = symbolic_func
    f2 = tvm.tir.stmt_functor.renew_defs(f1)
    tvm.ir.assert_structural_equal(f1, f2)


def test_buffer_map():
    @T.prim_func
    def main(a: T.handle, b: T.handle):
        m = T.int64()
        A = T.match_buffer(a, (m * 2,))
        B = T.match_buffer(b, (m, 2))
        for i, j in T.grid(m, 2):
            with T.block("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi * 2 + vj]

    f1 = main
    f2 = tvm.tir.stmt_functor.renew_defs(main)
    tvm.ir.assert_structural_equal(f1, f2)
    assert f1.buffer_map[f1.params[1]].shape[0] != f2.buffer_map[f2.params[1]].shape[0]


def test_gather():
    @T.prim_func(private=True)
    def take(
        A: T.Buffer((4096, 4096), "float16"),
        B: T.Buffer((1,), "int32"),
        T_take: T.Buffer((1, 4096), "float16"),
    ):
        for ax0, ax1 in T.grid(1, 4096):
            with T.block("T_take"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(A[B[v_ax0], v_ax1], B[v_ax0])
                T.writes(T_take[v_ax0, v_ax1])
                T_take[v_ax0, v_ax1] = A[B[v_ax0], v_ax1]

    f1 = take
    f2 = tvm.tir.stmt_functor.renew_defs(take)
    tvm.ir.assert_structural_equal(f1, f2)


def test_nonzero_iteration_domain_preserves_extent_and_execution():
    @T.prim_func
    def source(A: T.Buffer((12,), "int32"), B: T.Buffer((12,), "int32")):
        for i in T.serial(6):
            with T.block("copy"):
                vi = T.axis.spatial((3, 9), i + 3)
                B[vi] = T.if_then_else(vi >= 6, A[vi], T.int32(-7))

    renewed = tvm.tir.stmt_functor.renew_defs(source)
    tvm.ir.assert_structural_equal(source, renewed)
    values = np.arange(12, dtype="int32")
    output = tvm.nd.array(np.full(12, -99, "int32"))
    tvm.build(renewed, "llvm")(tvm.nd.array(values), output)
    expected = np.full(12, -99, "int32")
    expected[3:6] = -7
    expected[6:9] = values[6:9]
    np.testing.assert_array_equal(output.numpy(), expected)


def test_free_buffer_identity_and_access_only_reference():
    buffer = tvm.tir.decl_buffer((8,), "int32", name="free")
    block = tvm.tir.Block(
        [],
        [tvm.tir.BufferRegion(buffer, [tvm.ir.Range(0, 8)])],
        [],
        "opaque",
        tvm.tir.Evaluate(buffer.data),
    )
    func = tvm.tir.PrimFunc([], tvm.tir.BlockRealize([], True, block))
    renewed = tvm.tir.stmt_functor.renew_defs(func)
    tvm.ir.assert_structural_equal(func, renewed)
    assert renewed.body.block.reads[0].buffer.same_as(buffer)


if __name__ == "__main__":
    tvm.testing.main()
