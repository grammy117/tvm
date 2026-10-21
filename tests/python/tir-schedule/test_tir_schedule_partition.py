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
# pylint: disable=missing-function-docstring,missing-module-docstring
import pytest
import numpy as np
import tvm
import tvm.testing
from tvm import te, tir
from tvm.script import tir as T
from tvm.tir.schedule.testing import (
    verify_trace_roundtrip,
)

# pylint: disable=no-member,invalid-name,unused-variable


def assert_partition_equivalent(expected, actual):
    """Check behavior independently of the choice of partition coordinates."""
    assert tir.analysis.verify_well_formed(actual)
    loops = []
    tir.stmt_functor.post_order_visit(
        actual.body, lambda node: loops.append(node) if isinstance(node, tir.For) else None
    )
    assert all(isinstance(loop.min, tir.IntImm) and int(loop.min) == 0 for loop in loops)

    def execute(func):
        # On CPU, replace the opaque GPU sink only for this test, preserving
        # its destination and index argument. This checks partitioned addresses,
        # without claiming execution of a GPU fragment instruction.
        def lower_sink(stmt):
            if (
                isinstance(stmt, tir.Evaluate)
                and isinstance(stmt.value, tir.Call)
                and stmt.value.op == tvm.ir.Op.get("tir.tvm_fill_fragment")
            ):
                call = stmt.value
                buffer = next(b for b in func.buffer_map.values() if b.data.same_as(call.args[0]))
                index = call.args[-1]
                return tir.BufferStore(
                    buffer,
                    tir.const(17, buffer.dtype),
                    [index // buffer.shape[1], index % buffer.shape[1]],
                )
            return None

        lowered = func.with_body(tir.stmt_functor.ir_transform(func.body, None, lower_sink))
        args = []
        for index, param in enumerate(func.params):
            buffer = func.buffer_map[param]
            shape = tuple(int(dim) for dim in buffer.shape)
            values = ((np.arange(np.prod(shape)).reshape(shape) + 13 * index) % 37 - 18).astype(
                buffer.dtype
            )
            args.append(tvm.nd.array(values))
        tvm.build(lowered.with_attr("global_symbol", "main"), "llvm")(*args)
        return [value.numpy() for value in args]

    for reference, result in zip(execute(expected), execute(actual)):
        np.testing.assert_array_equal(reference, result)


@T.prim_func
def elementwise(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128, 128))
    B = T.match_buffer(b, (128, 128, 128))
    for i, j, k in T.grid(128, 128, 128):
        with T.block("B"):
            vi, vj, vk = T.axis.remap("SSS", [i, j, k])
            B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@T.prim_func
def elementwise_symbolic(a: T.handle, b: T.handle, n: T.int32) -> None:
    A = T.match_buffer(a, (128, 128, n))
    B = T.match_buffer(b, (128, 128, n))
    for i, j, k in T.grid(128, 128, n):
        with T.block("B"):
            vi, vj, vk = T.axis.remap("SSS", [i, j, k])
            B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@T.prim_func
def elementwise_with_anno(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128, 128))
    B = T.match_buffer(b, (128, 128, 128))
    for i, j in T.grid(128, 128):
        for k in T.serial(0, 128, annotations={"useless_annotation": True}):
            with T.block("B"):
                vi, vj, vk = T.axis.remap("SSS", [i, j, k])
                T.reads([A[vi, vj, vk]])
                T.writes([B[vi, vj, vk]])
                B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@T.prim_func
def elementwise_with_thread_binding(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128, 128))
    B = T.match_buffer(b, (128, 128, 128))
    for i, j in T.grid(128, 128):
        for k in T.thread_binding(0, 128, thread="threadIdx.x"):
            with T.block("B"):
                vi, vj, vk = T.axis.remap("SSS", [i, j, k])
                T.reads([A[vi, vj, vk]])
                T.writes([B[vi, vj, vk]])
                B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@T.prim_func
def elementwise_with_opaque_block(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128, 128))
    B = T.match_buffer(b, (128, 128, 128))
    for i, j, k in T.grid(128, 128, 128):
        with T.block("opaque"):
            T.reads([A[i, j, k]])
            T.writes([B[i, j, k]])
            with T.block("B"):
                vi, vj, vk = T.axis.remap("SSS", [i, j, k])
                T.reads([A[vi, vj, vk]])
                T.writes([B[vi, vj, vk]])
                B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@T.prim_func
def opaque_access(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [16, 16], "float32")
    B = T.match_buffer(b, [16, 16], "float32")
    for i, j in T.grid(16, 16):
        with T.block("A"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads([])
            T.writes([A[0:16, 0:16]])
            A[vi, vj] = 1
    for i, j in T.grid(16, 16):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads([])
            T.writes([B[0:16, 0:16]])
            T.evaluate(T.tvm_fill_fragment(B.data, 16, 16, 16, 0, vi * 16 + vj, dtype="handle"))


# pylint: enable=no-member,invalid-name,unused-variable


def test_loop_partition():
    sch = tir.Schedule(elementwise, debug_mask="all")
    block_b = sch.get_block("B")
    i, j, k = sch.get_loops(block_b)
    sch.loop_partition(i, factors=[2, 1, 64])

    block_b_partition = sch.get_block("B_i0")
    i, j, k = sch.get_loops(block_b_partition)
    sch.loop_partition(j, factors=[4, 32])

    assert_partition_equivalent(elementwise, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise)


def test_partition_with_inferred_factor():
    sch = tir.Schedule(elementwise, debug_mask="all")
    block_b = sch.get_block("B")
    i, j, k = sch.get_loops(block_b)
    sch.loop_partition(i, factors=[None, 1, 64])

    block_b_partition = sch.get_block("B_i1")
    i, j, k = sch.get_loops(block_b_partition)
    sch.loop_partition(k, factors=[1, 64, None])

    assert_partition_equivalent(elementwise, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise)


def test_partition_with_opaque_block():
    sch = tir.Schedule(elementwise_with_opaque_block, debug_mask="all")
    block_opaque = sch.get_block("opaque")
    i, _, _ = sch.get_loops(block_opaque)
    sch.loop_partition(i, factors=[None, 16])
    assert_partition_equivalent(elementwise_with_opaque_block, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise_with_opaque_block)


def test_partition_with_opaque_access():
    sch = tir.Schedule(opaque_access, debug_mask="all")
    block_a = sch.get_block("A")
    _, j = sch.get_loops(block_a)
    sch.loop_partition(j, factors=[None, 4])
    block_b = sch.get_block("B")
    _, j = sch.get_loops(block_b)
    sch.loop_partition(j, factors=[None, 4])
    assert_partition_equivalent(opaque_access, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=opaque_access)


def test_partition_int64_extent_with_mixed_factors():
    def _create_prim_func():
        m = te.const(384, "int64")
        A = te.placeholder((m,), name="A", dtype="float32")
        B = te.compute((m,), lambda i: A[i] + 1, name="B")
        return te.create_prim_func([A, B])

    mod = _create_prim_func()
    sch = tir.Schedule(mod, debug_mask="all")
    (i,) = sch.get_loops(sch.get_block("B"))
    sch.loop_partition(
        i,
        factors=[
            te.const(1, "int64"),
            te.const(51, "int32"),
        ],
    )


def test_partition_reduction_keeps_one_initialization():
    source = te.placeholder((3, 11), "int32", name="source")
    k = te.reduce_axis((0, 11), name="k")
    summed = te.compute((3,), lambda i: te.sum(source[i, k], axis=k), name="sum")
    original = te.create_prim_func([source, summed])
    sch = tir.Schedule(original, debug_mask="all")
    sch.loop_partition(sch.get_loops(sch.get_block("sum"))[-1], [2, 3])
    values = np.arange(33, dtype="int32").reshape(3, 11) - 17
    actual = tvm.nd.empty((3,), "int32")
    tvm.build(sch.mod, "llvm")(tvm.nd.array(values), actual)
    np.testing.assert_array_equal(actual.numpy(), values.sum(axis=1, dtype="int32"))
    verify_trace_roundtrip(sch, original)


def test_partition_nonzero_loop_min_and_single_partition():
    @T.prim_func
    def original(A: T.Buffer((12,), "int32"), B: T.Buffer((12,), "int32")):
        for i in T.serial(3, 9):
            with T.block("copy"):
                vi = T.axis.spatial((3, 9), i)
                B[vi] = A[vi] + 7

    for factors in ([2], [None]):
        sch = tir.Schedule(original, debug_mask="all")
        loops = sch.loop_partition(sch.get_loops(sch.get_block("copy"))[0], factors)
        assert all(int(sch.get(loop).min) == 0 for loop in loops)
        values = np.arange(12, dtype="int32")
        actual = tvm.nd.array(np.full(12, -99, "int32"))
        tvm.build(sch.mod, "llvm")(tvm.nd.array(values), actual)
        expected = np.full(12, -99, "int32")
        expected[3:9] = values[3:9] + 7
        np.testing.assert_array_equal(actual.numpy(), expected)
        verify_trace_roundtrip(sch, original)


def test_partition_preserves_private_definitions_and_consumers():
    @T.prim_func
    def original(A: T.Buffer((7, 3), "int32"), B: T.Buffer((7, 3), "int32")):
        for i in range(7):
            with T.block("row"):
                vi = T.axis.spatial(7, i)
                tmp = T.alloc_buffer((3,), "int32")
                for j in range(3):
                    with T.block("local"):
                        vj = T.axis.spatial(3, j)
                        tmp[vj] = A[vi, vj] + 2
                for j in range(3):
                    with T.block("out"):
                        vj = T.axis.spatial(3, j)
                        B[vi, vj] = tmp[vj] * 3

    sch = tir.Schedule(original, debug_mask="all")
    sch.loop_partition(sch.get_loops(sch.get_block("row"))[0], [2])
    assert_partition_equivalent(original, sch.mod["main"])
    assert tir.analysis.verify_well_formed(sch.mod["main"])
    definitions = []
    tir.stmt_functor.post_order_visit(
        sch.mod["main"].body,
        lambda n: definitions.extend(n.alloc_buffers) if isinstance(n, tir.Block) else None,
    )
    assert len(definitions) == 2 and not definitions[0].data.same_as(definitions[1].data)
    verify_trace_roundtrip(sch, original)

    a = te.placeholder((7,), "int32", name="a")
    b = te.compute((7,), lambda i: a[i] * 2, name="b")
    c = te.compute((7,), lambda i: b[i] + 3, name="c")
    sch = tir.Schedule(te.create_prim_func([a, c]), debug_mask="all")
    sch.loop_partition(sch.get_loops(sch.get_block("b"))[0], [2])
    scope = sch.get_child_blocks(sch.get_block("root"))[0]
    assert [sch.get(block).name_hint for block in sch.get_consumers(scope)] == ["c"]


def test_partition_then_copy_canonicalization_and_blockization():
    from tvm import topi

    a = te.placeholder((2, 3), "int32", name="duplicate")
    b = te.placeholder((2, 4), "int32", name="duplicate")
    out = topi.concatenate([a, b], axis=1)
    original = te.create_prim_func([a, b, out])
    saved = tvm.ir.save_json(original)
    sch = tir.Schedule(original, debug_mask="all")
    i, j = sch.get_loops(sch.get_block("T_concat"))
    sch.reorder(j, i)
    sch.loop_partition(j, [3])
    regions = []
    for name in ("T_concat_ax10", "T_concat_ax11"):
        block = sch.get_block(name)
        copied = sch.cache_write(block, 0, "global")
        sch.compute_inline(block)
        sch.blockize(sch.get_loops(copied)[0])
        region = sch.cache_write(copied, 0, "global")
        sch.compute_inline(copied)
        assert isinstance(sch.get(region).body.value, tir.BufferLoad)
        assert len(sch.get(region).reads) == 1
        regions.append(region)
    values = np.arange(6, dtype="int32").reshape(2, 3)
    other = np.arange(8, dtype="int32").reshape(2, 4) + 19
    actual = tvm.nd.empty((2, 7), "int32")
    tvm.build(sch.mod, "llvm")(tvm.nd.array(values), tvm.nd.array(other), actual)
    np.testing.assert_array_equal(actual.numpy(), np.concatenate([values, other], axis=1))
    verify_trace_roundtrip(sch, original)
    assert tvm.ir.save_json(original) == saved


def test_partition_fail_symbolic():
    sch = tir.Schedule(elementwise_symbolic, debug_mask="all")
    block_b = sch.get_block("B")
    _, _, k = sch.get_loops(block_b)
    with pytest.raises(tvm.tir.ScheduleError):
        sch.loop_partition(k, factors=[10, None])


def test_partition_fail_out_of_bound():
    sch = tir.Schedule(elementwise, debug_mask="all")
    block_b = sch.get_block("B")
    i, j, k = sch.get_loops(block_b)
    with pytest.raises(tvm.tir.ScheduleError):
        sch.loop_partition(i, factors=[1000, 2, 3])


def test_partition_with_non_positive_factors():
    sch = tir.Schedule(elementwise, debug_mask="all")
    block_b = sch.get_block("B")
    i, j, k = sch.get_loops(block_b)
    with pytest.raises(tvm.tir.ScheduleError):
        sch.loop_partition(i, factors=[-2, -64])
    with pytest.raises(tvm.tir.ScheduleError):
        sch.loop_partition(j, factors=[0, None])
    with pytest.raises(tvm.tir.ScheduleError):
        sch.loop_partition(k, factors=[None, -16])


def test_partition_fail_with_annotation():
    sch = tir.Schedule(elementwise_with_anno, debug_mask="all")
    block_b = sch.get_block("B")
    _, j, k = sch.get_loops(block_b)
    with pytest.raises(tvm.tir.ScheduleError):
        sch.loop_partition(k, factors=[None, 10])


def test_partition_fail_with_thread_binding():
    sch = tir.Schedule(elementwise_with_thread_binding, debug_mask="all")
    block_b = sch.get_block("B")
    _, j, k = sch.get_loops(block_b)
    with pytest.raises(tvm.tir.ScheduleError):
        sch.loop_partition(k, factors=[None, 10])


if __name__ == "__main__":
    tvm.testing.main()
