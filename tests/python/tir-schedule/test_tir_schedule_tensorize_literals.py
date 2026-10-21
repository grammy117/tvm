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
"""Tensorization compares literal representations rather than floating equality."""

import numpy as np
import pytest
import tvm
from tvm import te, tir
from tvm.script import tir as T
from tvm.tir.schedule.testing import verify_trace_roundtrip


def literal_program(bits):
    value = tir.FloatImm("float32", np.array(bits, "uint32").view("float32").item())
    output = te.compute((8,), lambda i: value, name="fill")
    return te.create_prim_func([output])


def literal_intrinsic(bits):
    word = tir.IntImm("uint32", bits)
    value = tir.FloatImm("float32", np.array(bits, "uint32").view("float32").item())

    @T.prim_func
    def desc(b: T.handle):
        B = T.match_buffer(b, (4,), "float32", offset_factor=1)
        with T.block("root"):
            T.reads()
            T.writes(B[0:4])
            for i in range(4):
                with T.block("fill"):
                    vi = T.axis.spatial(4, i)
                    B[vi] = value

    @T.prim_func
    def impl(b: T.handle):
        B = T.match_buffer(b, (4,), "float32", offset_factor=1)
        with T.block("root"):
            T.reads()
            T.writes(B[0:4])
            for i in range(4):
                B[i] = T.reinterpret("float32", word)

    name = f"test.literal_bits.{bits:08x}"
    tir.TensorIntrin.register(name, desc, impl, override=True)
    return name


@pytest.mark.parametrize("bits", [0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC12345, 0xFFC12345])
def test_tensorize_preserves_literal_bits_and_trace(bits):
    func = literal_program(bits)
    sch = tir.Schedule(func, debug_mask="all")
    _, inner = sch.split(sch.get_loops(sch.get_block("fill"))[0], [None, 4])
    sch.tensorize(inner, literal_intrinsic(bits))
    verify_trace_roundtrip(sch, func)
    output = tvm.nd.empty((8,), "float32")
    tvm.build(sch.mod, "llvm")(output)
    np.testing.assert_array_equal(output.numpy().view("uint32"), np.full(8, bits, "uint32"))


@pytest.mark.parametrize(
    "bits,other", [(0, 0x80000000), (0x7FC12345, 0x7FC54321), (0x7FC12345, 0xFFC12345)]
)
def test_tensorize_refuses_different_literal_bits(bits, other):
    sch = tir.Schedule(literal_program(bits), debug_mask="all")
    _, inner = sch.split(sch.get_loops(sch.get_block("fill"))[0], [None, 4])
    with pytest.raises(tir.ScheduleError):
        sch.tensorize(inner, literal_intrinsic(other))
