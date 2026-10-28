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
"""Direct TensorIR initialization without algebraic bias relocation."""

import unittest

import numpy as np
import pytest
import tvm
from tvm import relax, tir
from tvm.script import tir as T
from tvm.relax.backend.contrib.gemmini import prepare_gemmini_graph
from tvm.relax.backend.contrib.gemmini_tensorir import (
    make_tensorir_schedule, prepare_gemmini_tensorir_workloads,
)
import test_backend_contrib_gemmini_tensorir as reference


@T.prim_func
def seeded_product(A: T.Buffer((3, 5), "int8"), B: T.Buffer((5, 4), "int8"),
                   D: T.Buffer((4,), "int32"), C: T.Buffer((3, 4), "int32")):
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    for i, j, k in T.grid(3, 4, 5):
        with T.block("product"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = D[vj]
            C[vi, vj] = C[vi, vj] + T.Cast("int32", A[vi, vk]) * T.Cast("int32", B[vk, vj])


def bias_graph():
    a = relax.Var("a", relax.TensorStructInfo((3, 5), "int8"))
    b = relax.Var("b", relax.TensorStructInfo((5, 4), "int8"))
    d = relax.Var("bias", relax.TensorStructInfo((4,), "int32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [a, b, d]):
        with bb.dataflow():
            product = bb.emit(relax.op.matmul(a, b, out_dtype="int32"))
            result = bb.emit_output(relax.op.add(product, d))
        bb.emit_func_output(result)
    return bb.get()


def test_bias_epilogue_remains_after_zero_initialized_reduction():
    mod, references, normalizations = prepare_gemmini_tensorir_workloads(bias_graph())
    assert not references and not normalizations
    inits = []
    for func in mod.functions.values():
        if isinstance(func, tir.PrimFunc):
            assert not func.attrs or "gemmini.bias_preload" not in func.attrs
            tir.stmt_functor.post_order_visit(func.body, lambda n: inits.append(n.init.value)
                                             if isinstance(n, tir.Block) and n.init else None)
    assert inits and all(isinstance(value, tir.IntImm) and int(value) == 0 for value in inits)
    assert "bias" in mod.script()


def test_removed_bias_choice_is_not_silently_replayed():
    with pytest.raises(ValueError, match="Automatic bias preload was removed"):
        prepare_gemmini_tensorir_workloads(bias_graph(), bias_preload=True)


class GemminiDirectTests(unittest.TestCase):
    setUp = reference.GemminiTensorIRTests.setUp
    build = reference.GemminiTensorIRTests.build

    def test_explicit_seed_tensorizes_and_executes(self):
        result = make_tensorir_schedule(seeded_product)
        assert "tvm_gemmini_dma_acc_write" in result.scheduled_mod.script()
        runtime, control = self.build(result.scheduled_mod)
        a = np.arange(15, dtype="int8").reshape(3, 5) - 7
        b = np.arange(20, dtype="int8").reshape(5, 4) - 9
        d = np.array([-19, 23, 0, 1], dtype="int32")
        output = tvm.nd.empty((3, 4), "int32")
        runtime(tvm.nd.array(a), tvm.nd.array(b), tvm.nd.array(d), output)
        np.testing.assert_array_equal(output.numpy(), a.astype("int64") @ b.astype("int64") + d)
        assert control.emulator_faults() == 0
        assert control.emulator_pending() == 0

    def test_post_reduction_bias_executes_without_preload(self):
        report = []
        prepared = prepare_gemmini_graph(bias_graph(), report=report)
        assert [row["placement"] for row in report] == ["gemmini"]
        assert not report[0]["metadata"].get("bias_preloaded", False)
        func = next(f for f in prepared.functions.values() if isinstance(f, tir.PrimFunc))
        runtime, control = self.build(tvm.IRModule({"main": func.with_attr("global_symbol", "main")}))
        a = np.arange(15, dtype="int8").reshape(3, 5) - 7
        b = np.arange(20, dtype="int8").reshape(5, 4) - 9
        d = np.array([-19, 23, 0, 1], dtype="int32")
        output = tvm.nd.empty((3, 4), "int32")
        runtime(tvm.nd.array(a), tvm.nd.array(b), tvm.nd.array(d), output)
        np.testing.assert_array_equal(output.numpy(), a.astype("int64") @ b.astype("int64") + d)
        assert control.emulator_faults() == 0
        assert control.emulator_pending() == 0
