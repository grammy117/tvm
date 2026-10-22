# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Native checks of graph-derived static calls, mixed dtypes and live storage."""

import ctypes
from _ctypes import dlclose
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import tvm
from tvm import relax, tir

APP = Path(tvm.__file__).resolve().parents[2] / "apps/gemmini"
sys.path.insert(0, str(APP))
from baremetal import export_graph, export_graphs, plan_graph, preflight_graph


def aligned(shape, dtype):
    dtype = np.dtype(dtype)
    raw = np.empty(int(np.prod(shape)) * dtype.itemsize + 64, "uint8")
    start = -raw.ctypes.data % 64
    return raw[start:start + int(np.prod(shape)) * dtype.itemsize].view(dtype).reshape(shape)


def graph(dtype="int8", tuple_output=False, shape=(2, 3), weight_offset=0):
    x = relax.Var("x", relax.TensorStructInfo(shape, dtype))
    w = np.arange(12, dtype=dtype).reshape(3, 4) - 5 + weight_offset
    builder = relax.BlockBuilder()
    with builder.function("main", [x]):
        with builder.dataflow():
            mm = builder.emit(relax.op.matmul(x, relax.const(w), out_dtype="int32" if dtype == "int8" else dtype))
            a = builder.emit(relax.op.add(mm, relax.const(3, "int32" if dtype == "int8" else dtype)))
            b = builder.emit(relax.op.multiply(a, relax.const(2, "int32" if dtype == "int8" else dtype)))
            c = builder.emit(relax.op.subtract(b, relax.const(1, "int32" if dtype == "int8" else dtype)))
            output = builder.emit_output(relax.Tuple([mm, c, x, c]) if tuple_output else c)
        builder.emit_func_output(output)
    return relax.transform.LegalizeOps()(builder.get())


def custom_graph(*, alignment=64, multi_output=False, attribute=None, storage_alignment=None):
    buffers = [tir.decl_buffer((8,), "int32", name=name, data_alignment=alignment) for name in ("x", "dead", "live")]
    x, dead, live = buffers
    index = tir.Var("i", "int32")
    def write(buffer, amount):
        body = tir.BufferStore(buffer, tir.BufferLoad(x, [index]) + amount, [index])
        return tir.For(index, 0, 8, tir.ForKind.SERIAL, body)
    selected = buffers if multi_output else [x, live]
    body = tir.SeqStmt([write(live, 11), write(dead, -37)]) if multi_output else write(live, 11)
    if storage_alignment is not None:
        body = tir.AttrStmt(x.data, "storage_alignment", storage_alignment, body)
    primitive = tir.PrimFunc([buffer.data for buffer in selected], body,
                            buffer_map={buffer.data: buffer for buffer in selected}).with_attr("tir.noalias", True)
    if attribute:
        primitive = primitive.with_attr(*attribute)
    value = relax.Var("x", relax.TensorStructInfo((8,), "int32"))
    builder = relax.BlockBuilder()
    gv = builder.add_func(primitive, "producer")
    with builder.function("main", [value]):
        with builder.dataflow():
            info = relax.TensorStructInfo((8,), "int32")
            result = builder.emit(relax.call_tir(gv, [value], [info, info] if multi_output else info))
            live_result = builder.emit(relax.TupleGetItem(result, 1)) if multi_output else result
            output = builder.emit_output(relax.op.add(live_result, relax.const(1, "int32")))
        builder.emit_func_output(output)
    return relax.transform.LegalizeOps()(builder.get())


def residual_graph():
    value = relax.Var("x", relax.TensorStructInfo((8,), "int32"))
    builder = relax.BlockBuilder()
    with builder.function("main", [value]):
        with builder.dataflow():
            residual = builder.emit(relax.op.add(value, relax.const(np.full(8, 1, "int32"))))
            branch = builder.emit(relax.op.multiply(residual, relax.const(np.full(8, 2, "int32"))))
            branch = builder.emit(relax.op.add(branch, relax.const(np.full(8, 3, "int32"))))
            output = builder.emit_output(relax.op.add(residual, branch))
        builder.emit_func_output(output)
    return relax.transform.LegalizeOps()(builder.get())


class StaticGraphTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gemmini-static-graph-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def compile(self, mod):
        report = export_graph(mod, self.root, target="llvm")
        library = self.root / "model.so"
        subprocess.run(["cc", "-shared", "-fPIC", "-std=c11", "-O2", self.root / "model.c", self.root / "constants.S", self.root / "operators.o", "-o", library], check=True, capture_output=True)
        loaded = ctypes.CDLL(str(library))
        self.addCleanup(dlclose, loaded._handle)
        loaded.model_run.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint64]
        loaded.model_run.restype = ctypes.c_int32
        inputs = [aligned(item["shape"], item["dtype"]) for item in report["inputs"]]
        outputs = [aligned(item["shape"], item["dtype"]) for item in report["outputs"]]
        workspace = aligned((max(report["workspace_bytes"], 64),), "uint8")
        return report, loaded, inputs, outputs, workspace

    def invoke(self, loaded, inputs, outputs, workspace, size=None):
        ins = (ctypes.c_void_p * len(inputs))(*(array.ctypes.data for array in inputs))
        outs = (ctypes.c_void_p * len(outputs))(*(array.ctypes.data for array in outputs))
        return loaded.model_run(ins, outs, workspace.ctypes.data, workspace.nbytes if size is None else size)

    def test_absolute_constant_binding_reuses_operator_and_payload(self):
        report = export_graph(graph(), self.root, target="llvm")
        constants = aligned((report["constants_bytes"],), "uint8")
        constants[:] = np.fromfile(self.root / "constants.bin", dtype="uint8")
        library = self.root / "absolute.so"
        subprocess.run(["cc", "-shared", "-fPIC", "-std=c11", "-O2",
                        f"-DTVM_MODEL_CONSTANTS_ADDRESS=UINT64_C({constants.ctypes.data})",
                        self.root / "model.c", self.root / "operators.o", "-o", library],
                       check=True, capture_output=True)
        loaded = ctypes.CDLL(str(library))
        self.addCleanup(dlclose, loaded._handle)
        loaded.model_run.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint64]
        loaded.model_run.restype = ctypes.c_int32
        inputs = [aligned(item["shape"], item["dtype"]) for item in report["inputs"]]
        outputs = [aligned(item["shape"], item["dtype"]) for item in report["outputs"]]
        workspace = aligned((max(report["workspace_bytes"], 64),), "uint8")
        inputs[0][:] = np.arange(6).reshape(2, 3)
        expected = ((inputs[0].astype("int32") @ (np.arange(12, dtype="int32").reshape(3, 4) - 5)) + 3) * 2 - 1
        # No constants.S/model_constants symbol is linked. Both calls must use
        # the original bytes at the explicit address, including alias guards.
        for _ in range(2):
            self.assertEqual(self.invoke(loaded, inputs, outputs, workspace), 0)
            np.testing.assert_array_equal(outputs[0], expected)
        self.assertEqual(report["constants_binding"]["absolute_address_macro"], "TVM_MODEL_CONSTANTS_ADDRESS")

    def test_view_only_program_has_no_operator_calls_or_workspace(self):
        x = relax.Var("x", relax.TensorStructInfo((2, 3), "int32"))
        builder = relax.BlockBuilder()
        with builder.function("main", [x]):
            with builder.dataflow():
                first = builder.emit(relax.op.reshape(x, (3, 2)))
                second = builder.emit(relax.op.reshape(first, (6,)))
                output = builder.emit_output(relax.Tuple([first, second, x]))
            builder.emit_func_output(output)
        report, loaded, inputs, outputs, workspace = self.compile(builder.get())
        self.assertFalse(report["calls"])
        self.assertEqual(report["workspace_bytes"], 0)
        self.assertEqual(len(report["views"]), 2)
        self.assertEqual(report["memory_preflight"]["completion"]["live_tensor_bytes"], 24)
        inputs[0][:] = np.arange(6).reshape(2, 3) - 4
        self.assertEqual(self.invoke(loaded, inputs, outputs, workspace, size=0), 0)
        for actual in outputs:
            np.testing.assert_array_equal(actual.flat, inputs[0].flat)
        aliases = [inputs[0].reshape(outputs[0].shape), *outputs[1:]]
        self.assertEqual(self.invoke(loaded, inputs, aliases, workspace), -2)

    def test_constant_only_program_keeps_scalar_bits_and_zero_input_abi(self):
        scalar = relax.const(np.array(-0.0, "float32"))
        vector = relax.const(np.array([-2147483648, 7, 2147483647], "int32"))
        function = relax.Function([], relax.Tuple([scalar, vector]))
        report, loaded, inputs, outputs, workspace = self.compile(tvm.IRModule({"main": function}))
        self.assertFalse(report["calls"])
        self.assertFalse(inputs)
        self.assertEqual(report["workspace_bytes"], 0)
        self.assertEqual(loaded.model_run(None, (ctypes.c_void_p * 2)(*(a.ctypes.data for a in outputs)), None, 0), 0)
        self.assertEqual(int(outputs[0].reshape(1).view("uint32")[0]), 0x80000000)
        np.testing.assert_array_equal(outputs[1], [-2147483648, 7, 2147483647])

    def test_returned_view_retains_backing_storage_until_completion(self):
        x = relax.Var("x", relax.TensorStructInfo((8,), "int32"))
        builder = relax.BlockBuilder()
        with builder.function("main", [x]):
            with builder.dataflow():
                backing = builder.emit(x + relax.const(1, "int32"))
                view = builder.emit(relax.op.reshape(backing, (2, 4)))
                branch1 = builder.emit(x * relax.const(2, "int32"))
                branch2 = builder.emit(x * relax.const(3, "int32"))
                result = builder.emit(branch1 + branch2)
                output = builder.emit_output(relax.Tuple([view, result]))
            builder.emit_func_output(output)
        graph = relax.transform.RewriteDataflowReshape()(relax.transform.LegalizeOps()(builder.get()))
        plan = plan_graph(graph)
        self.assertEqual(len(plan["views"]), 1)
        self.assertEqual(plan["views"][0].storage.last_use, len(plan["calls"]))
        report, loaded, inputs, outputs, workspace = self.compile(graph)
        self.assertEqual(len(report["calls"]), 4)
        self.assertEqual(len(report["views"]), 1)
        inputs[0][:] = np.arange(8) - 4
        self.assertEqual(self.invoke(loaded, inputs, outputs, workspace), 0)
        np.testing.assert_array_equal(outputs[0], (inputs[0] + 1).reshape(2, 4))
        np.testing.assert_array_equal(outputs[1], inputs[0] * 5)

    def test_stage_bundle_shares_constants_and_links_distinct_entrypoints(self):
        modules = {"prefill": graph(), "decode": graph(shape=(1, 3)), "private": graph(shape=(1, 3), weight_offset=1)}
        before = {name: mod.script(show_meta=True) for name, mod in modules.items()}
        report = export_graphs(modules, self.root / "bundle", target="llvm")
        self.assertEqual(before, {name: mod.script(show_meta=True) for name, mod in modules.items()})
        folder = self.root / "bundle"
        self.assertLess(report["constants_bytes"], report["duplicated_stage_constants_bytes"])
        self.assertGreater(report["constants_bytes"], plan_graph(modules["prefill"])["constants_bytes"])
        self.assertFalse(report["session_execution"])
        self.assertEqual(report["state_routing"], "caller_owned")
        self.assertEqual(report["serial_workspace_bytes"], max(stage["workspace_bytes"] for stage in report["stages"].values()))
        library = folder / "model.so"
        sources = [folder / "runtime.c", folder / "constants.S"]
        sources.extend(folder / stage / file for stage in report["stages"] for file in ("model.c", "operators.o"))
        subprocess.run(["cc", "-shared", "-fPIC", "-std=c11", "-O2", *sources, "-o", library], check=True, capture_output=True)
        loaded = ctypes.CDLL(str(library))
        self.addCleanup(dlclose, loaded._handle)
        workspace = aligned((max(report["serial_workspace_bytes"], 64),), "uint8")
        retained = []
        for name in ("prefill", "decode", "private", "prefill", "decode"):
            stage = report["stages"][name]
            runner = getattr(loaded, stage["entrypoint"])
            runner.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint64]
            runner.restype = ctypes.c_int32
            value = aligned(stage["inputs"][0]["shape"], "int8")
            value[:] = np.arange(value.size).reshape(value.shape) - 3
            output = aligned(stage["outputs"][0]["shape"], "int32")
            ins, outs = (ctypes.c_void_p * 1)(value.ctypes.data), (ctypes.c_void_p * 1)(output.ctypes.data)
            expected = (value.astype("int64") @ (np.arange(12).reshape(3, 4) - 5 + (name == "private")) + 3) * 2 - 1
            self.assertEqual(runner(ins, outs, workspace.ctypes.data, workspace.nbytes), 0)
            np.testing.assert_array_equal(output, expected)
            retained.append((output, expected.copy()))
            self.assertEqual(runner(ins, ins, workspace.ctypes.data, workspace.nbytes), -2)
            for old, reference in retained:
                np.testing.assert_array_equal(old, reference)

    def test_stage_bundle_preflights_all_stages_before_artifacts(self):
        output = self.root / "refused"
        limit = preflight_graph(graph())["allocated"]["explicit_tensor_bytes"]
        with self.assertRaisesRegex(ValueError, "memory"):
            export_graphs({"small": graph(), "large": graph(shape=(20, 3))}, output, target="llvm", memory_limit_bytes=limit)
        self.assertFalse(output.exists())
        for modules in ({}, {"bad-name": graph()}, {1: graph()}):
            with self.assertRaisesRegex(ValueError, "C identifiers"):
                export_graphs(modules, output, target="llvm")
        self.assertFalse(output.exists())

    def test_float64_and_bfloat16_storage_preserve_declared_arithmetic(self):
        import ml_dtypes
        report, loaded, inputs, outputs, workspace = self.compile(graph(dtype="float64"))
        inputs[0][:] = np.arange(6).reshape(2, 3) / 4 + 2. ** -30
        expected = (inputs[0] @ (np.arange(12).reshape(3, 4) - 5) + 3) * 2 - 1
        self.assertTrue(np.any(expected != expected.astype("float32").astype("float64")))
        self.assertEqual(self.invoke(loaded, inputs, outputs, workspace), 0)
        np.testing.assert_array_equal(outputs[0], expected)
        self.root = self.root / "bf16"
        self.root.mkdir()
        x = relax.Var("x", relax.TensorStructInfo((8,), "bfloat16"))
        values = np.array([-3.5, -0., 0., 0.5, 1., 1.0078125, 2., 127.], dtype=ml_dtypes.bfloat16)
        constants = np.array([0., -0., 0., .001953125, .00390625, .00390625, .0078125, .25], dtype=ml_dtypes.bfloat16)
        constant = tvm.nd.empty((8,), "bfloat16")
        constant.copyfrom(constants.view("uint16"))
        builder = relax.BlockBuilder()
        with builder.function("main", [x]):
            result = builder.emit(relax.op.add(relax.op.astype(x, "float32"), relax.op.astype(relax.Constant(constant), "float32")))
            builder.emit_func_output(relax.op.astype(result, "bfloat16"))
        mod = relax.transform.LegalizeOps()(builder.get())
        report, loaded, inputs, outputs, workspace = self.compile(mod)
        inputs[0][:] = values
        self.assertEqual(self.invoke(loaded, inputs, outputs, workspace), 0)
        expected = (values.astype("float32") + constants.astype("float32")).astype(ml_dtypes.bfloat16)
        np.testing.assert_array_equal(outputs[0].view("uint16"), expected.view("uint16"))
        self.assertEqual(report["constants"][0]["bytes"], 16)
        original = np.fromfile(self.root / "constants.bin", "uint16", count=8)
        np.testing.assert_array_equal(original, constants.view("uint16"))
        riscv = export_graph(mod, self.root / "riscv")
        self.assertEqual(riscv["inputs"][0]["dtype"], "bfloat16")

    def test_mixed_dtype_calls_reuse_and_repeated_execution(self):
        report, loaded, inputs, outputs, workspace = self.compile(graph())
        self.assertEqual(len(report["calls"]), 4)
        self.assertEqual(report["workspace_bytes"], 128)
        offsets = [t["offset"] for t in report["intermediates"] if t["origin"] == "workspace"]
        self.assertEqual(offsets, [0, 64, 0])
        weights = np.arange(12, dtype="int64").reshape(3, 4) - 5
        for seed in (0, 1, 0):
            value = (np.arange(6).reshape(2, 3) - 3 + seed).astype("int8")
            inputs[0][:] = value
            outputs[0][:] = -777
            self.assertEqual(self.invoke(loaded, inputs, outputs, workspace), 0)
            np.testing.assert_array_equal(outputs[0], (value.astype("int64") @ weights + 3) * 2 - 1)
            np.testing.assert_array_equal(inputs[0], value)

    def test_bfloat16_fusion_preserves_intermediate_store_rounding(self):
        import ml_dtypes
        from tvm.relax.backend.contrib.gemmini import prepare_gemmini_graph_legacy as prepare_gemmini_graph

        values = np.array([1, 1.0078125, -1, -1.0078125], dtype=ml_dtypes.bfloat16)
        delta = np.full((4,), 2. ** -8, dtype=ml_dtypes.bfloat16)
        factor = np.full((4,), 1000, dtype=ml_dtypes.bfloat16)

        def round_bf16(value):
            words = np.asarray(value, dtype="float32").view("uint32")
            bias = np.uint32(0x7fff) + ((words >> np.uint32(16)) & np.uint32(1))
            return ((words + bias) >> np.uint32(16)).astype("uint16")

        stored = round_bf16(values.astype("float32") + delta.astype("float32"))
        decoded = (stored.astype("uint32") << np.uint32(16)).view("float32")
        expected = round_bf16(decoded * factor.astype("float32"))
        np.testing.assert_array_equal(expected, np.array([1000, 1016, -996, -1000], dtype=ml_dtypes.bfloat16).view("uint16"))
        params = [relax.Var(name, relax.TensorStructInfo((4,), "bfloat16")) for name in ("x", "delta", "factor")]
        builder = relax.BlockBuilder()
        with builder.function("main", params):
            with builder.dataflow():
                intermediate = builder.emit(relax.op.add(params[0], params[1]))
                result = builder.emit_output(relax.op.multiply(intermediate, params[2]))
            builder.emit_func_output(result)
        original = builder.get()
        root = self.root
        for optimized in (False, True):
            with self.subTest(optimized=optimized):
                self.root = root / ("optimized" if optimized else "baseline")
                report, loaded, inputs, outputs, workspace = self.compile(prepare_gemmini_graph(original, optimize=optimized))
                self.assertEqual(report["outputs"][0]["dtype"], "bfloat16")
                self.assertEqual(len(report["calls"]), 1 if optimized else 2)
                self.assertFalse(report["runtime_workspace_allocator_required"])
                for reversed_input in (False, True, False):
                    inputs[0][:] = values[::-1] if reversed_input else values
                    inputs[1][:], inputs[2][:] = delta, factor
                    before = [value.view("uint16").copy() for value in inputs]
                    outputs[0].view("uint16").fill(0x5555)
                    self.assertEqual(self.invoke(loaded, inputs, outputs, workspace), 0)
                    np.testing.assert_array_equal(outputs[0].view("uint16"), expected[::-1] if reversed_input else expected)
                    for value, saved in zip(inputs, before):
                        np.testing.assert_array_equal(value.view("uint16"), saved)

    def test_fp32_pointwise_inline_keeps_declared_precision(self):
        from tvm.relax.backend.contrib.gemmini import prepare_gemmini_graph_legacy as prepare_gemmini_graph

        params = [relax.Var(name, relax.TensorStructInfo((4,), "float32")) for name in ("x", "delta", "factor")]
        builder = relax.BlockBuilder()
        with builder.function("main", params):
            with builder.dataflow():
                intermediate = builder.emit(relax.op.add(params[0], params[1]))
                result = builder.emit_output(relax.op.multiply(intermediate, params[2]))
            builder.emit_func_output(result)
        mod = prepare_gemmini_graph(builder.get(), optimize=True)
        report, loaded, inputs, outputs, workspace = self.compile(mod)
        inputs[0][:] = [1, 1.0078125, -1, -1.0078125]
        inputs[1].fill(2. ** -8)
        inputs[2].fill(1000)
        self.assertEqual(self.invoke(loaded, inputs, outputs, workspace), 0)
        np.testing.assert_array_equal(outputs[0], (inputs[0] + inputs[1]) * inputs[2])
        self.assertEqual(len(report["calls"]), 1)
        self.assertEqual(report["workspace_bytes"], 0)

    def test_low_precision_scalar_and_vector_stores_survive_inline(self):
        from tvm import te
        from tvm.relax.backend.contrib.gemmini import _inline_cpu_producers

        # Structural coverage only: the static exporter has no FP8 tensor ABI.
        for scalar in ("bfloat16", "e4m3_float8", "e5m2_float8", "float32"):
            for lanes in (1, 4):
                dtype = scalar if lanes == 1 else scalar + "x" + str(lanes)
                with self.subTest(dtype=dtype):
                    x, delta, factor = [te.placeholder((4,), dtype=dtype, name=name) for name in ("x", "delta", "factor")]
                    intermediate = te.compute((4,), lambda i: x[i] + delta[i], name="stored_add")
                    result = te.compute((4,), lambda i: intermediate[i] * factor[i], name="multiply")
                    primitive = te.create_prim_func([x, delta, factor, result])
                    after = _inline_cpu_producers(tvm.IRModule({"main": primitive}))
                    if scalar == "float32":
                        self.assertEqual(len(after["main"].body.block.alloc_buffers), 0)
                    else:
                        tvm.ir.assert_structural_equal(after["main"], primitive)

    def test_retained_tuple_results_alias_copies_and_workspace_rejection(self):
        report, loaded, inputs, outputs, workspace = self.compile(graph(tuple_output=True))
        value = np.arange(6, dtype="int8").reshape(2, 3) - 3
        inputs[0][:] = value
        for output in outputs:
            output.fill(17)
        self.assertEqual(self.invoke(loaded, inputs, outputs, workspace, size=0), -1)
        self.assertTrue(all(np.all(output == 17) for output in outputs))
        self.assertEqual(self.invoke(loaded, inputs, outputs, workspace), 0)
        expected = value.astype("int64") @ (np.arange(12).reshape(3, 4) - 5)
        for actual, wanted in zip(outputs, [expected, (expected + 3) * 2 - 1, value, (expected + 3) * 2 - 1]):
            np.testing.assert_array_equal(actual, wanted)
        copies = [output.copy() for output in outputs]
        self.assertEqual(self.invoke(loaded, inputs, [outputs[0]] * 4, workspace), -2)
        for output, before in zip(outputs, copies):
            np.testing.assert_array_equal(output, before)
        self.assertEqual(self.invoke(loaded, inputs, outputs, workspace), 0)
        self.assertFalse(report["heap_required_by_orchestration"])

    def test_fused_fp32_graph_keeps_mathematics(self):
        mod = graph(dtype="float32")
        optimized = relax.get_pipeline("zero")(mod)
        report, loaded, inputs, outputs, workspace = self.compile(optimized)
        inputs[0][:] = np.arange(6, dtype="float32").reshape(2, 3) - 3
        self.assertEqual(self.invoke(loaded, inputs, outputs, workspace), 0)
        expected = (inputs[0] @ (np.arange(12, dtype="float32").reshape(3, 4) - 5) + 3) * 2 - 1
        np.testing.assert_array_equal(outputs[0], expected)
        self.assertLess(len(report["calls"]), 4)

    def test_large_fused_buffers_use_usmp_with_explicit_guarded_workspace(self):
        optimized = relax.get_pipeline("zero")(graph(dtype="float32", shape=(128, 3)))
        report, loaded, inputs, outputs, workspace = self.compile(optimized)
        self.assertTrue(report["operator_workspaces"])
        preflight = preflight_graph(optimized, target="llvm")
        self.assertEqual(preflight["allocated"], report["memory_preflight"]["allocated"])
        self.assertGreater(preflight["peak_live_workspace_bytes"], 0)
        self.assertGreaterEqual(report["workspace_bytes"], 128 * 4 * 4)
        llvm = (self.root / "operators.ll").read_text()
        self.assertNotIn("TVMBackendAllocWorkspace", llvm)
        self.assertNotIn("TVMBackendFreeWorkspace", llvm)
        inputs[0][:] = np.arange(128 * 3, dtype="float32").reshape(128, 3) - 13
        outputs[0].fill(17)
        self.assertEqual(self.invoke(loaded, inputs, outputs, workspace, size=workspace.nbytes-1), -1)
        self.assertTrue(np.all(outputs[0] == 17))
        self.assertEqual(self.invoke(loaded, inputs, outputs, workspace), 0)
        expected = (inputs[0] @ (np.arange(12, dtype="float32").reshape(3, 4) - 5) + 3) * 2 - 1
        np.testing.assert_array_equal(outputs[0], expected)

    def test_baseline_maxpool_padding_does_not_require_a_guest_heap(self):
        from tvm.relax.backend.contrib.gemmini import prepare_gemmini_graph_legacy as prepare_gemmini_graph
        value = relax.Var("x", relax.TensorStructInfo((1, 1, 64, 64), "int8"))
        builder = relax.BlockBuilder()
        with builder.function("main", [value]):
            output = builder.emit(relax.op.nn.max_pool2d(value, pool_size=(3, 3), strides=(2, 2), padding=(1, 1)))
            builder.emit_func_output(output)
        prepared = prepare_gemmini_graph(builder.get(), optimize=False)
        report, loaded, inputs, outputs, workspace = self.compile(prepared)
        self.assertEqual(len(report["calls"]), 1)
        self.assertNotIn("TVMBackendAllocWorkspace", (self.root / "operators.ll").read_text())
        for offset in (0, 11):
            image = ((np.arange(4096).reshape(1, 1, 64, 64) * 13 + offset) % 255 - 127).astype("int8")
            inputs[0][:] = image
            workspace.fill(0xA5)
            self.assertEqual(self.invoke(loaded, inputs, outputs, workspace), 0)
            padded = np.pad(image, ((0, 0), (0, 0), (1, 1), (1, 1)), constant_values=-128)
            expected = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(2, 3))[:, :, ::2, ::2].max(axis=(-1, -2))
            np.testing.assert_array_equal(outputs[0], expected)
            np.testing.assert_array_equal(inputs[0], image)

    def test_boolean_private_buffers_use_byte_sized_usmp_storage(self):
        from tvm import te
        source = te.placeholder((1024,), "int32", "source")
        mask = te.compute((1024,), lambda i: source[i] > 0, name="mask")
        axis = te.reduce_axis((0, 1024), "r")
        count = te.compute((1,), lambda _: te.sum(mask[axis].astype("int32"), axis=axis), name="count")
        function = te.create_prim_func([source, count])
        value = relax.Var("x", relax.TensorStructInfo((1024,), "int32"))
        builder = relax.BlockBuilder()
        gv = builder.add_func(function, "count")
        with builder.function("main", [value]):
            with builder.dataflow():
                output = builder.emit_output(relax.call_tir(gv, (value,), relax.TensorStructInfo((1,), "int32")))
            builder.emit_func_output(output)
        report, loaded, inputs, outputs, workspace = self.compile(builder.get())
        self.assertGreaterEqual(report["workspace_bytes"], 1024)
        self.assertTrue(report["operator_workspaces"])
        inputs[0][:] = np.arange(1024, dtype="int32") - 500
        self.assertEqual(self.invoke(loaded, inputs, outputs, workspace), 0)
        np.testing.assert_array_equal(outputs[0], [523])
        self.assertNotIn("TVMBackendAllocWorkspace", (self.root / "operators.ll").read_text())

    def test_unused_multi_output_sibling_cannot_alias_live_result(self):
        report, loaded, inputs, outputs, workspace = self.compile(custom_graph(multi_output=True))
        first, second = report["intermediates"][:2]
        inputs[0][:] = np.arange(8, dtype="int32")
        self.assertEqual(self.invoke(loaded, inputs, outputs, workspace), 0)
        np.testing.assert_array_equal(outputs[0], inputs[0] + 12)
        self.assertNotEqual(first["offset"], second["offset"])

    def test_stronger_external_alignment_is_rejected(self):
        for mod in (custom_graph(alignment=128), custom_graph(storage_alignment=128)):
            with self.subTest(), self.assertRaisesRegex(ValueError, "alignment"):
                plan_graph(mod)

    def test_unsupported_function_abi_and_target_are_rejected(self):
        for attribute in (("calling_conv", 1), ("calling_conv", 2), ("target", tvm.target.Target("cuda -arch=sm_50"))):
            with self.subTest(attribute=attribute), self.assertRaisesRegex(ValueError, "calling convention|target"):
                plan_graph(custom_graph(attribute=attribute))

    def test_unknown_shapes_dtypes_and_memory_are_rejected(self):
        for mod in (graph(dtype="float16"), graph(shape=(tir.Var("batch", "int64"), 3))):
            with self.assertRaises(ValueError):
                plan_graph(mod)
        with self.assertRaises(ValueError):
            export_graph(graph(), self.root, target="llvm", memory_limit_bytes=1)
        self.assertFalse((self.root / "model.c").exists())

    def test_residual_preflight_and_execution_keep_branch_live(self):
        mod = residual_graph()
        preflight = preflight_graph(mod)
        self.assertEqual(preflight["allocated"]["workspace_bytes"], 192)
        self.assertEqual(preflight["peak_live_workspace_bytes"], 96)
        self.assertEqual(preflight["peak_live_workspace_aligned_bytes"], 192)
        self.assertEqual(preflight["allocated"]["constants_bytes"], 192)
        self.assertEqual(preflight["peak_live_tensor_bytes"], 160)
        self.assertEqual(preflight["allocated"]["explicit_tensor_bytes"], 448)
        self.assertEqual(preflight["per_call"][2]["live_tensors"],
                         ["constant_2", "tensor_0", "tensor_1", "tensor_2"])
        self.assertIn("tensor_0", preflight["per_call"][3]["live_tensors"])
        report, loaded, inputs, outputs, workspace = self.compile(mod)
        self.assertEqual(report["memory_preflight"], preflight)
        for seed in (-100, 37, -100):
            value = np.arange(8, dtype="int32") + seed
            inputs[0][:] = value
            workspace[:] = 0xa5
            self.assertEqual(self.invoke(loaded, inputs, outputs, workspace), 0)
            np.testing.assert_array_equal(outputs[0], 3 * value + 6)
            np.testing.assert_array_equal(inputs[0], value)

    def test_duplicate_returns_have_unique_liveness_and_disjoint_storage(self):
        mod = graph(tuple_output=True)
        plan = plan_graph(mod)
        report = preflight_graph(mod)
        outputs = plan["outputs"]
        self.assertIs(outputs[1], outputs[3])
        self.assertEqual(report["allocated"]["output_bytes"], 102)
        self.assertEqual(report["completion"]["live_tensor_bytes"], 70)
        self.assertEqual(len(report["completion"]["live_tensors"]), 3)
        self.assertEqual([item["copy_required"] for item in report["output_bindings"]],
                         [False, False, True, True])
        for entry in report["per_call"]:
            self.assertEqual(len(entry["live_tensors"]), len(set(entry["live_tensors"])))
        self.assertEqual(report["allocated"]["explicit_tensor_bytes"],
                         plan["workspace_bytes"] + plan["constants_bytes"] + 6 + 102)

    def test_preflight_budget_matches_export_before_codegen_or_artifacts(self):
        mod = residual_graph()
        report = preflight_graph(mod)
        required = report["allocated"]["explicit_tensor_bytes"]
        self.assertEqual(preflight_graph(mod, memory_limit_bytes=required)["allocated"],
                         report["allocated"])
        with patch("baremetal.tvm.build", side_effect=AssertionError("unexpected code generation")):
            for limit in (required - 1, 0, True, 1.5):
                with self.subTest(limit=limit):
                    with self.assertRaisesRegex(ValueError, "memory budget"):
                        preflight_graph(mod, memory_limit_bytes=limit)
                    with self.assertRaisesRegex(ValueError, "memory budget"):
                        export_graph(mod, self.root, target="llvm", memory_limit_bytes=limit)
            preflight_graph(mod, memory_limit_bytes=required)
        self.assertEqual(list(self.root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
