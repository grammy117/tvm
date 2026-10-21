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
"""View elimination, native indexed copies, and static alias storage checks."""

import ctypes
import subprocess
import unittest

import numpy as np
import tvm
from tvm import relax, te, tir, meta_schedule as ms
from tvm.relax.backend.contrib.gemmini import prepare_gemmini_graph
from tvm.relax.backend.contrib.gemmini_elementwise import make_elementwise_schedule
from tvm.relax.backend.contrib.gemmini_meta_schedule import gemmini_space_generator
from tvm.relax.backend.contrib.gemmini_target import GemminiCapabilities
from tvm.relax.backend.contrib.gemmini_tensorir import lower_tensorir_schedule
from tvm.tir.schedule.testing import verify_trace_roundtrip

import test_backend_contrib_gemmini_tensorir as reference
import test_backend_contrib_gemmini_baremetal as static


class GemminiViewTests(unittest.TestCase):
    setUp = reference.GemminiTensorIRTests.setUp
    build = reference.GemminiTensorIRTests.build

    def static_run(self, graph, values):
        self.index += 1
        folder = self.root / f"static-{self.index}"
        report = static.export_graph(graph, folder, target="llvm")
        path = folder / "model.so"
        subprocess.run(
            [
                "cc",
                "-shared",
                "-fPIC",
                "-std=c11",
                "-O2",
                folder / "model.c",
                folder / "constants.S",
                folder / "operators.o",
                self.emulator,
                "-o",
                path,
            ],
            check=True,
            capture_output=True,
        )
        library = ctypes.CDLL(str(path))
        library.model_run.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_uint64,
        ]
        library.model_run.restype = ctypes.c_int32
        library.emulator_reset()
        inputs = [static.aligned(value.shape, value.dtype) for value in values]
        for source, value in zip(inputs, values):
            source[...] = value
        outputs = [static.aligned(item["shape"], item["dtype"]) for item in report["outputs"]]
        workspace = static.aligned((max(report["workspace_bytes"], 64),), "uint8")
        status = library.model_run(
            (ctypes.c_void_p * len(inputs))(*(x.ctypes.data for x in inputs)),
            (ctypes.c_void_p * len(outputs))(*(x.ctypes.data for x in outputs)),
            workspace.ctypes.data,
            report["workspace_bytes"],
        )
        self.assertEqual(status, 0)
        self.assertEqual(library.emulator_faults(), 0)
        self.assertEqual(library.emulator_pending(), 0)
        for source, value in zip(inputs, values):
            np.testing.assert_array_equal(source.view("uint8"), value.view("uint8"))
        return report, outputs, library

    def test_indexed_views_and_metadata_compile_through_normal_graph_passes(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is unavailable")
        from tvm.relax.frontend.torch import from_exported_program

        for kind in (
            "squeeze_dim",
            "squeeze_nonunit",
            "squeeze_dims",
            "unsqueeze",
            "view",
            "permute",
            "select",
            "slice",
            "split",
            "diagonal",
            "as_strided",
            "scalar",
        ):
            for dtype in ("int8", "int32", "float32"):
                with self.subTest(kind=kind, dtype=dtype):

                    class View(torch.nn.Module):
                        def forward(self, x):
                            if kind == "squeeze_dim":
                                return x.squeeze(0)
                            if kind == "squeeze_nonunit":
                                return x.squeeze(1)
                            if kind == "squeeze_dims":
                                return x.squeeze((0, 1))
                            if kind == "unsqueeze":
                                return x.unsqueeze(0)
                            if kind == "view":
                                return x.view(2, 6)
                            if kind == "permute":
                                return x.permute(2, 0, 1)
                            if kind == "select":
                                return x.select(1, -1)
                            if kind == "slice":
                                return torch.ops.aten.slice.Tensor(x, 1, 0, 3, 2)
                            if kind == "split":
                                return torch.split(x, [1, 2], dim=1)
                            if kind == "diagonal":
                                return x.diagonal(1, 1, 2)
                            if kind == "as_strided":
                                return torch.as_strided(x, (2, 2), (4, 1), 1)
                            return torch.scalar_tensor(
                                -0.0 if dtype == "float32" else -7, dtype=getattr(torch, dtype)
                            )

                    values = (np.arange(12).reshape(1, 3, 4) * 37 - 128).astype(dtype)
                    if dtype == "float32":
                        values = np.resize(
                            np.array([0x80000000, 0x7F812345, 1, 0xFF800000], "uint32"),
                            values.shape,
                        ).view("float32")
                    model = View()
                    example = torch.from_numpy(values)
                    graph = from_exported_program(
                        torch.export.export(model, (example,)), unwrap_unit_return_tuple=True
                    )
                    report = []
                    prepared = prepare_gemmini_graph(graph, report=report)
                    expect_no_call = kind in ("squeeze_nonunit", "scalar")
                    self.assertEqual(
                        sum(row["placement"] == "gemmini" for row in report),
                        0,
                        report,
                    )
                    manifest, outputs, library = self.static_run(prepared, [values])
                    expected = model(example)
                    expected = list(expected) if isinstance(expected, tuple) else [expected]
                    self.assertEqual(len(outputs), len(expected))
                    for output, value in zip(outputs, expected):
                        value = value.numpy()
                        self.assertEqual(output.shape, value.shape)
                        np.testing.assert_array_equal(
                            output.reshape(-1).view("uint8"),
                            np.ascontiguousarray(value).reshape(-1).view("uint8"),
                        )
                    self.assertEqual(library.emulator_commands(), 0)
                    if expect_no_call:
                        self.assertFalse(manifest["calls"])

    def test_stock_reshape_elision_preserves_alias_input_to_cpu_fallback(self):
        x = relax.Var("x", relax.TensorStructInfo((8,), "int32"))
        index = relax.Var("index", relax.TensorStructInfo((1,), "int64"))
        builder = relax.BlockBuilder()
        with builder.function("main", [x, index]):
            with builder.dataflow():
                viewed = builder.emit(relax.op.reshape(x, (2, 4)))
                output = builder.emit_output(relax.op.take(viewed, index, axis=0))
            builder.emit_func_output(output)
        report = []
        prepared = prepare_gemmini_graph(builder.get(), report=report)
        values = [np.arange(8, dtype="int32") - 4, np.array([1], "int64")]
        manifest, outputs, library = self.static_run(prepared, values)
        self.assertEqual(len(manifest["views"]), 1)
        self.assertEqual(len(manifest["calls"]), 1)
        self.assertEqual(library.emulator_commands(), 0)
        np.testing.assert_array_equal(outputs[0], values[0].reshape(2, 4)[[1]])

    def test_vector_broadcast_dma_uses_standard_traces_and_metaschedule(self):
        for input_dtype, output_dtype in (
            ("int8", "int8"),
            ("int32", "int32"),
            ("float32", "float32"),
            ("int8", "int32"),
        ):
            source = te.placeholder((8,), input_dtype, "source")
            copied = te.compute((3, 8), lambda i, j: source[j].astype(output_dtype), name="copied")
            original = te.create_prim_func([source, copied])
            saved = tvm.ir.save_json(original)
            result = make_elementwise_schedule(original)
            replay = tir.Schedule(original, debug_mask="all")
            result.trace.apply_to_schedule(replay, remove_postproc=False)
            verify_trace_roundtrip(replay, original)
            self.assertEqual(tvm.ir.save_json(original), saved)
            context = ms.TuneContext(
                mod=original,
                target=GemminiCapabilities().target(),
                space_generator=gemmini_space_generator(),
                num_threads=1,
            )
            values = (np.arange(8) * 37 - 128).astype(input_dtype)
            if input_dtype == "float32":
                values = np.array(
                    [0x80000000, 0x7F812345, 1, 0xFF800000, 0, 0x7FC12345, 0x00800000, 0x3F800000],
                    "uint32",
                ).view("float32")
            expected = np.tile(values.astype(output_dtype), (3, 1))
            designs = context.generate_design_space()
            self.assertGreaterEqual(len(designs), 1)
            for design in designs:
                design.enter_postproc()
                self.assertTrue(all(p.apply(design) for p in context.space_generator.postprocs))
                lowered = lower_tensorir_schedule(
                    original, design, workload_metadata={"operation": "copy_or_widen"}
                )
                runtime, control = self.build(lowered.scheduled_mod)
                output = tvm.nd.empty((3, 8), output_dtype)
                runtime(tvm.nd.array(values), output)
                np.testing.assert_array_equal(output.numpy().view("uint8"), expected.view("uint8"))
                self.assertEqual(control.emulator_faults(), 0)
