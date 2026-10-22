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
"""Fail-closed fixture admission before costly exported graph simulation."""

from pathlib import Path
import copy
import hashlib
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
import tvm

sys.path.insert(0, str(Path(tvm.__file__).resolve().parents[2] / "apps/gemmini"))
from verify_exported_graph import load_fixture, validate_manifest, validate_sequence, strict_json, guest_layout
from verify_baremetal_cpu import bind_cpu_runtime


class FixtureAdmission(unittest.TestCase):
    def test_explicit_guest_map_and_linker_share_the_memory_budget(self):
        from verify_baremetal_cpu import LINKER, GUEST_BYTES, GUEST_BASE, STACK_BYTES
        linker, argument = guest_layout(GUEST_BYTES)
        self.assertEqual(linker, LINKER)
        self.assertEqual(argument, "-m0x80000000:0x10000000")
        linker, argument = guest_layout(512 << 20)
        self.assertIn("LENGTH = 0x20000000", linker)
        self.assertEqual(argument, "-m0x80000000:0x20000000")
        for invalid in (True, 0, STACK_BYTES, GUEST_BYTES + 1, float(GUEST_BYTES), (1 << 64) - GUEST_BASE + 4096):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                guest_layout(invalid)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="gemmini-exported-fixture-")
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "fixture.npz"
        self.manifest = {"inputs": [{"shape": [2, 3], "dtype": "int8", "bytes": 6}],
                         "outputs": [{"shape": [2, 4], "dtype": "int32", "bytes": 32}, {"shape": [2, 4], "dtype": "float32", "bytes": 32}]}
        self.arrays = {"input_0": np.arange(12, dtype=np.int8).reshape(2, 2, 3),
                       "output_0": np.arange(16, dtype=np.int32).reshape(2, 2, 4),
                       "output_1": np.arange(16, dtype=np.float32).reshape(2, 2, 4)}

    def test_complete_mixed_output_stream_preserves_exact_bits(self):
        self.arrays["output_1"][0, 0, 0] = -0.0
        np.savez(self.path, **self.arrays)
        actual, samples = load_fixture(self.path, self.manifest)
        self.assertEqual(samples, 2)
        for name, array in self.arrays.items():
            self.assertEqual(actual[name].dtype, array.dtype)
            self.assertEqual(actual[name].tobytes(), array.tobytes())

    def test_bf16_raw_payload_and_float64_stream_preserve_exact_bits(self):
        manifest = {"inputs": [{"shape": [2, 3], "dtype": "bfloat16", "bytes": 12}], "outputs": [{"shape": [2, 4], "dtype": "float64", "bytes": 64}]}
        bits = np.array([0, 0x8000, 1, 0x8001, 0x3f80, 0xbf80, 0x7f7f, 0xff7f, 0x3f81, 0x3f7f, 0x4000, 0xc000], dtype=np.uint16).reshape(2, 2, 3)
        values = np.arange(16, dtype=np.float64).reshape(2, 2, 4) + 2**-30
        np.savez(self.path, input_0=bits, output_0=values)
        actual, samples = load_fixture(self.path, manifest)
        self.assertEqual(samples, 2)
        self.assertEqual(actual["input_0"].tobytes(), bits.tobytes())
        self.assertEqual(actual["output_0"].tobytes(), values.tobytes())
        for replacement in (bits.astype(np.float32), bits.view(np.int16)):
            np.savez(self.path, input_0=replacement, output_0=values)
            with self.assertRaisesRegex(ValueError, "dtype/shape mismatch"):
                load_fixture(self.path, manifest)
        np.savez(self.path, input_0=bits, output_0=values.astype(np.float32))
        with self.assertRaisesRegex(ValueError, "dtype/shape mismatch"):
            load_fixture(self.path, manifest)

    def test_nonfinite_bf16_and_float64_payloads_refused(self):
        manifest = {"inputs": [{"shape": [1], "dtype": "bfloat16", "bytes": 2}], "outputs": [{"shape": [1], "dtype": "float64", "bytes": 8}]}
        for bits in (0x7f80, 0xff80, 0x7f81, 0xffc1):
            np.savez(self.path, input_0=np.array([[bits]], dtype=np.uint16), output_0=np.array([[0]], dtype=np.float64))
            with self.subTest(bits=bits), self.assertRaisesRegex(ValueError, "finite"):
                load_fixture(self.path, manifest)
        for value in (np.nan, np.inf, -np.inf):
            np.savez(self.path, input_0=np.array([[0]], dtype=np.uint16), output_0=np.array([[value]], dtype=np.float64))
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite"):
                load_fixture(self.path, manifest)

    def test_cpu_archives_are_explicit_distinct_and_self_contained(self):
        bind, command = Mock(), Mock()
        arguments, metadata = bind_cpu_runtime([], {}, bind, command)
        self.assertEqual(arguments, [])
        self.assertEqual(metadata["archives"], [])
        bind.assert_not_called()
        command.assert_not_called()
        archive = Path(self.directory.name) / "runtime.a"
        archive.write_bytes(b"!<thin>\n")
        with self.assertRaisesRegex(ValueError, "distinct"):
            bind_cpu_runtime([archive, archive], {}, bind, command)
        bind.assert_not_called()
        bind.return_value = {"path": str(archive)}
        with self.assertRaisesRegex(ValueError, "thin"):
            bind_cpu_runtime([archive], {}, bind, command)
        command.assert_not_called()

    def test_implicit_dtype_conversion_and_truncated_stream_refused(self):
        for name, replacement in (("input_0", self.arrays["input_0"].astype(np.float32)),
                                  ("output_0", self.arrays["output_0"][:1]),
                                  ("output_1", self.arrays["output_1"][0]),
                                  ("output_0", self.arrays["output_0"][:, :, :3])):
            with self.subTest(name=name, shape=replacement.shape, dtype=replacement.dtype):
                arrays = dict(self.arrays, **{name: replacement})
                np.savez(self.path, **arrays)
                with self.assertRaises(ValueError):
                    load_fixture(self.path, self.manifest)

    def test_understated_overstated_and_invalid_tensor_bytes_refused(self):
        import copy
        np.savez(self.path, **self.arrays)
        malformed = [{"bytes": 31}, {"bytes": 33}, {"bytes": True},
                     {"shape": [2, 0]}, {"shape": [2, True]}, {"shape": [2, 4.0]},
                     {"dtype": "float64"}, {"shape": [1 << 62, 8], "bytes": 1 << 67}]
        for mutation in malformed:
            manifest = copy.deepcopy(self.manifest)
            manifest["outputs"][0].update(mutation)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                load_fixture(self.path, manifest)

    def test_internal_tensor_storage_envelope_is_checked(self):
        import copy
        manifest = dict(self.manifest, constants=[{"dtype": "int8", "shape": [3], "bytes": 3, "offset": 0}],
                        intermediates=[{"dtype": "int32", "shape": [8], "bytes": 32, "offset": 0, "origin": "workspace"}],
                        constants_bytes=64, workspace_bytes=64, explicit_tensor_bytes=198)
        validate_manifest(manifest)
        for group, mutation in (("constants", {"bytes": 2}), ("constants", {"offset": 64}),
                                ("intermediates", {"bytes": 16}), ("intermediates", {"offset": 1})):
            altered = copy.deepcopy(manifest)
            altered[group][0].update(mutation)
            with self.subTest(group=group, mutation=mutation), self.assertRaises(ValueError):
                validate_manifest(altered)
        with self.assertRaises(ValueError):
            validate_manifest(dict(manifest, explicit_tensor_bytes=197))

    def test_extra_missing_nonfinite_and_empty_streams_refused(self):
        malformed = [dict(self.arrays, extra=np.zeros(1)),
                     {name: value for name, value in self.arrays.items() if name != "output_1"},
                     dict(self.arrays, output_1=np.full((2, 2, 4), np.nan, np.float32)),
                     {name: value[:0] for name, value in self.arrays.items()}]
        for arrays in malformed:
            np.savez(self.path, **arrays)
            with self.assertRaises(ValueError):
                load_fixture(self.path, self.manifest)


class SequenceAdmission(unittest.TestCase):
    def setUp(self):
        tensor = {"shape": [2], "dtype": "int32", "bytes": 8}
        stage = {"inputs": [tensor], "outputs": [tensor, tensor], "constants": [], "intermediates": [],
                 "constants_bytes": 0, "constants_sha256": "shared", "workspace_bytes": 64, "explicit_tensor_bytes": 88,
                 "alignment_bytes": 64, "heap_required_by_orchestration": False, "runtime_workspace_allocator_required": False,
                 "target": "llvm -mtriple=riscv64-unknown-elf", "calls": []}
        self.package = {"stages": {"prefill": dict(stage, entrypoint="stage_1_model_run"), "cpu": dict(stage, entrypoint="stage_0_model_run")},
                        "constants_bytes": 0, "constants_sha256": "shared", "duplicated_stage_constants_bytes": 0, "serial_workspace_bytes": 64,
                        "maximum_stage_tensor_bytes": 88, "state_routing": "caller_owned", "session_execution": False}
        self.route = {"version": 1, "calls": [{"stage": "prefill", "inputs": [{"fixture": 0}]},
                                            {"stage": "cpu", "inputs": [{"output": [0, 1]}]},
                                            {"stage": "cpu", "inputs": [{"fixture": 0}]}]}

    def test_prior_output_fanout_fixture_reuse_and_all_outputs_retained(self):
        manifest, calls = validate_sequence(self.package, self.route)
        self.assertEqual(len(manifest["inputs"]), 1)
        self.assertEqual(len(manifest["outputs"]), 6)
        self.assertEqual(manifest["explicit_tensor_bytes"], 120)
        self.assertEqual(calls[1]["inputs"], [("output", 1)])
        self.assertEqual(calls[2]["outputs"], [4, 5])
        route = copy.deepcopy(self.route)
        route["calls"][2]["inputs"] = [{"output": [0, 1]}]
        validate_sequence(self.package, route)

    def test_future_cyclic_bool_sparse_unknown_and_extra_routes_refused(self):
        for binding in ({"output": [1, 0]}, {"output": [0, -1]}, {"output": [False, 0]}, {"output": [0, True]},
                        {"fixture": True}, {"fixture": 256}, {"fixture": 2}, {"transform": "cast"},
                        {"fixture": 0, "output": [0, 0]}):
            route = copy.deepcopy(self.route)
            route["calls"][1]["inputs"] = [binding]
            with self.subTest(binding=binding), self.assertRaises(ValueError):
                validate_sequence(self.package, route)
        for mutation in ({"version": True}, {"calls": []}, {"calls": self.route["calls"] * 22}, {"extra": 0}):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_sequence(self.package, dict(self.route, **mutation))
        route = copy.deepcopy(self.route)
        route["calls"][0]["stage"] = "unknown"
        with self.assertRaises(ValueError):
            validate_sequence(self.package, route)

    def test_static_signature_count_matches_the_call_bound(self):
        for count in (33, 64):
            package = copy.deepcopy(self.package)
            stage = package["stages"]["prefill"]
            package["stages"] = {f"signature_{index}": dict(stage, entrypoint=f"stage_{index}_model_run") for index in range(count)}
            route = {"version": 1, "calls": [{"stage": f"signature_{index}", "inputs": [{"fixture": 0} if index == 0 else {"output": [index - 1, 0]}]} for index in range(count)]}
            manifest, calls = validate_sequence(package, route)
            self.assertEqual(len(calls), count)
            self.assertEqual(len(manifest["outputs"]), count * 2)
            self.assertEqual(manifest["explicit_tensor_bytes"], count * 16 + 72)
        package["stages"]["signature_64"] = dict(stage, entrypoint="stage_64_model_run")
        with self.assertRaisesRegex(ValueError, "64 named static stages"):
            validate_sequence(package, route)
        del package["stages"]["signature_64"]
        route["calls"].append({"stage": "signature_0", "inputs": [{"fixture": 0}]})
        with self.assertRaisesRegex(ValueError, "64 calls"):
            validate_sequence(package, route)

    def test_stage_dtype_envelope_allocator_namespace_and_budget_refused(self):
        for field, value in (("inputs", [{"shape": [1, 2], "dtype": "int32", "bytes": 8}]),
                             ("inputs", [{"shape": [2], "dtype": "float32", "bytes": 8}]),
                             ("entrypoint", "stage_1_model_run"), ("entrypoint", "invalid-symbol"),
                             ("runtime_workspace_allocator_required", True), ("heap_required_by_orchestration", True),
                             ("alignment_bytes", 32), ("constants_sha256", "different")):
            package = copy.deepcopy(self.package)
            package["stages"]["cpu"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_sequence(package, self.route)
        for field in ("serial_workspace_bytes", "maximum_stage_tensor_bytes"):
            package = dict(self.package, **{field: 1})
            with self.assertRaises(ValueError):
                validate_sequence(package, self.route)
        for field in ("constants_bytes", "duplicated_stage_constants_bytes", "serial_workspace_bytes", "maximum_stage_tensor_bytes"):
            for value in (False, float(self.package[field])):
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, "plain"):
                    validate_sequence(dict(self.package, **{field: value}), self.route)
        package = copy.deepcopy(self.package)
        for stage in package["stages"].values():
            stage["workspace_bytes"] = 1 << 30
            stage["explicit_tensor_bytes"] = (1 << 30) + 24
        package.update(serial_workspace_bytes=1 << 30, maximum_stage_tensor_bytes=(1 << 30) + 24)
        with self.assertRaisesRegex(ValueError, "guest budget"):
            validate_sequence(package, self.route)
        manifest, _ = validate_sequence(package, self.route, guest_bytes=2 << 30)
        self.assertGreater(manifest["explicit_tensor_bytes"], 1 << 30)
        route = dict(self.route, calls=self.route["calls"][:1])
        with self.assertRaisesRegex(ValueError, "Every stage"):
            validate_sequence(self.package, route)

    def test_duplicate_route_json_fields_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "route.json"
            path.write_text('{"version":1,"version":1,"calls":[]}')
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                strict_json(path)

    def test_real_exported_sequence_with_independent_frozen_fixture(self):
        from tvm import relax
        from tvm.relax.backend.contrib.gemmini import prepare_gemmini_graph_legacy as prepare_gemmini_graph
        from baremetal import export_graphs
        with tempfile.TemporaryDirectory() as directory:
            selected = os.environ.get("TVM_EXPORTED_SEQUENCE_FIXTURE")
            root = Path(selected) if selected else Path(directory) / "fixture"
            root.mkdir(parents=True, exist_ok=False)
            weight = np.array([[1, -2], [3, 4], [-5, 6]], dtype="int8")
            modules = {}
            for name in ("prefill", "decode", "finish"):
                x = relax.Var("x", relax.TensorStructInfo((1, 3), "int8"))
                raw = relax.Var("raw", relax.TensorStructInfo((4,), "bfloat16"))
                precise = relax.Var("precise", relax.TensorStructInfo((2,), "float64"))
                state = relax.Var("state", relax.TensorStructInfo((1, 2), "int32"))
                builder = relax.BlockBuilder()
                inputs = [x, raw, precise] if name == "prefill" else [state, raw, precise] if name == "finish" else [x, state, raw, precise]
                with builder.function("main", inputs):
                    with builder.dataflow():
                        dot = state if name == "finish" else builder.emit(relax.op.matmul(x, relax.const(weight), out_dtype="int32"))
                        shifted = builder.emit(relax.op.add(dot, relax.const(np.array([[3, 3]], dtype="int32"))))
                        result = shifted if name in ("prefill", "finish") else builder.emit(relax.op.add(shifted, state))
                        high = builder.emit(relax.op.add(precise, relax.const(2**-40, "float64")))
                        output = builder.emit_output(relax.Tuple([result, raw, high]))
                    builder.emit_func_output(output)
                modules[name] = prepare_gemmini_graph(builder.get(), optimize=True)
            package = export_graphs(modules, root / "graphs")
            route = {"version": 1, "calls": [{"stage": "prefill", "inputs": [{"fixture": 0}, {"fixture": 1}, {"fixture": 2}]},
                                              {"stage": "decode", "inputs": [{"fixture": 3}, {"output": [0, 0]}, {"output": [0, 1]}, {"output": [0, 2]}]},
                                              {"stage": "decode", "inputs": [{"fixture": 4}, {"output": [1, 0]}, {"output": [0, 1]}, {"output": [1, 2]}]},
                                              {"stage": "decode", "inputs": [{"fixture": 3}, {"output": [2, 0]}, {"output": [2, 1]}, {"output": [2, 2]}]},
                                              {"stage": "finish", "inputs": [{"output": [3, 0]}, {"output": [3, 1]}, {"output": [3, 2]}]}]}
            manifest, calls = validate_sequence(package, route)
            arrays = {"input_0": np.array([[[2, -3, 4]], [[-7, 6, -5]]], dtype="int8"),
                      "input_1": np.array([[0, 0x8000, 1, 0xbf80], [0x3f81, 0x3f7f, 0x8001, 0x4000]], dtype="uint16"),
                      "input_2": np.array([[1 + 2**-30, -0.0], [-3 - 2**-30, 2**-40]], dtype="float64"),
                      "input_3": np.array([[[-1, 2, 3]], [[5, -4, 3]]], dtype="int8"),
                      "input_4": np.array([[[7, 0, -2]], [[-1, -2, -3]]], dtype="int8")}
            # Independent Python integer complete-K dot and exactly representable
            # binary FP64 increments; neither exported code nor TVM is evaluated.
            state = [[0, 0] for _ in range(2)]
            high = arrays["input_2"].copy()
            for ordinal, index in enumerate((0, 3, 4, 3)):
                for sample in range(2):
                    for column in range(2):
                        state[sample][column] += 3 + sum(int(arrays[f"input_{index}"][sample, 0, k]) * int(weight[k, column]) for k in range(3))
                high = high + 2**-40
                arrays[f"output_{ordinal * 3}"] = np.array(state, dtype="int32").reshape(2, 1, 2)
                arrays[f"output_{ordinal * 3 + 1}"] = arrays["input_1"].copy()
                arrays[f"output_{ordinal * 3 + 2}"] = high.copy()
            arrays["output_12"] = np.array([[value + 3 for value in row] for row in state], dtype="int32").reshape(2, 1, 2)
            arrays["output_13"] = arrays["input_1"].copy()
            arrays["output_14"] = high + 2**-40
            np.savez(root / "fixture.npz", **arrays)
            (root / "route.json").write_text(json.dumps(route, indent=2) + "\n")
            provenance = {"fixture_sha256": hashlib.sha256((root / "fixture.npz").read_bytes()).hexdigest(),
                          "reference_sources": [{"path": str(Path(__file__).resolve()), "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}],
                          "reference_contract": "Python integer complete-K dots +3 and explicit prior-state sum; final all-CPU state +3; raw uint16 BF16 copies; exact binary FP64 increments 2^-40. Two reset fixtures, five explicit invocations, all fifteen outputs retained; no TVM/reference-model evaluation.",
                          "sample_ids": ["bounded-sequence-fixture-0", "bounded-sequence-fixture-1"]}
            (root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
            actual, samples = load_fixture(root / "fixture.npz", manifest)
            self.assertEqual(samples, 2)
            self.assertEqual(len(manifest["outputs"]), 15)
            self.assertGreater(package["duplicated_stage_constants_bytes"], package["constants_bytes"])
            for key in arrays:
                self.assertEqual(actual[key].tobytes(), arrays[key].tobytes())


if __name__ == "__main__":
    unittest.main()
