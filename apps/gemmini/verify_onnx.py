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
"""Check synthetic PyTorch exports and focused ONNX regressions against Relax CPU execution."""

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback


BFLOAT_CASES = ("bfloat16_matmul", "bfloat16_gemm", "bfloat16_conv2d",
               "bfloat16_conv2d_same_upper", "bfloat16_conv2d_same_lower", "bfloat16_layer_norm",
               "bfloat16_softmax", "bfloat16_softmax_241", "bfloat16_gelu_none", "bfloat16_gelu_tanh",
               "bfloat16_sigmoid")
CASES = ("matmul", "batched_matmul", "conv2d", "layer_norm", "rms_norm")
IMPORTER_CASES = ("shape_add", "gather_negative", "gather_negative_constant", "expand_leading_dims", "constant_of_shape_scalar",
                  "bfloat16_initializer", "bfloat16_constant", "bfloat16_cast", "bfloat16_cast_like", "bfloat16_constant_cast_like", "cast_like_integer", "constant_integer_literals", "constant_float_literals", "cast_like_shape_scalar")
RTOL, ATOL = 1e-4, 1e-5


def allowed_path(value):
    """Reject restricted components before resolving links or opening any file."""
    path = Path(os.path.abspath(value))
    for component in path.parts:
        if any(word in component.lower() for word in ("hammer", "vlsi")):
            raise ValueError("Restricted path component")
    # Walk links individually so an intermediate forbidden target is never traversed.
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current = current / component
        if current.is_symlink():
            target = Path(os.readlink(current))
            current = allowed_path(target if target.is_absolute() else current.parent / target)
    return current


def identity(path):
    path = allowed_path(path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def tensors(values):
    return [{"shape": list(value.shape), "dtype": str(value.dtype)} for value in values]


def compare(actual, expected, np):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(f"Output mismatch: {actual.shape}/{actual.dtype} versus {expected.shape}/{expected.dtype}")
    # Widen only comparison arithmetic; execution operands/results keep BF16.
    comparison_actual = actual.astype("float64") if actual.dtype.name == "bfloat16" else actual
    comparison_expected = expected.astype("float64") if expected.dtype.name == "bfloat16" else expected
    delta = np.abs(comparison_actual - comparison_expected)
    errors = {"max_absolute_error": float(delta.max()), "max_relative_error": float((delta / np.maximum(np.abs(comparison_expected), ATOL)).max())}
    integer = np.issubdtype(expected.dtype, np.integer)
    errors["comparison"] = "exact_integer" if integer else "floating_tolerance"
    errors["passed"] = bool(np.array_equal(actual, expected) if integer else np.allclose(comparison_actual, comparison_expected, rtol=RTOL, atol=ATOL, equal_nan=False))
    return errors


def make_case(name, torch, np):
    rng = np.random.default_rng(194)

    if name in BFLOAT_CASES:
        import ml_dtypes

        def parameter(shape):
            return torch.tensor(rng.normal(0.2, 0.8, shape).astype("float32")).to(torch.bfloat16)

        class BFloatModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                if name.startswith("bfloat16_conv2d"):
                    weight_shape = (3, 2, 2 if "same" in name else 1, 17)
                    bias_shape = (3,)
                elif name == "bfloat16_layer_norm":
                    weight_shape, bias_shape = (17,), (17,)
                else:
                    weight_shape, bias_shape = (17, 4), (4,)
                self.register_buffer("weight", parameter(weight_shape))
                self.register_buffer("bias", parameter(bias_shape))

            def forward(self, value):
                if name == "bfloat16_sigmoid":
                    return torch.sigmoid(value)
                if name.startswith("bfloat16_gelu"):
                    return torch.nn.functional.gelu(value, approximate="tanh" if name.endswith("tanh") else "none")
                if name.startswith("bfloat16_softmax"):
                    return torch.nn.functional.softmax(value, dim=-1)
                if name == "bfloat16_matmul":
                    return torch.matmul(value, self.weight)
                if name == "bfloat16_gemm":
                    return torch.addmm(self.bias, value, self.weight, beta=0.5, alpha=0.75)
                if "conv2d_same" in name:
                    # Odd padding along height distinguishes SAME_UPPER/LOWER.
                    top, bottom = (0, 1) if name.endswith("upper") else (1, 0)
                    value = torch.nn.functional.pad(value, (8, 8, top, bottom))
                    return torch.nn.functional.conv2d(value, self.weight, self.bias, stride=2)
                if name == "bfloat16_conv2d":
                    return torch.nn.functional.conv2d(value, self.weight, self.bias)
                return torch.nn.functional.layer_norm(value, (17,), self.weight, self.bias, eps=1e-5)

        model = BFloatModel().eval()
        shape = (1, 2, 5, 19) if name.startswith("bfloat16_conv2d") else (2, 3, 17) if name == "bfloat16_layer_norm" else (3, 17)
        if name.startswith("bfloat16_softmax"):
            shape = (2, 3, 241 if name.endswith("241") else 17)
        elif name == "bfloat16_sigmoid":
            shape = (3, 257)
        value = rng.normal(0.2, 0.8, shape).astype("float32")
        if name == "bfloat16_sigmoid":
            value += np.linspace(-16, 16, 257, dtype="float32")
        if name.startswith("bfloat16_gelu"):
            value += np.linspace(-4, 4, 17, dtype="float32")
        if name == "bfloat16_layer_norm" or name.startswith("bfloat16_softmax"):
            # Offset rows exercise centered FP32 variance rather than the
            # cancellation-prone E[X^2]-E[X]^2 computation.
            value += np.array([[32, 100, -8], [128, -32, 2]], dtype="float32")[..., None]
        return model, [value.astype(ml_dtypes.bfloat16)]

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            if name == "conv2d":
                self.conv = torch.nn.Conv2d(2, 3, 3, stride=(2, 1), padding=(1, 1))
                with torch.no_grad():
                    self.conv.weight.copy_(torch.from_numpy(rng.uniform(-0.7, 0.7, (3, 2, 3, 3)).astype("float32")))
                    self.conv.bias.copy_(torch.tensor([-0.4, 0.2, 0.7]))
            if name in ("layer_norm", "rms_norm"):
                self.register_buffer("weight", torch.tensor([0.7, 1.3, -0.8, 1.8, 0.4]))
                self.register_buffer("bias", torch.tensor([0.3, -0.5, 0.2, 0.6, -0.1]))

        def forward(self, value, right=None):
            if name in ("matmul", "batched_matmul"):
                return torch.matmul(value, right)
            if name == "conv2d":
                return self.conv(value)
            if name == "layer_norm":
                return torch.nn.functional.layer_norm(value, (5,), self.weight, self.bias, eps=1e-5)
            return value * torch.rsqrt((value * value).mean(dim=-1, keepdim=True) + 1e-5) * self.weight

    if name == "matmul":
        shapes = [(3, 5), (5, 4)]
    elif name == "batched_matmul":
        shapes = [(2, 3, 5), (2, 5, 4)]
    elif name == "conv2d":
        shapes = [(1, 2, 5, 6)]
    else:
        shapes = [(2, 3, 5)]
    arrays = [rng.normal(0.2, 0.8, shape).astype("float32") for shape in shapes]
    if name in ("layer_norm", "rms_norm"):
        arrays[0] = arrays[0] * np.array([[0.2, 2.0, 5.0], [3.0, 0.4, 1.5]], dtype="float32")[..., None] + np.array([[-2.0, 0.3, 4.0], [1.0, -3.0, 0.7]], dtype="float32")[..., None]
    return Model().eval(), arrays


def make_importer_case(name, opset, onnx, np):
    """Build schema-valid ONNX cases with independent NumPy expectations."""
    helper, tensor, numpy_helper = onnx.helper, onnx.TensorProto, onnx.numpy_helper
    value = np.arange(6, dtype="float32").reshape(2, 3) - 2
    inputs, initializers = {"value": value}, []
    if name.startswith("bfloat16_"):
        import ml_dtypes

        def bf16_tensor(key, bits, shape, raw):
            result = tensor(name=key, data_type=tensor.BFLOAT16, dims=shape)
            bits = np.asarray(bits, dtype="uint16")
            if raw:
                result.raw_data = bits.astype("<u2").tobytes()
            else:
                result.int32_data.extend(bits.reshape(-1).tolist())
            return result

        if name == "bfloat16_initializer":
            inputs["value"] = value.astype(ml_dtypes.bfloat16)
            bits = np.array([0x3f81, 0xbeff, 0x8000, 0x0001, 0x4081, 0xc181], dtype="uint16").reshape(2, 3)
            initializers = [bf16_tensor("weight", bits, [2, 3], False)]
            nodes = [helper.make_node("Cast", ["value"], ["input_float"], to=tensor.FLOAT),
                     helper.make_node("Cast", ["weight"], ["weight_float"], to=tensor.FLOAT),
                     helper.make_node("Add", ["input_float", "weight_float"], ["output"])]
            expected = inputs["value"].astype("float32") + bits.view(ml_dtypes.bfloat16).astype("float32")
        elif name in ("bfloat16_constant", "bfloat16_constant_cast_like"):
            first = np.array([0x3f81, 0xbeff, 0x4081], dtype="uint16")
            second = np.array([0x8000, 0x0001, 0xc181], dtype="uint16")
            # ONNX 1.17's reference Concat loses its custom BF16 dtype;
            # Transpose retains it and provides an independent executable oracle.
            bits = np.stack((first, second), axis=1)
            nodes = [helper.make_node("Constant", [], ["joined"], value=bf16_tensor("joined_value", bits, [3, 2], True)),
                     helper.make_node("Transpose", ["joined"], ["weight"], perm=[1, 0]),
                     helper.make_node("Cast", ["weight"], ["weight_float"], to=tensor.FLOAT),
                     helper.make_node("Add", ["value", "weight_float"], ["output"])]
            expected = value + np.stack((first, second)).view(ml_dtypes.bfloat16).astype("float32")
            if name == "bfloat16_constant_cast_like":
                nodes[2] = helper.make_node("CastLike", ["weight", "value"], ["weight_float"])
        else:
            # Values lose nonzero low bits at the BF16 Cast. ONNX 1.17's
            # reference truncates BF16 casts; these below-halfway values agree
            # with the independent round-to-nearest ml_dtypes expectation.
            inputs["value"] = np.array([[1.001, 1.009, -1.001], [2.002, -2.002, 0.1001]], dtype="float32")
            nodes = [helper.make_node("Cast", ["value"], ["rounded"], to=tensor.BFLOAT16),
                     helper.make_node("Cast", ["rounded"], ["output"], to=tensor.FLOAT)]
            expected = inputs["value"].astype(ml_dtypes.bfloat16).astype("float32")
            if name == "bfloat16_cast_like":
                # Target shape and values do not participate in the conversion.
                inputs["target"] = np.array([123], dtype=ml_dtypes.bfloat16)
                nodes = [helper.make_node("CastLike", ["value", "target"], ["rounded"]), helper.make_node("CastLike", ["rounded", "value"], ["output"])]
    elif name == "cast_like_integer":
        inputs["value"] = np.array([[-2.9, -1.1, 0.1], [1.9, 2.1, 3.9]], dtype="float32")
        inputs["target"] = np.array([2**40 + 7], dtype="int64")
        nodes = [helper.make_node("CastLike", ["value", "target"], ["output"])]
        expected = inputs["value"].astype("int64")
    elif name in ("constant_integer_literals", "constant_float_literals"):
        inputs = {}
        integer = name == "constant_integer_literals"
        scalar = 2**40 + 7 if integer else -1.25
        vector = [-3, 0, 17] if integer else [0.1, 2.5, -3.75]
        scalar_attribute = {"value_int" if integer else "value_float": scalar}
        vector_attribute = {"value_ints" if integer else "value_floats": vector}
        nodes = [helper.make_node("Constant", [], ["scalar"], **scalar_attribute), helper.make_node("Constant", [], ["vector"], **vector_attribute), helper.make_node("Add", ["scalar", "vector"], ["output"])]
        dtype = "int64" if integer else "float32"
        expected = np.array(scalar, dtype=dtype) + np.array(vector, dtype=dtype)
    elif name == "cast_like_shape_scalar":
        initializers = [numpy_helper.from_array(np.array(-1, dtype="int64"), "index")]
        nodes = [helper.make_node("Shape", ["value"], ["shape"]), helper.make_node("Gather", ["shape", "index"], ["width"], axis=0), helper.make_node("CastLike", ["width", "value"], ["floating_width"]), helper.make_node("Sqrt", ["floating_width"], ["output"])]
        expected = np.sqrt(np.array(value.shape[-1], dtype="float32"))
    elif name == "shape_add":
        offset = np.array([1, 3, 2**40 + 7], dtype="int64")
        initializers = [numpy_helper.from_array(np.array(0, dtype="int64"), "index"), numpy_helper.from_array(offset, "offset")]
        nodes = [helper.make_node("Shape", ["value"], ["shape"]), helper.make_node("Gather", ["shape", "index"], ["rows"], axis=0), helper.make_node("Add", ["rows", "offset"], ["output"])]
        expected = offset + value.shape[0]
    elif name in ("gather_negative", "gather_negative_constant"):
        inputs["value"] = np.arange(12, dtype="float32").reshape(4, 3) - 5
        indices = np.array([-1, 0, -2], dtype="int64")
        if name == "gather_negative":
            inputs["indices"] = indices
        else:
            initializers = [numpy_helper.from_array(indices, "indices")]
        nodes = [helper.make_node("Gather", ["value", "indices"], ["output"], axis=0)]
        expected = np.take(inputs["value"], indices, axis=0)
    elif name == "expand_leading_dims":
        initializers = [numpy_helper.from_array(np.array([3], dtype="int64"), "shape")]
        nodes = [helper.make_node("Expand", ["value", "shape"], ["output"])]
        expected = value.copy()
    else:
        inputs["value"] = np.array(-1.25, dtype="float32")
        initializers = [numpy_helper.from_array(np.array([], dtype="int64"), "shape")]
        fill = numpy_helper.from_array(np.array([2.5], dtype="float32"), "fill")
        nodes = [helper.make_node("ConstantOfShape", ["shape"], ["filled"], value=fill), helper.make_node("Add", ["value", "filled"], ["output"])]
        expected = inputs["value"] + np.array(2.5, dtype="float32")
    graph_inputs = [helper.make_tensor_value_info(key, tensor.BFLOAT16 if value.dtype.name == "bfloat16" else
                                                tensor.INT64 if value.dtype == np.int64 else tensor.FLOAT,
                                                list(value.shape)) for key, value in inputs.items()]
    output = helper.make_tensor_value_info("output", tensor.INT64 if expected.dtype == np.int64 else tensor.FLOAT, list(expected.shape))
    graph = helper.make_graph(nodes, name, graph_inputs, [output], initializer=initializers)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)], producer_name="gemmini_frontend_verifier"), inputs, expected


def provenance(tvm, torch, onnx, np, override):
    import google.protobuf as protobuf
    from tvm._ffi.base import _LIB
    from tvm.relax.frontend.onnx import onnx_frontend

    root = allowed_path(Path(__file__).parents[2])
    python_path, library_path = allowed_path(tvm.__file__), allowed_path(_LIB._name)
    if not python_path.is_relative_to(root / "python"):
        raise RuntimeError(f"Wrong TVM Python package: {python_path}")
    library_dirs = [allowed_path(value) for value in os.environ.get("TVM_LIBRARY_PATH", "").split(os.pathsep) if value]
    if not library_dirs or library_path.parent not in library_dirs:
        raise RuntimeError(f"Loaded library {library_path} does not match explicit TVM_LIBRARY_PATH")
    if not tvm.runtime.enabled("llvm"):
        raise RuntimeError("Loaded TVM lacks LLVM support")
    importer = onnx_frontend
    if override is not None:
        source = allowed_path(override)
        spec = importlib.util.spec_from_file_location("tvm.relax.frontend.onnx._verification_importer", source)
        importer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(importer)
    marker = tvm.get_global_func("tir.transform.HostOnlyAutoCopyGuardEnabled", allow_missing=True)
    info = tvm.support.libinfo()
    return importer.from_onnx, {
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__, "onnx": onnx.__version__, "numpy": np.__version__, "protobuf": protobuf.__version__, "tvm": tvm.__version__},
        "dependency_locations": {name: str(allowed_path(module.__file__)) for name, module in (("torch", torch), ("onnx", onnx), ("numpy", np), ("protobuf", protobuf))},
        "tvm_python_location": str(python_path), "loaded_libtvm_path": str(library_path), "tvm_library_path": os.environ["TVM_LIBRARY_PATH"],
        "llvm_enabled": True, "build_info": {key: str(info[key]) for key in ("LLVM_VERSION", "USE_LLVM", "GIT_COMMIT_HASH") if key in info},
        "host_only_auto_copy_guard": {"available": marker is not None, "enabled": bool(marker()) if marker is not None else False},
        "frontend_source": identity(importer.__file__), "normal_frontend_source": identity(onnx_frontend.__file__), "importer_override": override is not None,
    }


def run_case(name, opset, output, from_onnx, torch, onnx, np, tvm):
    from onnx.reference import ReferenceEvaluator
    from tvm import relax

    record = {"case": name, "opset": opset, "status": "failed", "stage": "inputs", "rtol": RTOL, "atol": ATOL}
    try:
        directory = allowed_path(output / f"{name}_opset{opset}")
        directory.mkdir(parents=True, exist_ok=True)
        graph_path = allowed_path(directory / "model.onnx")
        if name in IMPORTER_CASES:
            graph, inputs, expected = make_importer_case(name, opset, onnx, np)
            names, arrays, reference_name = list(inputs), list(inputs.values()), "numpy"
            record["stage"] = "onnx_construction"
            onnx.save(graph, str(graph_path))
        else:
            model, arrays = make_case(name, torch, np)
            arguments = tuple(torch.from_numpy(value.view("uint16")).view(torch.bfloat16)
                              if value.dtype.name == "bfloat16" else torch.from_numpy(value) for value in arrays)
            names, reference_name = ["value", "right"][:len(arrays)], "torch"
            with torch.no_grad():
                result = model(*arguments)
                if result.dtype == torch.bfloat16:
                    import ml_dtypes

                    expected = result.view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
                else:
                    expected = result.numpy()
            record["stage"] = "torch_export"
            torch.onnx.export(model, arguments, str(graph_path), input_names=names, output_names=["output"], opset_version=opset, dynamo=False, do_constant_folding=True)
            graph = onnx.load(str(graph_path))
            if "conv2d_same" in name:
                # Encode ONNX's automatic padding against the independent
                # source's explicit asymmetric pad followed by CPU convolution.
                weights = [value for value in graph.graph.initializer if value.name in ("weight", "bias")]
                if len(weights) != 2:
                    raise AssertionError("Source convolution weight names changed")
                mode = "SAME_UPPER" if name.endswith("upper") else "SAME_LOWER"
                node = onnx.helper.make_node("Conv", [names[0], "weight", "bias"], ["output"],
                                             auto_pad=mode, strides=[2, 2])
                replacement = onnx.helper.make_graph([node], name, list(graph.graph.input),
                                                     list(graph.graph.output), initializer=weights)
                graph.CopyFrom(onnx.helper.make_model(replacement, opset_imports=[onnx.helper.make_opsetid("", opset)]))
                onnx.save(graph, str(graph_path))
        inputs = dict(zip(names, arrays))
        saved = {reference_name: expected, **inputs}
        record["reference"] = reference_name
        record["inputs"] = dict(zip(names, tensors(arrays)))
        record[f"{reference_name}_output"] = tensors([expected])[0]
        np.savez(allowed_path(directory / "numerical_outputs.npz"), **saved)
        record["onnx_graph"] = identity(graph_path)
        record["operators"] = dict(sorted(Counter(f"{node.domain or 'ai.onnx'}::{node.op_type}" for node in graph.graph.node).items()))
        record["stage"] = "onnx_reference"
        onnx.checker.check_model(graph, full_check=True)
        reference_supported = True
        if name == "bfloat16_cast_like":
            try:
                onnx.helper.np_dtype_to_tensor_dtype(inputs["target"].dtype)
            except ValueError:
                reference_supported = False
        if name in BFLOAT_CASES:
            # ONNX 1.17's reference kernels operate on the uint16/FP32
            # containers, not source BF16 contraction/normalization arithmetic.
            # The independent PyTorch CPU source is the numerical oracle here.
            record["onnx_reference"] = {"executed": False, "reason": "ONNX BF16 reference arithmetic is unsupported; independent PyTorch CPU reference used"}
            record["arithmetic_contract"] = {"operand_dtype": "bfloat16", "accumulator_or_statistics_dtype": "float32",
                                             "result_dtype": "bfloat16", "weight_precision_changed": False}
        elif not reference_supported:
            record["onnx_reference"] = {"executed": False, "reason": "Selected ONNX dependency cannot map ml_dtypes BF16 for CastLike; independent NumPy rounding reference used"}
        else:
            reference = ReferenceEvaluator(graph).run(None, inputs)[0]
            record["onnx_output"] = tensors([reference])[0]
            saved["onnx"] = reference
            np.savez(allowed_path(directory / "numerical_outputs.npz"), **saved)
            record[f"onnx_vs_{reference_name}"] = compare(reference, expected, np)
            if not record[f"onnx_vs_{reference_name}"]["passed"]:
                raise AssertionError(f"ONNX reference differs from {reference_name}")
        record["stage"] = "relax_import"
        mod = from_onnx(graph, shape_dict={key: list(value.shape) for key, value in zip(names, arrays)}, keep_params_in_input=False)
        if name == "bfloat16_initializer" and mod["main"].params[0].struct_info.dtype != "bfloat16":
            raise AssertionError("Importer changed the declared BF16 input ABI")
        if name == "bfloat16_cast" and "bfloat16" not in mod.script():
            raise AssertionError("Importer erased the declared BF16 rounding boundary")
        ir_path = allowed_path(directory / "imported_relax.py")
        ir_path.write_text(mod.script(), encoding="utf-8")
        record["relax_ir"] = identity(ir_path)
        if name in ("gather_negative", "gather_negative_constant") and from_onnx.__module__.endswith("._verification_importer"):
            record.update(status="not_executed", stage="ablation_index_safety", reason="Historical Gather may leave negative take indices unchecked; imported IR retained without executing it")
            return record
        record["stage"] = "relax_build"
        executable = relax.build(mod, target="llvm")
        record["stage"] = "relax_execute"
        device = tvm.cpu()
        result = relax.VirtualMachine(executable, device)["main"](*[tvm.nd.array(value, device) for value in arrays])
        actual = result.numpy() if hasattr(result, "numpy") else result[0].numpy()
        if name in BFLOAT_CASES:
            import ml_dtypes

            array = result if hasattr(result, "numpy") else result[0]
            if array.dtype != "bfloat16":
                raise AssertionError("Compiled source BF16 output dtype changed")
            actual = actual.view(ml_dtypes.bfloat16)
        record["relax_output"] = tensors([actual])[0]
        saved["relax"] = actual
        np.savez(allowed_path(directory / "numerical_outputs.npz"), **saved)
        record["stage"] = "numerical_compare"
        record[f"relax_vs_{reference_name}"] = compare(actual, expected, np)
        comparisons = [f"relax_vs_{reference_name}"]
        if "onnx_output" in record:
            record["relax_vs_onnx"] = compare(actual, reference, np)
            comparisons.append("relax_vs_onnx")
        if not all(record[key]["passed"] for key in comparisons):
            raise AssertionError("Relax output differs from reference")
        record.update(status="passed", stage="complete")
    except Exception as error:
        record["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--suite", choices=("torch", "importer", "bfloat16"), default="torch", help="Default case set; --case overrides this selection")
    parser.add_argument("--case", choices=CASES + IMPORTER_CASES + BFLOAT_CASES, action="append")
    parser.add_argument("--opset", choices=(17, 18, 22), type=int, action="append")
    parser.add_argument("--importer-source", type=Path, help="Isolated importer ablation; does not replace the normal module")
    args = parser.parse_args()
    output = allowed_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "failed", "scope": "Synthetic frontend checks, LLVM host CPU only; no models or Gemmini execution", "suite": args.suite, "seed": 194, "cases": [], "verifier_source": identity(__file__)}
    try:
        import numpy as np
        import onnx
        import torch
        import tvm

        torch.set_num_threads(1)
        torch.manual_seed(194)
        from_onnx, metadata = provenance(tvm, torch, onnx, np, args.importer_source)
        report.update(metadata)
        selected = args.case or {"torch": CASES, "importer": IMPORTER_CASES, "bfloat16": BFLOAT_CASES}[args.suite]
        for name in selected:
            # BF16 Conv enters the ONNX schema at opset22; earlier exports
            # deliberately fail full schema checks rather than dropping dtype.
            for opset in args.opset or ((22,) if args.suite == "bfloat16" else (17, 18)):
                record = run_case(name, opset, output, from_onnx, torch, onnx, np, tvm)
                report["cases"].append(record)
                print(f"{name} opset {opset}: {record['status']} ({record['stage']})", flush=True)
        if all(record["status"] == "passed" for record in report["cases"]):
            report["status"] = "passed"
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    report_path = allowed_path(output / "results.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{report['status']}: {report_path}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
