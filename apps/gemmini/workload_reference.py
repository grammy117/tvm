# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Independent NumPy references for bounded semantic Relax experiment graphs.

The evaluator reads constants and expressions, never compiled candidate code.
Integer contractions accumulate in checked INT64 and must fit their declared
output type. Every explicit floating operation retains its declared boundary.
This is intentionally a small interpreter: unknown/opaque operations refuse.
Synthetic inputs establish experiment correctness, not selected EL4 captures.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import tvm
from tvm import relax


def identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def checked_integer(value, dtype):
    value = np.asarray(value)
    bound = np.iinfo(dtype)
    if value.size and (value.min() < bound.min or value.max() > bound.max):
        raise ValueError("Independent integer result exceeds " + str(dtype))
    return value.astype(dtype)


def integer_matmul(a, b, dtype):
    if a.dtype.kind != "i" or b.dtype.kind != "i" or a.ndim != 2 or b.ndim != 2:
        raise ValueError("Reference matmul requires rank-2 signed integer operands")
    if a.shape[1] != b.shape[0] or a.dtype.itemsize > 4 or b.dtype.itemsize > 4:
        raise ValueError("Unsupported integer contraction dimensions or operand width")
    magnitude = max(abs(int(a.min())), abs(int(a.max())))
    magnitude *= max(abs(int(b.min())), abs(int(b.max()))) * a.shape[1]
    if magnitude > np.iinfo(np.int64).max:
        raise ValueError("Reference INT64 accumulation could overflow")
    return checked_integer(a.astype(np.int64) @ b.astype(np.int64), dtype)


def integer_conv2d(x, w, attrs):
    if (str(attrs.data_layout) != "NHWC" or str(attrs.kernel_layout) != "OHWI"
            or str(attrs.out_layout) not in ("", "NHWC") or int(attrs.groups) != 1
            or x.dtype.kind != "i" or w.dtype.kind != "i"
            or x.dtype.itemsize > 1 or w.dtype.itemsize > 1):
        raise ValueError("Reference convolution requires NHWC/OHWI ungrouped INT8")
    sh, sw = map(int, attrs.strides)
    dh, dw = map(int, attrs.dilation)
    top, left, bottom, right = map(int, attrs.padding)
    batch, height, width, channels = x.shape
    outputs, kh, kw, wc = w.shape
    if channels != wc:
        raise ValueError("Convolution channel mismatch")
    oh = (height + top + bottom - (kh - 1) * dh - 1) // sh + 1
    ow = (width + left + right - (kw - 1) * dw - 1) // sw + 1
    if min(oh, ow, sh, sw, dh, dw) < 1 or min(top, left, bottom, right) < 0:
        raise ValueError("Unsupported convolution extent")
    if kh * kw * channels * 128 * 128 > np.iinfo(np.int64).max:
        raise ValueError("Reference convolution INT64 accumulation could overflow")
    padded = np.pad(x, ((0, 0), (top, bottom), (left, right), (0, 0)))
    out = np.zeros((batch, oh, ow, outputs), dtype=np.int64)
    for i in range(kh):
        for j in range(kw):
            patch = padded[:, i * dh:i * dh + oh * sh:sh,
                           j * dw:j * dw + ow * sw:sw, :]
            product = integer_matmul(patch.reshape(-1, channels), w[:, i, j, :].T, "int64")
            out += product.reshape(out.shape)
    return checked_integer(out, str(attrs.out_dtype))


def _tensor(value, info):
    if not isinstance(info, relax.TensorStructInfo):
        raise ValueError("Expected static tensor type in reference")
    shape = tuple(int(v) for v in info.shape)
    result = np.asarray(value)
    if result.shape != shape or result.dtype.name != str(info.dtype):
        raise ValueError(f"Reference type mismatch: {result.shape}/{result.dtype} versus {info}")
    if result.dtype.kind == "f" and not np.isfinite(result).all():
        raise ValueError("Independent reference produced nonfinite output")
    return result


class Reference:
    def __init__(self, inputs, params, *, retain_checkpoints=False):
        if len(inputs) != len(params):
            raise ValueError("Reference input count mismatch")
        self.values = {p: _tensor(a, p.struct_info) for p, a in zip(params, inputs)}
        self.records = []
        self.checkpoints = {} if retain_checkpoints else None

    def expression(self, expr):
        if isinstance(expr, relax.Var):
            return self.values[expr]
        if isinstance(expr, relax.Constant):
            return expr.data.numpy()
        if isinstance(expr, relax.PrimValue):
            if not isinstance(expr.value, (tvm.tir.IntImm, tvm.tir.FloatImm)):
                raise ValueError("Reference requires literal scalar values")
            return expr.value.value
        if isinstance(expr, relax.Tuple):
            return tuple(self.expression(v) for v in expr.fields)
        if isinstance(expr, relax.TupleGetItem):
            return self.expression(expr.tuple_value)[expr.index]
        if isinstance(expr, relax.SeqExpr):
            for block in expr.blocks:
                for binding in block.bindings:
                    if not isinstance(binding, relax.VarBinding):
                        raise ValueError("Reference requires VarBinding")
                    value = self.expression(binding.value)
                    self.values[binding.var] = value
                    if isinstance(value, np.ndarray):
                        name = f"binding_{len(self.records):04d}_{binding.var.name_hint}"
                        self.records.append({"key": name, "name": binding.var.name_hint,
                                             "shape": list(value.shape), "dtype": value.dtype.name,
                                             "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
                                             "min": float(value.min()), "max": float(value.max())})
                        if self.checkpoints is not None:
                            self.checkpoints[name] = value
            return self.expression(expr.body)
        if not isinstance(expr, relax.Call) or not isinstance(expr.op, tvm.ir.Op):
            raise ValueError("Opaque or unsupported reference expression")
        args = [self.expression(v) for v in expr.args]
        name = expr.op.name
        unary = {"relax.abs": np.abs, "relax.negative": np.negative,
                 "relax.round": lambda a: np.copysign(np.floor(np.abs(a.astype(np.float64)) + .5), a).astype(a.dtype),
                 "relax.nn.relu": lambda a: np.maximum(a, 0)}
        binary = {"relax.add": np.add, "relax.subtract": np.subtract,
                  "relax.multiply": np.multiply, "relax.divide": np.divide,
                  "relax.floor_divide": np.floor_divide, "relax.maximum": np.maximum,
                  "relax.minimum": np.minimum, "relax.less": np.less}
        if name in unary:
            value = unary[name](*args)
        elif name in binary:
            value = binary[name](*args)
        elif name == "relax.clip":
            value = np.clip(*args)
        elif name == "relax.where":
            value = np.where(*args)
        elif name == "relax.astype":
            dtype = str(expr.attrs.dtype)
            value = checked_integer(args[0], dtype) if np.dtype(dtype).kind == "i" else args[0].astype(dtype)
        elif name == "relax.max":
            axis = None if expr.attrs.axis is None else tuple(int(v) for v in expr.attrs.axis)
            value = np.max(args[0], axis=axis, keepdims=bool(expr.attrs.keepdims))
        elif name == "relax.concat":
            value = np.concatenate(args[0], axis=int(expr.attrs.axis))
        elif name == "relax.matmul":
            value = integer_matmul(*args, str(expr.attrs.out_dtype))
        elif name == "relax.nn.conv2d":
            value = integer_conv2d(*args, expr.attrs)
        elif name == "relax.sigmoid":
            # Independent mathematical sigmoid, rounded at the Relax FP boundary.
            x = args[0].astype(np.float64)
            e = np.exp(-np.abs(x))
            value = np.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e)).astype(args[0].dtype)
        else:
            raise ValueError("Unsupported reference operator: " + name)
        return _tensor(value, expr.struct_info)


def evaluate(mod, inputs, *, retain_checkpoints=False):
    function = mod["main"]
    reference = Reference(inputs, function.params, retain_checkpoints=retain_checkpoints)
    result = reference.expression(function.body)
    outputs = list(result) if isinstance(result, tuple) else [result]
    if any(not isinstance(v, np.ndarray) for v in outputs):
        raise ValueError("Reference output must be a flat tuple of tensors")
    return outputs, reference


def synthetic_inputs(mod, seed=0, *, nonnegative_int8=False, integer_magnitude=31):
    if not 1 <= integer_magnitude <= 127:
        raise ValueError("INT8 fixture magnitude must be in [1,127]")
    rng = np.random.default_rng(seed)
    inputs, records = [], []
    for param in mod["main"].params:
        info = param.struct_info
        shape, dtype = tuple(int(v) for v in info.shape), str(info.dtype)
        if dtype == "float32":
            value = rng.integers(-128, 129, size=shape).astype("float32") / np.float32(128)
            distribution = "uniform dyadic grid [-1,1], step 1/128"
        elif dtype == "int8":
            low, high = (0 if nonnegative_int8 else -integer_magnitude), integer_magnitude + 1
            value = rng.integers(low, high, size=shape, dtype="int8")
            distribution = f"uniform integers [{low},{high - 1}]"
        else:
            raise ValueError("Synthetic input dtype not supported: " + dtype)
        inputs.append(value)
        records.append({"name": param.name_hint, "shape": list(shape), "dtype": dtype,
                        "distribution": distribution})
    return inputs, records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, help="NPZ containing actual input_0, etc.; preferred for real slices")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nonnegative-int8", action="store_true", help="For post-ReLU activation inputs")
    parser.add_argument("--integer-magnitude", type=int, default=31,
                        help="Bound synthetic INT8 inputs; default avoids pervasive surrogate requant saturation")
    parser.add_argument("--checkpoints", action="store_true")
    args = parser.parse_args(argv)
    mod = tvm.ir.load_json(args.program.read_text())
    if args.inputs:
        with np.load(args.inputs, allow_pickle=False) as data:
            inputs = [data[f"input_{i}"].copy() for i in range(len(mod["main"].params))]
        input_record = {"source": identity(args.inputs)}
    else:
        inputs, specifications = synthetic_inputs(mod, args.seed, nonnegative_int8=args.nonnegative_int8,
                                                  integer_magnitude=args.integer_magnitude)
        input_record = {"source": "synthetic boundary activations, not selected EL4 capture",
                        "seed": args.seed, "specifications": specifications}
    outputs, reference = evaluate(mod, inputs, retain_checkpoints=args.checkpoints)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    fixture = args.output_dir / "fixture.npz"
    arrays = {f"input_{i}": value for i, value in enumerate(inputs)}
    arrays.update({("expected" if len(outputs) == 1 else f"expected_{i}"): value
                   for i, value in enumerate(outputs)})
    np.savez(fixture, **arrays)
    if args.checkpoints:
        np.savez(args.output_dir / "reference-checkpoints.npz", **reference.checkpoints)
    sources = [identity(Path(__file__)), identity(args.program), identity(np.core._multiarray_umath.__file__)]
    if args.inputs:
        sources.append(identity(args.inputs))
    provenance = {"schema": "gemmini_independent_chunk_reference_v1",
                  "fixture_sha256": identity(fixture)["sha256"], "reference_sources": sources,
                  "reference_contract": "Independent NumPy semantic evaluator: checked INT64 integer contractions; explicit declared FP32 boundaries; Relax round ties away from zero (distinct from PyTorch ties-even); mathematical sigmoid via FP64 then declared rounding; no candidate execution",
                  "program": identity(args.program), "inputs": input_record,
                  "environment": {"python": platform.python_version(), "numpy": np.__version__},
                  "comparison": {"rtol": 1e-5, "atol": 2e-6, "integer": "exact", "finite": True},
                  "full_model_qualified": False}
    (args.output_dir / "fixture-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (args.output_dir / "reference-bindings.json").write_text(json.dumps(reference.records, indent=2) + "\n")
    print(json.dumps({"fixture": str(fixture), "outputs": [{"shape": list(v.shape), "dtype": v.dtype.name}
                                                           for v in outputs]}))


if __name__ == "__main__":
    main()
