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
"""Representative per-operator Relax programs for ResNet50, SmolVLA and TinyLlama.

Each program contains every distinct accelerator-eligible operator of one model
phase, written with the same Relax operators and integer recipes as the model
exports, with one call per occurrence in the model. ``extract_tasks`` therefore
deduplicates the operators and weights each task by its real call count, without
building or simulating the whole model. Weights are synthetic constants: shapes,
dtypes, layouts and fused epilogues determine the schedule, not the values.

* TinyLlama-1.1B: the W8A8 ``IntegerProjection`` recipe (row-quantized INT8
  activations, per-channel INT8 weights, FP32 dequantization). Decode uses one
  token; prefill uses 32-token chunks.
* SmolVLA: the same projection recipe for the vision encoder (64-token chunks
  of each 1024-patch image), connector, 241-token VLM prefix and 50-token action
  expert (ten denoising steps).
* ResNet50: the integer recipe of ``resnet_quantized.export_relax`` in NHWC:
  INT8 convolution, INT32 bias, power-of-two requantization and saturation.
"""

import numpy as np
import tvm
from tvm import relax

TINYLLAMA = {"hidden": 2048, "intermediate": 5632, "kv": 256, "layers": 22, "vocab": 32000}


class _Program:
    """Builds one Relax function that calls each operator ``count`` times."""

    def __init__(self, seed=0, operators=None):
        self.rng = np.random.default_rng(seed)
        self.builder = relax.BlockBuilder()
        self.params = []
        self.operators = []
        self.selection = None if operators is None else set(operators)
        self.operator_index = 0

    def include(self):
        index = self.operator_index
        self.operator_index += 1
        return self.selection is None or index in self.selection

    def const(self, shape, dtype):
        if dtype == "int8":
            value = self.rng.integers(-127, 128, shape, dtype="int8")
        elif dtype == "int32":
            value = self.rng.integers(-(1 << 16), 1 << 16, shape).astype("int32")
        else:
            value = (self.rng.uniform(0.5, 2.0, shape) / 64).astype(dtype)
        return relax.const(value, dtype)

    def input(self, name, shape, dtype):
        var = relax.Var(name, relax.TensorStructInfo(shape, dtype))
        self.params.append(var)
        return var


def _projection(bb, program, rows, k, n, count, results):
    """W8A8 IntegerProjection: int_mm, then FP32 * activation scale * weight scale."""
    if not program.include():
        return
    codes = program.input(f"codes_{rows}_{k}_{n}", (rows, k), "int8")
    scale = program.input(f"scale_{rows}_{k}_{n}", (rows, 1), "float32")
    weight = program.const((k, n), "int8")
    weight_scale = program.const((n,), "float32")
    for _ in range(count):
        product = bb.emit(relax.op.matmul(codes, weight, out_dtype="int32"))
        value = bb.emit(relax.op.astype(product, "float32"))
        value = bb.emit(relax.op.multiply(value, scale))
        results.append(bb.emit(relax.op.multiply(value, weight_scale)))
    program.operators.append({"kind": "projection", "m": rows, "k": k, "n": n, "count": count})


def _conv(bb, program, h, w, cin, cout, kernel, stride, count, results):
    """resnet_quantized conv site in NHWC: conv, bias, 2^e requant, saturation."""
    if not program.include():
        return
    padding = kernel // 2
    data = program.input(f"conv_{h}_{w}_{cin}_{cout}_{kernel}_{stride}", (1, h, w, cin), "int8")
    weight = program.const((cout, kernel, kernel, cin), "int8")
    bias = program.const((1, 1, 1, cout), "int32")
    exponent = program.rng.integers(-12, -6, (1, 1, 1, cout))
    divisor = relax.const(np.left_shift(np.int64(1), -exponent).astype("int64"), "int64")
    half = relax.const((np.left_shift(np.int64(1), -exponent) // 2).astype("int64"), "int64")
    for _ in range(count):
        product = bb.emit(
            relax.op.nn.conv2d(
                data, weight, strides=stride, padding=padding, data_layout="NHWC",
                kernel_layout="OHWI", out_dtype="int32",
            )
        )
        product = bb.emit(relax.op.add(product, bias))
        wide = bb.emit(relax.op.astype(product, "int64"))
        magnitude = bb.emit(relax.op.abs(wide))
        rounded = bb.emit(relax.op.floor_divide(bb.emit(relax.op.add(magnitude, half)), divisor))
        signed = bb.emit(
            relax.op.where(
                bb.emit(relax.op.less(wide, relax.const(0, "int64"))),
                bb.emit(relax.op.negative(rounded)),
                rounded,
            )
        )
        results.append(bb.emit(relax.op.astype(bb.emit(relax.op.clip(signed, -127, 127)), "int8")))
    program.operators.append(
        {"kind": "conv2d", "h": h, "w": w, "cin": cin, "cout": cout, "kernel": kernel, "stride": stride, "count": count}
    )


def _finish(program, emit):
    bb = program.builder
    results = []
    with bb.function("main"):
        with bb.dataflow():
            emit(bb, program, results)
            output = bb.emit_output(relax.Tuple(results))
        bb.emit_func_output(output, params=program.params)
    mod = bb.get()
    return mod, program.operators


def tinyllama(phase="decode", *, chunk=32, seed=0, operators=None):
    rows = 1 if phase == "decode" else chunk
    c = TINYLLAMA
    shapes = [
        (c["hidden"], c["hidden"], 2 * c["layers"]),  # q_proj, o_proj
        (c["hidden"], c["kv"], 2 * c["layers"]),  # k_proj, v_proj
        (c["hidden"], c["intermediate"], 2 * c["layers"]),  # gate_proj, up_proj
        (c["intermediate"], c["hidden"], c["layers"]),  # down_proj
    ]
    if phase == "decode":
        shapes.append((c["hidden"], c["vocab"], 1))  # lm_head on the generated token

    def emit(bb, program, results):
        for k, n, count in shapes:
            _projection(bb, program, rows, k, n, count, results)

    return _finish(_Program(seed, operators), emit)


def smolvla(part="expert", *, seed=0, operators=None):
    """SmolVLA phases: ``vision`` (per 64-token chunk), ``prefix`` (VLM), ``expert``."""
    if part == "vision":
        chunks = 16 * 3  # 1024 patches per image, three cameras
        shapes = [
            (64, 768, 768, 4 * 12 * chunks),  # q, k, v, o projections, 12 layers
            (64, 768, 3072, 12 * chunks),  # fc1
            (64, 3072, 768, 12 * chunks),  # fc2
            (64, 12288, 960, 3),  # connector, one 64-token row block per image
        ]
    elif part == "prefix":
        tokens = 241  # 3 x 64 image tokens, 48 language tokens, 1 state token
        shapes = [
            (tokens, 960, 960, 2 * 16),  # q_proj, o_proj
            (tokens, 960, 320, 2 * 16),  # k_proj, v_proj
            (tokens, 960, 2560, 2 * 16),  # gate_proj, up_proj
            (tokens, 2560, 960, 16),  # down_proj
        ]
    elif part == "expert":
        steps, actions = 10, 50
        shapes = [
            (actions, 720, 960, 16 * steps),  # q_proj
            (actions, 720, 320, 2 * 8 * steps),  # self-attention k_proj, v_proj
            (actions, 960, 720, 16 * steps),  # o_proj
            (actions, 720, 2048, 2 * 16 * steps),  # gate_proj, up_proj
            (actions, 2048, 720, 16 * steps),  # down_proj
            (actions, 32, 720, steps),  # action_in_proj
            (actions, 1440, 720, steps),  # action_time_mlp_in
            (actions, 720, 720, steps),  # action_time_mlp_out
            (actions, 720, 32, steps),  # action_out_proj
        ]
    else:
        raise ValueError("Unknown SmolVLA part")

    def emit(bb, program, results):
        for m, k, n, count in shapes:
            _projection(bb, program, m, k, n, count, results)

    return _finish(_Program(seed, operators), emit)


# torchvision ResNet50 convolutions at 224x224: (H, W, Cin, Cout, kernel, stride, count).
RESNET50 = [
    (224, 224, 3, 64, 7, 2, 1),
    (56, 56, 64, 64, 1, 1, 1),
    (56, 56, 64, 64, 3, 1, 3),
    (56, 56, 64, 256, 1, 1, 4),
    (56, 56, 256, 64, 1, 1, 2),
    (56, 56, 256, 128, 1, 1, 1),
    (56, 56, 128, 128, 3, 2, 1),
    (28, 28, 128, 512, 1, 1, 4),
    (56, 56, 256, 512, 1, 2, 1),
    (28, 28, 512, 128, 1, 1, 3),
    (28, 28, 128, 128, 3, 1, 3),
    (28, 28, 512, 256, 1, 1, 1),
    (28, 28, 256, 256, 3, 2, 1),
    (14, 14, 256, 1024, 1, 1, 6),
    (28, 28, 512, 1024, 1, 2, 1),
    (14, 14, 1024, 256, 1, 1, 5),
    (14, 14, 256, 256, 3, 1, 5),
    (14, 14, 1024, 512, 1, 1, 1),
    (14, 14, 512, 512, 3, 2, 1),
    (7, 7, 512, 2048, 1, 1, 3),
    (14, 14, 1024, 2048, 1, 2, 1),
    (7, 7, 2048, 512, 1, 1, 2),
    (7, 7, 512, 512, 3, 1, 2),
]


def resnet50(*, seed=0, layers=None, operators=None):
    selected = RESNET50 if layers is None else [RESNET50[i] for i in layers]

    def emit(bb, program, results):
        for h, w, cin, cout, kernel, stride, count in selected:
            _conv(bb, program, h, w, cin, cout, kernel, stride, count, results)
        # Classifier: the recipe's linear site is an INT8 matmul with bias.
        _projection(bb, program, 1, 2048, 1000, 1, results)

    return _finish(_Program(seed, operators), emit)


def smoke(*, seed=0, operators=None):
    """Small shapes exercising tails, padding and both epilogue kinds."""

    def emit(bb, program, results):
        _projection(bb, program, 50, 32, 32, 2, results)
        _projection(bb, program, 1, 64, 48, 1, results)

    return _finish(_Program(seed, operators), emit)


MODELS = {
    "smoke": smoke,
    "tinyllama-decode": lambda **kw: tinyllama("decode", **kw),
    "tinyllama-prefill": lambda **kw: tinyllama("prefill", **kw),
    "smolvla-vision": lambda **kw: smolvla("vision", **kw),
    "smolvla-prefix": lambda **kw: smolvla("prefix", **kw),
    "smolvla-expert": lambda **kw: smolvla("expert", **kw),
    "resnet50": resnet50,
}


def build(name, *, operators=None):
    mod, operators = MODELS[name](operators=operators)
    if not operators:
        raise ValueError("Operator selection is empty")
    if not tvm.relax.analysis.well_formed(mod):
        raise AssertionError("Representative program is not well formed")
    return mod, operators
