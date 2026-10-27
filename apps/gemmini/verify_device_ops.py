#!/usr/bin/env python3
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
"""Independent byte/affine/residual fixtures for the normal TensorIR pipeline.

Prepare a guarded positive/wrong-oracle ELF pair, then execute with the existing
verify_tensorir.py run --backend spike command. No model assets are required.
"""

import argparse
from pathlib import Path

import numpy as np
from tvm import relax, te, tir, topi

from verify_tensorir import prepare


def graph(func, input_count, constants=None):
    buffers = [func.buffer_map[param] for param in func.params]
    builder = relax.BlockBuilder()
    kernel = builder.add_func(func, "kernel")
    args = [
        relax.Var(f"input{i}", relax.TensorStructInfo(buffer.shape, buffer.dtype))
        for i, buffer in enumerate(buffers[:input_count])
    ]
    constants = constants or {}
    call_args = [relax.const(constants[i]) if i in constants else arg for i, arg in enumerate(args)]
    with builder.function("main", [arg for i, arg in enumerate(args) if i not in constants]):
        with builder.dataflow():
            result = builder.emit_output(
                relax.call_tir(
                    kernel,
                    call_args,
                    out_sinfo=relax.TensorStructInfo(buffers[-1].shape, buffers[-1].dtype),
                )
            )
        builder.emit_func_output(result)
    return builder.get()


def quantize(value, scale, relu=False):
    value = value.astype("float32") * tir.const(scale, "float32")
    if relu:
        value = te.max(value, tir.const(0, "float32"))
    return te.min(
        te.max(tir.nearbyint(value), tir.const(-128, "float32")), tir.const(127, "float32")
    ).astype("int8")


def fixtures():
    result = []
    shape = (17, 19)
    x = np.resize(np.arange(-128, 128, dtype="int16"), shape).astype("int8")
    y = np.flip(x).copy()
    for name, alpha, beta, offset, scale, relu in (
        ("subtract", 1, -1, 0, None, False),
        ("negate", -1, 0, 0, None, False),
        ("scalar_affine", 7, 0, -23, None, False),
        ("single_round", 1, 1, 0, 0.5, False),
        ("affine_relu", -3, 2, 19, 0.125, True),
    ):
        a, b = te.placeholder(shape, "int8", "a"), te.placeholder(shape, "int8", "b")

        def compute(i, j):
            value = a[i, j].astype("int32") * alpha + b[i, j].astype("int32") * beta + offset
            return value if scale is None else quantize(value, scale, relu)

        c = te.compute(shape, compute, name="affine")
        expected = x.astype("int64") * alpha + y.astype("int64") * beta + offset
        if scale is not None:
            expected = expected.astype("float32") * np.float32(scale)
            if relu:
                expected = np.maximum(expected, 0)
            expected = np.clip(np.rint(expected), -128, 127).astype("int8")
        else:
            expected = expected.astype("int32")
        result.append((name, graph(te.create_prim_func([a, b, c]), 2), [x, y], expected))
    # Every pair of byte operands: exercises ties, both signs, and saturation.
    for scales in ((0.5, 0.25), (0.1, 0.3)):
        shape = (256, 256)
        a, b = te.placeholder(shape, "int8", "a"), te.placeholder(shape, "int8", "b")
        c = te.compute(
            shape,
            lambda i, j: quantize(
                quantize(a[i, j], scales[0]).astype("int32")
                + quantize(b[i, j], scales[1]).astype("int32"),
                1.0,
            ),
            name="residual",
        )
        x = np.broadcast_to(np.arange(-128, 128, dtype="int16")[:, None], shape).astype("int8")
        y = x.T.copy()
        expected = np.clip(
            np.rint(x.astype("float32") * np.float32(scales[0]))
            + np.rint(y.astype("float32") * np.float32(scales[1])),
            -128,
            127,
        ).astype("int8")
        result.append(
            (
                "per_operand_" + str(scales[0]).replace(".", ""),
                graph(te.create_prim_func([a, b, c]), 2),
                [x, y],
                expected,
            )
        )
    for dtype in ("bool", "int16", "float16", "bfloat16", "int64", "float64"):
        shape = (3, 17)
        a = te.placeholder(shape, dtype, "source")
        c = te.compute((17, 3), lambda i, j: a[j, i], name="permuted")
        storage = np.dtype("uint16" if dtype == "bfloat16" else dtype)
        values = np.zeros(shape, storage)
        if dtype == "bool":
            values = (np.arange(np.prod(shape)).reshape(shape) % 2).astype("bool")
        elif dtype == "bfloat16":
            values = np.resize(np.array([0, 0x8000, 0x7FC1, 0x7F81, 1, 0xFF80], "uint16"), shape)
        elif storage.kind == "f":
            word = np.dtype("uint" + str(storage.itemsize * 8))
            bits = np.resize(
                np.array([0, 1, np.iinfo(word).max, 1 << (word.itemsize * 8 - 1)], dtype=word),
                shape,
            )
            values = bits.view(storage)
        else:
            values = (np.arange(np.prod(shape)).reshape(shape) * 8191 + 127).astype(storage)
        result.append(
            (
                "dma_" + dtype,
                graph(te.create_prim_func([a, c]), 1),
                [values],
                values.T.copy(),
                "gemmini",
                {"offload_copies": True, "movement_strategy": "scalar"},
            )
        )
    a = te.placeholder((3, 18), "int64", "source")
    c = te.compute((3, 18), lambda i, j: a[i, j], name="copied")
    values = np.resize(np.array([-(1 << 63), (1 << 63) - 1, -1, 0, 1], "int64"), (3, 18))
    result.append(
        (
            "dma_int64_rectangular",
            graph(te.create_prim_func([a, c]), 1),
            [values],
            values.copy(),
            "gemmini",
            {"offload_copies": True},
        )
    )
    for size, padding in ((2, 0), (3, 1)):
        shape = (1, 3, 5, 7)
        a = te.placeholder(shape, "int8", "image")
        c = topi.nn.pool2d(a, (size, size), (2, 2), (1, 1), (padding,) * 4, "max", layout="NCHW")
        values = np.resize(np.arange(-128, 0, dtype="int16"), shape).astype("int8")
        padded = np.pad(
            values, ((0, 0), (0, 0), (padding, padding), (padding, padding)), constant_values=-128
        )
        expected = np.empty(tuple(int(v) for v in c.shape), "int8")
        for n, channel, h, w in np.ndindex(expected.shape):
            expected[n, channel, h, w] = padded[
                n, channel, h * 2 : h * 2 + size, w * 2 : w * 2 + size
            ].max()
        result.append(
            (f"max_pool_{size}", graph(te.create_prim_func([a, c]), 1), [values], expected)
        )
    for depth in (4, 9):
        a = te.placeholder((3, 5, depth), "int8", "windows")
        k = te.reduce_axis((0, depth), "k")
        sums = te.compute(
            (3, 5), lambda i, j: te.sum(a[i, j, k].astype("int32"), axis=k), name="window_sum"
        )
        c = te.compute(
            (3, 5),
            lambda i, j: quantize(sums[i, j], float(np.float32(1 / depth))),
            name="window_average",
        )
        values = np.resize(np.arange(-128, 128, dtype="int16"), (3, 5, depth)).astype("int8")
        expected = np.clip(
            np.rint(values.astype("int64").sum(axis=2).astype("float32") * np.float32(1 / depth)),
            -128,
            127,
        ).astype("int8")
        result.append(
            (f"window_average_{depth}", graph(te.create_prim_func([a, c]), 1), [values], expected)
        )
    a = te.placeholder((3, 17), "int32", "accumulator")
    c = te.compute((3, 17), lambda i, j: quantize(a[i, j], 0.0031129, True), name="readout")
    values = np.resize(
        np.array([-(1 << 31), (1 << 31) - 1, -128, 0, 1, 161, 482, 1024], "int32"), (3, 17)
    )
    expected = np.clip(
        np.rint(np.maximum(values.astype("float32") * np.float32(0.0031129), 0)), -128, 127
    ).astype("int8")
    result.append(("standalone_readout", graph(te.create_prim_func([a, c]), 1), [values], expected))
    # A nonuniform frozen group-of-16 vector must retain every original value.
    # Raw INT32 Gemmini output plus host FP32 rescaling is an allowed exit.
    a, b = te.placeholder((3, 5), "int8", "a"), te.placeholder((5, 33), "int8", "b")
    scales = te.placeholder((33,), "float32", "scales")
    k = te.reduce_axis((0, 5), "k")
    product = te.compute(
        (3, 33), lambda i, j: te.sum(a[i, k].astype("int32") * b[k, j].astype("int32"), axis=k)
    )
    c = te.compute((3, 33), lambda i, j: product[i, j].astype("float32") * scales[j])
    left = np.arange(-8, 7, dtype="int8").reshape(3, 5)
    right = np.resize(np.arange(-7, 8, dtype="int8"), (5, 33))
    frozen = np.repeat(np.array([1 / 256, 3 / 256, 7 / 256], "float32"), 16)[:33]
    expected = (left.astype("int64") @ right.astype("int64")).astype("float32") * frozen
    result.append(
        (
            "group16_preserved",
            graph(te.create_prim_func([a, b, scales, c]), 3, {2: frozen}),
            [left, right],
            expected,
            "gemmini",
            {},
            2,
        )
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapter-receipt", type=Path, required=True)
    parser.add_argument("--riscv-gcc", type=Path, required=True)
    parser.add_argument("--cases", nargs="+")
    args = parser.parse_args()
    cases = fixtures()
    args.cases = args.cases or [case[0] for case in cases]
    args.timing_repeats, args.rtol, args.atol, args.libm = 0, 0.0, 0.0, False
    prepare(args, cases=cases)


if __name__ == "__main__":
    main()
