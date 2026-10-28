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
"""Bounded byte affine arithmetic and proof-bound quantized residual sums.

Only packing constants runs on the host.  Weighted terms execute on the mesh;
unit terms use accumulator loads.  This is a TensorIntrin family, not a whole
operation C implementation or a change to the mathematical program.
"""

import hashlib
import struct

import tvm
from tvm import tir

from .gemmini_epilogue import prove_readout
from .gemmini_intrin import ACC, SPAD, register_intrinsic_once
from .gemmini_elementwise import _canonical_description


def _replace(expression, old, new):
    return tir.stmt_functor.ir_transform(
        tir.Evaluate(expression),
        lambda node: new if tvm.ir.structural_equal(node, old) else None,
        None,
    ).value


def _byte_term(value):
    if not isinstance(value, tir.Cast) or value.dtype != "int32" or value.value.dtype != "int8":
        return None
    inner = value.value
    if isinstance(inner, tir.BufferLoad):
        return inner, 1.0
    loads = []
    tir.stmt_functor.post_order_visit(
        inner, lambda node: loads.append(node) if isinstance(node, tir.BufferLoad) else None
    )
    if not loads or any(
        load.dtype != "int8" or not tvm.ir.structural_equal(load, loads[0]) for load in loads
    ):
        return None
    source = loads[0]
    proxy = tir.decl_buffer((1,), "int32", name="input_code")[0]

    def promote(node):
        if (
            isinstance(node, tir.Cast)
            and node.dtype == "float32"
            and tvm.ir.structural_equal(node.value, source)
        ):
            return proxy.astype("float32")
        return None

    promoted = tir.stmt_functor.ir_transform(tir.Evaluate(inner), promote, None).value
    try:
        proof = prove_readout(promoted, proxy)
    except ValueError:
        return None
    scale = struct.unpack("f", struct.pack("I", proof["scale_bits"]))[0]
    if proof["relu"] or not 0 < scale <= 1:
        return None
    return source, scale


def _linear(value):
    if isinstance(value, tir.IntImm) and value.dtype == "int32":
        return [], int(value)
    term = _byte_term(value)
    if term is not None:
        return [(term[0], 1, term[1])], 0
    if isinstance(value, (tir.Add, tir.Sub)) and value.dtype == "int32":
        left, right = _linear(value.a), _linear(value.b)
        if left is None or right is None:
            return None
        sign = -1 if isinstance(value, tir.Sub) else 1
        return left[0] + [
            (load, sign * coefficient, scale) for load, coefficient, scale in right[0]
        ], left[1] + sign * right[1]
    if isinstance(value, tir.Mul) and value.dtype == "int32":
        for expression, factor in ((value.a, value.b), (value.b, value.a)):
            if isinstance(factor, tir.IntImm):
                inner = _linear(expression)
                if inner is not None:
                    coefficient = int(factor)
                    return [
                        (load, coefficient * c, scale) for load, c, scale in inner[0]
                    ], coefficient * inner[1]
    return None


def affine_match(node):
    """Recognize one/two signed-byte affine terms on the complete byte domain."""
    if node.init is not None or not isinstance(node.body, tir.BufferStore):
        return None
    output, expression = node.body.buffer, node.body.value
    if output.dtype not in ("int8", "int32") or not 1 <= len(node.iter_vars) <= 5:
        return None
    if any(
        axis.iter_type != tir.IterVar.DataPar
        or not isinstance(axis.dom.extent, tir.IntImm)
        or int(axis.dom.extent) <= 0
        or not isinstance(axis.dom.min, tir.IntImm)
        or int(axis.dom.min) != 0
        for axis in node.iter_vars
    ):
        return None
    if not tvm.ir.structural_equal(list(node.body.indices), [axis.var for axis in node.iter_vars]):
        return None
    candidates = []
    if output.dtype == "int32":
        candidates = [expression]
    else:
        tir.stmt_functor.post_order_visit(
            expression,
            lambda value: (
                candidates.append(value)
                if isinstance(value, (tir.Add, tir.Sub, tir.Mul, tir.Cast, tir.BufferLoad))
                and value.dtype == "int32"
                else None
            ),
        )
        candidates.reverse()
    for candidate in candidates:
        # A free i32 input has no byte-source bound. Producer readout belongs
        # to its contraction/epilogue proof, not this standalone affine family.
        linear = _linear(candidate)
        if linear is None or not 1 <= len(linear[0]) <= 2:
            continue
        terms, offset = linear
        if any(
            not -128 <= c <= 127
            or c == 0
            or load.buffer.data.same_as(output.data)
            or not tvm.ir.structural_equal(load.indices, node.body.indices)
            or not tvm.ir.structural_equal(load.buffer.shape, output.shape)
            for load, c, _ in terms
        ):
            continue
        # This bound covers every prefix in either load/mesh order.
        accumulator_bound = abs(offset) + sum(128 * abs(c) for _, c, _ in terms)
        if accumulator_bound > (1 << 31) - 1:
            continue
        scaled_operands = any(scale != 1.0 for _, _, scale in terms)
        if scaled_operands and (
            output.dtype != "int8"
            or len(node.iter_vars) != 2
            or len(terms) != 2
            or offset != 0
            or any(c != 1 for _, c, _ in terms)
        ):
            continue
        proof = None
        if output.dtype == "int8":
            proxy = tir.decl_buffer((1,), "int32", name="affine_accumulator")[0]
            try:
                proof = prove_readout(_replace(expression, candidate, proxy), proxy)
            except ValueError:
                continue
        # Leave the existing widening path and its copy opt-in unchanged.
        if proof is None and len(terms) == 1 and terms[0][1:] == (1, 1.0) and offset == 0:
            continue
        return {
            "terms": terms,
            "offset": offset,
            "readout": proof,
            "per_operand_rounding": scaled_operands,
            "accumulator_partial_abs_bound": accumulator_bound,
            "pe_partial_abs_bound": max((128 * abs(c) for _, c, _ in terms if c != 1), default=0),
        }
    return None


def _ptr(buffer, mode):
    return buffer.access_ptr(mode)


def register_affine_intrinsic(node, rows, rank):
    """One-column tiles keep all host accesses strided DMA, without gather math."""
    match = affine_match(node)
    if match is None or rank not in (1, 2) or not 1 <= rows <= 16:
        raise ValueError("Unsupported bounded byte affine tile")
    shape = (1,) if rank == 1 else (rows, 1)
    sources, original = [], []
    for load, _, _ in match["terms"]:
        if not any(load.buffer.same_as(buffer) for buffer in original):
            original.append(load.buffer)
            sources.append(
                tir.decl_buffer(
                    shape,
                    load.dtype,
                    name=f"source{len(sources)}",
                    strides=[tir.Var(f"stride{len(sources)}", "int64"), 1][-rank:],
                    offset_factor=1,
                )
            )
    output = tir.decl_buffer(
        shape,
        node.body.buffer.dtype,
        name="result",
        strides=[tir.Var("out_stride", "int64"), 1][-rank:],
        offset_factor=1,
    )
    buffers = sources + [output]
    params = [tir.Var(buffer.name, "handle") for buffer in buffers]
    loops = [tir.Var(f"i{i}", "int32") for i in range(rank)]
    axes = [tir.Var(f"v{i}", "int32") for i in range(rank)]

    def substitute(value):
        if isinstance(value, tir.BufferLoad):
            for old, new in zip(original, sources):
                if value.buffer.same_as(old):
                    return new[tuple(axes)]
        return None

    expression = tir.stmt_functor.ir_transform(
        tir.Evaluate(node.body.value), substitute, None
    ).value
    expression = tir.stmt_functor.substitute(
        expression, {axis.var: value for axis, value in zip(node.iter_vars[-rank:], axes)}
    )

    def region(buffer):
        return tir.BufferRegion(buffer, [tvm.ir.Range(0, extent) for extent in buffer.shape])

    reads, writes = [region(buffer) for buffer in sources], [region(output)]
    point_reads = [
        tir.BufferRegion(buffer, [tvm.ir.Range.from_min_extent(axis, 1) for axis in axes])
        for buffer in sources
    ]
    point_writes = [
        tir.BufferRegion(output, [tvm.ir.Range.from_min_extent(axis, 1) for axis in axes])
    ]
    block = tir.Block(
        [
            tir.IterVar(tvm.ir.Range(0, extent), axis, tir.IterVar.DataPar)
            for extent, axis in zip(shape, axes)
        ],
        point_reads,
        point_writes,
        "affine",
        tir.BufferStore(output, expression, axes),
    )
    body = tir.BlockRealize(loops, True, block)
    for loop, extent in reversed(list(zip(loops, shape))):
        body = tir.For(loop, 0, extent, tir.ForKind.SERIAL, body)
    desc = tir.PrimFunc(
        params,
        tir.BlockRealize([], True, tir.Block([], reads, writes, "root", body)),
        buffer_map=dict(zip(params, buffers)),
    )
    acc = tir.decl_buffer((rows, 16), "int32", name="acc", scope=ACC)
    spad = tir.decl_buffer((rows, 16), "int8", name="operand", scope=SPAD)
    weight = tir.decl_buffer((1, 16), "int8", name="weight", scope=SPAD)
    scalar = tir.decl_buffer((), "int8", name="coefficient", scope="global")
    bias = tir.decl_buffer((), "int32", name="offset", scope="global")
    statements = []

    def call(symbol, *args):
        return tir.Evaluate(tir.call_extern("void", symbol, *args))

    initialized = match["offset"] != 0
    if initialized:
        statements += [
            tir.BufferStore(bias, tir.const(match["offset"], "int32"), []),
            call(
                "tvm_gemmini_dma_acc_write",
                _ptr(bias, "r"),
                _ptr(acc, "w"),
                rows,
                1,
                tir.IntImm("int64", 0),
                0,
            ),
        ]
    for load, coefficient, scale in match["terms"]:
        source = sources[
            next(i for i, buffer in enumerate(original) if load.buffer.same_as(buffer))
        ]
        stride = source.strides[0] if rank == 2 else tir.IntImm("int64", 1)
        if coefficient == 1:
            bits = struct.unpack("I", struct.pack("f", scale))[0]
            statements.append(
                call(
                    "tvm_gemmini_dma_scaled_in",
                    _ptr(source, "r"),
                    _ptr(acc, "rw" if initialized else "w"),
                    rows,
                    1,
                    stride,
                    tir.IntImm("uint32", bits),
                    int(initialized),
                    0,
                )
            )
        else:
            statements += [
                tir.BufferStore(scalar, tir.const(coefficient, "int8"), []),
                call("tvm_gemmini_dma_a", _ptr(source, "r"), _ptr(spad, "w"), rows, 1, stride, 0),
                call(
                    "tvm_gemmini_dma_b",
                    _ptr(scalar, "r"),
                    _ptr(weight, "w"),
                    1,
                    1,
                    tir.IntImm("int64", 1),
                    0,
                ),
                call(
                        "tvm_gemmini_compute_ws",
                    _ptr(spad, "r"),
                    _ptr(weight, "r"),
                    _ptr(acc, "rw" if initialized else "w"),
                    rows,
                    1,
                    1,
                    int(initialized),
                ),
            ]
        initialized = True
    out_stride = output.strides[0] if rank == 2 else tir.IntImm("int64", 1)
    proof = match["readout"]
    if proof is None:
        statements.append(
            call(
                "tvm_gemmini_dma_out",
                _ptr(acc, "r"),
                _ptr(output, "w"),
                rows,
                1,
                tir.IntImm("int64", 16),
                out_stride,
            )
        )
    else:
        statements.append(
            call(
                "tvm_gemmini_dma_scaled_out",
                _ptr(acc, "r"),
                _ptr(output, "w"),
                rows,
                1,
                tir.IntImm("int64", 16),
                out_stride,
                tir.IntImm("uint32", proof["scale_bits"]),
                int(proof["relu"]),
            )
        )
    inner = tir.Block(
        [],
        reads,
        writes,
        "affine_commands",
        tir.SeqStmt(statements),
        alloc_buffers=[acc, spad, weight, scalar, bias],
    )
    impl = tir.PrimFunc(
        params,
        tir.BlockRealize(
            [], True, tir.Block([], reads, writes, "root", tir.BlockRealize([], True, inner))
        ),
        buffer_map=dict(zip(params, buffers)),
    )
    desc = _canonical_description(desc)
    key = hashlib.sha256(tvm.ir.save_json(desc).encode()).hexdigest()[:24]
    name = f"gemmini.affine.{rank}d.{rows}.{key}"
    register_intrinsic_once(name, desc, impl)
    return name
