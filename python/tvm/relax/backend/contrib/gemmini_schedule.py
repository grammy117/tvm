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

"""Bounded mathematical TensorIR schedules for the Gemmini primitive C ABI.

The semantic module uses ordinary buffer copies and exact integer reductions.
TensorIntrin substitution replaces those blocks with load/compute/store calls;
TVM retains the macro-tile loops, operand reuse, reduction order and row layout.
These schedules are synchronous and do not implement DMA/compute overlap.
The fixed DIM=16, scratchpad/accumulator layout and PE width are also enforced
by apps/gemmini/matmul.c. This is not a general fused-contraction recognizer.
"""

from dataclasses import dataclass
from functools import lru_cache

import tvm
from tvm import tir
from tvm.script import tir as T

_DIM = 16
_SP_ROWS = 16384
_ACC_ROWS = 1024
_MAX_K = 131071


@dataclass(frozen=True)
class GemminiSchedule:
    """Separate semantic/device modules and the actual tensorization trace."""

    semantic_mod: tvm.IRModule
    scheduled_mod: tvm.IRModule
    trace: tir.schedule.Trace
    lowering_steps: tuple
    metadata: dict


def _check_shape(m, n, k, tile_i, tile_j):
    if any(type(x) is not int or x <= 0 for x in (m, n, k)) or k > _MAX_K:
        raise ValueError("Gemmini requires positive static dimensions and K <= 131071")
    if any(type(x) is not int or x not in (1, 2, 4) for x in (tile_i, tile_j)):
        raise ValueError("Gemmini macro tiles must be 1, 2 or 4 tiles in each dimension")
    if max(m * k, k * n, m * n * 4) > (1 << 63) - 1 or max(k, n * 4) > (1 << 32) - 1:
        raise ValueError("Gemmini byte spans or load/store strides exceed the ABI")


def _regions(extent, tiles):
    """Full macro regions plus one tail, each with fixed micro-tile extents."""
    full, tail = divmod(extent, tiles * _DIM)
    result = []
    if full:
        result.append((full, 0, [(0, tiles, _DIM)], True))
    if tail:
        whole, partial = divmod(tail, _DIM)
        groups = ([(0, whole, _DIM)] if whole else []) + ([(whole, 1, partial)] if partial else [])
        result.append((1, full * tiles * _DIM, groups, False))
    return result


def make_semantic_matmul(m, n, k, tile_i=1, tile_j=1):
    """Create CPU-executable mathematics before any intrinsic substitution.

    A macro tile retains tile_i A tiles and tile_j B tiles per K chunk. Each
    accumulator tile is overwritten on the first chunk and updated thereafter.
    Tail blocks have fixed extents, so no padded operand is read or computed.
    """
    _check_shape(m, n, k, tile_i, tile_j)
    lines = ["@T.prim_func", f"def main(A: T.Buffer((T.int64({m}), T.int64({k})), 'int8'), B: T.Buffer((T.int64({k}), T.int64({n})), 'int8'), C: T.Buffer((T.int64({m}), T.int64({n})), 'int32')):"]
    attrs = {"tir.noalias": True, "gemmini.m": m, "gemmini.n": n, "gemmini.k": k, "gemmini.tile_i": tile_i, "gemmini.tile_j": tile_j}
    # Tensorize unifies the index dtype of all buffers in a descriptor. Use
    # int64 throughout; only bounded device row addresses are cast to uint32.
    lines += [f"    T.func_attr({attrs!r})", f"    AL = T.alloc_buffer((T.int64({tile_i}), T.int64(16), T.int64(16)), 'int8', strides=[T.int64(256), T.int64(16), T.int64(1)])",
              f"    BL = T.alloc_buffer((T.int64({tile_j}), T.int64(16), T.int64(16)), 'int8', strides=[T.int64(256), T.int64(16), T.int64(1)])",
              f"    CL = T.alloc_buffer((T.int64({tile_i}), T.int64({tile_j}), T.int64(16), T.int64(16)), 'int32', strides=[T.int64({tile_j * 256}), T.int64(256), T.int64(16), T.int64(1)])"]
    serial = 0

    def emit(indent, text):
        lines.append("    " * indent + text)

    def block_name(kind, rows, cols, depth=0):
        nonlocal serial
        serial += 1
        return f"gemmini_{kind}_{rows}_{cols}_{depth}_{serial}"

    for mr in _regions(m, tile_i):
        for nr in _regions(n, tile_j):
            serial += 1
            region = serial
            emit(1, f"for mi{region} in T.serial(T.int64({mr[0]})):")
            emit(2, f"for nj{region} in T.serial(T.int64({nr[0]})):")
            mb = f"mi{region} * T.int64({tile_i * 16})" if mr[3] else f"T.int64({mr[1]})"
            nb = f"nj{region} * T.int64({tile_j * 16})" if nr[3] else f"T.int64({nr[1]})"

            def phase(indent, depth, kb, update):
                for start, count, rows in mr[2]:
                    name = block_name("a", rows, depth)
                    emit(indent, f"for ai in T.serial({count}):")
                    emit(indent + 1, f"with T.block('{name}'):")
                    emit(indent + 2, f"T.reads(A[{mb} + (ai + {start}) * 16: {mb} + (ai + {start}) * 16 + {rows}, {kb}: {kb} + {depth}])")
                    emit(indent + 2, f"T.writes(AL[ai + {start}, 0:{rows}, 0:{depth}])")
                    emit(indent + 2, f"for x, y in T.grid({rows}, {depth}):")
                    emit(indent + 3, f"AL[ai + {start}, x, y] = A[{mb} + (ai + {start}) * 16 + x, {kb} + y]")
                for start, count, cols in nr[2]:
                    name = block_name("b", depth, cols)
                    emit(indent, f"for bj in T.serial({count}):")
                    emit(indent + 1, f"with T.block('{name}'):")
                    emit(indent + 2, f"T.reads(B[{kb}: {kb} + {depth}, {nb} + (bj + {start}) * 16: {nb} + (bj + {start}) * 16 + {cols}])")
                    emit(indent + 2, f"T.writes(BL[bj + {start}, 0:{depth}, 0:{cols}])")
                    emit(indent + 2, f"for x, y in T.grid({depth}, {cols}):")
                    emit(indent + 3, f"BL[bj + {start}, x, y] = B[{kb} + x, {nb} + (bj + {start}) * 16 + y]")
                for ast, acount, rows in mr[2]:
                    for bst, bcount, cols in nr[2]:
                        name = block_name("update" if update else "overwrite", rows, cols, depth)
                        emit(indent, f"for ai, bj in T.grid({acount}, {bcount}):")
                        emit(indent + 1, f"with T.block('{name}'):")
                        reads = f"AL[ai + {ast}, 0:{rows}, 0:{depth}], BL[bj + {bst}, 0:{depth}, 0:{cols}]"
                        if update:
                            reads += f", CL[ai + {ast}, bj + {bst}, 0:{rows}, 0:{cols}]"
                        emit(indent + 2, f"T.reads({reads})")
                        emit(indent + 2, f"T.writes(CL[ai + {ast}, bj + {bst}, 0:{rows}, 0:{cols}])")
                        if not update:
                            emit(indent + 2, f"for x, y in T.grid({rows}, {cols}):")
                            emit(indent + 3, f"CL[ai + {ast}, bj + {bst}, x, y] = T.int32(0)")
                        emit(indent + 2, f"for x, y, r in T.grid({rows}, {cols}, {depth}):")
                        emit(indent + 3, f"CL[ai + {ast}, bj + {bst}, x, y] += T.Cast('int32', AL[ai + {ast}, x, r]) * T.Cast('int32', BL[bj + {bst}, r, y])")

            phase(3, min(k, 16), "T.int64(0)", False)
            if k // 16 > 1:
                emit(3, f"for ko in T.serial(T.int64({k // 16 - 1})):")
                phase(4, 16, "(ko + T.int64(1)) * T.int64(16)", True)
            if k > 16 and k % 16:
                phase(3, k % 16, f"T.int64({k // 16 * 16})", True)
            for ast, acount, rows in mr[2]:
                for bst, bcount, cols in nr[2]:
                    name = block_name("store", rows, cols)
                    emit(3, f"for ai, bj in T.grid({acount}, {bcount}):")
                    emit(4, f"with T.block('{name}'):")
                    emit(5, f"T.reads(CL[ai + {ast}, bj + {bst}, 0:{rows}, 0:{cols}])")
                    emit(5, f"T.writes(C[{mb} + (ai + {ast}) * 16: {mb} + (ai + {ast}) * 16 + {rows}, {nb} + (bj + {bst}) * 16: {nb} + (bj + {bst}) * 16 + {cols}])")
                    emit(5, f"for x, y in T.grid({rows}, {cols}):")
                    emit(6, f"C[{mb} + (ai + {ast}) * 16 + x, {nb} + (bj + {bst}) * 16 + y] = CL[ai + {ast}, bj + {bst}, x, y]")
    # Explicit read/write regions need the same index dtype as the buffers;
    # the pinned LowerMatchBuffer does not implicitly cast range extents.
    for extent in range(1, 17):
        lines = [line.replace(f"0:{extent}]", f"T.int64(0):T.int64({extent})]").replace(f"0:{extent},", f"T.int64(0):T.int64({extent}),") for line in lines]
    func = tvm.script.from_source("\n".join(lines), {"T": T})
    # Canonicalize indices of extent-one loops for the pinned matcher. This
    # remains ordinary CPU mathematics and preserves the explicit tile loops.
    return tir.transform.Simplify()(tvm.IRModule({"main": func}))


@lru_cache(None)
def _intrinsic(kind, rows, cols, depth, bbase):
    """Register a fixed-shape descriptor, including each genuine tail shape."""
    name = f"gemmini_primitive_{kind}_{rows}_{cols}_{depth}_{bbase}"
    compute = kind in ("overwrite", "update")
    prefix = ["@T.prim_func", "def main(a: T.handle, b: T.handle" + (", c: T.handle):" if compute else "):")]
    if compute:
        buffers = [("A", "a", rows, depth, "int8"), ("B", "b", depth, cols, "int8"), ("C", "c", rows, cols, "int32")]
        for buf, arg, x, y, dtype in buffers:
            prefix.append(f"    {buf} = T.match_buffer({arg}, ({x}, {y}), '{dtype}', strides=[16, 1], elem_offset=T.int64(), offset_factor=1)")
        prefix += ["    with T.block('root'):", f"        T.reads(A[0:{rows}, 0:{depth}], B[0:{depth}, 0:{cols}]" + (f", C[0:{rows}, 0:{cols}])" if kind == "update" else ")"),
                   f"        T.writes(C[0:{rows}, 0:{cols}])"]
        body = []
        if kind == "overwrite":
            body += [f"        for x, y in T.grid({rows}, {cols}):", "            C[x, y] = T.int32(0)"]
        body += [f"        for x, y, r in T.grid({rows}, {cols}, {depth}):", "            C[x, y] += T.Cast('int32', A[x, r]) * T.Cast('int32', B[r, y])"]
        call = f"T.call_extern('void', 'tvm_gemmini_compute', T.Cast('uint32', A.elem_offset // 16), T.Cast('uint32', {bbase} + B.elem_offset // 16), T.Cast('uint32', C.elem_offset // 16), T.uint32({rows}), T.uint32({cols}), T.uint32({depth}), T.int32({int(kind == 'update')}))"
    else:
        dtype = "int32" if kind == "store" else "int8"
        srcstride, dststride = ("16", "T.int64()") if kind == "store" else ("T.int64()", "16")
        prefix += [f"    A = T.match_buffer(a, ({rows}, {cols}), '{dtype}', strides=[{srcstride}, 1], elem_offset=T.int64(), offset_factor=1)",
                   f"    B = T.match_buffer(b, ({rows}, {cols}), '{dtype}', strides=[{dststride}, 1], elem_offset=T.int64(), offset_factor=1)",
                   "    with T.block('root'):", f"        T.reads(A[0:{rows}, 0:{cols}])", f"        T.writes(B[0:{rows}, 0:{cols}])"]
        body = [f"        for x, y in T.grid({rows}, {cols}):", "            B[x, y] = A[x, y]"]
        if kind == "store":
            args = "B.access_ptr('w'), T.Cast('uint32', A.elem_offset // 16)"
        else:
            args = f"A.access_ptr('r'), T.Cast('uint32', {bbase if kind == 'b' else 0} + B.elem_offset // 16)"
        symbol = "store" if kind == "store" else f"load_{kind}"
        call = f"T.call_extern('void', 'tvm_gemmini_{symbol}', {args}, T.uint32({rows}), T.uint32({cols}))"
    desc = tvm.script.from_source("\n".join(prefix + body), {"T": T})
    impl = tvm.script.from_source("\n".join(prefix + [f"        T.evaluate({call})"]), {"T": T})
    tir.TensorIntrin.register(name, desc, impl)
    return name


def tensorize_gemmini_matmul(semantic_mod):
    """Substitute actual TensorIntrins, then add synchronous ABI admission.

    This accepts the tiled semantic module, not arbitrary already-fused TIR.
    Keep semantic IR available for independent execution/analysis before calling
    this function. LowerMatchBuffer must precede CompactBufferAllocation: row
    addresses refer to physical 16x16 slots, even for a smaller tail region.
    """
    func = semantic_mod["main"]
    m, n, k, tile_i, tile_j = (int(func.attrs[f"gemmini.{key}"]) for key in ("m", "n", "k", "tile_i", "tile_j"))
    _check_shape(m, n, k, tile_i, tile_j)
    blocks = []
    tir.stmt_functor.post_order_visit(func.body, lambda node: blocks.append(node.name_hint) if isinstance(node, tir.Block) and node.name_hint.startswith("gemmini_") else None)
    if not blocks:
        raise ValueError("No Gemmini semantic blocks to tensorize")
    # The pinned TVM's optional cached region-cover verifier disagrees with
    # recomputation for sibling partial update blocks after substitution.
    # Keep sref-tree verification; Tensorize's structural matching is unchanged.
    schedule = tir.Schedule(semantic_mod, debug_mask=1)
    for block in blocks:
        _, kind, rows, cols, depth, _ = block.split("_")
        intrinsic = _intrinsic(kind, int(rows), int(cols), int(depth), _SP_ROWS - tile_j * _DIM)
        schedule.tensorize(schedule.get_block(block), intrinsic)
    # Resolve offsets while physical tile strides are intact, before TVM's
    # default compaction. Subsequent lowering removes unused staging buffers.
    resolved = tir.transform.LowerMatchBuffer()(schedule.mod)
    device = resolved["main"]
    a, b, c = [device.buffer_map[param] for param in device.params]
    dims = [tir.IntImm("int64", value) for value in (m, n, k, k, n, n)]
    status = tir.Var("gemmini_status", "int32")
    admission = tir.call_extern("int32", "tvm_gemmini_validate_matmul_i8_i32", a.data, b.data, c.data, *dims)
    error = tir.SeqStmt([tir.Evaluate(tir.call_extern("void", "TVMAPISetLastError", tir.StringImm("Gemmini scheduled matmul rejected its buffers"))), tir.Evaluate(tir.tvm_throw_last_error())])
    execute = tir.SeqStmt([tir.Evaluate(tir.call_extern("void", "tvm_gemmini_begin", *dims[3:])), device.body, tir.Evaluate(tir.call_extern("void", "tvm_gemmini_end"))])
    device = device.with_body(tir.LetStmt(status, admission, tir.IfThenElse(status != 0, error, execute)))
    mt, nt, kt = ((value + 15) // 16 for value in (m, n, k))
    metadata = {"m": m, "n": n, "k": k, "tile_i": tile_i, "tile_j": tile_j, "spad_rows": 16 * (tile_i + tile_j), "acc_rows": 16 * tile_i * tile_j,
                "load_a": mt * ((n + tile_j * 16 - 1) // (tile_j * 16)) * kt,
                "load_b": nt * ((m + tile_i * 16 - 1) // (tile_i * 16)) * kt, "compute": mt * nt * kt, "store": mt * nt,
                "pe_output_bits": 20, "max_chunk_k": 16, "instruction_policy": "primitive_no_fsm"}
    if metadata["spad_rows"] > _SP_ROWS or metadata["acc_rows"] > _ACC_ROWS:
        raise ValueError("Gemmini schedule exceeds physical row capacity")
    return GemminiSchedule(semantic_mod, tvm.IRModule({"main": device}), schedule.trace,
                           ("semantic Simplify", "Schedule.tensorize", "LowerMatchBuffer", "ABI admission/begin/end", "default TVM lowering"), metadata)


def make_gemmini_matmul(m, n, k, tile_i=1, tile_j=1):
    """Convenience composition; semantic construction and substitution are separate."""
    return tensorize_gemmini_matmul(make_semantic_matmul(m, n, k, tile_i, tile_j))
