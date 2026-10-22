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
"""Opt-in ordered arithmetic for versioned Merlin ONNX extension domains.

These contracts reproduce selected source arithmetic. They are not standard ONNX
operators, generic Gemmini optimizations, or automatic model-name rewrites.
Imported lazily by _get_convert_map after OnnxOpConverter has been defined.
"""
import math

import tvm
from tvm import relax, tir

from .onnx_frontend import OnnxOpConverter, constant_numpy


def _fixed_tree_input(value):
    """Admit bounded static FP32 operands for the versioned ordered CPU contract."""
    info = getattr(value, "struct_info", None)
    if not isinstance(info, relax.TensorStructInfo) or info.dtype != "float32" or not isinstance(info.shape, relax.ShapeExpr):
        raise ValueError("Fixed-tree operators require static FP32 tensors")
    shape = tuple(int(dim) for dim in info.shape.values if isinstance(dim, tir.IntImm))
    if len(shape) != len(info.shape.values) or not shape or any(dim <= 0 for dim in shape):
        raise ValueError("Fixed-tree operators require positive static dimensions")
    k = shape[-1]
    if k > 4096 or k & (k - 1) or math.prod(shape) > ((1 << 63) - 1) // 4:
        raise ValueError("Fixed-tree operators require power-of-two K <= 4096 and bounded FP32 storage")
    return shape


def _fixed_tree_call(bb, inputs, dot):
    """Separate FP32 products and adjacent-pair additions; never reassociate a reduction."""
    from tvm.script import tir as T  # pylint: disable=import-outside-toplevel

    shapes = [_fixed_tree_input(value) for value in inputs]
    k = shapes[0][-1]
    typed_shape = lambda shape: "(" + ", ".join(f"T.int64({dim})" for dim in shape) + ",)"
    if dot:
        if any(len(shape) != 4 for shape in shapes) or shapes[0][:2] != shapes[1][:2] or k != shapes[1][-1]:
            raise ValueError("FixedTreeDot requires [B,H,M,K] and [B,H,N,K] with matching B/H/K")
        batch, heads, rows, _ = shapes[0]
        columns = shapes[1][2]
        output_shape = (batch, heads, rows, columns)
        if math.prod(output_shape) > ((1 << 63) - 1) // 4:
            raise ValueError("FixedTreeDot output storage exceeds the FP32 byte span")
        lines = ["@T.prim_func", f"def fixed_tree_dot(A: T.Buffer({typed_shape(shapes[0])}, 'float32'), B: T.Buffer({typed_shape(shapes[1])}, 'float32'), C: T.Buffer({typed_shape(output_shape)}, 'float32'), scratch: T.Buffer((T.int64({k}),), 'float32')):"]
        lines += ["    T.func_attr({'tir.noalias': True, 'op_pattern': 8})", f"    for b, h, m, n in T.grid(T.int64({batch}), T.int64({heads}), T.int64({rows}), T.int64({columns})):", f"        for i in range(T.int64({k})):", "            scratch[i] = A[b, h, m, i] * B[b, h, n, i]"]
        final = "C[b, h, m, n] = scratch[0]"
        arguments = inputs
        name = "fixed_tree_dot_fp32_v1"
    else:
        rows = math.prod(shapes[0][:-1])
        output_shape = shapes[0][:-1] + (1,)
        lines = ["@T.prim_func", f"def fixed_tree_mean_squares(A: T.Buffer({typed_shape((rows, k))}, 'float32'), C: T.Buffer({typed_shape((rows, 1))}, 'float32'), scratch: T.Buffer((T.int64({k}),), 'float32')):"]
        lines += ["    T.func_attr({'tir.noalias': True, 'op_pattern': 8})", f"    for row in range(T.int64({rows})):", f"        for i in range(T.int64({k})):", "            scratch[i] = A[row, i] * A[row, i]"]
        final = f"C[row, 0] = scratch[0] * T.float32({1.0 / k!r})"
        arguments = [bb.normalize(relax.op.reshape(inputs[0], (rows, k)))]
        name = "fixed_tree_mean_squares_fp32_v1"
    width = k
    while width > 1:
        width //= 2
        lines += [f"        for i in range(T.int64({width})):", "            scratch[i] = scratch[2 * i] + scratch[2 * i + 1]"]
    lines.append("        " + final)
    # Preserve the root block expected by Relax's TIR analyses. Scratch is an
    # explicit private output so static planning accounts for it without a heap.
    lines = lines[:3] + ["    with T.block('root'):"] + ["    " + line for line in lines[3:]]
    prim_func = tvm.script.from_source("\n".join(lines), extra_vars={"T": T}).without_attr("global_symbol")
    global_var = bb.add_func(prim_func, name)
    result_shape = output_shape if dot else (rows, 1)
    outputs = [relax.TensorStructInfo(result_shape, "float32"), relax.TensorStructInfo((k,), "float32")]
    result = bb.normalize(relax.TupleGetItem(bb.emit(relax.call_tir(global_var, arguments, outputs)), 0))
    return result if dot else relax.op.reshape(bb.normalize(result), output_shape)


class FixedTreeDot(OnnxOpConverter):
    """merlin.fixed_order v1: separately rounded FP32 products and balanced adjacent sums."""

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        if len(inputs) != 2 or any(value is None for value in inputs):
            raise ValueError("FixedTreeDot requires two tensor operands")
        return _fixed_tree_call(bb, inputs, dot=True)


class FixedTreeMeanSquares(OnnxOpConverter):
    """merlin.fixed_order v1: ordered FP32 square sum times the exact power-of-two reciprocal."""

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        if len(inputs) != 1 or inputs[0] is None:
            raise ValueError("FixedTreeMeanSquares requires one tensor operand")
        return _fixed_tree_call(bb, inputs, dot=False)


def _source_silu_call(bb, value):
    """Bound PyTorch AVX512 FP32 vector SiLU; LLVM CPU lowering only.

    SLEEF 3.8 xexpf, Naoki Shibata and contributors 2010-2024:
    https://github.com/shibatch/sleef/blob/3.8/src/libm/sleefsimdsp.c#L1856
    Eight explicit fused FP32 rounds; every other FP32 stage rounds separately.
    Native scalar std::exp tails are outside this paired32 vector contract.

    Boost Software License - Version 1.0 - August 17th, 2003

    Permission is hereby granted, free of charge, to any person or organization
    obtaining a copy of the software and accompanying documentation covered by
    this license (the "Software") to use, reproduce, display, distribute,
    execute, and transmit the Software, and to prepare derivative works of the
    Software, and to permit third-parties to whom the Software is furnished to
    do so, all subject to the following:

    The copyright notices in the Software and this entire statement, including
    the above license grant, this restriction and the following disclaimer,
    must be included in all copies of the Software, in whole or in part, and
    all derivative works of the Software, unless such copies or derivative
    works are solely in the form of machine-executable object code generated by
    a source language processor.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
    SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
    FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
    ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
    """
    from tvm.script import tir as T  # pylint: disable=import-outside-toplevel

    info = getattr(value, "struct_info", None)
    if not isinstance(info, relax.TensorStructInfo) or info.dtype != "float32" or not isinstance(info.shape, relax.ShapeExpr):
        raise ValueError("Source SiLU requires static FP32 tensors")
    shape = tuple(int(dim) for dim in info.shape.values if isinstance(dim, tir.IntImm))
    if len(shape) != len(info.shape.values) or not shape or any(dim <= 0 for dim in shape) or shape[-1] % 32:
        raise ValueError("Source SiLU requires positive static shapes with last dimension divisible by32")
    count = math.prod(shape)
    if count > ((1 << 63) - 1) // 4:
        raise ValueError("Source SiLU FP32 storage exceeds the byte span")
    # Clamp/sanitize before FP32-to-int conversion, including inadmissible NaNs.
    # llvm.fma guarantees fused rounding; tir.fma lowers to optional fmuladd.
    source = f"""@T.prim_func
def source_silu(A:T.Buffer((T.int64({count}),), 'float32'), Y:T.Buffer((T.int64({count}),), 'float32')):
 T.func_attr({{'tir.noalias':True,'op_pattern':8}})
 with T.block('root'):
  for i in range(T.int64({count})):
   d=-A[i]
   core=T.Select(d!=d,T.float32(0),T.min(T.max(d,T.float32(-104)),T.float32(100)))
   multiplied=core*T.reinterpret('float32',T.uint32(0x3fb8aa3b))
   lower=T.floor(multiplied)
   fraction=multiplied-lower
   base=T.Cast('int32',lower)
   q=base+T.Select((fraction>T.float32(.5))|((fraction==T.float32(.5))&(T.bitwise_and(base,1)!=0)),1,0)
   qf=T.Cast('float32',q)
   s0=T.call_llvm_pure_intrin('float32','llvm.fma',T.uint32(3),qf,T.reinterpret('float32',T.uint32(0xbf317200)),core)
   s=T.call_llvm_pure_intrin('float32','llvm.fma',T.uint32(3),qf,T.reinterpret('float32',T.uint32(0xb5bfbe8e)),s0)
   u0=T.reinterpret('float32',T.uint32(0x39502bda))
   u1=T.call_llvm_pure_intrin('float32','llvm.fma',T.uint32(3),u0,s,T.reinterpret('float32',T.uint32(0x3ab696c9)))
   u2=T.call_llvm_pure_intrin('float32','llvm.fma',T.uint32(3),u1,s,T.reinterpret('float32',T.uint32(0x3c0888a6)))
   u3=T.call_llvm_pure_intrin('float32','llvm.fma',T.uint32(3),u2,s,T.reinterpret('float32',T.uint32(0x3d2aaa7a)))
   u4=T.call_llvm_pure_intrin('float32','llvm.fma',T.uint32(3),u3,s,T.reinterpret('float32',T.uint32(0x3e2aaaab)))
   u5=T.call_llvm_pure_intrin('float32','llvm.fma',T.uint32(3),u4,s,T.float32(.5))
   square=s*s
   polynomial=T.call_llvm_pure_intrin('float32','llvm.fma',T.uint32(3),square,u5,s)
   unit=T.float32(1)+polynomial
   half=T.shift_right(q,1)
   first=unit*T.reinterpret('float32',T.shift_left(half+127,23))
   scaled=first*T.reinterpret('float32',T.shift_left(q-half+127,23))
   exponent=T.Select(d!=d,d,T.Select(d<T.float32(-104),T.float32(0),T.Select(d>T.float32(100),T.float32('inf'),scaled)))
   denominator=T.float32(1)+exponent
   Y[i]=A[i]/denominator
"""
    prim = tvm.script.from_source(source, extra_vars={"T": T}).without_attr("global_symbol")
    global_var = bb.add_func(prim, "torch_silu_avx512_fp32_v1")
    flat = bb.normalize(relax.op.reshape(value, (count,)))
    result = bb.normalize(bb.emit(relax.call_tir(global_var, [flat], relax.TensorStructInfo((count,), "float32"))))
    return relax.op.reshape(result, shape)


def _source_mean_call(bb, value, width):
    """Pinned PyTorch FP32 AVX2 cascade sum followed by direct division by 720 or 960.

    Independently transcribed from PyTorch revision 449b1768410104d3ed79d3bcfe4ba1d65c7f22c0,
    aten/src/ATen/native/cpu/SumKernel.cpp row_sum/vectorized_inner_sum and
    aten/src/ATen/native/ReduceOps.cpp mean_out CPU sum_out/div_ path.
    Source: https://github.com/pytorch/pytorch/tree/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0
    PyTorch contributors, BSD-3-Clause: https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/LICENSE
    Already-squared inputs remain external. Native-source identity and geometry
    admission belong to the exporter; this operator specifies the eight-lane
    arithmetic for positive static shapes ending in the declared width. Native equivalence on
    arbitrary runtime values or dispatches is outside the scoped workload proof.
    The private 65-word output makes 260 scratch bytes visible to static planning.
    """
    from tvm.script import tir as T  # pylint: disable=import-outside-toplevel

    tail_groups, remainder_vectors, remainder_start = {720: (6, 2, 88), 960: (14, 0, 120)}[width]
    info = getattr(value, "struct_info", None)
    if not isinstance(info, relax.TensorStructInfo) or info.dtype != "float32" or not isinstance(info.shape, relax.ShapeExpr):
        raise ValueError(f"Source mean{width} requires static FP32 tensors")
    shape = tuple(int(dim) for dim in info.shape.values if isinstance(dim, tir.IntImm))
    if len(shape) != len(info.shape.values) or not shape or any(dim <= 0 for dim in shape) or shape[-1] != width:
        raise ValueError(f"Source mean{width} requires positive static shapes with last dimension {width}")
    rows = math.prod(shape[:-1])
    if rows * width > ((1 << 63) - 1) // 4:
        raise ValueError(f"Source mean{width} FP32 storage exceeds the byte span")
    remainder = f"""            for v in range(T.int64({remainder_vectors})):
                for l in range(T.int64(8)):
                    S[l]=S[l]+A[r,(v+{remainder_start})*8+l]
""" if remainder_vectors else ""
    source = f"""@T.prim_func
def source_mean{width}(A:T.Buffer((T.int64({rows}),T.int64({width})), 'float32'), Y:T.Buffer((T.int64({rows}),T.int64(1)), 'float32'), S:T.Buffer((T.int64(65),), 'float32')):
    T.func_attr({{'tir.noalias':True,'op_pattern':8}})
    with T.block('root'):
        for r in range(T.int64({rows})):
            for j in range(T.int64(65)):
                S[j]=T.float32(0)
            for g in range(T.int64(16)):
                for k,l in T.grid(T.int64(4),T.int64(8)):
                    S[k*8+l]=S[k*8+l]+A[r,(g*4+k)*8+l]
            for j in range(T.int64(32)):
                S[32+j]=S[32+j]+S[j]
                S[j]=T.float32(0)
            for g in range(T.int64({tail_groups})):
                for k,l in T.grid(T.int64(4),T.int64(8)):
                    S[k*8+l]=S[k*8+l]+A[r,((g+16)*4+k)*8+l]
            for j in range(T.int64(32)):
                S[j]=S[j]+S[32+j]
                S[j]=S[j]+T.float32(0)
                S[j]=S[j]+T.float32(0)
{remainder}            for k in range(T.int64(3)):
                for l in range(T.int64(8)):
                    S[l]=S[l]+S[(k+1)*8+l]
            for l in range(T.int64(8)):
                S[64]=S[64]+S[l]
            Y[r,0]=(T.float32(0)+S[64])/T.float32({width})
"""
    prim = tvm.script.from_source(source, extra_vars={"T": T}).without_attr("global_symbol")
    global_var = bb.add_func(prim, f"torch_mean{width}_fp32_v1")
    flat = bb.normalize(relax.op.reshape(value, (rows, width)))
    outputs = [relax.TensorStructInfo((rows, 1), "float32"), relax.TensorStructInfo((65,), "float32")]
    result = bb.normalize(relax.TupleGetItem(bb.emit(relax.call_tir(global_var, [flat], outputs)), 0))
    return relax.op.reshape(result, shape[:-1] + (1,))


def _source_mean720_call(bb, value):
    return _source_mean_call(bb, value, 720)


def _source_mean960_call(bb, value):
    return _source_mean_call(bb, value, 960)


def _source_layernorm768_call(bb, value, gamma, beta, epsilon):
    """Pinned Torch AVX2 BF16 width768, source FMA/cascade order, CPU LLVM only.

    PyTorch revision449b1768410104d3ed79d3bcfe4ba1d65c7f22c0,
    moments_utils.h and layer_norm_kernel.cpp, BSD-3-Clause contributors.
    Scalar source contraction and affine FMA sites are independently bound to
    the installed native binary. Global CPU capability AVX512 does not identify
    this kernel: its AVX512 registration is null and selected dispatch is AVX2.
    Workload proof admits finite values and positive finite variance+epsilon.
    Native-source/dispatch identity belongs to the model-aware exporter.
    The private80-word output exposes320 scratch bytes to static planning.
    """
    from tvm.script import tir as T

    for operand, wanted_shape in ((value, (1, 1024, 768)), (gamma, (768,)), (beta, (768,))):
        info = getattr(operand, "struct_info", None)
        if not isinstance(info, relax.TensorStructInfo) or info.dtype != "bfloat16" or not isinstance(info.shape, relax.ShapeExpr):
            raise ValueError("Source LayerNorm768 requires static BF16 tensors")
        shape = tuple(int(dim) for dim in info.shape.values if isinstance(dim, tir.IntImm))
        if len(shape) != len(info.shape.values) or shape != wanted_shape:
            raise ValueError("Source LayerNorm768 requires x[1,1024,768] and affine[768]")
        if math.prod(shape) > ((1 << 63) - 1) // 2:
            raise ValueError("Source LayerNorm768 BF16 storage exceeds byte span")
    if not isinstance(epsilon, relax.Constant) or epsilon.struct_info.dtype != "float32" or epsilon.struct_info.ndim != 0:
        raise ValueError("Source LayerNorm768 requires original constant FP32 scalar epsilon")
    if int(constant_numpy(epsilon).view("uint32").item()) != 0x358637bd:
        raise ValueError("Source LayerNorm768 requires original epsilon word0x358637bd")
    source = """@T.prim_func
 def source_layernorm768(X:T.Buffer((T.int64(1024),T.int64(768)), 'bfloat16'), G:T.Buffer((T.int64(768),), 'bfloat16'), B:T.Buffer((T.int64(768),), 'bfloat16'), E:T.Buffer((), 'float32'), Y:T.Buffer((T.int64(1024),T.int64(768)), 'bfloat16'), S:T.Buffer((T.int64(80),), 'float32')):
  T.func_attr({'tir.noalias':True,'op_pattern':8})
  with T.block('root'):
   for row in range(T.int64(1024)):
    for slot in range(T.int64(80)):
     S[slot]=T.float32(0)
    for chunk in range(T.int64(3)):
     for slot in range(T.int64(32)):
      S[T.int64(32)+slot]=T.float32(0)
     for j in range(T.int64(16)):
      S[71]=T.float32(1)/T.Cast('float32',j+T.int64(1))
      for half,lane in T.grid(T.int64(2),T.int64(8)):
       xv=T.reinterpret('float32',T.shift_left(T.Cast('uint32',T.reinterpret('uint16',X[row,chunk*T.int64(256)+j*T.int64(16)+half*T.int64(8)+lane])),T.uint32(16)))
       S[72]=xv-S[T.int64(32)+half*T.int64(16)+lane]
       S[T.int64(32)+half*T.int64(16)+lane]=T.call_llvm_pure_intrin('float32','llvm.fma',T.uint32(3),S[72],S[71],S[T.int64(32)+half*T.int64(16)+lane])
       S[73]=xv-S[T.int64(32)+half*T.int64(16)+lane]
       S[T.int64(40)+half*T.int64(16)+lane]=T.call_llvm_pure_intrin('float32','llvm.fma',T.uint32(3),S[72],S[73],S[T.int64(40)+half*T.int64(16)+lane])
     for half in range(T.int64(2)):
      oldcount=T.Select(chunk==T.int64(1),T.int64(32)+half*T.int64(16),half*T.int64(16))
      S[71]=T.float32(16)/T.Cast('float32',oldcount+T.int64(16))
      for lane in range(T.int64(8)):
       S[72]=S[T.int64(32)+half*T.int64(16)+lane]-S[lane]
       S[76]=S[T.int64(8)+lane]+S[T.int64(40)+half*T.int64(16)+lane]
       S[74]=S[71]*S[72]
       S[75]=S[72]*T.Cast('float32',oldcount)
       S[lane]=S[lane]+S[74]
       S[T.int64(8)+lane]=T.call_llvm_pure_intrin('float32','llvm.fma',T.uint32(3),S[75],S[74],S[76])
     if chunk==T.int64(1):
      S[71]=T.float32(64)/T.float32(64)
      for lane in range(T.int64(8)):
       S[72]=S[lane]-S[T.int64(16)+lane]
       S[76]=S[T.int64(24)+lane]+S[T.int64(8)+lane]
       S[74]=S[71]*S[72]
       S[75]=S[72]*T.float32(0)
       S[T.int64(16)+lane]=S[T.int64(16)+lane]+S[74]
       S[T.int64(24)+lane]=T.call_llvm_pure_intrin('float32','llvm.fma',T.uint32(3),S[75],S[74],S[76])
      for slot in range(T.int64(16)):
       S[slot]=T.float32(0)
    S[71]=T.float32(64)/T.float32(96)
    for lane in range(T.int64(8)):
     S[72]=S[T.int64(16)+lane]-S[lane]
     S[76]=S[T.int64(8)+lane]+S[T.int64(24)+lane]
     S[74]=S[71]*S[72]
     S[75]=S[72]*T.float32(32)
     S[lane]=S[lane]+S[74]
     S[T.int64(8)+lane]=T.call_llvm_pure_intrin('float32','llvm.fma',T.uint32(3),S[75],S[74],S[76])
    for lane in range(T.int64(8)):
     S[71]=T.float32(96)/T.Cast('float32',(lane+T.int64(1))*T.int64(96))
     S[72]=S[lane]-S[64]
     S[64]=T.call_llvm_pure_intrin('float32','llvm.fma',T.uint32(3),S[71],S[72],S[64])
     S[77]=S[72]*S[72]
     S[78]=S[77]*S[71]
     S[79]=T.call_llvm_pure_intrin('float32','llvm.fma',T.uint32(3),S[78],T.Cast('float32',lane*T.int64(96)),S[T.int64(8)+lane])
     S[65]=S[65]+S[79]
    S[66]=S[65]/T.float32(768)
    S[67]=S[66]+E[()]
    S[68]=T.call_llvm_pure_intrin('float32','llvm.sqrt',T.uint32(1),S[67])
    S[69]=T.float32(1)/S[68]
    S[70]=(-S[69])*S[64]
    for col in range(T.int64(768)):
     xv=T.reinterpret('float32',T.shift_left(T.Cast('uint32',T.reinterpret('uint16',X[row,col])),T.uint32(16)))
     gv=T.reinterpret('float32',T.shift_left(T.Cast('uint32',T.reinterpret('uint16',G[col])),T.uint32(16)))
     bv=T.reinterpret('float32',T.shift_left(T.Cast('uint32',T.reinterpret('uint16',B[col])),T.uint32(16)))
     normalized=T.call_llvm_pure_intrin('float32','llvm.fma',T.uint32(3),xv,S[69],S[70])
     affine=T.call_llvm_pure_intrin('float32','llvm.fma',T.uint32(3),normalized,gv,bv)
     word=T.reinterpret('uint32',affine)
     narrowed=T.Cast('uint16',T.shift_right(word+T.uint32(0x7fff)+T.bitwise_and(T.shift_right(word,T.uint32(16)),T.uint32(1)),T.uint32(16)))
     Y[row,col]=T.reinterpret('bfloat16',narrowed)
"""
    # Remove one artificial indentation level used by this generated snippet.
    source = "\n".join(line[1:] if line.startswith(" ") else line for line in source.splitlines())
    prim = tvm.script.from_source(source, extra_vars={"T": T}).without_attr("global_symbol")
    global_var = bb.add_func(prim, "torch_layernorm_avx2_bf16_width768_v1")
    flat = bb.normalize(relax.op.reshape(value, (1024, 768)))
    outputs = [relax.TensorStructInfo((1024, 768), "bfloat16"), relax.TensorStructInfo((80,), "float32")]
    result = bb.normalize(relax.TupleGetItem(bb.emit(relax.call_tir(global_var, [flat, gamma, beta, epsilon], outputs)), 0))
    return relax.op.reshape(result, (1, 1024, 768))


def _source_conv3x16_call(bb, value, weight, bias):
    """Pinned oneDNN software-BF16 AVX512 convolution for the Smol patch cut.

    oneDNN revision 8d263e693366ef8db40acc569cc7d8edf644556d,
    jit_avx512_core_bf16_conv_kernel.cpp and jit_avx512_core_bf16cvt.hpp,
    Intel contributors, Apache-2.0. Emulated vdpbf16ps fixes the per-patch
    channel order1,0,zero-partner,2. Bias adds after all FP32 fused rounds.
    Native-source identity and finite normal-product admission belong to the
    exporter and scoped workload proof; arbitrary dispatches are unqualified.
    The private32-word output exposes128 scratch bytes to static planning.
    """
    from tvm.script import tir as T  # pylint: disable=import-outside-toplevel

    for operand, expected in ((value, (1, 3, 512, 512)), (weight, (768, 3, 16, 16)), (bias, (768,))):
        info = getattr(operand, "struct_info", None)
        if not isinstance(info, relax.TensorStructInfo) or info.dtype != "bfloat16" or not isinstance(info.shape, relax.ShapeExpr):
            raise ValueError("Source Conv3x16 requires static original BF16 operands")
        shape = tuple(int(dim) for dim in info.shape.values if isinstance(dim, tir.IntImm))
        if len(shape) != len(info.shape.values) or shape != expected:
            raise ValueError("Source Conv3x16 requires x[1,3,512,512], weight[768,3,16,16], bias[768]")
        if math.prod(shape) > ((1 << 63) - 1) // 2:
            raise ValueError("Source Conv3x16 BF16 storage exceeds the byte span")
    if 768 * 32 * 32 > ((1 << 63) - 1) // 2 or 32 > ((1 << 63) - 1) // 4:
        raise ValueError("Source Conv3x16 output or scratch storage exceeds the byte span")
    source = """@T.prim_func
def source_conv3x16(X:T.Buffer((T.int64(1),T.int64(3),T.int64(512),T.int64(512)), 'bfloat16'), W:T.Buffer((T.int64(768),T.int64(3),T.int64(16),T.int64(16)), 'bfloat16'), B:T.Buffer((T.int64(768),), 'bfloat16'), Y:T.Buffer((T.int64(1),T.int64(768),T.int64(32),T.int64(32)), 'bfloat16'), S:T.Buffer((T.int64(32),), 'float32')):
 T.func_attr({'tir.noalias':True,'op_pattern':8})
 with T.block('root'):
  for oc,oh in T.grid(T.int64(768),T.int64(32)):
   for ow in range(T.int64(32)):
    S[ow]=T.float32(0)
   for kh,kw in T.grid(T.int64(16),T.int64(16)):
    for ow in range(T.int64(32)):
     x1=T.reinterpret('float32',T.shift_left(T.Cast('uint32',T.reinterpret('uint16',X[0,1,oh*T.int64(16)+kh,ow*T.int64(16)+kw])),T.uint32(16)))
     w1=T.reinterpret('float32',T.shift_left(T.Cast('uint32',T.reinterpret('uint16',W[oc,1,kh,kw])),T.uint32(16)))
     S[ow]=T.call_llvm_pure_intrin('float32','llvm.fma',T.uint32(3),x1,w1,S[ow])
     x0=T.reinterpret('float32',T.shift_left(T.Cast('uint32',T.reinterpret('uint16',X[0,0,oh*T.int64(16)+kh,ow*T.int64(16)+kw])),T.uint32(16)))
     w0=T.reinterpret('float32',T.shift_left(T.Cast('uint32',T.reinterpret('uint16',W[oc,0,kh,kw])),T.uint32(16)))
     S[ow]=T.call_llvm_pure_intrin('float32','llvm.fma',T.uint32(3),x0,w0,S[ow])
     S[ow]=T.call_llvm_pure_intrin('float32','llvm.fma',T.uint32(3),T.float32(0),T.float32(0),S[ow])
     x2=T.reinterpret('float32',T.shift_left(T.Cast('uint32',T.reinterpret('uint16',X[0,2,oh*T.int64(16)+kh,ow*T.int64(16)+kw])),T.uint32(16)))
     w2=T.reinterpret('float32',T.shift_left(T.Cast('uint32',T.reinterpret('uint16',W[oc,2,kh,kw])),T.uint32(16)))
     S[ow]=T.call_llvm_pure_intrin('float32','llvm.fma',T.uint32(3),x2,w2,S[ow])
   for ow in range(T.int64(32)):
    bv=T.reinterpret('float32',T.shift_left(T.Cast('uint32',T.reinterpret('uint16',B[oc])),T.uint32(16)))
    affine=S[ow]+bv
    word=T.reinterpret('uint32',affine)
    narrowed=T.Cast('uint16',T.shift_right(word+T.uint32(0x7fff)+T.bitwise_and(T.shift_right(word,T.uint32(16)),T.uint32(1)),T.uint32(16)))
    Y[0,oc,oh,ow]=T.reinterpret('bfloat16',narrowed)
"""
    prim = tvm.script.from_source(source, extra_vars={"T": T}).without_attr("global_symbol")
    global_var = bb.add_func(prim, "torch_conv2d_avx512_bf16_ic3_kernel16_stride16_v1")
    outputs = [relax.TensorStructInfo((1, 768, 32, 32), "bfloat16"), relax.TensorStructInfo((32,), "float32")]
    return bb.normalize(relax.TupleGetItem(bb.emit(relax.call_tir(global_var, [value, weight, bias], outputs)), 0))


class TorchConv2dAvx512Bf16IC3Kernel16Stride16(OnnxOpConverter):
    """merlin.source_preserving v1: software-BF16 patch convolution source order."""

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        if len(inputs) != 3 or any(value is None for value in inputs):
            raise ValueError("Source Conv3x16 requires three original BF16 operands")
        return _source_conv3x16_call(bb, *inputs)


class TorchLayerNormAvx2Bf16Width768(OnnxOpConverter):
    """merlin.source_preserving v1: pinned BF16 LayerNorm768 source order."""

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        if len(inputs) != 4 or any(value is None for value in inputs):
            raise ValueError("Source LayerNorm768 requires four operands")
        return _source_layernorm768_call(bb, *inputs)


def _source_softmax_call(bb, value, width):
    """Pinned CPU FP32 last-axis softmax, with the valid vector tail.

    PyTorch 449b1768410104d3ed79d3bcfe4ba1d65c7f22c0 SoftMaxKernel.cpp
    _vec_softmax_lastdim<float> and functional_base.h AVX512 reduce_all:
    https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/native/cpu/SoftMaxKernel.cpp
    PyTorch contributors, BSD-3-Clause (the license linked in _source_mean720_call).
    SLEEF 3.8 xexpf uses the eight fused rounds and constants transcribed below;
    the complete Boost license notice in _source_silu_call also applies here.
    Source maximum, stored FP32 exponentials, sixteen-lane sums, reciprocal and
    multiplication retain separate rounds. Only exp uses explicit llvm.fma.

    Exporter/probe admission binds native dispatch/library/headers and finite
    logits *and* finite shifted subtractions, with positive finite sums. These
    are proof conditions, not exported runtime guards. This approximation is
    source arithmetic, not a universally correctly rounded mathematical exp.
    The private width+32-word output exposes its scratch bytes to planning.
    """
    from tvm.script import tir as T  # pylint: disable=import-outside-toplevel

    if width not in (177, 227):
        raise ValueError("Source softmax requires an admitted width")
    scratch, work = width + 32, width + 16
    groups, tail_start, tail = width // 16 - 1, width // 16 * 16, width % 16
    info = getattr(value, "struct_info", None)
    if not isinstance(info, relax.TensorStructInfo) or info.dtype != "float32" or not isinstance(info.shape, relax.ShapeExpr):
        raise ValueError("Source softmax requires a static FP32 tensor")
    shape = tuple(int(dim) for dim in info.shape.values if isinstance(dim, tir.IntImm))
    if len(shape) != len(info.shape.values) or not shape or any(dim <= 0 for dim in shape) or shape[-1] != width:
        raise ValueError("Source softmax requires positive static dimensions ending in its admitted width")
    rows = math.prod(shape[:-1])
    if math.prod(shape) > ((1 << 63) - 1) // 4:
        raise ValueError("Source softmax FP32 storage exceeds the byte span")
    source = f"""@T.prim_func
def source_softmax{width}(A:T.Buffer((T.int64({rows}),T.int64({width})), 'float32'), Y:T.Buffer((T.int64({rows}),T.int64({width})), 'float32'), S:T.Buffer((T.int64({scratch}),), 'float32')):
    T.func_attr({{'tir.noalias':True,'op_pattern':8}})
    with T.block('root'):
        for r in range(T.int64({rows})):
            for l in range(T.int64(16)):
                S[{width}+l]=A[r,l]
            for g,l in T.grid(T.int64({groups}),T.int64(16)):
                S[{width}+l]=T.max(S[{width}+l],A[r,(g+1)*16+l])
            for t in range(T.int64({tail})):
                S[{width}+t]=T.max(S[{width}+t],A[r,{tail_start}+t])
            for l in range(T.int64(16)):
                S[{work}+l]=T.max(S[{width}+l],S[{width}+T.bitwise_xor(l,T.int64(8))])
            for l in range(T.int64(16)):
                S[{width}+l]=S[{work}+l]
            for l in range(T.int64(16)):
                S[{work}+l]=T.max(S[{width}+l],S[{width}+T.bitwise_xor(l,T.int64(4))])
            for l in range(T.int64(16)):
                S[{width}+l]=S[{work}+l]
            for l in range(T.int64(16)):
                S[{work}+l]=T.max(S[{width}+l],S[{width}+T.bitwise_xor(l,T.int64(2))])
            for l in range(T.int64(16)):
                S[{width}+l]=S[{work}+l]
            for l in range(T.int64(16)):
                S[{work}+l]=T.max(S[{width}+l],S[{width}+T.bitwise_xor(l,T.int64(1))])
            for l in range(T.int64(16)):
                S[{width}+l]=S[{work}+l]
            maximum=S[{width}]
            for i in range(T.int64({width})):
                d=A[r,i]-maximum
                core=T.max(d,T.float32(-104))
                multiplied=core*T.reinterpret('float32',T.uint32(0x3fb8aa3b))
                lower=T.floor(multiplied)
                fraction=multiplied-lower
                base=T.Cast('int32',lower)
                q=base+T.Select((fraction>T.float32(.5))|((fraction==T.float32(.5))&(T.bitwise_and(base,1)!=0)),1,0)
                qf=T.Cast('float32',q)
                s0=T.call_llvm_pure_intrin('float32','llvm.fma',T.uint32(3),qf,T.reinterpret('float32',T.uint32(0xbf317200)),core)
                s=T.call_llvm_pure_intrin('float32','llvm.fma',T.uint32(3),qf,T.reinterpret('float32',T.uint32(0xb5bfbe8e)),s0)
                u0=T.reinterpret('float32',T.uint32(0x39502bda))
                u1=T.call_llvm_pure_intrin('float32','llvm.fma',T.uint32(3),u0,s,T.reinterpret('float32',T.uint32(0x3ab696c9)))
                u2=T.call_llvm_pure_intrin('float32','llvm.fma',T.uint32(3),u1,s,T.reinterpret('float32',T.uint32(0x3c0888a6)))
                u3=T.call_llvm_pure_intrin('float32','llvm.fma',T.uint32(3),u2,s,T.reinterpret('float32',T.uint32(0x3d2aaa7a)))
                u4=T.call_llvm_pure_intrin('float32','llvm.fma',T.uint32(3),u3,s,T.reinterpret('float32',T.uint32(0x3e2aaaab)))
                u5=T.call_llvm_pure_intrin('float32','llvm.fma',T.uint32(3),u4,s,T.float32(.5))
                square=s*s
                polynomial=T.call_llvm_pure_intrin('float32','llvm.fma',T.uint32(3),square,u5,s)
                unit=T.float32(1)+polynomial
                half=T.shift_right(q,1)
                first=unit*T.reinterpret('float32',T.shift_left(half+127,23))
                scaled=first*T.reinterpret('float32',T.shift_left(q-half+127,23))
                S[i]=T.Select(d<T.float32(-104),T.float32(0),scaled)
            for l in range(T.int64(16)):
                S[{width}+l]=S[l]
            for g,l in T.grid(T.int64({groups}),T.int64(16)):
                S[{width}+l]=S[{width}+l]+S[(g+1)*16+l]
            for t in range(T.int64({tail})):
                S[{width}+t]=S[{width}+t]+S[{tail_start}+t]
            for l in range(T.int64(16)):
                S[{work}+l]=S[{width}+l]+S[{width}+T.bitwise_xor(l,T.int64(8))]
            for l in range(T.int64(16)):
                S[{width}+l]=S[{work}+l]
            for l in range(T.int64(16)):
                S[{work}+l]=S[{width}+l]+S[{width}+T.bitwise_xor(l,T.int64(4))]
            for l in range(T.int64(16)):
                S[{width}+l]=S[{work}+l]
            for l in range(T.int64(16)):
                S[{work}+l]=S[{width}+l]+S[{width}+T.bitwise_xor(l,T.int64(2))]
            for l in range(T.int64(16)):
                S[{width}+l]=S[{work}+l]
            for l in range(T.int64(16)):
                S[{work}+l]=S[{width}+l]+S[{width}+T.bitwise_xor(l,T.int64(1))]
            for l in range(T.int64(16)):
                S[{width}+l]=S[{work}+l]
            reciprocal=T.float32(1)/S[{width}]
            for i in range(T.int64({width})):
                Y[r,i]=S[i]*reciprocal
"""
    prim = tvm.script.from_source(source, extra_vars={"T": T}).without_attr("global_symbol")
    global_var = bb.add_func(prim, f"torch_softmax_avx512_fp32_lastdim{width}_v1")
    flat = bb.normalize(relax.op.reshape(value, (rows, width)))
    outputs = [relax.TensorStructInfo((rows, width), "float32"), relax.TensorStructInfo((scratch,), "float32")]
    result = bb.normalize(relax.TupleGetItem(bb.emit(relax.call_tir(global_var, [flat], outputs)), 0))
    return relax.op.reshape(result, shape)


def _source_softmax177_call(bb, value):
    return _source_softmax_call(bb, value, 177)


def _source_softmax227_call(bb, value):
    return _source_softmax_call(bb, value, 227)


def _source_mkl_sgemm177_call(bb, left, right, kind, width=177):
    """Scoped native oneMKL AVX512 QK/PV: increasing-K IEEE FP32 FMA.

    Reconstructed from the pinned libtorch_cpu.so MKL kernel entry/packing
    observations: QK uses sgemm_kernel_0_b0; PV uses sgemm_kernel_nocopy_NN_b0.
    Both keep one accumulator per output, initialized to positive zero, and
    visit logical K in order. Native alpha is one and beta is zero. Source
    identity, one-thread dispatch and original physical layouts belong to
    exporter admission. This is not a general oneMKL MatMul contract: only
    the finite, RNE/gradual-underflow manual-denoiser geometries are proved.
    Explicit llvm.fma is mandatory; tir.fma allows optional fusion.
    The four-byte accumulator is private but exposed for memory accounting.
    """
    from tvm.script import tir as T  # pylint: disable=import-outside-toplevel

    if (kind, width) not in (("qk", 177), ("pv", 177), ("qk", 227)):
        raise ValueError("Unsupported source SGEMM geometry")
    reduction, columns = (64, width) if kind == "qk" else (width, 64)
    expected = ((1, 15, 50, reduction), (1, 15, reduction, columns))
    for value, required in zip((left, right), expected):
        info = getattr(value, "struct_info", None)
        if not isinstance(info, relax.TensorStructInfo) or info.dtype != "float32" or not isinstance(info.shape, relax.ShapeExpr):
            raise ValueError("Source SGEMM requires static FP32 tensors")
        dims = info.shape.values
        if any(not isinstance(dim, tir.IntImm) for dim in dims):
            raise ValueError("Source SGEMM requires static integer dimensions")
        shape = tuple(int(dim) for dim in dims)
        if any(dim <= 0 for dim in shape) or math.prod(shape) > ((1 << 63) - 1) // 4:
            raise ValueError("Source SGEMM FP32 storage exceeds the byte span")
        if shape != required:
            raise ValueError("Source SGEMM requires the admitted manual-denoiser geometry")
    left_count, right_count = (math.prod(shape) for shape in expected)
    count = 15 * 50 * columns
    source = f"""@T.prim_func
def source_mkl_sgemm(A:T.Buffer((T.int64({left_count}),),'float32'), B:T.Buffer((T.int64({right_count}),),'float32'), Y:T.Buffer((T.int64({count}),),'float32'), S:T.Buffer((T.int64(1),),'float32')):
 T.func_attr({{'tir.noalias':True,'op_pattern':8}})
 with T.block('root'):
  for h in range(T.int64(15)):
   for m in range(T.int64(50)):
    for n in range(T.int64({columns})):
     S[0]=T.float32(0)
     for k in range(T.int64({reduction})):
      S[0]=T.call_llvm_pure_intrin('float32','llvm.fma',T.uint32(3),A[(h*T.int64(50)+m)*T.int64({reduction})+k],B[(h*T.int64({reduction})+k)*T.int64({columns})+n],S[0])
     Y[(h*T.int64(50)+m)*T.int64({columns})+n]=S[0]
"""
    prim = tvm.script.from_source(source, extra_vars={"T": T}).without_attr("global_symbol")
    global_var = bb.add_func(prim, f"torch_mkl_sgemm_fp32_{kind}{width}_v1")
    args = [bb.normalize(relax.op.reshape(value, (size,))) for value, size in zip((left, right), (left_count, right_count))]
    outputs = [relax.TensorStructInfo((count,), "float32"), relax.TensorStructInfo((1,), "float32")]
    result = bb.normalize(relax.TupleGetItem(bb.emit(relax.call_tir(global_var, args, outputs)), 0))
    return relax.op.reshape(result, (1, 15, 50, columns))


def _source_flash_vision1024_call(bb, query, key, value, mask):
    """Pinned finite vision Flash arithmetic with source-proved positive-zero masks.

    The exporter authenticates the original mask DAG, native layout and runtime.
    This is a bounded source profile, not general masked SDPA. The kernel retains
    FP32 sums before BF16 probability stores and the installed 256+256 PV order.
    Its expf dependency is host-library bound; target equivalence is unqualified.
    Both workspaces are visible outputs and reused serially across heads/tiles.
    """
    from tvm.script import tir as T  # pylint: disable=import-outside-toplevel

    expected = [(1, 12, 1024, 64)] * 3 + [(1, 1, 1024, 1024)]
    operands = [query, key, value, mask]
    for operand, required in zip(operands, expected):
        info = getattr(operand, "struct_info", None)
        if not isinstance(info, relax.TensorStructInfo) or info.dtype != "bfloat16" or not isinstance(info.shape, relax.ShapeExpr):
            raise ValueError("Source Flash requires static BF16 tensors")
        if any(not isinstance(dim, tir.IntImm) for dim in info.shape.values) or tuple(int(dim) for dim in info.shape.values) != required:
            raise ValueError("Source Flash requires the admitted vision geometry")
    def widen(expression):
        return f"T.reinterpret('float32',T.shift_left(T.Cast('uint32',T.reinterpret('uint16',{expression})),T.uint32(16)))"

    # QK0..131071; row maxima131072; sums131328; dst131584;
    # two reduction vectors147968; GEMM widening148000; PV dot180768.
    source = f"""@T.prim_func
def source_flash(Q:T.Buffer((T.int64(1),T.int64(12),T.int64(1024),T.int64(64)), 'bfloat16'), K:T.Buffer((T.int64(1),T.int64(12),T.int64(1024),T.int64(64)), 'bfloat16'), V:T.Buffer((T.int64(1),T.int64(12),T.int64(1024),T.int64(64)), 'bfloat16'), Mask:T.Buffer((T.int64(1),T.int64(1),T.int64(1024),T.int64(1024)), 'bfloat16'), Y:T.Buffer((T.int64(1),T.int64(12),T.int64(1024),T.int64(64)), 'bfloat16'), F:T.Buffer((T.int64(180832),), 'float32'), P:T.Buffer((T.int64(256),T.int64(512)), 'bfloat16')):
    T.func_attr({{'tir.noalias':True,'op_pattern':8}})
    with T.block('root'):
        for head,query_tile in T.grid(T.int64(12),T.int64(4)):
            for row in range(T.int64(256)):
                F[131072+row]=T.reinterpret('float32',T.uint32(0xff800000))
                F[131328+row]=T.float32(0)
            for key_tile in range(T.int64(2)):
                for inner,col in T.grid(T.int64(64),T.int64(512)):
                    F[148000+inner*512+col]={widen('K[0,head,key_tile*512+col,inner]')}
                for row,group in T.grid(T.int64(256),T.int64(32)):
                    for lane in T.vectorized(T.int64(16)):
                        F[row*512+group*16+lane]=T.float32(0)
                    for inner in range(T.int64(64)):
                        q={widen('Q[0,head,query_tile*256+row,inner]')}
                        for lane in T.vectorized(T.int64(16)):
                            F[row*512+group*16+lane]=T.call_llvm_pure_intrin('float32','llvm.fma',T.uint32(3),q,F[148000+inner*512+group*16+lane],F[row*512+group*16+lane])
                for row in range(T.int64(256)):
                    for col in range(T.int64(512)):
                        mask={widen('Mask[0,0,query_tile*256+row,key_tile*512+col]')}
                        F[row*512+col]=T.call_llvm_pure_intrin('float32','llvm.fma',T.uint32(3),F[row*512+col],T.float32(.125),mask)
                    for lane in range(T.int64(16)):
                        F[147968+lane]=F[row*512+lane]
                    for group,lane in T.grid(T.int64(31),T.int64(16)):
                        value=F[row*512+(group+1)*16+lane]
                        F[147968+lane]=T.if_then_else(F[147968+lane]>value,F[147968+lane],value)
                    for stage in T.unroll(T.int64(4)):
                        stride=T.shift_right(T.int64(8),stage)
                        for lane in range(T.int64(16)):
                            other=F[147968+T.bitwise_xor(lane,stride)]
                            F[147984+lane]=T.if_then_else(F[147968+lane]>other,F[147968+lane],other)
                        for lane in range(T.int64(16)):
                            F[147968+lane]=F[147984+lane]
                    maximum=T.if_then_else(F[131072+row]>F[147968],F[131072+row],F[147968])
                    for lane in range(T.int64(16)):
                        F[147968+lane]=T.float32(0)
                    for group,lane in T.grid(T.int64(32),T.int64(16)):
                        col=group*16+lane
                        delta=F[row*512+col]-maximum
                        src=delta*T.reinterpret('float32',T.uint32(0x3fb8aa3b))
                        fractional=src-T.floor(src)
                        p0=T.call_llvm_pure_intrin('float32','llvm.fma',T.uint32(3),fractional,T.reinterpret('float32',T.uint32(0xbda235d5)),T.reinterpret('float32',T.uint32(0xbe65b8f5)))
                        p1=T.call_llvm_pure_intrin('float32','llvm.fma',T.uint32(3),fractional,p0,T.reinterpret('float32',T.uint32(0x3e9b69f0)))
                        p2=T.call_llvm_pure_intrin('float32','llvm.fma',T.uint32(3),fractional,p1,T.reinterpret('float32',T.uint32(0x38e077a1)))
                        corrected=src-p2
                        affine=T.call_llvm_pure_intrin('float32','llvm.fma',T.uint32(3),T.reinterpret('float32',T.uint32(0x4b000000)),corrected,T.reinterpret('float32',T.uint32(0x4e7e0000)))
                        encoded=T.if_then_else(delta<T.reinterpret('float32',T.uint32(0xc2aeac50)),T.uint32(0),T.Cast('uint32',T.Cast('int32',affine)))
                        exponential=T.reinterpret('float32',encoded)
                        F[147968+lane]=F[147968+lane]+exponential
                        narrowed=T.Cast('uint16',T.shift_right(encoded+T.uint32(0x7fff)+T.bitwise_and(T.shift_right(encoded,T.uint32(16)),T.uint32(1)),T.uint32(16)))
                        P[row,col]=T.reinterpret('bfloat16',narrowed)
                    for stage in T.unroll(T.int64(4)):
                        stride=T.shift_right(T.int64(8),stage)
                        for lane in range(T.int64(16)):
                            F[147984+lane]=F[147968+lane]+F[147968+T.bitwise_xor(lane,stride)]
                        for lane in range(T.int64(16)):
                            F[147968+lane]=F[147984+lane]
                    old_rescale=T.call_extern('float32','expf',F[131072+row]-maximum)
                    F[131328+row]=T.call_llvm_pure_intrin('float32','llvm.fma',T.uint32(3),old_rescale,F[131328+row],F[147968])
                    F[131072+row]=maximum
                    if key_tile>0:
                        for col in range(T.int64(64)):
                            F[131584+row*64+col]=F[131584+row*64+col]*old_rescale
                for inner,col in T.grid(T.int64(512),T.int64(64)):
                    F[148000+inner*64+col]={widen('V[0,head,key_tile*512+inner,col]')}
                for row in range(T.int64(256)):
                    # Installed nocopy NN K512 driver partitions256+256.
                    # Each chunk starts+0; beta1 adds existing D only after its dot.
                    for chunk,group in T.grid(T.int64(2),T.int64(4)):
                        for lane in T.vectorized(T.int64(16)):
                            F[180768+group*16+lane]=T.float32(0)
                        for inner in range(T.int64(256)):
                            probability={widen('P[row,chunk*256+inner]')}
                            for lane in T.vectorized(T.int64(16)):
                                F[180768+group*16+lane]=T.call_llvm_pure_intrin('float32','llvm.fma',T.uint32(3),probability,F[148000+(chunk*256+inner)*64+group*16+lane],F[180768+group*16+lane])
                        for lane in T.vectorized(T.int64(16)):
                            if key_tile==0 and chunk==0:
                                F[131584+row*64+group*16+lane]=F[180768+group*16+lane]
                            else:
                                F[131584+row*64+group*16+lane]=F[180768+group*16+lane]+F[131584+row*64+group*16+lane]
            for row in range(T.int64(256)):
                denominator=T.if_then_else(F[131328+row]==T.float32(0),T.float32(1),F[131328+row])
                reciprocal=T.float32(1)/denominator
                for col in range(T.int64(64)):
                    product=F[131584+row*64+col]*reciprocal
                    word=T.reinterpret('uint32',product)
                    narrowed=T.Cast('uint16',T.shift_right(word+T.uint32(0x7fff)+T.bitwise_and(T.shift_right(word,T.uint32(16)),T.uint32(1)),T.uint32(16)))
                    Y[0,head,query_tile*256+row,col]=T.reinterpret('bfloat16',narrowed)
"""
    prim = tvm.script.from_source(source, extra_vars={'T': T}).without_attr('global_symbol')
    kernel = bb.add_func(prim, "torch_flash_cpu_bf16_vision1024_v1")
    outputs = [relax.TensorStructInfo((1, 12, 1024, 64), "bfloat16"), relax.TensorStructInfo((180832,), "float32"), relax.TensorStructInfo((256, 512), "bfloat16")]
    return bb.normalize(relax.TupleGetItem(bb.emit(relax.call_tir(kernel, operands, outputs)), 0))


class TorchFlashBf16Vision1024(OnnxOpConverter):
    """merlin.source_preserving v1: original finite, zero-mask vision Flash."""

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        if len(inputs) != 4 or any(operand is None for operand in inputs):
            raise ValueError("Source Flash requires original Q/K/V and positive-zero mask")
        return _source_flash_vision1024_call(bb, *inputs)


class TorchMklSgemmF32QK177(OnnxOpConverter):
    """merlin.source_preserving v1: bounded native FP32 QK contraction."""

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        if len(inputs) != 2 or any(value is None for value in inputs):
            raise ValueError("Source QK SGEMM requires two tensor operands")
        return _source_mkl_sgemm177_call(bb, inputs[0], inputs[1], "qk")


class TorchMklSgemmF32PV177(OnnxOpConverter):
    """merlin.source_preserving v1: bounded native FP32 PV contraction."""

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        if len(inputs) != 2 or any(value is None for value in inputs):
            raise ValueError("Source PV SGEMM requires two tensor operands")
        return _source_mkl_sgemm177_call(bb, inputs[0], inputs[1], "pv")


def _source_onednn_bf16_pv_call(bb, left, right, width):
    """Bounded oneDNN AVX512 emulated BF16 GEMM source-family recurrence.

    Installed JIT bytes and all four isolated compute variants establish the
    high-partner then low-partner FP32 FMA order, including the odd positive-zero
    partner. Actual native terminals agree; original internal packing/selected
    variant were not observed. Alpha one and beta zero follow ordinary MatMul
    semantics. Source identity, dispatch, physical layouts and finite/RNE/gradual
    context are exporter admission obligations. One terminal BF16 RNE is explicit;
    private FP32 accumulator storage is exposed for memory accounting.
    """
    from tvm.script import tir as T  # pylint: disable=import-outside-toplevel

    if width not in (177, 227):
        raise ValueError("Source BF16 PV requires an admitted width")
    queries = 177 if width == 177 else 50
    k, n, dtype, itemsize = width, 64, "bfloat16", 2
    expected = ((1, 15, queries, k), (1, 15, k, n))
    for value, required in zip((left, right), expected):
        info = getattr(value, "struct_info", None)
        if not isinstance(info, relax.TensorStructInfo) or info.dtype != dtype or not isinstance(info.shape, relax.ShapeExpr):
            raise ValueError("Source contraction requires static tensors of admitted dtype")
        if any(not isinstance(d, tir.IntImm) for d in info.shape.values):
            raise ValueError("Source contraction requires static integer dimensions")
        shape = tuple(int(d) for d in info.shape.values)
        if any(d <= 0 for d in shape) or math.prod(shape) > ((1 << 63) - 1) // itemsize:
            raise ValueError("Source contraction storage exceeds int64 byte span")
        if shape != required:
            raise ValueError("Source contraction requires admitted attention geometry")
    left_count, right_count = (math.prod(s) for s in expected)
    count = 15 * queries * n
    arithmetic = f"""     for z in range(T.int64({k // 2})):
      S[0]=T.call_llvm_pure_intrin('float32','llvm.fma',T.uint32(3),T.Cast('float32',A[(h*T.int64({queries})+m)*T.int64({k})+z*T.int64(2)+T.int64(1)]),T.Cast('float32',B[(h*T.int64({k})+z*T.int64(2)+T.int64(1))*T.int64(64)+n]),S[0])
      S[0]=T.call_llvm_pure_intrin('float32','llvm.fma',T.uint32(3),T.Cast('float32',A[(h*T.int64({queries})+m)*T.int64({k})+z*T.int64(2)]),T.Cast('float32',B[(h*T.int64({k})+z*T.int64(2))*T.int64(64)+n]),S[0])
     S[0]=T.call_llvm_pure_intrin('float32','llvm.fma',T.uint32(3),T.float32(0),T.Cast('float32',A[(h*T.int64({queries})+m)*T.int64({k})+T.int64({k - 1})]),S[0])
     S[0]=T.call_llvm_pure_intrin('float32','llvm.fma',T.uint32(3),T.Cast('float32',A[(h*T.int64({queries})+m)*T.int64({k})+T.int64({k - 1})]),T.Cast('float32',B[(h*T.int64({k})+T.int64({k - 1}))*T.int64(64)+n]),S[0])
     bits=T.reinterpret('uint32',S[0])
     rounded=bits+T.uint32(32767)+T.bitwise_and(T.shift_right(bits,T.uint32(16)),T.uint32(1))
     Y[(h*T.int64({queries})+m)*T.int64(64)+n]=T.reinterpret('bfloat16',T.Cast('uint16',T.shift_right(rounded,T.uint32(16))))
"""
    source = f"""@T.prim_func
def source_attention{width}(A:T.Buffer((T.int64({left_count}),),'{dtype}'),B:T.Buffer((T.int64({right_count}),),'{dtype}'),Y:T.Buffer((T.int64({count}),),'{dtype}'),S:T.Buffer((T.int64(1),),'float32')):
 T.func_attr({{'tir.noalias':True,'op_pattern':8}})
 with T.block('root'):
  for h in range(T.int64(15)):
   for m in range(T.int64({queries})):
    for n in range(T.int64({n})):
     S[0]=T.float32(0)
{arithmetic}"""
    prim = tvm.script.from_source(source, extra_vars={"T": T}).without_attr("global_symbol")
    global_var = bb.add_func(prim, f"torch_onednn_gemm_bf16_pv{width}_v1")
    args = [bb.normalize(relax.op.reshape(value, (size,))) for value, size in zip((left, right), (left_count, right_count))]
    outputs = [relax.TensorStructInfo((count,), dtype), relax.TensorStructInfo((1,), "float32")]
    result = bb.normalize(relax.TupleGetItem(bb.emit(relax.call_tir(global_var, args, outputs)), 0))
    return relax.op.reshape(result, (1, 15, queries, n))


def _source_onednn_bf16_pv227_call(bb, left, right):
    return _source_onednn_bf16_pv_call(bb, left, right, 227)

class TorchMklSgemmF32QK227(OnnxOpConverter):
    """merlin.source_preserving v1: bounded native FP32 QK227 contraction."""

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        if len(inputs) != 2 or any(value is None for value in inputs):
            raise ValueError("Source QK SGEMM requires two tensor operands")
        return _source_mkl_sgemm177_call(bb, inputs[0], inputs[1], "qk", 227)


class TorchOnednnGemmBf16PV177(OnnxOpConverter):
    """merlin.source_preserving v1: bounded native BF16 prefix PV177."""

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        if len(inputs) != 2 or any(value is None for value in inputs):
            raise ValueError("Source BF16 PV requires two tensor operands")
        return _source_onednn_bf16_pv_call(bb, inputs[0], inputs[1], 177)


class TorchOnednnGemmBf16PV227(OnnxOpConverter):
    """merlin.source_preserving v1: bounded native BF16 PV227 contraction."""

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        if len(inputs) != 2 or any(value is None for value in inputs):
            raise ValueError("Source BF16 PV requires two tensor operands")
        return _source_onednn_bf16_pv227_call(bb, inputs[0], inputs[1])


class TorchSoftmaxAvx512F32LastDim177(OnnxOpConverter):
    """merlin.source_preserving v1: bound SmolVLA CPU FLOAT softmax arithmetic."""

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        if len(inputs) != 1 or inputs[0] is None:
            raise ValueError("Source softmax177 requires one tensor operand")
        return _source_softmax177_call(bb, inputs[0])


class TorchMean720Fp32(OnnxOpConverter):
    """merlin.source_preserving v1: FP32 eight-lane cascade mean, last-axis keepdim."""

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        if len(inputs) != 1 or inputs[0] is None:
            raise ValueError("Source mean720 requires one tensor operand")
        return _source_mean720_call(bb, inputs[0])


class TorchMean960Fp32(OnnxOpConverter):
    """merlin.source_preserving v1: width960 FP32 cascade mean, last-axis keepdim."""

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        if len(inputs) != 1 or inputs[0] is None:
            raise ValueError("Source mean960 requires one tensor operand")
        return _source_mean960_call(bb, inputs[0])


class TorchSiluAvx512F32(OnnxOpConverter):
    """merlin.source_preserving v1: native vector exp/add/direct-div SiLU."""

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        if len(inputs) != 1 or inputs[0] is None:
            raise ValueError("Source SiLU requires one tensor operand")
        return _source_silu_call(bb, inputs[0])


class TorchSoftmaxAvx512F32LastDim227(OnnxOpConverter):
    """merlin.source_preserving v1: bound SmolVLA CPU FLOAT softmax arithmetic."""

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        if len(inputs) != 1 or inputs[0] is None:
            raise ValueError("Source softmax227 requires one tensor operand")
        return _source_softmax227_call(bb, inputs[0])


CONVERTERS = {
        "TorchSiluAvx512F32": TorchSiluAvx512F32,
        "TorchMean720Fp32": TorchMean720Fp32,
        "TorchMean960Fp32": TorchMean960Fp32,
        "TorchFlashBf16Vision1024": TorchFlashBf16Vision1024,
        "TorchMklSgemmF32QK177": TorchMklSgemmF32QK177,
        "TorchMklSgemmF32PV177": TorchMklSgemmF32PV177,
        "TorchMklSgemmF32QK227": TorchMklSgemmF32QK227,
        "TorchOnednnGemmBf16PV177": TorchOnednnGemmBf16PV177,
        "TorchOnednnGemmBf16PV227": TorchOnednnGemmBf16PV227,
        "TorchSoftmaxAvx512F32LastDim177": TorchSoftmaxAvx512F32LastDim177,
        "TorchSoftmaxAvx512F32LastDim227": TorchSoftmaxAvx512F32LastDim227,
        "TorchLayerNormAvx2Bf16Width768": TorchLayerNormAvx2Bf16Width768,
        "TorchConv2dAvx512Bf16IC3Kernel16Stride16": TorchConv2dAvx512Bf16IC3Kernel16Stride16,
        "FixedTreeDot": FixedTreeDot,
        "FixedTreeMeanSquares": FixedTreeMeanSquares,
}
