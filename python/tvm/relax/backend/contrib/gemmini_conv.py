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
"""Static integer convolution decomposition before Gemmini matmul scheduling.

This opt-in pass emits ordinary CPU mathematical packing/restoration and a
semantic Relax matmul. Run it before FoldConstant and prepare_gemmini_graph;
it neither emits device instructions nor changes bias/quantization semantics.
The complete im2col and int32 product are explicit graph tensors, not hidden
scratch allocations. The exporter remains responsible for whole-graph capacity.
"""

from dataclasses import dataclass
from math import prod

import tvm
from tvm import relax, te, tir
from tvm.relax.expr_functor import PyExprMutator, mutator

_MAX_K = ((1 << 31) - 1) // (128 * 128)
_MAX_BYTES = (1 << 63) - 1


@dataclass(frozen=True)
class Conv2DContract:
    """Admitted semantic shapes; byte counts describe tensors, not peak memory."""

    data_shape: tuple
    weight_shape: tuple
    output_shape: tuple
    strides: tuple
    padding: tuple
    dilation: tuple

    @property
    def matmul_shape(self):
        batch, channels, height, width = self.output_shape
        return batch * height * width, channels, prod(self.weight_shape[1:])

    @property
    def tensor_bytes(self):
        m, n, k = self.matmul_shape
        return {"input": prod(self.data_shape), "weight": prod(self.weight_shape), "im2col": m * k, "packed_weight": k * n, "matmul_output": 4 * m * n, "output": 4 * prod(self.output_shape)}


def _shape(info, dtype):
    if not isinstance(info, relax.TensorStructInfo) or info.dtype != dtype or info.ndim != 4:
        return None
    if not isinstance(info.shape, relax.ShapeExpr):
        return None
    if info.vdevice is not None and info.vdevice.target.kind.name not in ("llvm", "c"):
        return None
    if any(not isinstance(dim, tir.IntImm) or dim.value <= 0 for dim in info.shape):
        return None
    return tuple(int(dim.value) for dim in info.shape)


def get_conv2d_contract(call):
    """Return the static zero-arithmetic-padding contract, or None for fallback.

    Raw signed-int8 operands have no implicit affine zero point. Bias,
    zero-point correction, activation and requantization stay outside this pass.
    The reduction bound protects every partial sum for all signed-int8 values.
    """
    if not isinstance(call, relax.Call) or not isinstance(call.op, tvm.ir.Op) or call.op.name != "relax.nn.conv2d":
        return None
    attrs = call.attrs
    if int(attrs.groups) != 1 or attrs.data_layout != "NCHW" or attrs.kernel_layout != "OIHW" or attrs.out_layout not in ("", "NCHW"):
        return None
    data, weight = (_shape(arg.struct_info, "int8") for arg in call.args)
    output = _shape(call.struct_info, "int32")
    if data is None or weight is None or output is None:
        return None
    strides, padding, dilation = (tuple(int(value) for value in values) for values in (attrs.strides, attrs.padding, attrs.dilation))
    if len(strides) != 2 or len(dilation) != 2 or len(padding) != 4 or min(*strides, *dilation) <= 0 or min(padding) < 0:
        return None
    batch, channels, height, width = data
    out_channels, inner, kh, kw = weight
    oh = (height + padding[0] + padding[2] - dilation[0] * (kh - 1) - 1) // strides[0] + 1
    ow = (width + padding[1] + padding[3] - dilation[1] * (kw - 1) - 1) // strides[1] + 1
    if channels != inner or min(oh, ow) <= 0 or output != (batch, out_channels, oh, ow):
        return None
    contract = Conv2DContract(data, weight, output, strides, padding, dilation)
    _, n, k = contract.matmul_shape
    if k > _MAX_K or n * 4 > (1 << 32) - 1 or max(contract.tensor_bytes.values()) > _MAX_BYTES:
        return None
    # Index arithmetic and padding attributes must themselves fit signed int64.
    if max(*data, *weight, *strides, *padding, *dilation, height + padding[0] + padding[2], width + padding[1] + padding[3], dilation[0] * (kh - 1), dilation[1] * (kw - 1)) > _MAX_BYTES:
        return None
    return contract


def _im2col(data, contract):
    _, _, oh, ow = contract.output_shape
    _, _, height, width = contract.data_shape
    _, _, kh, kw = contract.weight_shape
    m, _, k = contract.matmul_shape

    def patch(p, r):
        y = (p // ow % oh) * contract.strides[0] + (r // kw % kh) * contract.dilation[0] - contract.padding[0]
        x = (p % ow) * contract.strides[1] + (r % kw) * contract.dilation[1] - contract.padding[1]
        return tir.if_then_else(tir.all(y >= 0, y < height, x >= 0, x < width), data[p // (oh * ow), r // (kh * kw), y, x], tir.const(0, "int8"))

    return te.compute((tir.IntImm("int64", m), tir.IntImm("int64", k)), patch, name="im2col")


def _pack_weight(weight, contract):
    _, _, kh, kw = contract.weight_shape
    _, n, k = contract.matmul_shape
    return te.compute((tir.IntImm("int64", k), tir.IntImm("int64", n)), lambda r, o: weight[o, r // (kh * kw), r // kw % kh, r % kw], name="pack_weight")


def _restore(product, contract):
    batch, channels, oh, ow = contract.output_shape
    return te.compute(tuple(tir.IntImm("int64", size) for size in (batch, channels, oh, ow)), lambda n, o, y, x: product[(n * oh + y) * ow + x, o], name="restore_nchw")


@mutator
class _Conv2DLowerer(PyExprMutator):
    def visit_call_(self, call):
        call = self.visit_expr_post_order(call)
        contract = get_conv2d_contract(call)
        if contract is None:
            return call
        patch = self.builder_.emit_te(lambda data: _im2col(data, contract), call.args[0], primfunc_name_hint="gemmini_im2col")
        weight = self.builder_.emit_te(lambda value: _pack_weight(value, contract), call.args[1], primfunc_name_hint="gemmini_pack_weight")
        product = self.builder_.emit(relax.op.matmul(patch, weight, out_dtype="int32"))
        return self.builder_.call_te(lambda value: _restore(value, contract), product, primfunc_name_hint="gemmini_restore_nchw")

    def visit_function_(self, func):
        if func.attrs and ("Codegen" in func.attrs or "Composite" in func.attrs):
            return func
        return super().visit_function_(func)


@tvm.transform.module_pass(opt_level=0, name="DecomposeGemminiConv2D")
class DecomposeGemminiConv2D:
    """Decompose eligible static groups=1 NCHW/OIHW int8/int32 convolution.

    Full materialization preserves a compact buffer-only graph ABI. Call before
    mathematical constant folding and Gemmini scheduled-matmul substitution.
    Unsupported operations and externally owned functions remain unchanged.
    """

    def transform_module(self, mod, _ctx):
        lowerer = _Conv2DLowerer(mod)
        for gv, func in list(mod.functions_items()):
            if isinstance(func, relax.Function):
                lowerer.builder_.update_func(gv, lowerer.visit_expr(func))
        return lowerer.builder_.get()
