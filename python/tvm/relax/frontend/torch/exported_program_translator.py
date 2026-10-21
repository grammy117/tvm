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

# pylint: disable=invalid-name, inconsistent-return-statements, unidiomatic-typecheck
# pylint: disable=import-outside-toplevel
"""PyTorch ExportedProgram of Relax."""

from collections import ChainMap, OrderedDict
from functools import partial
from typing import Callable, Dict, List, Tuple

import torch
import tvm
from tvm import relax

from .base_fx_graph_translator import BaseFXGraphImporter


class ExportedProgramImporter(BaseFXGraphImporter):
    """An importer from ExportedProgram to Relax."""

    from torch import fx

    def _captured_dtype(self, node: fx.Node) -> str:
        captured = node.meta.get("val")
        if not isinstance(captured, torch.Tensor):
            raise NotImplementedError("Arithmetic import requires the captured output dtype")
        return self._convert_data_type(captured.dtype)

    def _numeric_operand(self, value, dtype):
        if isinstance(value, relax.Expr):
            return (
                value
                if value.struct_info.dtype == dtype
                else self.block_builder.emit(relax.op.astype(value, dtype))
            )
        if dtype.startswith("int"):
            bits = tvm.DataType(dtype).bits
            value = (int(value) + (1 << (bits - 1))) % (1 << bits) - (1 << (bits - 1))
        return relax.const(value, dtype)

    def _numeric_values(self, lhs, rhs, dtype, operation, alpha=1):
        import numpy as np
        from ._numeric import binary_int64, binary_narrow_integer, fused_alpha_add

        if dtype == "bool":
            lhs, rhs = self._numeric_operand(lhs, dtype), self._numeric_operand(rhs, dtype)
            if operation == "multiply":
                return self.block_builder.emit(relax.op.logical_and(lhs, rhs))
            if operation == "add":
                scaled = rhs if alpha else relax.const(False, "bool")
                return self.block_builder.emit(relax.op.logical_or(lhs, scaled))
            raise NotImplementedError("Boolean subtraction is not a valid Torch operation")
        if dtype.startswith("int"):
            lhs, rhs = self._numeric_operand(lhs, dtype), self._numeric_operand(rhs, dtype)
            bits = tvm.DataType(dtype).bits
            if not isinstance(alpha, int) or not -(1 << (bits - 1)) <= alpha < (1 << (bits - 1)):
                raise NotImplementedError("Integer alpha must fit the result's signed dtype")
            if dtype == "int64":
                return self.block_builder.emit_te(
                    binary_int64,
                    lhs,
                    rhs,
                    operation,
                    alpha,
                    primfunc_name_hint="wrapping_" + operation,
                )
            # Two signed INT32 operands and an admitted INT32 alpha have a
            # product-plus-input within INT64. Narrow only after the arithmetic.
            return self.block_builder.emit_te(
                binary_narrow_integer,
                lhs,
                rhs,
                operation,
                alpha,
                primfunc_name_hint="wrapping_" + operation,
            )
        opmath = "float32" if dtype == "float16" else dtype
        # Add/sub convert a weak scalar to the result dtype. Reduced-precision
        # mul/div preserve a scalar in opmath precision (ATen BinaryOpsKernel).
        if dtype == "float16" and operation in ("add", "subtract"):
            lhs, rhs = self._numeric_operand(lhs, dtype), self._numeric_operand(rhs, dtype)
        lhs, rhs = self._numeric_operand(lhs, opmath), self._numeric_operand(rhs, opmath)
        if operation in ("add", "subtract"):
            if dtype == "float16" and alpha not in (1, -1):
                raise NotImplementedError(
                    "Exact FP16 nonunit alpha is unqualified: the captured CPU reference "
                    "uses different scalar-tail and vector intermediate rounding"
                )
            alpha = float(np.float16(alpha)) if dtype == "float16" else alpha
            alpha = -alpha if operation == "subtract" else alpha
            if alpha == 1:
                result = self.block_builder.emit(relax.op.add(lhs, rhs))
            elif alpha == -1:
                result = self.block_builder.emit(relax.op.subtract(lhs, rhs))
            else:
                result = self.block_builder.emit_te(
                    fused_alpha_add, lhs, rhs, alpha, primfunc_name_hint="alpha_add"
                )
        else:
            op = relax.op.multiply if operation == "multiply" else relax.op.divide
            result = self.block_builder.emit(op(lhs, rhs))
        return self._numeric_operand(result, dtype)

    def _numeric_binary(self, node: fx.Node, *, operation: str) -> relax.Var:
        args = self.retrieve_args(node)
        alpha = args[2] if len(args) > 2 else node.kwargs.get("alpha", 1)
        return self._numeric_values(args[0], args[1], self._captured_dtype(node), operation, alpha)

    def _numeric_negative(self, node: fx.Node) -> relax.Var:
        x = self.retrieve_args(node)[0]
        dtype = self._captured_dtype(node)
        if dtype.startswith("int"):
            return self._numeric_values(0, x, dtype, "subtract")
        return self.block_builder.emit(relax.op.negative(x))

    def _matmul_values(self, lhs, rhs, dtype):
        from tvm import tir
        from ._numeric import modular_matmul

        prototype = self.block_builder.normalize(relax.op.matmul(lhs, rhs, out_dtype=dtype))
        shape = prototype.struct_info.shape
        depth = self.shape_of(lhs)[-1]
        if isinstance(depth, tir.IntImm) and int(depth) == 0:
            return self.block_builder.emit(relax.op.zeros(shape, dtype))
        if dtype in ("int32", "int64"):
            if not isinstance(depth, tir.IntImm):
                raise NotImplementedError(
                    "Modular integer matmul requires a static reduction extent to establish empty initialization"
                )
            return self.block_builder.emit_te(
                modular_matmul, lhs, rhs, tuple(shape), primfunc_name_hint="modular_matmul"
            )
        compute_dtype = "float32" if dtype == "float16" else dtype
        if dtype == "int8":
            if not isinstance(depth, tir.IntImm):
                raise NotImplementedError("INT8 matmul requires a static accumulation bound")
            max_product = (1 << (tvm.DataType(lhs.struct_info.dtype).bits - 1)) ** 2
            compute_dtype = "int32" if int(depth) * max_product <= (1 << 31) - 1 else "int64"
            if int(depth) * max_product > (1 << 63) - 1:
                raise NotImplementedError("INT8 matmul exceeds the proven INT64 prefix bound")
        result = self.block_builder.emit(relax.op.matmul(lhs, rhs, out_dtype=compute_dtype))
        return self._numeric_operand(result, dtype)

    def _matmul(self, node: fx.Node) -> relax.Var:
        lhs, rhs = self.retrieve_args(node)
        return self._matmul_values(lhs, rhs, self._captured_dtype(node))

    def _addmm(self, node: fx.Node) -> relax.Var:
        bias, lhs, rhs = self.retrieve_args(node)
        dtype = self._captured_dtype(node)
        alpha, beta = node.kwargs.get("alpha", 1), node.kwargs.get("beta", 1)
        shape = self.block_builder.normalize(
            relax.op.matmul(lhs, rhs, out_dtype=dtype)
        ).struct_info.shape
        if alpha == 0:
            if beta == 0:
                return self.block_builder.emit(relax.op.zeros(shape, dtype))
            scaled = bias if beta == 1 else self._numeric_values(bias, beta, dtype, "multiply")
            return self.block_builder.emit(relax.op.broadcast_to(scaled, shape))
        product = self._matmul_values(lhs, rhs, dtype)
        if alpha != 1:
            product = self._numeric_values(product, alpha, dtype, "multiply")
        if beta == 0:
            return product
        scaled = bias if beta == 1 else self._numeric_values(bias, beta, dtype, "multiply")
        return self._numeric_values(scaled, product, dtype, "add")

    ########## Unary Ops ##########

    def _hardtanh(self, node: fx.Node) -> relax.Expr:
        args = self.retrieve_args(node)
        x = args[0]
        min_val = node.args[1] if len(args) > 1 else node.kwargs("min_val", -1.0)
        max_val = node.args[2] if len(args) > 2 else node.kwargs("max_val", 1.0)
        return self.block_builder.emit(relax.op.clip(x, min_val, max_val))

    ########## Neural Network ##########

    def _batch_norm_legit_no_training(self, node: fx.Node) -> relax.Var:
        import numpy as np

        x = self.env[node.args[0]]
        channel = int(self.shape_of(x)[1])
        dtype = x.struct_info.dtype
        weight = self.env.get(node.args[1], relax.const(np.ones(channel), dtype=dtype))
        bias = self.env.get(node.args[2], relax.const(np.zeros(channel), dtype=dtype))
        running_mean = self.env.get(node.args[3], relax.const(np.zeros(channel), dtype=dtype))
        running_var = self.env.get(node.args[4], relax.const(np.ones(channel), dtype=dtype))
        momentum = node.args[5] if len(node.args) > 5 else node.kwargs.get("momentum", 0.1)
        eps = node.args[6] if len(node.args) > 6 else node.kwargs.get("eps", 1e-05)

        return self.block_builder.emit(
            relax.op.nn.batch_norm(
                x,
                weight,
                bias,
                running_mean,
                running_var,
                axis=1,
                epsilon=eps,
                momentum=momentum,
            )
        )

    def _group_norm(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        num_groups = node.args[1]
        gamma = self.env[node.args[2]] if len(node.args) > 2 else None
        beta = self.env[node.args[3]] if len(node.args) > 3 else None
        eps = node.args[4] if len(node.args) > 4 else 1e-05

        dim = len(self.shape_of(x))
        return self.block_builder.emit(
            relax.op.nn.group_norm(
                x,
                gamma,
                beta,
                num_groups=num_groups,
                channel_axis=1,
                axes=list(range(2, dim)),
                epsilon=eps,
            )
        )

    def _upsample_impl(
        self, x: relax.Expr, size, align_corners: bool, scale_factor, method: str
    ) -> relax.Var:
        coord_trans = "align_corners" if align_corners else "half_pixel"

        if size is None:
            shape = self.shape_of(x)
            assert isinstance(shape, relax.ShapeExpr)
            if isinstance(scale_factor, (tuple, list)):
                assert len(scale_factor) == len(shape) - 2
                size = tuple(
                    int(shape[i].value * scale_factor[i - 2]) for i in range(2, len(shape))
                )
            else:
                size = tuple(int(shape[i].value * scale_factor) for i in range(2, len(shape)))

        return self.block_builder.emit(
            relax.op.image.resize2d(
                x, size, layout="NCHW", method=method, coordinate_transformation_mode=coord_trans
            )
        )

    def _upsample_bilinear2d(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        size = node.args[1] if len(node.args) > 1 else node.kwargs.get("size", None)
        align_corners = (
            node.args[2] if len(node.args) > 2 else node.kwargs.get("align_corners", True)
        )
        scale_factor = node.args[3] if len(node.args) > 3 else node.kwargs.get("scale_factor", None)
        return self._upsample_impl(x, size, align_corners, scale_factor, "linear")

    def _upsample_nearest2d(self, node: fx.node) -> relax.Var:
        x = self.env[node.args[0]]
        size = node.args[1] if len(node.args) > 1 else node.kwargs.get("size", None)
        align_corners = (
            node.args[2] if len(node.args) > 2 else node.kwargs.get("align_corners", True)
        )
        scale_factor = node.args[3] if len(node.args) > 3 else node.kwargs.get("scale_factor", None)
        return self._upsample_impl(x, size, align_corners, scale_factor, "nearest_neighbor")

    ########## Manipulation ##########

    def _diagonal(self, node: fx.Node) -> relax.Var:
        from tvm import te, tir

        args = self.retrieve_args(node)
        x = args[0]
        shape = self.shape_of(x)
        if len(shape) < 2 or not all(isinstance(size, tir.IntImm) for size in shape):
            raise NotImplementedError(
                "Diagonal import requires a static shape with rank at least two"
            )
        offset = args[1] if len(args) > 1 else node.kwargs.get("offset", 0)
        dim1 = args[2] if len(args) > 2 else node.kwargs.get("dim1", 0)
        dim2 = args[3] if len(args) > 3 else node.kwargs.get("dim2", 1)
        if not isinstance(offset, int) or any(
            not isinstance(dim, int) or not -len(shape) <= dim < len(shape) for dim in (dim1, dim2)
        ):
            raise ValueError("Diagonal offset and axes must be valid static integers")
        dim1, dim2 = dim1 % len(shape), dim2 % len(shape)
        if dim1 == dim2:
            raise ValueError("Diagonal axes must be distinct")
        remaining = [i for i in range(len(shape)) if i not in (dim1, dim2)]
        length = max(0, min(int(shape[dim1]) + min(offset, 0), int(shape[dim2]) - max(offset, 0)))

        def compute(data):
            def value(*indices):
                access = [None] * len(shape)
                for i, axis in enumerate(remaining):
                    access[axis] = indices[i]
                access[dim1], access[dim2] = (
                    indices[-1] + max(-offset, 0),
                    indices[-1] + max(offset, 0),
                )
                return data[tuple(access)]

            return te.compute(
                tuple(shape[i] for i in remaining) + (length,), value, name="diagonal"
            )

        return self.block_builder.emit_te(compute, x, primfunc_name_hint="diagonal")

    def _as_strided(self, node: fx.Node) -> relax.Var:
        import math
        from tvm import te, tir

        args = self.retrieve_args(node)
        x = args[0]
        source = node.args[0].meta.get("val")
        if (
            not isinstance(source, torch.Tensor)
            or not source.is_contiguous()
            or not isinstance(source.storage_offset(), int)
        ):
            raise NotImplementedError(
                "as_strided import requires a contiguous source with a static storage offset; hidden storage aliases are not represented in Relax"
            )
        shape = tuple(self.shape_of(x))
        sizes = args[1] if len(args) > 1 else node.kwargs["size"]
        strides = args[2] if len(args) > 2 else node.kwargs["stride"]
        offset = args[3] if len(args) > 3 else node.kwargs.get("storage_offset", None)
        offset = source.storage_offset() if offset is None else offset
        if len(sizes) != len(strides) or not isinstance(offset, int) or offset < 0:
            raise ValueError("as_strided requires matching sizes/strides and a nonnegative offset")
        offset -= source.storage_offset()
        if not all(isinstance(size, tir.IntImm) for size in shape) or any(
            not isinstance(value, int) or value < 0 for value in (*sizes, *strides)
        ):
            raise NotImplementedError("as_strided requires static nonnegative shapes and strides")
        empty = any(size == 0 for size in sizes)
        last = offset + sum((size - 1) * stride for size, stride in zip(sizes, strides))
        if not empty and (offset < 0 or last >= math.prod(int(size) for size in shape)):
            raise NotImplementedError(
                "as_strided cannot read storage outside the logical Relax input"
            )

        def compute(data):
            def value(*indices):
                if empty:
                    return tir.const(0, data.dtype)
                linear = offset + sum(index * stride for index, stride in zip(indices, strides))
                access = []
                for axis, size in enumerate(shape):
                    divisor = math.prod(int(dim) for dim in shape[axis + 1 :])
                    access.append(linear // divisor % int(size))
                return data[tuple(access)]

            return te.compute(tuple(sizes), value, name="as_strided")

        return self.block_builder.emit_te(compute, x, primfunc_name_hint="as_strided")

    def _scalar_tensor(self, node: fx.Node) -> relax.Expr:
        args = self.retrieve_args(node)
        value = args[0]
        if not isinstance(value, (int, float, bool)):
            raise NotImplementedError("scalar_tensor import currently requires a constant scalar")
        captured = node.meta.get("val")
        dtype = node.kwargs.get("dtype") or (
            captured.dtype if isinstance(captured, torch.Tensor) else torch.get_default_dtype()
        )
        return relax.const(value, self._convert_data_type(dtype))

    def _full(self, node: fx.Node, *, like: bool = False) -> relax.Var:
        args = self.retrieve_args(node)
        source = args[0]
        value = args[1] if len(args) > 1 else node.kwargs["fill_value"]
        captured = node.meta.get("val")
        if not isinstance(captured, torch.Tensor):
            raise NotImplementedError("Full import requires the captured output dtype")
        if not isinstance(value, (int, float, bool)):
            raise NotImplementedError("Full import currently requires a constant scalar fill value")
        # The captured result preserves PyTorch's default dtype for full and
        # its input/explicit dtype policy for full_like, without guessing from
        # the scalar's Python type or inheriting full_like's input dtype.
        dtype = self._convert_data_type(captured.dtype)
        fill = relax.const(value, dtype)
        result = (
            relax.op.full_like(source, fill, dtype) if like else relax.op.full(source, fill, dtype)
        )
        return self.block_builder.emit(result)

    def _pad(self, node: fx.Node, *, mode=None) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        padding = args[1] if len(args) > 1 else node.kwargs["pad" if mode is None else "padding"]
        if mode is None:
            mode = args[2] if len(args) > 2 else node.kwargs.get("mode", "constant")
            value = args[3] if len(args) > 3 else node.kwargs.get("value", None)
        else:
            value = args[2] if len(args) > 2 else node.kwargs.get("value", None)
        mode = "edge" if mode == "replicate" else mode
        if mode not in ("constant", "reflect", "edge"):
            raise NotImplementedError(f"Unsupported PyTorch padding mode: {mode}")
        ndim = len(self.shape_of(x))
        if len(padding) % 2 or len(padding) > 2 * ndim:
            raise ValueError("Padding must contain pairs for at most the input rank")
        widths = [0] * (2 * ndim)
        for axis in range(len(padding) // 2):
            offset = 2 * (ndim - axis - 1)
            widths[offset : offset + 2] = padding[2 * axis : 2 * axis + 2]
        return self.block_builder.emit(
            relax.op.nn.pad(x, widths, 0 if value is None else value, pad_mode=mode)
        )

    def _select(self, node: fx.Node) -> relax.Var:
        from tvm import tir

        x = self.env[node.args[0]]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs["dim"]
        index = node.args[2] if len(node.args) > 2 else node.kwargs["index"]
        shape = self.shape_of(x)
        if (
            not isinstance(dim, int)
            or not -len(shape) <= dim < len(shape)
            or not isinstance(index, int)
        ):
            raise ValueError("Select requires a valid static dimension and integer index")
        extent = shape[dim]
        if index < 0:
            if not isinstance(extent, tir.IntImm):
                raise NotImplementedError("Negative select indices require a static extent")
            index += int(extent)
        if isinstance(extent, tir.IntImm) and not 0 <= index < int(extent):
            raise ValueError("Select index is outside the input extent")
        # Keep a static selection as graph-level slicing and shape metadata.
        # This avoids introducing a runtime index tensor for a constant index.
        selected = self.block_builder.emit(
            relax.op.strided_slice(x, [dim], [index], [index + 1], [1])
        )
        return self.block_builder.emit(relax.op.squeeze(selected, dim))

    def _slice(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        axes = [node.args[1]]
        begin = [node.args[2]]
        end = [node.args[3]]
        stride = [node.args[4] if len(node.args) > 4 else 1]
        return self.block_builder.emit(relax.op.strided_slice(x, axes, begin, end, stride))

    def _scatter_slice(self, node: fx.Node, *, select=False) -> relax.Var:
        """Import functional slice updates as ordinary, target-independent TOPI."""
        import numpy as np
        from tvm import tir, topi

        args = self.retrieve_args(node)
        x, source = args[:2]
        shape = self.shape_of(x)
        if not all(isinstance(dim, tir.IntImm) for dim in shape):
            raise NotImplementedError(
                "Scatter slice import currently requires a static input shape"
            )
        shape = tuple(int(dim) for dim in shape)
        rank = len(shape)
        axis = args[2] if len(args) > 2 else node.kwargs.get("dim", 0)
        if not isinstance(axis, int) or not -rank <= axis < rank:
            raise ValueError("Scatter slice dimension is outside the input rank")
        axis %= rank
        if select:
            index = args[3] if len(args) > 3 else node.kwargs["index"]
            if not isinstance(index, int) or not -shape[axis] <= index < shape[axis]:
                raise ValueError("Scatter select index is outside the input extent")
            start, stop, step = index % shape[axis], index % shape[axis] + 1, 1
            source = self.block_builder.emit(relax.op.expand_dims(source, axis))
        else:
            start = args[3] if len(args) > 3 else node.kwargs.get("start", None)
            stop = args[4] if len(args) > 4 else node.kwargs.get("end", None)
            step = args[5] if len(args) > 5 else node.kwargs.get("step", 1)
            if not isinstance(step, int) or step <= 0:
                raise ValueError("PyTorch scatter slice requires a positive static step")
            if any(value is not None and not isinstance(value, int) for value in (start, stop)):
                raise NotImplementedError("Scatter slice bounds must be static integers")
            start, stop, step = slice(start, stop, step).indices(shape[axis])
        expected = list(shape)
        expected[axis] = len(range(start, stop, step))
        if (
            tuple(self.shape_of(source)) != tuple(expected)
            or source.struct_info.dtype != x.struct_info.dtype
        ):
            raise ValueError("Scatter source shape and dtype must match the selected slice")
        if not expected[axis]:
            return self.block_builder.emit_te(topi.identity, x)
        begin, end, strides = [0] * rank, list(shape), [1] * rank
        begin[axis], end[axis], strides[axis] = start, stop, step
        if any(value > (1 << 31) - 1 for value in (*shape, *strides)):
            raise NotImplementedError("TOPI scatter slice requires signed INT32 index bounds")

        def compute(data, updates):
            return topi.strided_set(
                data,
                updates,
                topi.utils.const_vector(np.array(begin, "int32")),
                topi.utils.const_vector(np.array(end, "int32")),
                topi.utils.const_vector(np.array(strides, "int32")),
            )

        return self.block_builder.emit_te(compute, x, source, primfunc_name_hint="scatter_slice")

    ########## Others ##########

    def create_convert_map(
        self,
    ) -> Dict[str, Callable[[fx.Node], relax.Var]]:
        import operator

        return {
            # unary
            "acos.default": self._unary_op(relax.op.acos),
            "acosh.default": self._unary_op(relax.op.acosh),
            "asin.default": self._unary_op(relax.op.asin),
            "asinh.default": self._unary_op(relax.op.asinh),
            "atan.default": self._unary_op(relax.op.atan),
            "atanh.default": self._unary_op(relax.op.atanh),
            "clamp.default": self._clamp,
            "cos.default": self._unary_op(relax.op.cos),
            "cosh.default": self._unary_op(relax.op.cosh),
            "dropout.default": lambda node: self.env[node.args[0]],
            "exp.default": self._unary_op(relax.op.exp),
            "gelu.default": self._gelu,
            "hardsigmoid.default": self._hardsigmoid,
            "hardswish.default": self._hardswish,
            "hardtanh.default": self._hardtanh,
            "leaky_relu.default": self._leakyrelu,
            "log_softmax.int": self._log_softmax,
            "neg.default": self._numeric_negative,
            "relu.default": self._unary_op(relax.op.nn.relu),
            "round.default": self._round,
            "rsqrt.default": self._unary_op(relax.op.rsqrt),
            "sigmoid.default": self._unary_op(relax.op.sigmoid),
            "silu.default": self._unary_op(relax.op.nn.silu),
            "sin.default": self._unary_op(relax.op.sin),
            "sinh.default": self._unary_op(relax.op.sinh),
            "softmax.int": self._softmax,
            "sqrt.default": self._unary_op(relax.op.sqrt),
            "tan.default": self._unary_op(relax.op.tan),
            "tanh.default": self._unary_op(relax.op.tanh),
            "tril.default": self._tril_triu(relax.op.tril),
            "triu.default": self._tril_triu(relax.op.triu),
            # binary
            "add.Tensor": partial(self._numeric_binary, operation="add"),
            "add.Scalar": partial(self._numeric_binary, operation="add"),
            "div.Tensor": partial(self._numeric_binary, operation="divide"),
            "div.Scalar": partial(self._numeric_binary, operation="divide"),
            "eq.Scalar": self._binary_op(relax.op.equal, operator.eq),
            "eq.Tensor": self._binary_op(relax.op.equal, operator.eq),
            "floor_divide.default": self._binary_op(relax.op.floor_divide, operator.floordiv),
            "lt.Scalar": self._binary_op(relax.op.less, operator.lt),
            "lt.Tensor": self._binary_op(relax.op.less, operator.lt),
            "matmul.default": self._matmul,
            "mm.default": self._matmul,
            "max.other": self._binary_op(relax.op.maximum, max),
            "mul.Tensor": partial(self._numeric_binary, operation="multiply"),
            "mul.Scalar": partial(self._numeric_binary, operation="multiply"),
            "pow.Tensor_Scalar": self._binary_op(relax.op.power, operator.pow),
            "pow.Tensor_Tensor": self._binary_op(relax.op.power, operator.pow),
            "sub.Tensor": partial(self._numeric_binary, operation="subtract"),
            "sub.Scalar": partial(self._numeric_binary, operation="subtract"),
            # neural network
            "_native_batch_norm_legit_no_training.default": self._batch_norm_legit_no_training,
            "adaptive_avg_pool2d.default": self._adaptive_avg_pool2d,
            "addmm.default": self._addmm,
            "avg_pool2d.default": self._avg_pool2d,
            "baddbmm.default": self._baddbmm,
            "bmm.default": self._matmul,
            "conv_transpose1d.default": self._conv_transpose1d,
            "conv_transpose2d.input": self._conv_transpose2d,
            "conv1d.default": self._conv1d,
            "conv2d.default": self._conv2d,
            "conv3d.default": self._conv3d,
            "einsum.default": self._einsum,
            "embedding.default": lambda node: self._embedding_impl(
                self.env[node.args[1]], self.env[node.args[0]]
            ),
            "group_norm.default": self._group_norm,
            "layer_norm.default": self._layer_norm,
            "linear.default": self._linear,
            "max_pool2d.default": self._max_pool2d,
            "scaled_dot_product_attention.default": self._scaled_dot_product_attention,
            "unbind.int": self._unbind,
            "upsample_bilinear2d.vec": self._upsample_bilinear2d,
            "upsample_nearest2d.vec": self._upsample_nearest2d,
            # statistical
            "mean.default": self._mean,
            "mean.dim": self._mean,
            "sum.default": self._sum,
            "sum.dim_IntList": self._sum,
            # search
            "argmax.default": self._argmax_argmin(relax.op.argmax),
            "argmin.default": self._argmax_argmin(relax.op.argmin),
            # tensor manipulation
            "as_strided.default": self._as_strided,
            "cat.default": self._cat,
            "concat.default": self._cat,
            "cumsum.default": self._cumsum,
            "diagonal.default": self._diagonal,
            "constant_pad_nd.default": partial(self._pad, mode="constant"),
            "expand.default": self._expand,
            "pad.default": self._pad,
            "permute.default": self._permute,
            "reflection_pad1d.default": partial(self._pad, mode="reflect"),
            "reflection_pad2d.default": partial(self._pad, mode="reflect"),
            "reflection_pad3d.default": partial(self._pad, mode="reflect"),
            "repeat.default": self._repeat,
            "reshape.default": self._reshape,
            "replication_pad1d.default": partial(self._pad, mode="edge"),
            "replication_pad2d.default": partial(self._pad, mode="edge"),
            "replication_pad3d.default": partial(self._pad, mode="edge"),
            "select.int": self._select,
            "select_scatter.default": partial(self._scatter_slice, select=True),
            "slice.Tensor": self._slice,
            "slice_scatter.default": self._scatter_slice,
            "split.Tensor": self._split,
            "split_with_sizes.default": self._split,
            "squeeze.default": self._squeeze,
            "squeeze.dim": self._squeeze,
            "squeeze.dims": self._squeeze,
            "tile.default": self._tile,
            "transpose.int": self._transpose,
            "unsqueeze.default": lambda node: self.block_builder.emit(
                relax.op.expand_dims(self.env[node.args[0]], node.args[1])
            ),
            "view.default": self._reshape,
            # tensor creation
            "_to_copy.default": self._to_copy,
            "arange.start": self._arange,
            "clone.default": lambda node: self.env[node.args[0]],
            "empty.memory_format": self._empty,
            "fill.Scalar": self._fill,
            "full.default": self._full,
            "full_like.default": partial(self._full, like=True),
            "scalar_tensor.default": self._scalar_tensor,
            "new_ones.default": self._new_ones,
            # other
            "getitem": self._getitem,
        }

    def create_input_vars(
        self, exported_program: torch.export.ExportedProgram
    ) -> Tuple[Dict[str, relax.Var], Dict[str, relax.Var]]:
        """Create relax input vars."""
        parameters_buffers_constants = OrderedDict()
        user_inputs = OrderedDict()
        for spec in exported_program.graph_signature.input_specs:
            name_hint = spec.arg.name
            if spec.kind is torch.export.graph_signature.InputKind.CONSTANT_TENSOR:
                shape = exported_program.tensor_constants[spec.target].shape
                torch_dtype = exported_program.tensor_constants[spec.target].dtype
            elif spec.kind is torch.export.graph_signature.InputKind.USER_INPUT:
                for node in exported_program.graph.find_nodes(op="placeholder", target=spec.target):
                    if node.name == name_hint:
                        shape = node.meta["tensor_meta"].shape
                        torch_dtype = node.meta["tensor_meta"].dtype
                        break
            else:
                # PARAMETER or BUFFER
                shape = exported_program.state_dict[spec.target].shape
                torch_dtype = exported_program.state_dict[spec.target].dtype

            dtype = self._convert_data_type(torch_dtype)
            relax_var = relax.Var(name_hint, relax.TensorStructInfo(shape, dtype))
            if spec.kind is torch.export.graph_signature.InputKind.USER_INPUT:
                user_inputs[name_hint] = relax_var
            else:
                parameters_buffers_constants[name_hint] = relax_var

        return parameters_buffers_constants, user_inputs

    def from_exported_program(
        self,
        exported_program: torch.export.ExportedProgram,
        keep_params_as_input: bool,
        unwrap_unit_return_tuple: bool,
        no_bind_return_tuple: bool,
    ) -> tvm.IRModule:
        """Convert a PyTorch ExportedProgram to a Relax program."""
        from torch import fx  # type: ignore

        # Create input variables.
        parameter_buffer_constant_vars, user_input_vars = self.create_input_vars(exported_program)
        inputs_vars = user_input_vars.copy()
        inputs_vars.update(parameter_buffer_constant_vars)

        # Initialize the block builder with a function and a dataflow block.
        self.block_builder = relax.BlockBuilder()
        func_name = "main"
        func_attrs = {"num_input": len(user_input_vars)} if keep_params_as_input else None

        nodes: List[fx.Node] = exported_program.graph.nodes
        with self.block_builder.function(
            name=func_name, params=list(inputs_vars.values()).copy(), attrs=func_attrs
        ):
            output = None
            with self.block_builder.dataflow():
                # Translate the model.
                for node in nodes:
                    if node.op == "placeholder":
                        if "grapharg" in node.meta and node.meta["grapharg"].fake_tensor is None:
                            # Ignore sym input
                            continue

                        self.env[node] = inputs_vars[node.name]
                    elif node.op == "output":
                        args = self.retrieve_args(node)
                        assert len(args) == 1
                        assert isinstance(args[0], (tuple, relax.Tuple))

                        if unwrap_unit_return_tuple and len(args[0]) == 1:
                            output = self.block_builder.emit_output(args[0][0])
                        elif no_bind_return_tuple:
                            output = []
                            for ret in args[0]:
                                output.append(self.block_builder.emit_output(ret))
                        else:
                            output = self.block_builder.emit_output(args[0])
                        break
                    elif node.op == "get_attr":
                        self.env[node] = getattr(exported_program.graph_module, node.target)
                    elif node.op == "call_function":
                        func_name = node.target.__name__
                        assert (
                            func_name in self.convert_map
                        ), f"Unsupported function type {func_name}"
                        self.env[node] = self.convert_map[func_name](node)
                    else:
                        raise ValueError(f"Unsupported op {node.op}")
            assert output is not None
            self.block_builder.emit_func_output(output)

        to_bind_parameters = ChainMap(
            OrderedDict(exported_program.named_buffers()), exported_program.constants
        )
        if not keep_params_as_input:
            to_bind_parameters = to_bind_parameters.new_child(
                OrderedDict(exported_program.named_parameters())
            )

        binding = {}
        for tensor_name, tensor_value in to_bind_parameters.items():
            # find relax var name from graph signature
            for spec in exported_program.graph_signature.input_specs:
                if tensor_name == spec.target:
                    bind_name = spec.arg.name
                    break
            binding[bind_name] = tvm.nd.from_dlpack(tensor_value.detach())

        mod = self.block_builder.get()
        mod = relax.transform.BindParams("main", binding)(mod)

        if keep_params_as_input:
            parameters = dict(exported_program.named_parameters())
            params = [tvm.nd.from_dlpack(p.detach()) for p in parameters.values()]
            mod["main"] = mod["main"].with_attr("params", params)

        return mod


def from_exported_program(
    exported_program: torch.export.ExportedProgram,
    *,
    keep_params_as_input: bool = False,
    unwrap_unit_return_tuple: bool = False,
    no_bind_return_tuple: bool = False,
) -> tvm.IRModule:
    """Convert a PyTorch ExportedProgram to a Relax program

    Parameters
    ----------
    exported_program : torch.export.ExportedProgram
        The PyTorch ExportedProgram to convert.

    keep_params_as_input : bool
        Whether to keep model parameters as input variables.

    unwrap_unit_return_tuple : bool
        A boolean flag indicating if to the return value when it is an unit tuple.
        When the return value is not a unit tuple, no unwrap will take place.

    no_bind_return_tuple : bool
        A boolean flag indicating whether to bind the return tuple as a relax var.
        If the flag is true and the return value is a tuple, it will not bind it to a var.

    Returns
    -------
    output : tvm.IRModule
        The import result IRModule, with the function "main" containing the
        translated logic.

    Examples
    --------
    Users can use the torch.export.export() to extract a torch.export.ExportedProgram
    from a PyTorch model. The following codes show how to convert a PyTorch model to
    a Relax program.

    .. code-block:: python

        # Import the importer.
        import tvm
        from tvm.relax.frontend.torch import from_exported_program
        import torch
        from torch.export import export

        # Define the module
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(in_features=10, out_features=7, bias=True)

            def forward(self, input):
                return self.linear(input)

        # Instantiate the model and create the input info dict.
        torch_model = MyModule()

        # Use torch.export.export() to convert the PyTorch model into ExportedProgram.
        example_args = (torch.rand(128, 10, dtype=torch.float32),)
        exported_program = export(torch_model, args=example_args)

        # Use the importer to import the ExportedProgram to Relax.
        mod: tvm.IRModule = from_exported_program(exported_program)
    """
    # decompose into Core ATen operators
    exported_program.run_decompositions()

    return ExportedProgramImporter().from_exported_program(
        exported_program,
        keep_params_as_input,
        unwrap_unit_return_tuple,
        no_bind_return_tuple,
    )
