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

import operator
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
            bits = tvm.DataType(dtype).bits
            if not isinstance(alpha, int) or not -(1 << (bits - 1)) <= alpha < (1 << (bits - 1)):
                raise NotImplementedError("Integer alpha must fit the result's signed dtype")
            if operation == "subtract" and isinstance(rhs, int):
                # Fold a known scalar offset in the source modulo ring. This
                # target-independent identity exposes ordinary addition without
                # requiring a runtime negation or a different rounding policy.
                rhs, alpha, operation = -alpha * rhs, 1, "add"
            lhs, rhs = self._numeric_operand(lhs, dtype), self._numeric_operand(rhs, dtype)
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

    def _amax(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        axes = args[1] if len(args) > 1 else node.kwargs.get("dim", ())
        keepdim = args[2] if len(args) > 2 else node.kwargs.get("keepdim", False)
        return self.block_builder.emit(relax.op.max(args[0], axis=axes or None, keepdims=keepdim))

    def _rms_norm(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        data, shape = args[:2]
        if data.struct_info.dtype != "float32":
            raise NotImplementedError("RMSNorm currently qualifies FP32 source arithmetic")
        weight = args[2] if len(args) > 2 else node.kwargs.get("weight")
        epsilon = args[3] if len(args) > 3 else node.kwargs.get("eps")
        epsilon = torch.finfo(torch.float32).eps if epsilon is None else epsilon
        if weight is None:
            weight = self.block_builder.emit(relax.op.ones(shape, "float32"))
        axes = list(range(-len(shape), 0))
        return self.block_builder.emit(
            relax.op.nn.rms_norm(data, weight, axes=axes, epsilon=epsilon)
        )

    def _flatten(self, node: fx.Node) -> relax.Var:
        import math

        args = self.retrieve_args(node)
        data = args[0]
        shape = tuple(self.shape_of(data))
        if not shape:
            return self.block_builder.emit(relax.op.reshape(data, (1,)))
        start = args[1] if len(args) > 1 else node.kwargs.get("start_dim", 0)
        end = args[2] if len(args) > 2 else node.kwargs.get("end_dim", -1)
        if not -len(shape) <= start < len(shape) or not -len(shape) <= end < len(shape):
            raise ValueError("Flatten axes are outside the input rank")
        start, end = start % len(shape), end % len(shape)
        if start > end:
            raise ValueError("Flatten start axis follows its end axis")
        output = shape[:start] + (math.prod(shape[start : end + 1]),) + shape[end + 1 :]
        return self.block_builder.emit(relax.op.reshape(data, output))

    def _chunk(self, node: fx.Node) -> relax.Var:
        from tvm import tir

        args = self.retrieve_args(node)
        data = args[0]
        chunks = args[1] if len(args) > 1 else node.kwargs["chunks"]
        axis = args[2] if len(args) > 2 else node.kwargs.get("dim", 0)
        extent = self.shape_of(data)[axis]
        if not isinstance(extent, tir.IntImm):
            raise NotImplementedError("Chunk import requires a static split extent")
        if type(chunks) is not int or chunks <= 0:
            raise ValueError("Chunk count must be a positive integer")
        size = (int(extent) + chunks - 1) // chunks
        cuts = list(range(size, int(extent), size)) if size else [0] * (chunks - 1)
        if int(extent) > 0 and int(extent) % chunks == 0:
            cuts = chunks
        return self.block_builder.emit(relax.op.split(data, cuts, axis))

    def _batch_norm_inference(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        training = args[5] if len(args) > 5 else node.kwargs.get("training", False)
        if training is not False or args[3] is None or args[4] is None:
            raise NotImplementedError(
                "BatchNorm training/statistics updates are outside inference import"
            )
        data, weight, bias, mean, variance = args[:5]
        if weight is None:
            weight = self.block_builder.emit(
                relax.op.ones(self.shape_of(mean), data.struct_info.dtype)
            )
        if bias is None:
            bias = self.block_builder.emit(
                relax.op.zeros(self.shape_of(mean), data.struct_info.dtype)
            )
        momentum = args[6] if len(args) > 6 else node.kwargs.get("momentum", 0.1)
        epsilon = args[7] if len(args) > 7 else node.kwargs.get("eps", 1e-5)
        result = self.block_builder.emit(
            relax.op.nn.batch_norm(
                data, weight, bias, mean, variance, axis=1, epsilon=epsilon, momentum=momentum
            )
        )
        return self.block_builder.emit(relax.TupleGetItem(result, 0))

    def _where(self, node: fx.Node) -> relax.Var:
        condition, lhs, rhs = self.retrieve_args(node)
        dtype = self._captured_dtype(node)
        return self.block_builder.emit(
            relax.op.where(
                condition, self._numeric_operand(lhs, dtype), self._numeric_operand(rhs, dtype)
            )
        )

    def _assert_tensor_metadata(self, node: fx.Node):
        # Export inserts these assertions for already typed internal values.
        # Physical storage assertions are outside Relax's tensor contract.
        args = self.retrieve_args(node)
        value = args[0]
        requested = dict(node.kwargs)
        requested.update(zip(("size", "stride", "dtype"), node.args[1:]))
        if requested.get("stride") is not None or requested.get("layout") not in (
            None,
            torch.strided,
        ):
            raise NotImplementedError(
                "Tensor metadata assertion requires physical storage semantics"
            )
        if requested.get("size") is not None:
            actual = tuple(int(size) for size in self.shape_of(value))
            if actual != tuple(requested["size"]):
                raise ValueError("Tensor metadata size assertion disagrees with captured type")
        if requested.get("dtype") is not None:
            if value.struct_info.dtype != self._convert_data_type(requested["dtype"]):
                raise ValueError("Tensor metadata dtype assertion disagrees with captured type")
        if requested.get("device") is not None and str(requested["device"]) != "cpu":
            raise NotImplementedError("Tensor metadata assertion names a non-CPU source device")
        return None

    def _qdq(self, node: fx.Node, *, dequantize=False, channel=False):
        import math
        import numpy as np
        from ._quantization import (
            quantize_per_tensor,
            dequantize_per_tensor,
            dequantize_per_channel,
        )

        args = self.retrieve_args(node)
        data = args[0]
        if channel:
            if not dequantize:
                raise NotImplementedError("Per-channel runtime quantization is not yet qualified")
            scales, zeros, axis, low, high, source_dtype = args[1:]
            if scales in self._frozen_values:
                scales = self._convert_torch_tensor_to_relax(self._frozen_values[scales])
            if zeros in self._frozen_values:
                zeros = self._convert_torch_tensor_to_relax(self._frozen_values[zeros])
            if not isinstance(scales, relax.Constant) or not isinstance(zeros, relax.Constant):
                raise NotImplementedError(
                    "Channel Q/DQ requires frozen scale and zero-point tensors"
                )
            scale_values, zero_values = scales.data.numpy(), zeros.data.numpy()
            if not np.isfinite(scale_values).all() or not (scale_values > 0).all():
                raise ValueError("Channel scales must be finite and positive")
            if not ((zero_values >= low) & (zero_values <= high)).all():
                raise NotImplementedError("Channel zero points must lie in the quantized range")
            rank = len(self.shape_of(data))
            if not -rank <= axis < rank:
                raise ValueError("Channel Q/DQ axis is outside the input rank")
            axis %= rank
            if scales.data.shape != zeros.data.shape or len(scales.data.shape) != 1:
                raise ValueError("Channel Q/DQ parameters must be equal-size vectors")
            if int(self.shape_of(data)[axis]) != scales.data.shape[0]:
                raise ValueError("Channel Q/DQ parameter count disagrees with the tensor")
            implementation, parameters = dequantize_per_channel, (scales, zeros, axis)
        else:
            scale, zero, low, high, source_dtype = args[1:]
            if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
                raise ValueError("Tensor Q/DQ requires a finite positive static scale")
            if type(zero) is not int or not low <= zero <= high:
                raise ValueError("Tensor Q/DQ requires a zero point in the quantized range")
            implementation = dequantize_per_tensor if dequantize else quantize_per_tensor
            parameters = (
                (scale, zero)
                if dequantize
                else (1.0 / scale, zero, low, high, self._convert_data_type(source_dtype))
            )
        dtype = self._convert_data_type(source_dtype)
        if dtype not in ("int8", "uint8") or not -(1 << 7) <= low <= high <= 255:
            raise NotImplementedError("Q/DQ currently qualifies signed/unsigned byte codes")
        limits = np.iinfo(dtype)
        if low < limits.min or high > limits.max:
            raise ValueError("Q/DQ bounds exceed the declared code dtype")
        if dequantize:
            out_dtype = node.kwargs.get("out_dtype") or torch.float32
            if data.struct_info.dtype != dtype or out_dtype != torch.float32:
                raise NotImplementedError(
                    "Q/DQ currently requires matching byte input and FP32 output"
                )
            if not channel and scale == 1 and zero == 0:
                return self.block_builder.emit(relax.op.astype(data, "float32"))
        elif data.struct_info.dtype not in ("float16", "float32"):
            raise NotImplementedError("Q/DQ currently qualifies FP16/FP32 quantization input")
        return self.block_builder.emit_te(implementation, data, *parameters)

    def _frozen_array(self, value):
        """Read a source constant for an exactness proof, never a runtime input."""
        visited = set()
        while isinstance(value, relax.Var) and value not in visited:
            visited.add(value)
            frozen = self._frozen_values.get(value)
            if frozen is not None:
                value = frozen
                break
            value = self.block_builder.lookup_binding(value)
        if isinstance(value, relax.Constant):
            return value.data.numpy()
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return None

    def _integer_bias(self, bias, depth, weight_codes):
        """Commute an integral frozen FP32 bias only under an exact prefix bound."""
        import numpy as np
        from tvm import tir

        values = self._frozen_array(bias)
        if (
            values is None
            or values.dtype != np.float32
            or not values.size
            or not np.isfinite(values).all()
            or not np.equal(values, np.rint(values)).all()
            or not isinstance(depth, tir.IntImm)
        ):
            return None
        bound = int(depth) * 128 * 128
        weights = self._frozen_array(weight_codes)
        if weights is not None and weights.dtype == np.int8 and weights.ndim >= 2:
            bound = 128 * int(
                np.abs(weights.astype("int64")).reshape(weights.shape[0], -1).sum(axis=1).max()
            )
        if bound + float(np.max(np.abs(values))) > (1 << 24):
            return None
        return self.block_builder.emit(relax.op.astype(bias, "int32"))

    def _integer_carrier_pair(self, lhs, rhs, depth):
        """Prove exact FP32 contraction of signed-byte integer carriers.

        This target-independent identity changes no scales or rounding policy.
        Every possible prefix must fit FP32's exact integer range, including
        arbitrary reassociation/FMA. Larger or unproved cases stay floating.
        """
        from tvm import tir

        if not isinstance(depth, tir.IntImm) or int(depth) <= 0:
            return None
        codes = []
        for value in (lhs, rhs):
            binding = (
                self.block_builder.lookup_binding(value) if isinstance(value, relax.Var) else value
            )
            if not (
                isinstance(binding, relax.Call)
                and binding.op == tvm.ir.Op.get("relax.astype")
                and binding.attrs.dtype == "float32"
                and binding.args[0].struct_info.dtype == "int8"
            ):
                return None
            codes.append(binding.args[0])
        if int(depth) > (1 << 24) // (128 * 128):
            # Learned Linear/Conv weights are output-major. A frozen right
            # operand supplies a tighter, order-independent bound for every
            # output and every reduction prefix: 128 * sum(abs(weight)).
            # This is a proof from input constants, not a changed recipe.
            import numpy as np

            weights = self._frozen_array(codes[1])
            if weights is None:
                return None
            if (
                weights.dtype != np.int8
                or weights.ndim < 2
                or int(np.prod(weights.shape[1:])) != int(depth)
            ):
                return None
            bound = 128 * np.abs(weights.astype("int64")).reshape(weights.shape[0], -1).sum(axis=1)
            if not len(bound) or int(bound.max()) > (1 << 24):
                return None
        return codes

    def _linear(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        lhs, rhs = args[:2]
        bias = args[2] if len(args) > 2 else None
        codes = self._integer_carrier_pair(lhs, rhs, self.shape_of(lhs)[-1])
        if codes is None:
            return super()._linear(node)
        product = self.block_builder.emit(relax.op.linear(*codes, out_dtype="int32"))
        integer_bias = (
            self._integer_bias(bias, self.shape_of(lhs)[-1], codes[1]) if bias is not None else None
        )
        if integer_bias is not None:
            product = self.block_builder.emit(relax.op.add(product, integer_bias))
            bias = None
        result = self.block_builder.emit(relax.op.astype(product, "float32"))
        return result if bias is None else self.block_builder.emit(relax.op.add(result, bias))

    def _int_mm(self, node: fx.Node) -> relax.Var:
        """Preserve ATen's signed-byte product with an INT32 result.

        This is mathematical legalization, independent of accelerator selection.
        A type-derived prefix bound admits ordinary INT32 reduction; larger
        static reductions retain the source modulo-32 result on the CPU.
        """
        from tvm import tir

        lhs, rhs = self.retrieve_args(node)
        if any(value.struct_info.dtype != "int8" for value in (lhs, rhs)):
            raise ValueError("aten._int_mm requires signed INT8 operands")
        if any(len(self.shape_of(value)) != 2 for value in (lhs, rhs)):
            raise ValueError("aten._int_mm requires rank-two operands")
        if self._captured_dtype(node) != "int32":
            raise ValueError("aten._int_mm requires an INT32 result")
        depth = self.shape_of(lhs)[-1]
        if not isinstance(depth, tir.IntImm):
            raise NotImplementedError("aten._int_mm requires a static accumulation bound")
        if int(depth) == 0:
            shape = (self.shape_of(lhs)[0], self.shape_of(rhs)[1])
            return self.block_builder.emit(relax.op.zeros(shape, "int32"))
        if int(depth) * (1 << 14) <= (1 << 31) - 1:
            return self.block_builder.emit(relax.op.matmul(lhs, rhs, out_dtype="int32"))
        # The generic modular contraction's output dtype follows its operands.
        # Sign extension before that contraction preserves _int_mm's wide result.
        return self._matmul_values(
            self._numeric_operand(lhs, "int32"),
            self._numeric_operand(rhs, "int32"),
            "int32",
        )

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

    @staticmethod
    def _convolution_axes(value, dim, *, positive=True):
        values = (value,) * dim if isinstance(value, int) else tuple(value)
        if len(values) == 1:
            values *= dim
        if len(values) != dim or any(
            not isinstance(item, int) or item < (1 if positive else 0) for item in values
        ):
            raise ValueError("Convolution geometry requires static integers in its valid range")
        return values

    def _convolution(self, node: fx.Node, *, dim=None, transpose=False, direct=False):
        import math
        from tvm import tir, topi
        from ._convolution import modular_convolution

        args = self.retrieve_args(node)
        data, weight = args[:2]

        def argument(index, name, default):
            return args[index] if len(args) > index else node.kwargs.get(name, default)

        dim = len(self.shape_of(weight)) - 2 if dim is None else dim
        if dim not in (1, 2, 3):
            raise NotImplementedError("Convolution import requires one to three spatial axes")
        bias = argument(2, "bias", None)
        strides = self._convolution_axes(argument(3, "stride", 1), dim)
        padding = self._convolution_axes(argument(4, "padding", 0), dim, positive=False)
        if direct:
            dilation = self._convolution_axes(argument(5, "dilation", 1), dim)
            transpose = argument(6, "transposed", False)
            output_padding = self._convolution_axes(
                argument(7, "output_padding", 0), dim, positive=False
            )
            groups = argument(8, "groups", 1)
        elif transpose:
            output_padding = self._convolution_axes(
                argument(5, "output_padding", 0), dim, positive=False
            )
            groups = argument(6, "groups", 1)
            dilation = self._convolution_axes(argument(7, "dilation", 1), dim)
        else:
            dilation = self._convolution_axes(argument(5, "dilation", 1), dim)
            groups, output_padding = argument(6, "groups", 1), (0,) * dim
        if not isinstance(groups, int) or groups <= 0:
            raise ValueError("Convolution groups must be a positive static integer")
        unbatched = len(self.shape_of(data)) == dim + 1
        if unbatched:
            data = self.block_builder.emit(relax.op.expand_dims(data, 0))
        if len(self.shape_of(data)) != dim + 2:
            raise ValueError("Convolution input rank does not match its weight")
        dtype = self._captured_dtype(node)
        if dtype not in ("int8", "int32", "int64", "float16", "float32", "float64"):
            raise NotImplementedError("Convolution dtype is outside the admitted numeric contract")
        if data.struct_info.dtype != dtype or weight.struct_info.dtype != dtype:
            raise ValueError("Convolution data, weight and captured result must have one dtype")
        data_shape, weight_shape = tuple(self.shape_of(data)), tuple(self.shape_of(weight))
        out_channels = weight_shape[1] * groups if transpose else weight_shape[0]
        spatial = tuple(
            (size - 1) * stride - 2 * pad + dil * (kernel - 1) + 1 + extra
            if transpose
            else (size + 2 * pad - dil * (kernel - 1) - 1) // stride + 1
            for size, kernel, stride, pad, dil, extra in zip(
                data_shape[2:], weight_shape[2:], strides, padding, dilation, output_padding
            )
        )
        shape = data_shape[:1] + (out_channels,) + spatial
        compute_dtype = "float32" if dtype == "float16" else dtype
        integer_carriers = None
        if dtype == "float32" and not transpose:
            extents = weight_shape[2:] + (data_shape[1] // groups,)
            if all(isinstance(size, tir.IntImm) for size in extents):
                depth = tir.IntImm("int64", math.prod(int(size) for size in extents))
                integer_carriers = self._integer_carrier_pair(data, weight, depth)
                if integer_carriers is not None:
                    compute_dtype = "int32"
        if dtype == "int8":
            extents = weight_shape[2:] + (data_shape[1] // groups,)
            if not all(isinstance(size, tir.IntImm) for size in extents):
                raise NotImplementedError("INT8 convolution requires a static accumulation bound")
            magnitude = 1 << (tvm.DataType(dtype).bits - 1)
            bound = math.prod(int(size) for size in extents) * magnitude**2 + (
                magnitude if bias is not None else 0
            )
            if bound > (1 << 63) - 1:
                raise NotImplementedError("INT8 convolution exceeds the proven INT64 prefix bound")
            compute_dtype = "int32" if bound <= (1 << 31) - 1 else "int64"
        if dtype in ("int32", "int64"):
            if not all(isinstance(size, tir.IntImm) for size in data_shape + weight_shape):
                raise NotImplementedError("Modular convolution requires static tensor extents")
            result = self.block_builder.emit_te(
                modular_convolution,
                data,
                weight,
                shape,
                strides,
                padding,
                dilation,
                groups,
                transpose,
                primfunc_name_hint="modular_convolution",
            )
        else:
            lhs, rhs = integer_carriers if integer_carriers is not None else (data, weight)
            if dtype == "float16":
                lhs = self.block_builder.emit(relax.op.astype(lhs, compute_dtype))
                rhs = self.block_builder.emit(relax.op.astype(rhs, compute_dtype))
            if transpose and (
                dim == 3 or any(extra >= stride for extra, stride in zip(output_padding, strides))
            ):
                result = self.block_builder.emit_te(
                    topi.nn.group_conv_transpose_ncx,
                    lhs,
                    rhs,
                    strides,
                    padding * 2,
                    output_padding,
                    dilation,
                    groups,
                    compute_dtype,
                    primfunc_name_hint=f"conv{dim}d_transpose",
                )
            else:
                op = getattr(relax.op.nn, f"conv{dim}d" + ("_transpose" if transpose else ""))
                kwargs = {"output_padding": output_padding} if transpose else {}
                result = self.block_builder.emit(
                    op(
                        lhs,
                        rhs,
                        strides=strides,
                        padding=padding * 2,
                        dilation=dilation,
                        groups=groups,
                        out_dtype=compute_dtype,
                        **kwargs,
                    )
                )
        if integer_carriers is not None:
            integer_bias = (
                self._integer_bias(bias, depth, integer_carriers[1]) if bias is not None else None
            )
            if integer_bias is not None:
                bias = integer_bias
            else:
                # Unproved/fractional bias keeps the source FP32 addition.
                result = self.block_builder.emit(relax.op.astype(result, "float32"))
                compute_dtype = "float32"
        if bias is not None:
            bias = self.block_builder.emit(relax.op.reshape(bias, (1, -1) + (1,) * dim))
            if dtype in ("int32", "int64"):
                result = self._numeric_values(result, bias, dtype, "add")
            else:
                if bias.struct_info.dtype != compute_dtype:
                    bias = self.block_builder.emit(relax.op.astype(bias, compute_dtype))
                result = self.block_builder.emit(relax.op.add(result, bias))
        if result.struct_info.dtype != dtype:
            result = self.block_builder.emit(relax.op.astype(result, dtype))
        if unbatched:
            result = self.block_builder.emit(relax.op.squeeze(result, 0))
        return result

    def _convolution_backward(self, node: fx.Node):
        from tvm import tir
        from tvm.relax.op._op_gradient import conv2d_grad
        from ._convolution import convolution_input_gradient, convolution_weight_gradient

        args = self.retrieve_args(node)
        gradient, data, weight = args[:3]
        dim = len(self.shape_of(weight)) - 2
        if dim not in (1, 2, 3) or len(self.shape_of(data)) != dim + 2:
            raise NotImplementedError(
                "Convolution backward requires batched one/two/three-dimensional inputs"
            )
        dtype = data.struct_info.dtype
        if dtype not in ("float32", "float64"):
            raise NotImplementedError(
                "Convolution backward currently admits FP32/FP64 CPU contracts; integer gradients "
                "are not source CPU operations and FP16 backward is unqualified on this CPU build"
            )
        if weight.struct_info.dtype != dtype or gradient.struct_info.dtype != dtype:
            raise ValueError("Convolution gradients, data and weight must have one dtype")
        strides = self._convolution_axes(args[4], dim)
        padding = self._convolution_axes(args[5], dim, positive=False)
        dilation = self._convolution_axes(args[6], dim)
        transpose, groups, mask = args[7], args[9], tuple(args[10])
        data_shape, weight_shape = tuple(self.shape_of(data)), tuple(self.shape_of(weight))
        if not all(
            isinstance(size, tir.IntImm)
            for size in data_shape + weight_shape + tuple(self.shape_of(gradient))
        ):
            raise NotImplementedError(
                "Convolution adjoint admission currently requires static tensor extents"
            )
        results = [None, None, None]
        if int(data_shape[0]) == 0:
            shapes = (data_shape, weight_shape, (self.shape_of(gradient)[1],))
            return tuple(
                self.block_builder.emit(relax.op.zeros(shape, dtype)) if requested else None
                for shape, requested in zip(shapes, mask)
            )
        if not transpose and dim < 3 and (mask[0] or mask[1]):
            lhs, rhs, grad = data, weight, gradient
            s2, p2, d2 = strides, padding, dilation
            if dim == 1:
                lhs = self.block_builder.emit(relax.op.expand_dims(lhs, 2))
                rhs = self.block_builder.emit(relax.op.expand_dims(rhs, 2))
                grad = self.block_builder.emit(relax.op.expand_dims(grad, 2))
                s2, p2, d2 = (1,) + strides, (0,) + padding, (1,) + dilation
            original = self.block_builder.emit(
                relax.op.nn.conv2d(
                    lhs,
                    rhs,
                    strides=s2,
                    padding=p2 * 2,
                    dilation=d2,
                    groups=groups,
                    out_dtype=dtype,
                )
            )
            mathematical = conv2d_grad(
                original, self.block_builder.lookup_binding(original), grad, self.block_builder
            )
            for index in (0, 1):
                if mask[index]:
                    value = self.block_builder.emit(mathematical[index])
                    results[index] = (
                        self.block_builder.emit(relax.op.squeeze(value, 2)) if dim == 1 else value
                    )
        else:
            if mask[0]:
                results[0] = self.block_builder.emit_te(
                    convolution_input_gradient,
                    gradient,
                    weight,
                    data_shape,
                    strides,
                    padding,
                    dilation,
                    groups,
                    transpose,
                    primfunc_name_hint="convolution_input_gradient",
                )
            if mask[1]:
                results[1] = self.block_builder.emit_te(
                    convolution_weight_gradient,
                    data,
                    gradient,
                    weight_shape,
                    strides,
                    padding,
                    dilation,
                    groups,
                    transpose,
                    primfunc_name_hint="convolution_weight_gradient",
                )
        if mask[2]:
            results[2] = self.block_builder.emit(
                relax.op.sum(gradient, axis=[0] + list(range(2, dim + 2)))
            )
        return tuple(results)

    def _pool_input(self, node, dim):
        x = self.retrieve_args(node)[0]
        rank = len(self.shape_of(x))
        if rank not in (dim + 1, dim + 2):
            raise ValueError("Pooling input rank does not match its spatial dimensions")
        unbatched = rank == dim + 1
        if unbatched:
            x = self.block_builder.emit(relax.op.expand_dims(x, 0))
        dtype = x.struct_info.dtype
        if dtype == "float16":
            x = self.block_builder.emit(relax.op.astype(x, "float32"))
        return x, dtype, unbatched

    def _average_pool(self, node: fx.Node, *, dim: int) -> relax.Var:
        from tvm import tir
        import math
        from ._pooling import average_pool, output_extent

        args = self.retrieve_args(node)
        x, dtype, unbatched = self._pool_input(node, dim)

        def argument(index, name, default):
            return args[index] if len(args) > index else node.kwargs.get(name, default)

        def dimensions(value):
            result = [value] * dim if isinstance(value, int) else list(value)
            result = result * dim if len(result) == 1 else result
            if len(result) != dim or any(not isinstance(v, int) for v in result):
                raise ValueError("Pooling parameters require one integer per spatial axis")
            return tuple(result)

        kernel = dimensions(argument(1, "kernel_size", None))
        stride_arg = argument(2, "stride", None)
        strides = kernel if stride_arg is None or stride_arg == [] else dimensions(stride_arg)
        padding = dimensions(argument(3, "padding", 0))
        ceil = argument(4, "ceil_mode", False)
        include = argument(5, "count_include_pad", True)
        divisor = argument(6, "divisor_override", None) if dim > 1 else None
        if any(k <= 0 or s <= 0 or p < 0 for k, s, p in zip(kernel, strides, padding)):
            raise ValueError("Pooling requires positive kernels/strides and nonnegative padding")
        if divisor is not None and (not isinstance(divisor, int) or divisor == 0):
            raise ValueError("Average pooling divisor must be a nonzero static integer")
        if dtype == "int64" or (divisor is not None and (ceil or divisor != math.prod(kernel))):
            result = self.block_builder.emit_te(
                average_pool,
                x,
                kernel,
                strides,
                padding,
                ceil,
                include,
                divisor,
                primfunc_name_hint="average_pool",
            )
        else:
            op = getattr(relax.op.nn, f"avg_pool{dim}d")
            result = self.block_builder.emit(
                op(
                    x,
                    pool_size=kernel,
                    strides=strides,
                    padding=padding,
                    ceil_mode=ceil,
                    count_include_pad=True if divisor is not None else include,
                    layout={1: "NCW", 2: "NCHW", 3: "NCDHW"}[dim],
                )
            )
            if ceil:
                spatial = tuple(self.shape_of(x))[-dim:]
                if not all(isinstance(size, tir.IntImm) for size in spatial):
                    raise NotImplementedError(
                        "Ceil-mode pooling requires static spatial extents for the ATen final-window rule"
                    )
                desired = [
                    output_extent(int(size), k, s, p, True)
                    for size, k, s, p in zip(spatial, kernel, strides, padding)
                ]
                actual = tuple(self.shape_of(result))[-dim:]
                if any(
                    not isinstance(size, tir.IntImm) or int(size) != extent
                    for size, extent in zip(actual, desired)
                ):
                    result = self.block_builder.emit(
                        relax.op.strided_slice(
                            result, list(range(2, dim + 2)), [0] * dim, desired, [1] * dim
                        )
                    )
        if dtype == "float16":
            result = self.block_builder.emit(relax.op.astype(result, dtype))
        if unbatched:
            result = self.block_builder.emit(relax.op.squeeze(result, 0))
        return result

    def _adaptive_average_pool(self, node: fx.Node, *, dim: int) -> relax.Var:
        args = self.retrieve_args(node)
        x, dtype, unbatched = self._pool_input(node, dim)
        if not dtype.startswith("float"):
            raise NotImplementedError(
                "Adaptive average pooling import requires a floating CPU contract"
            )
        size = args[1] if len(args) > 1 else node.kwargs.get("output_size")
        if isinstance(size, int):
            size = (size,) * dim
        elif size is not None:
            size = tuple(
                extent if value is None else value
                for value, extent in zip(size, tuple(self.shape_of(x))[-dim:])
            )
        result = self.block_builder.emit(
            getattr(relax.op.nn, f"adaptive_avg_pool{dim}d")(
                x,
                size,
                layout={1: "NCW", 2: "NCHW", 3: "NCDHW"}[dim],
            )
        )
        if dtype == "float16":
            result = self.block_builder.emit(relax.op.astype(result, dtype))
        if unbatched:
            result = self.block_builder.emit(relax.op.squeeze(result, 0))
        return result

    @staticmethod
    def _window_pair(value, *, positive=True):
        values = (value, value) if isinstance(value, int) else tuple(value)
        if len(values) == 1:
            values = values * 2
        if len(values) != 2 or any(
            not isinstance(item, int) or item < (1 if positive else 0) for item in values
        ):
            raise ValueError("Window geometry requires two static integers in its valid range")
        return values

    def _pool_backward(self, node: fx.Node, *, adaptive=False) -> relax.Var:
        from ._overlap import adaptive_pool_backward, average_pool_backward

        args = self.retrieve_args(node)
        gradient, data = args[:2]
        shape = self.shape_of(data)
        unbatched = len(shape) == 3
        if len(shape) not in (3, 4) or len(self.shape_of(gradient)) != len(shape):
            raise ValueError("Pool backward requires matching batched or unbatched 2D ranks")
        dtype = data.struct_info.dtype
        if gradient.struct_info.dtype != dtype:
            raise ValueError("Pool gradient dtype must match its input")
        allowed = ("float16", "float32", "float64") + (() if adaptive else ("int64",))
        if dtype not in allowed:
            raise NotImplementedError("Pool backward dtype is outside the admitted CPU contract")
        if unbatched:
            gradient = self.block_builder.emit(relax.op.expand_dims(gradient, 0))
            data = self.block_builder.emit(relax.op.expand_dims(data, 0))
            shape = self.shape_of(data)
        if adaptive:
            result = self.block_builder.emit_te(
                adaptive_pool_backward,
                gradient,
                tuple(shape),
                primfunc_name_hint="adaptive_pool_backward",
            )
        else:

            def argument(index, name, default):
                return args[index] if len(args) > index else node.kwargs.get(name, default)

            kernel = self._window_pair(argument(2, "kernel_size", None))
            stride_arg = argument(3, "stride", [])
            strides = (
                kernel
                if stride_arg is None or stride_arg == [] or stride_arg == ()
                else self._window_pair(stride_arg)
            )
            padding = self._window_pair(argument(4, "padding", 0), positive=False)
            ceil = argument(5, "ceil_mode", False)
            include = argument(6, "count_include_pad", True)
            divisor = argument(7, "divisor_override", None)
            if divisor is not None and (not isinstance(divisor, int) or divisor == 0):
                raise ValueError("Average-pool backward divisor must be a nonzero static integer")
            if dtype in ("float32", "float64") and divisor is None:
                result = self.block_builder.emit(
                    relax.op.grad.avg_pool2d_backward(
                        gradient,
                        data,
                        kernel,
                        strides,
                        padding * 2,
                        ceil_mode=ceil,
                        count_include_pad=include,
                    )
                )
            else:
                result = self.block_builder.emit_te(
                    average_pool_backward,
                    gradient,
                    tuple(shape),
                    kernel,
                    strides,
                    padding,
                    include,
                    divisor,
                    primfunc_name_hint="average_pool_backward",
                )
        if unbatched:
            result = self.block_builder.emit(relax.op.squeeze(result, 0))
        return result

    def _col2im(self, node: fx.Node) -> relax.Var:
        from ._overlap import col2im

        args = self.retrieve_args(node)
        data = args[0]
        if len(self.shape_of(data)) not in (2, 3):
            raise ValueError("Col2im requires a batched or unbatched column matrix")
        if data.struct_info.dtype not in ("float16", "float32", "float64", "bool"):
            raise NotImplementedError("Col2im dtype is outside the admitted CPU contract")

        def argument(index, name, default):
            return args[index] if len(args) > index else node.kwargs.get(name, default)

        return self.block_builder.emit_te(
            col2im,
            data,
            self._window_pair(argument(1, "output_size", None)),
            self._window_pair(argument(2, "kernel_size", None)),
            self._window_pair(argument(3, "dilation", 1)),
            self._window_pair(argument(4, "padding", 0), positive=False),
            self._window_pair(argument(5, "stride", 1)),
            primfunc_name_hint="col2im",
        )

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

    def _max_pool2d_with_unused_indices(self, node: fx.Node):
        """max_pool2d_with_indices whose indices output is dead: lower the values only.

        Core-ATen export decomposes max_pool2d into this op. Refuse if any consumer reads the
        indices rather than inventing them.
        """
        for user in node.users:
            reads_values = user.target is operator.getitem and user.args[1] == 0
            if not reads_values and len(user.users):
                raise ValueError("max_pool2d_with_indices: the indices output is consumed; unsupported")
        return (self._max_pool2d(node), None)

    def _softmax_no_half_to_float(self, node: fx.Node):
        """aten._softmax(x, dim, half_to_float): the ordinary softmax when half_to_float is False."""
        if len(node.args) > 2 and node.args[2]:
            raise ValueError("_softmax with half_to_float=True is unsupported")
        return self._softmax(node)

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
            "_int_mm.default": self._int_mm,
            "amax.default": self._amax,
            "max.other": self._binary_op(relax.op.maximum, max),
            "mul.Tensor": partial(self._numeric_binary, operation="multiply"),
            "mul.Scalar": partial(self._numeric_binary, operation="multiply"),
            "pow.Tensor_Scalar": self._binary_op(relax.op.power, operator.pow),
            "pow.Tensor_Tensor": self._binary_op(relax.op.power, operator.pow),
            "sub.Tensor": partial(self._numeric_binary, operation="subtract"),
            "sub.Scalar": partial(self._numeric_binary, operation="subtract"),
            # neural network
            "_native_batch_norm_legit_no_training.default": self._batch_norm_legit_no_training,
            "batch_norm.default": self._batch_norm_inference,
            "adaptive_avg_pool1d.default": partial(self._adaptive_average_pool, dim=1),
            "adaptive_avg_pool2d.default": partial(self._adaptive_average_pool, dim=2),
            "adaptive_avg_pool3d.default": partial(self._adaptive_average_pool, dim=3),
            "_adaptive_avg_pool2d.default": partial(self._adaptive_average_pool, dim=2),
            "_adaptive_avg_pool3d.default": partial(self._adaptive_average_pool, dim=3),
            "addmm.default": self._addmm,
            "avg_pool1d.default": partial(self._average_pool, dim=1),
            "avg_pool2d.default": partial(self._average_pool, dim=2),
            "avg_pool3d.default": partial(self._average_pool, dim=3),
            "avg_pool2d_backward.default": self._pool_backward,
            "_adaptive_avg_pool2d_backward.default": partial(self._pool_backward, adaptive=True),
            "col2im.default": self._col2im,
            "baddbmm.default": self._baddbmm,
            "bmm.default": self._matmul,
            "convolution.default": partial(self._convolution, direct=True),
            "convolution_backward.default": self._convolution_backward,
            "conv_transpose1d.default": partial(self._convolution, dim=1, transpose=True),
            "conv_transpose2d.input": partial(self._convolution, dim=2, transpose=True),
            "conv_transpose3d.input": partial(self._convolution, dim=3, transpose=True),
            "conv1d.default": partial(self._convolution, dim=1),
            "conv2d.default": partial(self._convolution, dim=2),
            "conv3d.default": partial(self._convolution, dim=3),
            "einsum.default": self._einsum,
            "embedding.default": lambda node: self._embedding_impl(
                self.env[node.args[1]], self.env[node.args[0]]
            ),
            "group_norm.default": self._group_norm,
            "layer_norm.default": self._layer_norm,
            "linear.default": self._linear,
            "rms_norm.default": self._rms_norm,
            "max_pool2d.default": self._max_pool2d,
            "max_pool2d_with_indices.default": self._max_pool2d_with_unused_indices,
            "_softmax.default": self._softmax_no_half_to_float,
            "arange.start_step": self._arange,
            "le.Tensor": self._binary_op(relax.op.less_equal, operator.le),
            "alias.default": lambda node: self.env[node.args[0]],
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
            "chunk.default": self._chunk,
            "stack.default": self._stack,
            "cumsum.default": self._cumsum,
            "diagonal.default": self._diagonal,
            "constant_pad_nd.default": partial(self._pad, mode="constant"),
            "expand.default": self._expand,
            "flatten.using_ints": self._flatten,
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
            "to.dtype": self._to_copy,
            "to.dtype_layout": self._to_copy,
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
            "where.self": self._where,
            "where.ScalarSelf": self._where,
            "where.ScalarOther": self._where,
            "_assert_tensor_metadata.default": self._assert_tensor_metadata,
            "quantize_per_tensor.default": self._qdq,
            "dequantize_per_tensor.default": partial(self._qdq, dequantize=True),
            "dequantize_per_channel.default": partial(self._qdq, dequantize=True, channel=True),
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
                        metadata = node.meta.get("tensor_meta", node.meta.get("val"))
                        if metadata is None or not hasattr(metadata, "shape"):
                            raise ValueError(
                                "Exported input lacks tensor shape/dtype metadata: " + name_hint
                            )
                        shape = metadata.shape
                        torch_dtype = metadata.dtype
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
        frozen = dict(exported_program.named_buffers()) | dict(exported_program.constants)
        if not keep_params_as_input:
            frozen.update(exported_program.named_parameters())
        self._frozen_values = {
            parameter_buffer_constant_vars[spec.arg.name]: frozen[spec.target]
            for spec in exported_program.graph_signature.input_specs
            if spec.target in frozen
        }
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
                        assert func_name in self.convert_map, (
                            f"Unsupported function type {func_name}"
                        )
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
            # Loaded export archives may expose storage views with offsets
            # below TVM's alignment guarantee. Own a contiguous Torch allocation
            # before DLPack transfer; preserve every dtype's bits, including BF16.
            binding[bind_name] = tvm.nd.from_dlpack(tensor_value.detach().contiguous().clone())

        mod = self.block_builder.get()
        mod = relax.transform.BindParams("main", binding)(mod)

        if keep_params_as_input:
            parameters = dict(exported_program.named_parameters())
            params = [
                tvm.nd.from_dlpack(p.detach().contiguous().clone()) for p in parameters.values()
            ]
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
