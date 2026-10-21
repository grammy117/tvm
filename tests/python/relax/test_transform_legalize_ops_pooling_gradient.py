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
"""Average-pool adjoints against an independently constructed window Jacobian."""

import numpy as np
import pytest
import tvm
from tvm import relax


@pytest.mark.parametrize("layout", ["NCHW", "NHWC"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("ceil", [False, True])
@pytest.mark.parametrize("include", [False, True])
@pytest.mark.parametrize("padding", [(1, 1, 1, 1), (1, 0, 0, 1)])
def test_average_pool_gradient_clips_ceil_windows_to_declared_padding(
    layout, dtype, ceil, include, padding
):
    shape = (1, 2, 5, 6)
    kernel, strides = (3, 3), (2, 2)
    output_shape = shape[:2] + tuple(
        (size + before + after - k + (stride - 1 if ceil else 0)) // stride + 1
        for size, before, after, k, stride in zip(
            shape[-2:], padding[:2], padding[2:], kernel, strides
        )
    )
    values = ((np.arange(np.prod(shape)).reshape(shape) % 31 - 17) / 4).astype(dtype)
    grad = ((np.arange(np.prod(output_shape)).reshape(output_shape) % 19 - 11) / 4).astype(dtype)
    expected = np.zeros(shape, dtype)
    forward = np.empty(output_shape, dtype)
    for y in range(output_shape[-2]):
        for x in range(output_shape[-1]):
            start_y, start_x = y * 2 - padding[0], x * 2 - padding[1]
            end_y = min(start_y + 3, shape[-2] + padding[2])
            end_x = min(start_x + 3, shape[-1] + padding[3])
            valid_y = list(range(max(start_y, 0), min(end_y, shape[-2])))
            valid_x = list(range(max(start_x, 0), min(end_x, shape[-1])))
            divisor = (
                (end_y - start_y) * (end_x - start_x) if include else len(valid_y) * len(valid_x)
            )
            assert divisor > 0
            forward[:, :, y, x] = 0
            delta = grad[:, :, y, x] / divisor
            for iy in valid_y:
                for ix in valid_x:
                    expected[:, :, iy, ix] += delta
                    forward[:, :, y, x] += values[:, :, iy, ix] / divisor
    if layout == "NHWC":
        values, grad, expected = [
            np.transpose(array, (0, 2, 3, 1)).copy() for array in (values, grad, expected)
        ]
    bb = relax.BlockBuilder()
    gv = relax.Var("gradient", relax.TensorStructInfo(grad.shape, dtype))
    xv = relax.Var("input", relax.TensorStructInfo(values.shape, dtype))
    with bb.function("main", [gv, xv]):
        with bb.dataflow():
            result = bb.emit_output(
                relax.op.grad.avg_pool2d_backward(
                    gv,
                    xv,
                    kernel,
                    strides,
                    padding,
                    ceil_mode=ceil,
                    count_include_pad=include,
                    layout=layout,
                )
            )
        bb.emit_func_output(result)
    vm = relax.VirtualMachine(relax.build(bb.get(), "llvm"), tvm.cpu())
    actual = vm["main"](tvm.nd.array(grad), tvm.nd.array(values)).numpy()
    rtol = 1e-6 if dtype == "float32" else 1e-12
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=rtol)
    # The adjoint must preserve the inner product for arbitrary upstream values.
    lhs = np.sum(
        forward.astype("float64") * np.moveaxis(grad, -1, 1)
        if layout == "NHWC"
        else forward.astype("float64") * grad
    )
    rhs = np.sum(values.astype("float64") * actual)
    np.testing.assert_allclose(lhs, rhs, rtol=rtol, atol=rtol)
