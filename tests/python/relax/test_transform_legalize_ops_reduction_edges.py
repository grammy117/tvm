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
"""Execute scalar and empty reductions without target-specific scheduling."""

import warnings
import numpy as np
import pytest
import tvm
from tvm import relax


@pytest.mark.parametrize("kind,dtype", [("sum", "int64"), ("sum", "float32"), ("mean", "float32")])
@pytest.mark.parametrize(
    "shape,axes,keepdims",
    [
        ((0, 3), (0,), False),
        ((2, 0, 3), (0, 1), True),
        ((0, 3), (1,), False),
        ((), None, False),
        ((2, 3), (), True),
        ((2, 0), (), False),
    ],
)
def test_reduction_identity_and_empty_result(shape, axes, keepdims, kind, dtype):
    source = relax.Var("source", relax.TensorStructInfo(shape, dtype))
    builder = relax.BlockBuilder()
    with builder.function("main", [source]):
        with builder.dataflow():
            output = builder.emit_output(getattr(relax.op, kind)(source, axes, keepdims=keepdims))
        builder.emit_func_output(output)
    vm = relax.VirtualMachine(relax.build(builder.get(), "llvm"), tvm.cpu())
    values = (np.arange(int(np.prod(shape))).reshape(shape) - 3).astype(dtype)
    actual = vm["main"](tvm.nd.array(values)).numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expected = getattr(np, kind)(values, axis=axes, keepdims=keepdims)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    if dtype == "float32":
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0, equal_nan=True)
    else:
        np.testing.assert_array_equal(actual, expected)
