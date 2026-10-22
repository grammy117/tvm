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
"""Explicit small ATen contracts for the complete requested operator inventory.

This is verification input, never production compiler dispatch or cost data.
"""

import numpy as np

ARRAY_OPERATORS = {"mm", "bmm", "addmm", "convolution", "sum.dim_IntList"}
ACCUMULATOR_OPERATORS = {"add.Tensor", "add.Scalar", "sub.Scalar"}
MIXED_OPERATORS = {"convolution_backward", "avg_pool1d", "avg_pool2d", "avg_pool3d"}
CPU_OPERATORS = {
    "sub.Tensor",
    "mul.Scalar",
    "div.Scalar",
    "neg",
    "mean.dim",
    "mean.default",
    "adaptive_avg_pool1d",
    "_adaptive_avg_pool2d",
    "_adaptive_avg_pool3d",
    "avg_pool2d_backward",
    "_adaptive_avg_pool2d_backward",
    "col2im",
}
ELIDED_OPERATORS = {"clone", "fill.Scalar", "full", "full_like", "scalar_tensor"}

REQUESTED_OPERATORS = [
    "mm",
    "bmm",
    "addmm",
    "convolution",
    "convolution_backward",
    "add.Tensor",
    "sub.Tensor",
    "add.Scalar",
    "sub.Scalar",
    "mul.Scalar",
    "div.Scalar",
    "neg",
    "relu",
    "mean.dim",
    "avg_pool1d",
    "avg_pool2d",
    "avg_pool3d",
    "adaptive_avg_pool1d",
    "_adaptive_avg_pool2d",
    "_adaptive_avg_pool3d",
    "avg_pool2d_backward",
    "_adaptive_avg_pool2d_backward",
    "col2im",
    "clone",
    "repeat",
    "cat",
    "slice_scatter",
    "select_scatter",
    "fill.Scalar",
    "full",
    "full_like",
    "constant_pad_nd",
    "reflection_pad1d",
    "reflection_pad2d",
    "reflection_pad3d",
    "replication_pad2d",
    "replication_pad3d",
    "_to_copy",
    "as_strided",
    "diagonal",
    "permute",
    "scalar_tensor",
    "select.int",
    "slice.Tensor",
    "split_with_sizes",
    "squeeze.dim",
    "squeeze.dims",
    "unsqueeze",
    "view",
    "mean.default",
    "sum.dim_IntList",
]


def contract_model(name):
    import torch

    class Contract(torch.nn.Module):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def forward(self, x, y, w):
            name = self.name
            if name == "mm":
                return torch.ops.aten.mm.default(x, y)
            if name == "bmm":
                return torch.ops.aten.bmm.default(x, y)
            if name == "addmm":
                return torch.ops.aten.addmm.default(w, x, y)
            if name == "convolution":
                return torch.ops.aten.convolution.default(
                    x, y, w, [1, 1], [0, 0], [1, 1], False, [0, 0], 1
                )
            if name == "convolution_backward":
                return torch.ops.aten.convolution_backward.default(
                    x, y, w, [2], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]
                )
            if name == "add.Tensor":
                return torch.ops.aten.add.Tensor(x, y)
            if name == "sub.Tensor":
                return torch.ops.aten.sub.Tensor(x, y)
            if name == "add.Scalar":
                return torch.ops.aten.add.Scalar(x, 17)
            if name == "sub.Scalar":
                return torch.ops.aten.sub.Scalar(x, 17)
            if name == "mul.Scalar":
                return torch.ops.aten.mul.Scalar(x, -3)
            if name == "div.Scalar":
                return torch.ops.aten.div.Scalar(x, 3)
            if name == "neg":
                return torch.ops.aten.neg.default(x)
            if name == "relu":
                return torch.ops.aten.relu.default(x)
            if name == "mean.dim":
                return torch.ops.aten.mean.dim(x, [1], False)
            if name == "mean.default":
                return torch.ops.aten.mean.default(x)
            if name == "sum.dim_IntList":
                return torch.ops.aten.sum.dim_IntList(x, [1], False)
            if name.startswith("avg_pool") and not name.endswith("backward"):
                dim = int(name[-2])
                k = [1] * (dim - 1) + [3]
                s = [1] * (dim - 1) + [2]
                p = [0] * (dim - 1) + [1]
                if dim == 1:
                    return torch.ops.aten.avg_pool1d.default(x, k, s, p, False, True)
                return getattr(torch.ops.aten, name).default(x, k, s, p, False, True, None)
            if name in ("adaptive_avg_pool1d", "_adaptive_avg_pool2d", "_adaptive_avg_pool3d"):
                return getattr(torch.ops.aten, name).default(x, [2] * int(name[-2]))
            if name == "avg_pool2d_backward":
                return torch.ops.aten.avg_pool2d_backward.default(
                    x, y, [2, 2], [1, 1], [0, 0], False, True, None
                )
            if name == "_adaptive_avg_pool2d_backward":
                return torch.ops.aten._adaptive_avg_pool2d_backward.default(x, y)
            if name == "col2im":
                return torch.ops.aten.col2im.default(x, [3, 4], [2, 2], [1, 1], [0, 0], [1, 1])
            if name == "clone":
                return torch.ops.aten.clone.default(x)
            if name == "repeat":
                return torch.ops.aten.repeat.default(x, [2, 1])
            if name == "cat":
                return torch.ops.aten.cat.default([x, y], 0)
            if name == "slice_scatter":
                return torch.ops.aten.slice_scatter.default(x, y, 1, 1, 5, 2)
            if name == "select_scatter":
                return torch.ops.aten.select_scatter.default(x, y, 1, -1)
            if name == "fill.Scalar":
                return torch.ops.aten.fill.Scalar(x, -17)
            if name == "full":
                return torch.ops.aten.full.default([2, 3], -17, dtype=torch.int32)
            if name == "full_like":
                return torch.ops.aten.full_like.default(x, -17)
            if name == "constant_pad_nd":
                return torch.ops.aten.constant_pad_nd.default(x, [1, 1], -17)
            if name.startswith("reflection_pad") or name.startswith("replication_pad"):
                return getattr(torch.ops.aten, name).default(x, [1] * (2 * int(name[-2])))
            if name == "_to_copy":
                return torch.ops.aten._to_copy.default(x, dtype=torch.int32)
            if name == "as_strided":
                return torch.ops.aten.as_strided.default(x, [3, 2], [1, 3], None)
            if name == "diagonal":
                return torch.ops.aten.diagonal.default(x, 0, 0, 1)
            if name == "permute":
                return torch.ops.aten.permute.default(x, [1, 0])
            if name == "scalar_tensor":
                return torch.ops.aten.scalar_tensor.default(-0.0, dtype=torch.float32)
            if name == "select.int":
                return torch.ops.aten.select.int(x, 1, -1)
            if name == "slice.Tensor":
                return torch.ops.aten.slice.Tensor(x, 1, 1, 5, 2)
            if name == "split_with_sizes":
                return torch.ops.aten.split_with_sizes.default(x, [2, 3], 1)
            if name == "squeeze.dim":
                return torch.ops.aten.squeeze.dim(x, 0)
            if name == "squeeze.dims":
                return torch.ops.aten.squeeze.dims(x, [0, 2])
            if name == "unsqueeze":
                return torch.ops.aten.unsqueeze.default(x, 1)
            if name == "view":
                return torch.ops.aten.view.default(x, [3, 2])
            raise AssertionError(name)

    return Contract(name)


def arrays_for(name):
    dtype = "int8"
    shapes = [(2, 3), (2, 3), (3,)]
    if name in ("mm", "addmm"):
        shapes = [(2, 3), (3, 2), (2,)]
    elif name == "bmm":
        shapes = [(2, 2, 3), (2, 3, 2), (2,)]
    elif name == "convolution":
        shapes = [(1, 1, 3, 4), (2, 1, 2, 2), (2,)]
    elif name == "convolution_backward":
        dtype = "float32"
        shapes = [(1, 2, 2, 3), (1, 1, 3, 4), (2, 1, 2, 2)]
    elif name in (
        "add.Tensor",
        "sub.Tensor",
        "add.Scalar",
        "sub.Scalar",
        "mul.Scalar",
        "div.Scalar",
        "neg",
        "fill.Scalar",
        "full",
        "full_like",
    ):
        dtype = "int32"
    elif name in ("mean.dim", "mean.default", "scalar_tensor"):
        dtype = "float32"
    elif name.startswith("avg_pool") and not name.endswith("backward"):
        dtype = "float32"
        dim = int(name[-2])
        shapes[0] = (1, 1) + (1,) * max(dim - 2, 0) + ((2,) if dim > 1 else ()) + (4,)
    elif name in ("adaptive_avg_pool1d", "_adaptive_avg_pool2d", "_adaptive_avg_pool3d"):
        dtype = "float32"
        dim = int(name[-2])
        shapes[0] = (1, 1) + tuple(3 + i for i in range(dim))
    elif name in ("avg_pool2d_backward", "_adaptive_avg_pool2d_backward"):
        dtype = "float32"
        shapes = [
            (1, 1, 2, 3) if name == "avg_pool2d_backward" else (1, 1, 2, 2),
            (1, 1, 3, 4),
            (3,),
        ]
    elif name == "col2im":
        dtype = "float32"
        shapes[0] = (1, 4, 6)
    elif name == "cat":
        shapes[1] = (1, 3)
    elif name == "slice_scatter":
        dtype = "int32"
        shapes[:2] = [(2, 5), (2, 2)]
    elif name == "select_scatter":
        dtype = "float32"
        shapes[1] = (2,)
    elif name.startswith("reflection_pad") or name.startswith("replication_pad"):
        dim = int(name[-2])
        shapes[0] = (1, 1) + (2,) * (dim - 1) + (3,)
    elif name == "as_strided":
        dtype = "int32"
    elif name == "diagonal":
        shapes[0] = (3, 4)
    elif name == "select.int":
        dtype = "float32"
        shapes[0] = (2, 3, 4)
    elif name == "slice.Tensor":
        shapes[0] = (2, 5)
    elif name == "split_with_sizes":
        dtype = "int32"
        shapes[0] = (2, 5)
    elif name == "squeeze.dim":
        dtype = "int32"
        shapes[0] = (1, 2, 3)
    elif name == "squeeze.dims":
        dtype = "float32"
        shapes[0] = (1, 2, 1, 3)
    arrays = [
        (np.arange(np.prod(s)).reshape(s) + i - 3).astype(dtype) for i, s in enumerate(shapes)
    ]
    if name in (
        "add.Tensor",
        "sub.Tensor",
        "add.Scalar",
        "sub.Scalar",
        "mul.Scalar",
        "div.Scalar",
        "neg",
    ):
        arrays[0] = np.resize(
            np.array([-(1 << 31), (1 << 31) - 1, -1, 0, 1, 17], "int32"), shapes[0]
        )
    return arrays


def execution_kind(name):
    """Required execution for these exact runtime-input contracts and profile."""
    if name in ARRAY_OPERATORS:
        return "array"
    if name in ACCUMULATOR_OPERATORS:
        return "accumulator"
    if name == "relu":
        return "readout_relu"
    if name in MIXED_OPERATORS:
        return "movement_and_cpu"
    if name in CPU_OPERATORS:
        return "cpu"
    if name in ELIDED_OPERATORS:
        return "elided_or_abi_copy"
    return "movement"


def flatten(value):
    if hasattr(value, "numpy"):
        return [value.numpy()]
    return [array for item in value for array in flatten(item)]


def fixtures():
    """Yield verification tuples plus explicit source and execution metadata."""
    import torch
    from tvm.relax.frontend.torch import from_exported_program

    assert len(REQUESTED_OPERATORS) == len(set(REQUESTED_OPERATORS)) == 51
    for name in REQUESTED_OPERATORS:
        arrays = arrays_for(name)
        model = contract_model(name)
        args = tuple(torch.from_numpy(array) for array in arrays)
        expected = flatten(model(*args))
        exported = torch.export.export(model, args)
        graph = from_exported_program(exported, unwrap_unit_return_tuple=True)
        kind = execution_kind(name)
        placement = (
            "host"
            if kind in ("cpu", "elided_or_abi_copy")
            else "mixed"
            if kind == "movement_and_cpu"
            else "gemmini"
        )
        selected = 0 if placement == "host" else 2 if name == "convolution_backward" else 1
        metadata = {
            "requested_operator": name,
            "converter_key": name if "." in name else name + ".default",
            "source_ops": [
                str(node.target) for node in exported.graph.nodes if node.op == "call_function"
            ],
            "execution_kind": kind,
            "placement_policy": "explicit primitive capability exercise; scalar DMA is not production placement",
        }
        yield (
            "requested_" + name.replace(".", "_"),
            graph,
            arrays,
            expected[0] if len(expected) == 1 else tuple(expected),
            placement,
            {"offload_copies": True, "movement_strategy": "scalar"},
            selected,
            metadata,
        )
