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
"""Compile structural TensorIR fixtures and run bounded Spike/Verilator checks.

Numerical references are independently computed with INT64 NumPy arithmetic.
The guest executes original TVM operators, checks every output and input guard,
and observes actual primitive calls. These runs establish functional evidence,
not hardware timing, model quality, or coverage of unimplemented operators.
"""

import argparse
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import time

from verify_baremetal_cpu import STARTUP, LINKER, identity
from verify_matmul import audit_instructions


TIMING_GUEST = r"""
static inline uint64_t dse_cycles(void) {
  uint64_t value;
  asm volatile("fence rw,rw; csrr %0,cycle" : "=r"(value) :: "memory");
  return value;
}
static char* dse_u64(char* out,uint64_t value) {
  char text[24]; unsigned used=0;
  do { text[used++]=(char)('0'+value%10); value/=10; } while(value);
  for(unsigned i=0;i<used/2;++i) {char a=text[i];text[i]=text[used-i-1];text[used-i-1]=a;}
  for(unsigned i=0;i<used;++i)*out++=text[i];return out;
}
static void dse_sample(unsigned stage,unsigned sample,uint64_t cycles) {
  char line[128];char* out=line;const char* text="DSE_CYCLES stage=";
  while(*text)*out++=*text++;out=dse_u64(out,stage);text=" sample=";
  while(*text)*out++=*text++;out=dse_u64(out,sample);text=" cycles=";
  while(*text)*out++=*text++;out=dse_u64(out,cycles);*out++='\n';*out=0;puts_htif(line);
}
"""


GUEST = r"""
#include <stdint.h>
#include <stddef.h>
volatile uint64_t tohost __attribute__((section(".tohost"),aligned(64)));
volatile uint64_t fromhost __attribute__((section(".tohost"),aligned(64)));
static void puts_htif(const char* text) {
  static volatile uint64_t request[4] __attribute__((aligned(64)));
  size_t n=0;while(text[n])++n;
  request[0]=64;request[1]=1;request[2]=(uintptr_t)text;request[3]=n;
  asm volatile("fence rw,rw":::"memory");tohost=(uintptr_t)request;
  while(!fromhost){}fromhost=0;asm volatile("fence rw,rw":::"memory");
}
void* memset(void* p,int c,size_t n) { volatile unsigned char* d=p;for(size_t i=0;i<n;++i)d[i]=c;return p; }
void* memcpy(void* d,const void* s,size_t n) { volatile unsigned char* o=d;const volatile unsigned char* p=s;for(size_t i=0;i<n;++i)o[i]=p[i];return d; }
int printf(const char* format,...) { (void)format;puts_htif("VENDOR_DIAGNOSTIC\n");return 0; }
__attribute__((noreturn)) void exit(int code) { tohost=((uint64_t)(code?code:99)<<1)|1;for(;;){} }
static unsigned computes,zeros,dmas,accumulates,acc_writes,relus,scaled_outs,pool_outs;
void __real_tvm_gemmini_dma_a(const int8_t*,uint32_t,uint32_t,uint32_t,int64_t,int32_t);
void __real_tvm_gemmini_dma_b(const int8_t*,uint32_t,uint32_t,uint32_t,int64_t,int32_t);
void __real_tvm_gemmini_dma_out(int32_t*,uint32_t,uint32_t,uint32_t,int64_t,int32_t);
void __wrap_tvm_gemmini_dma_a(const int8_t* p,uint32_t c,uint32_t m,uint32_t n,int64_t s,int32_t w) {
  ++dmas;__real_tvm_gemmini_dma_a(p,c,m,n,s,w);
}
void __wrap_tvm_gemmini_dma_b(const int8_t* p,uint32_t c,uint32_t m,uint32_t n,int64_t s,int32_t w) {
  ++dmas;__real_tvm_gemmini_dma_b(p,c,m,n,s,w);
}
void __wrap_tvm_gemmini_dma_out(int32_t* p,uint32_t c,uint32_t m,uint32_t n,int64_t s,int32_t w) {
  ++dmas;__real_tvm_gemmini_dma_out(p,c,m,n,s,w);
}
void __real_tvm_gemmini_dma_scaled_in(const int8_t*,uint32_t,uint32_t,uint32_t,int64_t,uint32_t,int32_t,int32_t);
void __wrap_tvm_gemmini_dma_scaled_in(const int8_t* p,uint32_t c,uint32_t m,uint32_t n,int64_t s,uint32_t scale,int32_t add,int32_t w) {
  ++dmas;if(add)++accumulates;
  __real_tvm_gemmini_dma_scaled_in(p,c,m,n,s,scale,add,w);
}
void __real_tvm_gemmini_dma_pool_out(int8_t*,uint32_t,uint32_t,int32_t);
void __wrap_tvm_gemmini_dma_pool_out(int8_t* p,uint32_t c,uint32_t size,int32_t w) {
  ++dmas;++pool_outs;__real_tvm_gemmini_dma_pool_out(p,c,size,w);
}
void __real_tvm_gemmini_compute(uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,int32_t);
void __real_tvm_gemmini_zero_acc(uint32_t,uint32_t,uint32_t);
void __wrap_tvm_gemmini_compute(uint32_t a,uint32_t b,uint32_t c,uint32_t m,uint32_t n,uint32_t k,int32_t add) {
  ++computes;__real_tvm_gemmini_compute(a,b,c,m,n,k,add);
}
void __real_tvm_gemmini_compute_ws(uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,int32_t);
void __wrap_tvm_gemmini_compute_ws(uint32_t a,uint32_t b,uint32_t c,uint32_t m,uint32_t n,uint32_t k,int32_t add) {
  ++computes;__real_tvm_gemmini_compute_ws(a,b,c,m,n,k,add);
}
void __wrap_tvm_gemmini_zero_acc(uint32_t c,uint32_t m,uint32_t n) { ++zeros;__real_tvm_gemmini_zero_acc(c,m,n); }
void __real_tvm_gemmini_dma_acc_write(const int32_t*,uint32_t,uint32_t,uint32_t,int64_t,int32_t);
void __real_tvm_gemmini_dma_acc_add(const int32_t*,uint32_t,uint32_t,uint32_t,int64_t,int32_t);
void __real_tvm_gemmini_dma_widen(const int8_t*,uint32_t,uint32_t,uint32_t,int64_t,int32_t);
void __real_tvm_gemmini_dma_spad_out(int8_t*,uint32_t,uint32_t,uint32_t,int64_t,int32_t);
void __real_tvm_gemmini_dma_relu_out(int8_t*,uint32_t,uint32_t,uint32_t,int64_t,int32_t);
void __wrap_tvm_gemmini_dma_acc_write(const int32_t* p,uint32_t c,uint32_t m,uint32_t n,int64_t s,int32_t w) {
  ++dmas;++acc_writes;__real_tvm_gemmini_dma_acc_write(p,c,m,n,s,w);
}
void __wrap_tvm_gemmini_dma_acc_add(const int32_t* p,uint32_t c,uint32_t m,uint32_t n,int64_t s,int32_t w) {
  ++dmas;++accumulates;__real_tvm_gemmini_dma_acc_add(p,c,m,n,s,w);
}
void __wrap_tvm_gemmini_dma_widen(const int8_t* p,uint32_t c,uint32_t m,uint32_t n,int64_t s,int32_t w) {
  ++dmas;__real_tvm_gemmini_dma_widen(p,c,m,n,s,w);
}
void __wrap_tvm_gemmini_dma_spad_out(int8_t* p,uint32_t c,uint32_t m,uint32_t n,int64_t s,int32_t w) {
  ++dmas;__real_tvm_gemmini_dma_spad_out(p,c,m,n,s,w);
}
void __wrap_tvm_gemmini_dma_relu_out(int8_t* p,uint32_t c,uint32_t m,uint32_t n,int64_t s,int32_t w) {
  ++dmas;++relus;__real_tvm_gemmini_dma_relu_out(p,c,m,n,s,w);
}
void __real_tvm_gemmini_dma_scaled_out(int8_t*,uint32_t,uint32_t,uint32_t,int64_t,uint32_t,int32_t,int32_t);
void __wrap_tvm_gemmini_dma_scaled_out(int8_t* p,uint32_t c,uint32_t m,uint32_t n,int64_t s,uint32_t scale,int32_t relu,int32_t w) {
  ++dmas;++scaled_outs;if(relu)++relus;
  __real_tvm_gemmini_dma_scaled_out(p,c,m,n,s,scale,relu,w);
}
static uint64_t hash(const void* p,size_t n) {
  const volatile unsigned char* b=p;uint64_t h=1469598103934665603ULL;
  for(size_t i=0;i<n;++i)h=(h^b[i])*1099511628211ULL;return h;
}
"""


def fixtures():
    import numpy as np
    from tvm import relax, te, topi

    result = []
    for kind in (
        "native1d",
        "native2d",
        "native3d",
        "wide",
        "transpose_wide",
        "backward",
        "transpose_backward",
    ):
        import torch
        from tvm.relax.frontend.torch import from_exported_program

        native = kind.startswith("native")
        backward = "backward" in kind
        transpose = kind.startswith("transpose")
        dim = int(kind[-2]) if native else 3 if backward else 2
        dtype = "int8" if native else "float32" if backward else "int64"

        class Convolution(torch.nn.Module):
            def forward(self, x, w, bias, g):
                if backward:
                    return torch.ops.aten.convolution_backward.default(
                        g,
                        x,
                        w,
                        [g.shape[1]],
                        [1] * dim,
                        [0] * dim,
                        [1] * dim,
                        transpose,
                        [0] * dim,
                        1,
                        [True, True, True],
                    )
                return torch.ops.aten.convolution.default(
                    x,
                    w,
                    bias,
                    [1] * dim,
                    [0] * dim,
                    [1] * dim,
                    transpose,
                    [0] * dim,
                    1,
                )

        spatial = (2,) * (dim - 1) + (3,)
        shape = (1, 1) + spatial
        w_shape = (1, 2) + (2,) * dim if transpose else (2, 1) + (2,) * dim
        values = (np.arange(np.prod(shape)).reshape(shape) * 37 - 128).astype(dtype)
        weights = (np.arange(np.prod(w_shape)).reshape(w_shape) * 29 - 128).astype(dtype)
        if dtype == "int64":
            seed = np.array([-(1 << 63), (1 << 63) - 1, -17, 33], "int64")
            values, weights = np.resize(seed, shape), np.resize(seed[::-1], w_shape)
        elif backward:
            values = (np.arange(np.prod(shape)).reshape(shape) % 13 - 7).astype(dtype)
            weights = (np.arange(np.prod(w_shape)).reshape(w_shape) % 7 - 4).astype(dtype)
        bias = np.array([-17, 23], dtype)
        op = getattr(torch.nn.functional, f"conv{'_transpose' if transpose else ''}{dim}d")
        expected_forward = op(torch.from_numpy(values), torch.from_numpy(weights))
        g = (np.arange(expected_forward.numel()).reshape(expected_forward.shape) % 11 - 6).astype(
            dtype
        )
        arrays = [values, weights, bias, g]
        model = Convolution()
        args = tuple(torch.from_numpy(array) for array in arrays)
        expected = model(*args)
        expected = tuple(value.numpy() for value in expected) if backward else expected.numpy()
        mod = from_exported_program(torch.export.export(model, args), unwrap_unit_return_tuple=True)
        result.append(
            (f"convolution_{kind}", mod, arrays, expected, "gemmini" if native else "host")
        )
    for kind in ("average", "override", "adaptive", "integer", "holes", "col2im", "boolean"):
        import torch
        from tvm.relax.frontend.torch import from_exported_program

        class Overlap(torch.nn.Module):
            def forward(self, g, x):
                if kind in ("col2im", "boolean"):
                    return torch.ops.aten.col2im.default(g, [3, 4], [2, 2], [1, 1], [0, 0], [1, 1])
                if kind == "adaptive":
                    return torch.ops.aten._adaptive_avg_pool2d_backward.default(g, x)
                if kind == "holes":
                    return torch.ops.aten.avg_pool2d_backward.default(
                        g, x, [1], [3], [0], False, True, 1
                    )
                return torch.ops.aten.avg_pool2d_backward.default(
                    g,
                    x,
                    [3, 3],
                    [1, 1] if kind == "integer" else [2, 2],
                    [1, 1],
                    kind != "integer",
                    True,
                    5 if kind == "override" else 1 if kind == "integer" else None,
                )

        dtype = (
            "bool" if kind == "boolean" else "int64" if kind in ("integer", "holes") else "float32"
        )
        g_shape = (
            (1, 4, 6)
            if kind in ("col2im", "boolean")
            else (1, 1, 1, 2)
            if kind == "holes"
            else (1, 1, 3, 4)
            if kind == "integer"
            else (1, 1, 2, 3)
        )
        gradient = (np.arange(np.prod(g_shape)).reshape(g_shape) - 3).astype(dtype)
        if kind == "integer":
            gradient = np.resize(np.array([-(1 << 63), (1 << 63) - 1, -17, 33], "int64"), g_shape)
        elif kind == "average":
            gradient *= 18
        elif kind == "override":
            gradient *= 5
        elif kind == "adaptive":
            gradient *= 6
        elif kind == "col2im":
            gradient /= 4
        x_shape = (1, 1, 5, 6) if kind == "adaptive" else (1, 1, 3, 4)
        values = np.zeros(x_shape, dtype)
        model = Overlap()
        args = (torch.from_numpy(gradient), torch.from_numpy(values))
        mod = from_exported_program(torch.export.export(model, args), unwrap_unit_return_tuple=True)
        result.append((f"overlap_{kind}", mod, [gradient, values], model(*args).numpy(), "host"))
    for kind, dim in (
        ("avg", 1),
        ("avg", 2),
        ("avg", 3),
        ("crop", 2),
        ("override", 2),
        ("adaptive", 1),
        ("adaptive", 2),
        ("adaptive", 3),
        ("long", 2),
    ):
        import torch
        from tvm.relax.frontend.torch import from_exported_program

        class Pool(torch.nn.Module):
            def forward(self, x):
                if kind == "adaptive":
                    return getattr(torch.nn.functional, f"adaptive_avg_pool{dim}d")(x, (2,) * dim)
                kernel = (1,) * (dim - 1) + (2 if kind == "crop" else 3,)
                strides = (1,) * (dim - 1) + (2,)
                padding = (0,) * (dim - 1) + (1,)
                kwargs = {"divisor_override": 5} if kind == "override" else {}
                return getattr(torch.nn.functional, f"avg_pool{dim}d")(
                    x, kernel, strides, padding, ceil_mode=kind == "crop", **kwargs
                )

        spatial = (
            tuple(5 + i for i in range(dim))
            if kind == "adaptive"
            else (1,) * max(dim - 2, 0) + ((2,) if dim > 1 else ()) + (3 if kind == "crop" else 4,)
        )
        shape = (1, 1) + spatial
        values = (np.arange(np.prod(shape)).reshape(shape) - 12).astype(
            "int64" if kind == "long" else "float32"
        )
        if kind != "long":
            values /= 4
        model = Pool()
        example = torch.from_numpy(values)
        mod = from_exported_program(
            torch.export.export(model, (example,)), unwrap_unit_return_tuple=True
        )
        result.append(
            (
                f"pool_{kind}{dim}d",
                mod,
                [values],
                model(example).numpy(),
                "mixed" if kind in ("avg", "crop") else "host",
            )
        )
    for kind, dtype in (
        ("mm", "int8"),
        ("bmm", "int8"),
        ("addmm", "int8"),
        ("add", "int32"),
        ("scalar_add", "int32"),
        ("subtract", "int32"),
        ("multiply", "int32"),
        ("divide", "int32"),
        ("alpha", "float32"),
        ("int64_add", "int64"),
        ("int64_mm", "int64"),
    ):
        import torch
        from tvm.relax.frontend.torch import from_exported_program

        class Arithmetic(torch.nn.Module):
            def forward(self, x, y, bias):
                if kind in ("mm", "int64_mm"):
                    return torch.mm(x, y)
                if kind == "bmm":
                    return torch.bmm(x, y)
                if kind == "addmm":
                    return torch.addmm(bias, x, y)
                if kind in ("add", "int64_add"):
                    return torch.add(x, y)
                if kind == "scalar_add":
                    return torch.ops.aten.add.Scalar(x, 17)
                if kind == "subtract":
                    return torch.sub(x, y, alpha=3)
                if kind == "multiply":
                    return torch.ops.aten.mul.Scalar(x, -3)
                if kind == "divide":
                    return torch.ops.aten.div.Scalar(x, 3)
                return torch.add(x, y, alpha=1.1)

        shapes = (
            ((2, 3), (3, 4), (4,))
            if kind in ("mm", "addmm", "int64_mm")
            else (((2, 2, 3), (2, 3, 4), (4,)) if kind == "bmm" else ((2, 3), (2, 3), (3,)))
        )
        seed = (
            np.array([-3.75, 1.125, -0.0, 1, 0.25, -0.5], dtype)
            if dtype == "float32"
            else np.array([np.iinfo(dtype).min, np.iinfo(dtype).max, -1, 0, 1, 17], dtype)
        )
        values = [
            np.resize(seed if i % 2 == 0 else seed[::-1], shape) for i, shape in enumerate(shapes)
        ]
        examples = [torch.from_numpy(value) for value in values]
        model = Arithmetic()
        graph = from_exported_program(
            torch.export.export(model, tuple(examples)), unwrap_unit_return_tuple=True
        )
        result.append(
            (
                "aten_" + kind,
                graph,
                values,
                model(*examples).numpy(),
                "gemmini" if kind in ("mm", "bmm", "addmm", "add", "scalar_add") else "host",
            )
        )
    for kind in ("full", "full_like"):
        for dtype, fill in (("int8", -128), ("int32", -2147483648), ("float32", -0.0)):
            import torch
            from tvm.relax.frontend.torch import from_exported_program

            class Creation(torch.nn.Module):
                def forward(self, x):
                    return (
                        torch.full((2, 3), fill, dtype=getattr(torch, dtype))
                        if kind == "full"
                        else torch.full_like(x, fill)
                    )

            values = np.arange(6).reshape(2, 3).astype(dtype)
            example = torch.from_numpy(values)
            model = Creation()
            graph = from_exported_program(
                torch.export.export(model, (example,)), unwrap_unit_return_tuple=True
            )
            for optimized in (False, True):
                result.append(
                    (
                        f"creation_{kind}_{dtype}_{'folded' if optimized else 'native'}",
                        graph,
                        [values],
                        model(example).numpy(),
                        "host" if optimized else "gemmini",
                        {"optimize": optimized},
                    )
                )
    for name, shape in (("relu_i8", (16, 16)), ("relu_tail", (3, 17))):
        a = (np.arange(np.prod(shape)).reshape(shape) % 256 - 128).astype("int8")
        av = relax.Var("a", relax.TensorStructInfo(shape, "int8"))
        builder = relax.BlockBuilder()
        with builder.function("main", [av]):
            with builder.dataflow():
                output = builder.emit_output(relax.op.nn.relu(av))
            builder.emit_func_output(output)
        result.append((name, builder.get(), [a], np.maximum(a, 0)))
    bits = np.array(
        [
            0,
            0x80000000,
            0x7F800000,
            0xFF800000,
            0x7FC12345,
            0x7F812345,
            0xFFA00001,
            1,
            0x007FFFFF,
            0x00800000,
            0x3F800000,
            0xBF800000,
        ],
        "uint32",
    )
    for name, kind, dtype in (
        ("view_diagonal_i8", "diagonal", "int8"),
        ("view_as_strided_i32", "as_strided", "int32"),
        ("view_select_f32", "select", "float32"),
        ("view_permute_i8", "permute", "int8"),
        ("view_split_i32", "split", "int32"),
        ("view_squeeze_i32", "squeeze", "int32"),
        ("view_squeeze_dims_f32", "squeeze_dims", "float32"),
        ("view_unsqueeze_i8", "unsqueeze", "int8"),
        ("view_reshape_f32", "reshape", "float32"),
        ("view_slice_i8", "slice", "int8"),
        ("view_noop_i32", "noop", "int32"),
        ("view_scalar_f32", "scalar", "float32"),
    ):
        import torch
        from tvm.relax.frontend.torch import from_exported_program

        class View(torch.nn.Module):
            def forward(self, x):
                if kind == "diagonal":
                    return x.diagonal(1, 1, 2)
                if kind == "as_strided":
                    return torch.as_strided(x, (2, 2), (4, 1), 1)
                if kind == "select":
                    return x.select(1, -1)
                if kind == "permute":
                    return x.permute(2, 0, 1)
                if kind == "split":
                    return torch.split(x, [1, 2], dim=1)
                if kind == "squeeze":
                    return x.squeeze(0)
                if kind == "squeeze_dims":
                    return x.squeeze((0, 1))
                if kind == "unsqueeze":
                    return x.unsqueeze(0)
                if kind == "reshape":
                    return x.reshape(2, 6)
                if kind == "slice":
                    return torch.ops.aten.slice.Tensor(x, 1, 0, 3, 2)
                if kind == "noop":
                    return x.squeeze(1)
                return torch.scalar_tensor(-0.0)

        values = (
            bits.reshape(1, 3, 4).view("float32")
            if dtype == "float32"
            else (np.arange(12).reshape(1, 3, 4) * 37 - 128).astype(dtype)
        )
        example = torch.from_numpy(values)
        model = View()
        graph = from_exported_program(
            torch.export.export(model, (example,)), unwrap_unit_return_tuple=True
        )
        output = model(example)
        expected = tuple(x.numpy() for x in output) if isinstance(output, tuple) else output.numpy()
        result.append(
            (name, graph, [values], expected, "host" if kind in ("noop", "scalar") else "gemmini")
        )
    x = relax.Var("x", relax.TensorStructInfo((8,), "int32"))
    index = relax.Var("index", relax.TensorStructInfo((1,), "int64"))
    builder = relax.BlockBuilder()
    with builder.function("main", [x, index]):
        with builder.dataflow():
            viewed = builder.emit(relax.op.reshape(x, (2, 4)))
            output = builder.emit_output(relax.op.take(viewed, index, axis=0))
        builder.emit_func_output(output)
    values = np.arange(8, dtype="int32") * 13 - 128
    result.append(
        (
            "view_alias_i32",
            builder.get(),
            [values, np.array([1], "int64")],
            values.reshape(2, 4)[[1]],
            "host",
        )
    )
    for name, dtype, shape, sizes, axis in (
        ("cat_i8", "int8", (3, 3), (3, 4, 1), 1),
        ("cat_i32", "int32", (3, 3), (3, 2, 1), 0),
        ("cat_f32", "float32", (2, 1, 3), (1, 2, 1), 1),
    ):
        arrays = []
        for i, size in enumerate(sizes):
            dims = list(shape)
            dims[axis] = size
            arrays.append(
                np.resize(bits, dims).view("float32")
                if dtype == "float32"
                else (np.arange(np.prod(dims)).reshape(dims) * 37 + 19 * i - 128).astype(dtype)
            )
        variables = [
            relax.Var(f"input_{i}", relax.TensorStructInfo(a.shape, dtype))
            for i, a in enumerate(arrays)
        ]
        builder = relax.BlockBuilder()
        with builder.function("main", variables):
            with builder.dataflow():
                output = builder.emit_output(relax.op.concat(variables, axis))
            builder.emit_func_output(output)
        result.append((name, builder.get(), arrays, np.concatenate(arrays, axis=axis)))
    for name, dtype, shape, axis, start, stop, step, select in (
        ("scatter_i8", "int8", (3, 7), 1, 1, 7, 2, False),
        ("scatter_i32", "int32", (3, 7), 1, 2, 5, 1, False),
        ("scatter_f32", "float32", (7, 3), 0, 0, 7, 3, False),
        ("select_i32", "int32", (3, 7), 1, 6, 7, 1, True),
        ("select_f32", "float32", (3, 7), 0, 1, 2, 1, True),
    ):
        update_shape = list(shape)
        update_shape[axis] = len(range(start, stop, step))
        if select:
            update_shape.pop(axis)
        base = (np.arange(np.prod(shape)).reshape(shape) * 7 - 128).astype(dtype)
        updates = (
            np.resize(bits, update_shape).view("float32")
            if dtype == "float32"
            else (np.arange(np.prod(update_shape)).reshape(update_shape) * 17 - 128).astype(dtype)
        )
        variables = [
            relax.Var("base", relax.TensorStructInfo(shape, dtype)),
            relax.Var("updates", relax.TensorStructInfo(update_shape, dtype)),
        ]
        begin, end, strides = [0] * len(shape), list(shape), [1] * len(shape)
        begin[axis], end[axis], strides[axis] = start, stop, step

        def compute(data, source):
            return topi.strided_set(
                data,
                source,
                topi.utils.const_vector(np.array(begin, "int32")),
                topi.utils.const_vector(np.array(end, "int32")),
                topi.utils.const_vector(np.array(strides, "int32")),
            )

        builder = relax.BlockBuilder()
        with builder.function("main", variables):
            with builder.dataflow():
                source = (
                    builder.emit(relax.op.expand_dims(variables[1], axis))
                    if select
                    else variables[1]
                )
                output = builder.emit_output(builder.call_te(compute, variables[0], source))
            builder.emit_func_output(output)
        expected = base.copy()
        selection = [slice(None)] * len(shape)
        selection[axis] = start if select else slice(start, stop, step)
        expected[tuple(selection)] = updates
        result.append((name, builder.get(), [base, updates], expected))
    for name in ("copy_f32", "repeat_f32", "fill_f32"):
        a = (bits.reshape(3, 4) if name != "fill_f32" else np.array(0x7F812345, "uint32")).view(
            "float32"
        )
        av = relax.Var("a", relax.TensorStructInfo(a.shape, "float32"))
        builder = relax.BlockBuilder()
        if name == "copy_f32":
            source = te.placeholder(a.shape, "float32")
            copied = te.compute(a.shape, lambda *i: source[i], name="copied")
            gv = builder.add_func(te.create_prim_func([source, copied]), "copy")
        with builder.function("main", [av]):
            with builder.dataflow():
                output = builder.emit_output(
                    relax.call_tir(gv, (av,), relax.TensorStructInfo(a.shape, "float32"))
                    if name == "copy_f32"
                    else relax.op.tile(av, (2, 1))
                    if name == "repeat_f32"
                    else relax.op.full((3, 4), av, "float32")
                )
            builder.emit_func_output(output)
        expected_bits = (
            bits.reshape(3, 4)
            if name == "copy_f32"
            else np.tile(bits.reshape(3, 4), (2, 1))
            if name == "repeat_f32"
            else np.full((3, 4), 0x7F812345, "uint32")
        )
        result.append((name, builder.get(), [a], expected_bits.view("float32")))
    for name, shape, widths, dtype, mode in (
        ("reflect1d", (1, 1, 3), ((0, 0), (0, 0), (1, 1)), "int8", "reflect"),
        ("reflect2d", (1, 1, 2, 3), ((0, 0), (0, 0), (1, 0), (1, 0)), "int32", "reflect"),
        (
            "reflect3d",
            (1, 1, 2, 2, 3),
            ((0, 0), (0, 0), (1, 0), (1, 0), (1, 0)),
            "float32",
            "reflect",
        ),
        ("edge2d", (1, 1, 2, 3), ((0, 0), (0, 0), (0, 1), (2, 0)), "float32", "edge"),
        (
            "edge3d",
            (1, 1, 2, 2, 2),
            ((0, 0), (0, 0), (1, 0), (0, 1), (2, 0)),
            "int8",
            "edge",
        ),
    ):
        a = (
            np.resize(bits, shape).view("float32")
            if dtype == "float32"
            else (np.arange(np.prod(shape)).reshape(shape) * 17 - 128).astype(dtype)
        )
        av = relax.Var("a", relax.TensorStructInfo(shape, dtype))
        builder = relax.BlockBuilder()
        with builder.function("main", [av]):
            with builder.dataflow():
                output = builder.emit_output(
                    relax.op.nn.pad(av, tuple(x for pair in widths for x in pair), pad_mode=mode)
                )
            builder.emit_func_output(output)
        result.append((name, builder.get(), [a], np.pad(a, widths, mode=mode)))
    for name, dtype, value in (
        ("constant_i8", "int8", -128),
        ("constant_i32", "int32", 2147483647),
        ("constant_f32", "float32", -3.5),
        ("constant_zero", "float32", -0.0),
        ("constant_nan", "float32", float("-nan")),
        ("constant_crop", "int32", -7),
    ):
        shape = (3, 4)
        widths = ((-1, 1), (2, -1)) if name == "constant_crop" else ((1, 1), (2, 1))
        a = (
            bits.reshape(shape).view("float32")
            if dtype == "float32"
            else (np.arange(12).reshape(shape) * 17 - 128).astype(dtype)
        )
        av = relax.Var("a", relax.TensorStructInfo(shape, dtype))
        builder = relax.BlockBuilder()
        with builder.function("main", [av]):
            with builder.dataflow():
                output = builder.emit_output(
                    relax.op.nn.pad(av, tuple(x for pair in widths for x in pair), value)
                )
            builder.emit_func_output(output)
        expected = np.pad(
            a, [(max(left, 0), max(right, 0)) for left, right in widths], constant_values=value
        )
        expected = expected[
            tuple(
                slice(max(-left, 0), extent - max(-right, 0))
                for (left, right), extent in zip(widths, expected.shape)
            )
        ].copy()
        result.append((name, builder.get(), [a], expected))
    for name, shape, dims, keepdims, dtype in (
        ("sum64", (3, 17), (1,), False, None),
        ("sum64_axes", (2, 3, 4), (1, 2), True, None),
        ("sum64_scalar", (17,), None, False, None),
        ("sum32_export", (3, 17), (1,), False, "int32"),
        ("sum8_export", (3, 17), (1,), True, "int8"),
    ):
        import torch
        from tvm.relax.frontend.torch import from_exported_program

        class ExportedSum(torch.nn.Module):
            def forward(self, source):
                result_dtype = None if dtype is None else getattr(torch, dtype)
                if dims is None:
                    return source.sum(dtype=result_dtype)
                return source.sum(dim=dims, keepdim=keepdims, dtype=result_dtype)

        a = (np.arange(np.prod(shape)).reshape(shape) * 13 - 128).astype("int8")
        example = torch.from_numpy(a)
        model = ExportedSum()
        graph = from_exported_program(
            torch.export.export(model, (example,)), unwrap_unit_return_tuple=True
        )
        expected = a.astype("int64").sum(axis=dims, keepdims=keepdims).astype(dtype or "int64")
        np.testing.assert_array_equal(expected, model(example).numpy())
        result.append((name, graph, [a], expected))
    for name, m, n, k, batches, fused in (
        ("small", 3, 4, 5, 1, False),
        ("full", 16, 16, 16, 1, True),
        ("tail", 17, 3, 17, 1, True),
        ("batched", 3, 4, 5, 2, True),
        ("reuse", 16, 32, 8, 1, False),
    ):
        ashape = (batches, m, k) if batches > 1 else (m, k)
        bshape = (batches, k, n) if batches > 1 else (k, n)
        a = ((np.arange(np.prod(ashape)).reshape(ashape) * 13) % 256 - 128).astype("int8")
        b = ((np.arange(np.prod(bshape)).reshape(bshape) * 7) % 256 - 128).astype("int8")
        if name == "small":
            a.fill(-128)
            b.fill(-128)
        bias = np.arange(n, dtype="int32") - 9
        av = relax.Var("a", relax.TensorStructInfo(ashape, "int8"))
        bv = relax.Var("b", relax.TensorStructInfo(bshape, "int8"))
        parameters, arrays = [av, bv], [a, b]
        if fused:
            dv = relax.Var("bias", relax.TensorStructInfo((n,), "int32"))
            parameters.append(dv)
            arrays.append(bias)
        builder = relax.BlockBuilder()
        with builder.function("main", parameters):
            with builder.dataflow():
                value = builder.emit(relax.op.matmul(av, bv, out_dtype="int32"))
                if fused:
                    value = builder.emit(relax.op.nn.relu(relax.op.add(value, dv)))
                output = builder.emit_output(value)
            builder.emit_func_output(output)
        expected = a.astype("int64") @ b.astype("int64")
        if fused:
            expected = np.maximum(expected + bias, 0)
        result.append((name, builder.get(), arrays, expected.astype("int32")))
    a = (np.arange(9).reshape(1, 1, 3, 3) * 13 - 60).astype("int8")
    b = (np.arange(8).reshape(2, 1, 2, 2) * 11 - 30).astype("int8")
    av = relax.Var("image", relax.TensorStructInfo(a.shape, "int8"))
    bv = relax.Var("filter", relax.TensorStructInfo(b.shape, "int8"))
    builder = relax.BlockBuilder()
    with builder.function("main", [av, bv]):
        with builder.dataflow():
            value = builder.emit_output(relax.op.nn.conv2d(av, bv, out_dtype="int32"))
        builder.emit_func_output(value)
    expected = np.zeros((1, 2, 2, 2), "int64")
    for _, oc, y, x in np.ndindex(expected.shape):
        expected[0, oc, y, x] = sum(
            int(a[0, 0, y + ky, x + kx]) * int(b[oc, 0, ky, kx])
            for ky in range(2)
            for kx in range(2)
        )
    result.append(("conv", builder.get(), [a, b], expected.astype("int32")))
    extremes = np.array([0, 1, 2147483647, -2147483648, -1, 305419896], "int32")
    for kind in ("add_i32", "copy_i8", "copy_i32", "widen"):
        if kind in ("copy_i8", "widen"):
            a = (
                (np.arange(256).reshape(16, 16) - 128).astype("int8")
                if kind == "widen"
                else (np.arange(15).reshape(3, 5) * 17 - 128).astype("int8")
            )
        else:
            a = np.resize(extremes, (3, 4))
        av = relax.Var("a", relax.TensorStructInfo(a.shape, str(a.dtype)))
        parameters, arrays = [av], [a]
        builder = relax.BlockBuilder()
        copy_gv = None
        if kind.startswith("copy"):
            source = te.placeholder(a.shape, str(a.dtype))
            copied = te.compute(a.shape, lambda *i: source[i], name="copied")
            copy_gv = builder.add_func(te.create_prim_func([source, copied]), "copy")
        if kind == "add_i32":
            b = np.resize(
                np.array([-1, 2147483647, 1, -1, -2147483648, 2147483647], "int32"), a.shape
            )
            bv = relax.Var("b", relax.TensorStructInfo(b.shape, "int32"))
            parameters.append(bv)
            arrays.append(b)
            left = te.placeholder(a.shape, "int32", "left")
            right = te.placeholder(b.shape, "int32", "right")
            wrapped = te.compute(
                a.shape,
                lambda *i: (left[i].astype("int64") + right[i].astype("int64")).astype("int32"),
                name="wrapping_sum",
            )
            add_gv = builder.add_func(te.create_prim_func([left, right, wrapped]), "wrapping_add")
        with builder.function("main", parameters):
            with builder.dataflow():
                if kind == "add_i32":
                    value = relax.call_tir(
                        add_gv, (av, bv), relax.TensorStructInfo(a.shape, "int32")
                    )
                elif kind == "widen":
                    value = relax.op.astype(av, "int32")
                else:
                    value = relax.call_tir(
                        copy_gv, (av,), relax.TensorStructInfo(a.shape, str(a.dtype))
                    )
                output = builder.emit_output(value)
            builder.emit_func_output(output)
        expected = (
            (a.astype("int64") + b.astype("int64")).astype("int32")
            if kind == "add_i32"
            else a.astype("int32")
            if kind == "widen"
            else a
        )
        result.append((kind, builder.get(), arrays, expected, "gemmini",
                       {"offload_copies": kind != "add_i32"}))
    for kind in ("permute", "repeat"):
        a = (np.arange(6).reshape(2, 3) * 17 - 128).astype("int8")
        av = relax.Var("a", relax.TensorStructInfo(a.shape, "int8"))
        builder = relax.BlockBuilder()
        with builder.function("main", [av]):
            with builder.dataflow():
                value = (
                    relax.op.permute_dims(av, [1, 0])
                    if kind == "permute"
                    else relax.op.tile(av, [2, 2])
                )
                output = builder.emit_output(value)
            builder.emit_func_output(output)
        expected = a.T if kind == "permute" else np.tile(a, (2, 2))
        result.append((kind, builder.get(), [a], expected))
    for kind in ("fill_i8", "fill_i32", "fill_widen", "add_scalar"):
        source_dtype = "int8" if kind in ("fill_i8", "fill_widen") else "int32"
        scalar = np.array(
            1 if kind == "add_scalar" else -128 if source_dtype == "int8" else -2147483648,
            source_dtype,
        )
        sv = relax.Var("scalar", relax.TensorStructInfo((), source_dtype))
        parameters, arrays = [sv], [scalar]
        builder = relax.BlockBuilder()
        if kind == "add_scalar":
            x = np.resize(np.array([2147483647, -2147483648, -1, 0, 17, 1234], "int32"), (3, 4))
            xv = relax.Var("x", relax.TensorStructInfo(x.shape, "int32"))
            parameters, arrays = [xv, sv], [x, scalar]
            a = te.placeholder(x.shape, "int32", "x")
            b = te.placeholder((), "int32", "scalar")
            c = te.compute(
                x.shape,
                lambda *i: (a[i].astype("int64") + b[()].astype("int64")).astype("int32"),
                name="scalar_sum",
            )
            gv = builder.add_func(te.create_prim_func([a, b, c]), "scalar_add")
        with builder.function("main", parameters):
            with builder.dataflow():
                value = (
                    relax.call_tir(gv, (xv, sv), relax.TensorStructInfo(x.shape, "int32"))
                    if kind == "add_scalar"
                    else relax.op.full((3, 4), sv, "int32" if kind != "fill_i8" else "int8")
                )
                output = builder.emit_output(value)
            builder.emit_func_output(output)
        expected = (
            (x.astype("int64") + int(scalar)).astype("int32")
            if kind == "add_scalar"
            else np.full((3, 4), int(scalar), "int8" if kind == "fill_i8" else "int32")
        )
        result.append((kind, builder.get(), arrays, expected))
    for name, shape, axes, keepdims in (
        ("sum", (3, 17), (1,), False),
        ("sum_full", (2, 32), (1,), False),
        ("sum_scalar", (17,), None, False),
        ("sum_axes", (2, 3, 4), (0, 2), False),
        ("sum_keepdims", (2, 3, 4), (1, 2), True),
    ):
        a = (np.arange(np.prod(shape)).reshape(shape) % 256 - 128).astype("int8")
        if name in ("sum", "sum_scalar"):
            a.fill(-128)
        av = relax.Var("a", relax.TensorStructInfo(shape, "int8"))
        builder = relax.BlockBuilder()
        with builder.function("main", [av]):
            with builder.dataflow():
                output = builder.emit_output(
                    relax.op.sum(relax.op.astype(av, "int32"), axis=axes, keepdims=keepdims)
                )
            builder.emit_func_output(output)
        expected = a.astype("int64").sum(axis=axes, keepdims=keepdims).astype("int32")
        result.append((name, builder.get(), [a], expected))
    for name, bias_shape, wrapping in (
        ("bias_prefix", (4,), False),
        ("bias_matrix", (3, 4), False),
        ("bias_column", (3, 1), False),
        ("bias_scalar", (), False),
        ("bias_wrap", (4,), True),
    ):
        a = np.full((3, 32), 127, "int8")
        if not wrapping:
            a[:, 16:] = -127
            a[1] = -a[1]
        b = np.full((32, 4), 127, "int8")
        bias = np.resize(np.array([2147483644, -2147483645, 17, -23], "int32"), bias_shape)
        left = te.placeholder(a.shape, "int8", "left")
        right = te.placeholder(b.shape, "int8", "right")
        initial = te.placeholder(bias_shape, "int32", "bias")
        reduction = te.reduce_axis((0, 32), "reduction")
        product = te.compute(
            (3, 4),
            lambda i, j: te.sum(
                left[i, reduction].astype("int32") * right[reduction, j].astype("int32"),
                axis=reduction,
            ),
            name="unbiased_product",
        )

        def bias_value(i, j):
            if bias_shape == ():
                return initial[()]
            if len(bias_shape) == 1:
                return initial[j]
            return initial[i, 0 if bias_shape[1] == 1 else j]

        def biased_value(i, j):
            value, seed = product[i, j], bias_value(i, j)
            return (
                (value.astype("int64") + seed.astype("int64")).astype("int32")
                if wrapping
                else value + seed
            )

        output = te.compute((3, 4), biased_value, name="biased_output")
        builder = relax.BlockBuilder()
        gv = builder.add_func(te.create_prim_func([left, right, initial, output]), "biased_matmul")
        arrays = [a, b, bias]
        parameters = [
            relax.Var(label, relax.TensorStructInfo(array.shape, str(array.dtype)))
            for label, array in zip(("a", "b", "bias"), arrays)
        ]
        with builder.function("main", parameters):
            with builder.dataflow():
                result_var = builder.emit_output(
                    relax.call_tir(gv, parameters, relax.TensorStructInfo((3, 4), "int32"))
                )
            builder.emit_func_output(result_var)
        expected = (a.astype("int64") @ b.astype("int64") + bias).astype("int32")
        result.append((name, builder.get(), arrays, expected))
    from operator_contracts import fixtures as requested_fixtures

    result.extend(requested_fixtures())
    return result


def fixture_guest(
    index, arrays, expected, workspace, operation, placement="gemmini", *,
    timing_repeats=0, rtol=0.0, atol=0.0, required_scaled=False,
):
    """Check every flat output independently, including tuples and host-only cases."""
    import numpy as np

    if type(timing_repeats) is not int or not 0 <= timing_repeats <= 10:
        raise ValueError("Timing repeats must be an integer in [0,10]")
    if any(not math.isfinite(value) or value < 0 for value in (rtol, atol)):
        raise ValueError("Comparison tolerances must be finite and nonnegative")

    def flatten(value):
        if isinstance(value, (tuple, list)):
            return [array for item in value for array in flatten(item)]
        return [np.asarray(value)]

    def payload(value):
        array = np.asarray(value)
        ctype = {
            "int8": "int8_t",
            "int32": "int32_t",
            "int64": "int64_t",
            "float32": "uint32_t",
            "bool": "uint8_t",
            "uint8": "uint8_t", "int16": "int16_t", "uint16": "uint16_t",
            "float16": "uint16_t", "uint32": "uint32_t", "uint64": "uint64_t",
            "float64": "uint64_t",
        }.get(str(array.dtype))
        if ctype is None or not array.size:
            raise ValueError("Unsupported or empty verification tensor")
        data = array.view("uint" + str(array.itemsize * 8)) if array.dtype.kind == "f" else array
        return ctype, ",".join(str(int(x)) for x in data.flat)

    guest, checks = [], []
    for arg, array in enumerate(arrays):
        ctype, values = payload(array)
        guest.append(
            f"static const {ctype} in_{index}_{arg}[] __attribute__((aligned(64)))={{{values}}};"
        )
    outputs = flatten(expected)
    for number, array in enumerate(outputs):
        ctype, values = payload(array)
        guest.append(f"static const {ctype} expected_{index}_{number}[]={{{values}}};")
        guest.append(
            f"static struct {{ uint8_t head[64]; {ctype} data[{array.size}]; uint8_t tail[64]; }} out_{index}_{number} __attribute__((aligned(64)));"
        )
    guest.append(
        f"static struct {{uint8_t head[64]; uint8_t data[{max(workspace, 1)}]; uint8_t tail[64];}} "
        f"work_{index} __attribute__((aligned(64)));"
    )
    pointers = ",".join(f"in_{index}_{arg}" for arg in range(len(arrays)))
    results = ",".join(f"out_{index}_{number}.data" for number in range(len(outputs)))
    checks.extend(
        [
            f"const void* inputs_{index}[]={{{pointers}}}; void* outputs_{index}[]={{{results}}};",
            f"unsigned before_{index}=computes, zero_before_{index}=zeros;",
            f"unsigned dma_before_{index}=dmas, add_before_{index}=accumulates;",
            f"unsigned write_before_{index}=acc_writes, relu_before_{index}=relus;",
        ]
    )
    if required_scaled:
        checks.append(f"unsigned scaled_before_{index}=scaled_outs;")
    for number in range(len(outputs)):
        checks.append(f"memset(&out_{index}_{number},0x6d,sizeof(out_{index}_{number}));")
    checks.append(f"memset(&work_{index},0x6d,sizeof(work_{index}));")
    for arg in range(len(arrays)):
        checks.append(
            f"uint64_t hash_{index}_{arg}=hash(in_{index}_{arg},sizeof(in_{index}_{arg}));"
        )
    if placement == "host":
        primitive_check = f"if(computes!=before_{index} || dmas!=dma_before_{index} || zeros!=zero_before_{index})return 21;"
    elif operation == "graph_array":
        primitive_check = f"if(computes==before_{index})return 21;"
    elif operation == "matmul_bias":
        primitive_check = f"if(computes==before_{index} || acc_writes==write_before_{index} || zeros!=zero_before_{index})return 21;"
    elif operation and operation != "sum_i8_i32":
        primitive_check = f"if(dmas==dma_before_{index})return 21;"
    else:
        primitive_check = f"if(computes==before_{index} || zeros==zero_before_{index})return 21;"
    if operation is not None and operation.startswith("add_i32"):
        primitive_check += f"if(accumulates==add_before_{index})return 21;"
    if operation == "relu_i8":
        primitive_check += f"if(relus==relu_before_{index})return 21;"
    if operation == "max_pool_i8":
        checks.append(f"unsigned pool_before_{index}=pool_outs;")
        primitive_check += f"if(pool_outs==pool_before_{index})return 21;"
    if placement == "mixed":
        primitive_check += f"if(computes!=before_{index})return 21;"
    if required_scaled:
        primitive_check += f"if(scaled_outs==scaled_before_{index})return 21;"
    checks.extend(
        [
            f"if(stage_{index}_model_run(inputs_{index},outputs_{index},work_{index}.data,{workspace})) return 20;",
            primitive_check,
        ]
    )
    validation = []
    for number, array in enumerate(outputs):
        if array.dtype == np.float32 and (rtol or atol):
            validation.append(
                f"for(unsigned i=0;i<{array.size};++i){{"
                f"union {{uint32_t u;float f;}} actual={{.u=out_{index}_{number}.data[i]}};"
                f"union {{uint32_t u;float f;}} expected={{.u=expected_{index}_{number}[i]^(BAD_ORACLE?UINT32_C(0x40000000):0)}};"
                f"float tolerance={float(atol).hex()}f+{float(rtol).hex()}f*(expected.f<0?-expected.f:expected.f);"
                'if(!(actual.f>=expected.f-tolerance && actual.f<=expected.f+tolerance)){puts_htif("TENSORIR_NUMERICAL_FAILURE\\n");return 22;}}'
            )
        else:
            validation.append(
                f'for(unsigned i=0;i<{array.size};++i)if((uint64_t)(int64_t)out_{index}_{number}.data[i]!=((uint64_t)(int64_t)expected_{index}_{number}[i]^BAD_ORACLE)){{puts_htif("TENSORIR_NUMERICAL_FAILURE\\n");return 22;}}'
            )
        validation.append(
            f"for(unsigned i=0;i<64;++i)if(out_{index}_{number}.head[i]!=0x6d || out_{index}_{number}.tail[i]!=0x6d)return 23;"
        )
    for arg in range(len(arrays)):
        validation.append(
            f"if(hash_{index}_{arg}!=hash(in_{index}_{arg},sizeof(in_{index}_{arg})))return 24;"
        )
    validation.append(
        f"for(unsigned i=0;i<64;++i)if(work_{index}.head[i]!=0x6d || work_{index}.tail[i]!=0x6d)return 23;"
    )
    checks.extend(validation)
    if timing_repeats:
        checks.append(f"uint64_t cycle_samples_{index}[{timing_repeats}];")
        checks.append(f"for(unsigned repeat=0;repeat<{timing_repeats};++repeat){{")
        checks.append("uint64_t cycle_begin=dse_cycles();")
        checks.append(f"int status=stage_{index}_model_run(inputs_{index},outputs_{index},work_{index}.data,{workspace});")
        checks.append(f"cycle_samples_{index}[repeat]=dse_cycles()-cycle_begin;")
        checks.append("if(status)return 20;")
        checks.extend(validation)
        checks.append("}")
        checks.append(f"for(unsigned repeat=0;repeat<{timing_repeats};++repeat)dse_sample({index},repeat,cycle_samples_{index}[repeat]);")
    return guest, checks


def prepare(args, *, cases=None, prepared_reports=None, additional_bindings=()):
    from tvm.relax.backend.contrib.gemmini import prepare_gemmini_graph
    from tvm.relax.backend.contrib.gemmini_target import GemminiCapabilities
    from baremetal import export_graphs

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    adapter_path = args.adapter_receipt.resolve(strict=True)
    adapter = json.loads(adapter_path.read_text())
    app = Path(__file__).resolve().parent
    for name, digest in adapter["adapter_sources"].items():
        if identity(app / name)["sha256"] != digest:
            raise ValueError("Adapter source changed: " + name)
    adapter_object = adapter_path.parent / "matmul.o"
    if identity(adapter_object)["sha256"] != adapter["object_sha256"]:
        raise ValueError("Adapter object changed")
    available = fixtures() if cases is None else cases
    cases = [case for case in available if case[0] in args.cases]
    if len(cases) != len(set(args.cases)) or not cases:
        raise ValueError("Unknown or empty verification case selection")
    modules, reports = {}, {}
    for case in cases:
        name, mod = case[:2]
        placement = case[4] if len(case) > 4 else "gemmini"
        if prepared_reports is None:
            report = []
            options = {"col_tiles": 2, "reduction_chunk": 8} if name == "reuse" else None
            modules[name] = prepare_gemmini_graph(
                mod, report=report, schedule_options=options, **(case[5] if len(case) > 5 else {})
            )
        else:
            modules[name], report = mod, prepared_reports[name]
        expected_functions = case[6] if len(case) > 6 else (0 if placement == "host" else 1)
        if sum(row["placement"] == "gemmini" for row in report) != expected_functions:
            raise ValueError("Fixture did not use the structural accelerator rule: " + name)
        reports[name] = report
    package = out / "package"
    manifest = export_graphs(modules, package, memory_limit_bytes=16 << 20)
    timing_repeats = getattr(args, "timing_repeats", 0)
    rtol, atol = getattr(args, "rtol", 0.0), getattr(args, "atol", 0.0)
    guest = [GUEST]
    if getattr(args, "libm", False):
        # Newlib exposes errno through this documented ABI. This guest runs
        # one hart without threads; no libc I/O, allocator or syscall stubs are
        # supplied. The selected libm implements all floating arithmetic.
        guest.append("#include <errno.h>\nstatic int guest_errno;\nint* __errno(void){return &guest_errno;}\n")
    if timing_repeats:
        guest.append(TIMING_GUEST)
    guest.append('#include "package/model.h"')
    checks = [
        "extern int32_t tvm_gemmini_validate_tensor_output(void*,int64_t);",
        "uint32_t admission_probe=0;",
        "if(tvm_gemmini_validate_tensor_output(&admission_probe,sizeof(admission_probe))!=0 || "
        "tvm_gemmini_validate_tensor_output(NULL,4)!=-1 || "
        "tvm_gemmini_validate_tensor_output(&admission_probe,0)!=-1 || "
        "tvm_gemmini_validate_tensor_output(&admission_probe,-1)!=-1 || "
        "tvm_gemmini_validate_tensor_output((uint8_t*)&admission_probe+1,4)!=-1 || "
        "tvm_gemmini_validate_tensor_output((void*)(UINTPTR_MAX-3),8)!=-2)return 25;",
    ]
    for index, case in enumerate(cases):
        name, _, arrays, expected = case[:4]
        placement = case[4] if len(case) > 4 else "gemmini"
        operation = next(
            (
                row["metadata"].get("operation")
                for row in reports[name]
                if row["placement"] == "gemmini"
            ),
            None,
        )
        declarations, case_checks = fixture_guest(
            index,
            arrays,
            expected,
            manifest["stages"][name]["workspace_bytes"],
            operation,
            placement,
            timing_repeats=timing_repeats,
            rtol=rtol,
            atol=atol,
            required_scaled=any(row.get("metadata", {}).get("readout_fused") or row.get("metadata", {}).get("scaled_readout") for row in reports[name]),
        )
        guest.extend(declarations)
        checks.append(f'puts_htif("TENSORIR_CASE {name}\\n");')
        checks.extend(case_checks)
    guest.append(
        "int main(void){\n"
        + "\n".join(checks)
        + f'\nputs_htif("TENSORIR_PASS cases={len(cases)} exact={int(not (rtol or atol))} guards=1 primitives=1\\n");return 0;}}\n'
    )
    (out / "runner.c").write_text("\n".join(guest))
    (out / "start.S").write_text(STARTUP)
    (out / "link.ld").write_text(LINKER)
    gcc = args.riscv_gcc.resolve(strict=True)
    flags = [
        "-std=gnu11",
        "-march=rv64gc",
        "-mabi=lp64d",
        "-mcmodel=medany",
        "-msmall-data-limit=0",
        "-O2",
        "-ffreestanding",
        "-fno-builtin",
        "-fno-stack-protector",
        "-fno-tree-loop-distribute-patterns",
        "-nostdlib",
        "-nostartfiles",
        "-static",
        "-Wl,--no-relax",
        "-Wl,-T," + str(out / "link.ld"),
        "-Wl,--wrap=tvm_gemmini_compute",
        "-Wl,--wrap=tvm_gemmini_compute_ws",
        "-Wl,--wrap=tvm_gemmini_zero_acc",
        "-Wl,--wrap=tvm_gemmini_dma_acc_write",
        "-Wl,--wrap=tvm_gemmini_dma_acc_add",
        "-Wl,--wrap=tvm_gemmini_dma_widen",
        "-Wl,--wrap=tvm_gemmini_dma_spad_out",
        "-Wl,--wrap=tvm_gemmini_dma_relu_out",
        "-Wl,--wrap=tvm_gemmini_dma_scaled_out",
        "-Wl,--wrap=tvm_gemmini_dma_scaled_in",
        "-Wl,--wrap=tvm_gemmini_dma_pool_out",
        "-Wl,--wrap=tvm_gemmini_dma_a",
        "-Wl,--wrap=tvm_gemmini_dma_b",
        "-Wl,--wrap=tvm_gemmini_dma_out",
        "-I" + str(out),
    ]
    sources = [out / "start.S", out / "runner.c", package / "runtime.c", package / "constants.S"]
    math_libraries = []
    if getattr(args, "libm", False):
        for name in ("libm.a", "libgcc.a"):
            path = Path(subprocess.check_output([str(gcc), "-march=rv64gc", "-mabi=lp64d",
                                                "-print-file-name=" + name], text=True).strip())
            math_libraries.append(path.resolve(strict=True))
    for name in modules:
        sources.extend([package / name / "model.c", package / name / "operators.o"])
    policy = adapter["instruction_policy"]
    forbidden = {
        value for key, value in policy["derived_definitions"].items() if key.startswith("k_LOOP_")
    }
    commands, elfs = [], {}
    for label, wrong in (("valid", 0), ("bad-oracle", 1)):
        elf = out / (label + ".elf")
        command = [
            str(gcc),
            *flags,
            "-DBAD_ORACLE=" + str(wrong),
            *map(str, sources),
            str(adapter_object),
            *map(str, math_libraries),
            "-o",
            str(elf),
        ]
        process = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=45
        )
        (out / (label + "-link.log")).write_text(process.stdout)
        if process.returncode:
            raise RuntimeError(process.stdout)
        audit = audit_instructions(elf.read_bytes(), policy["custom_opcode"], forbidden)
        if audit["prohibited_hits"]:
            raise ValueError("Hardware-loop instructions in final ELF")
        nm = gcc.with_name(gcc.name[:-3] + "nm")
        unresolved = subprocess.check_output([nm, "-u", elf], text=True)
        if unresolved.strip():
            raise ValueError("Unresolved guest symbols: " + unresolved)
        commands.append({"argv": command, "exit_code": process.returncode})
        elfs[label] = dict(identity(elf), instruction_audit=audit)
    source = app.parents[1]
    bindings = [identity(adapter_path), identity(adapter_object), identity(gcc)]
    bindings.extend(identity(path) for path in math_libraries)
    bindings.extend(identity(path) for path in additional_bindings)
    from tvm._ffi.base import _LIB  # pylint: disable=import-outside-toplevel

    bindings.append(identity(_LIB._name))
    bindings.extend(identity(path) for path in sorted(package.rglob("*")) if path.is_file())
    bindings.extend(identity(out / name) for name in ("runner.c", "start.S", "link.ld"))
    for relative in (
        "apps/gemmini/verify_tensorir.py",
        "apps/gemmini/operator_contracts.py",
        "apps/gemmini/baremetal.py",
        "apps/gemmini/matmul.c",
        "apps/gemmini/matmul.h",
        "apps/gemmini/verify_device_ops.py",
        "src/tir/schedule/primitive/reduction.cc",
        "src/tir/schedule/state.cc",
        "src/tir/schedule/primitive/decompose_padding.cc",
        "src/tir/schedule/primitive/blockize_tensorize.cc",
        "src/tir/schedule/ir_comparator.cc",
        "src/tir/schedule/primitive/block_annotate.cc",
        "src/tir/schedule/primitive/loop_transformation.cc",
        "src/tir/transforms/renew_defs.cc",
        "src/relax/op/nn/pooling.cc",
        "src/te/operation/create_primfunc.cc",
        "src/tir/ir/stmt_functor.cc",
        "src/tir/ir/data_type_rewriter.cc",
        "include/tvm/topi/nn/pooling.h",
        "python/tvm/topi/transform.py",
        "python/tvm/tir/function.py",
        "python/tvm/tir/transform/rounding.py",
        "python/tvm/relax/transform/legalize_ops/statistical.py",
        "python/tvm/relax/transform/legalize_ops/nn.py",
        "python/tvm/relax/transform/legalize_ops/grad.py",
        "python/tvm/relax/frontend/torch/exported_program_translator.py",
        "python/tvm/relax/frontend/torch/base_fx_graph_translator.py",
        "python/tvm/relax/frontend/torch/_numeric.py",
        "python/tvm/relax/frontend/torch/_quantization.py",
        "python/tvm/relax/frontend/torch/_pooling.py",
        "python/tvm/relax/frontend/torch/_overlap.py",
        "python/tvm/relax/frontend/torch/_convolution.py",
        "python/tvm/relax/op/_op_gradient.py",
        "python/tvm/topi/nn/__init__.py",
        "python/tvm/topi/nn/conv_transpose.py",
        *[
            "python/tvm/relax/backend/contrib/" + name + ".py"
            for name in (
                "gemmini",
                "gemmini_tensorir",
                "gemmini_intrin",
                "gemmini_memory",
                "gemmini_target",
                "gemmini_meta_schedule",
                "gemmini_elementwise",
                "gemmini_reduction",
                "gemmini_bias",
                "gemmini_regions",
                "gemmini_search",
                "gemmini_epilogue",
                "gemmini_affine",
                "gemmini_bytes",
                "gemmini_pool",
                "gemmini_sketch",
            )
        ],
    ):
        bindings.append(identity(source / relative))
    bundle = {
        "status": "prepared",
        "scope": "Bounded functional fixtures; no model/performance qualification",
        "capabilities": GemminiCapabilities().fingerprint,
        "ELFs": elfs,
        "scheduling": reports,
        "commands": commands,
        "bindings": bindings,
        "case_count": len(cases),
        "timing_repeats": timing_repeats,
        "comparison": {"rtol": rtol, "atol": atol, "integer": "exact"},
        "cases": [case[0] for case in cases],
        "placements": {case[0]: case[4] if len(case) > 4 else "gemmini" for case in cases},
        "preparation_options": {case[0]: case[5] if len(case) > 5 else {} for case in cases},
        "operator_contracts": {case[0]: case[7] for case in cases if len(case) > 7},
    }
    (out / "bundle.json").write_text(json.dumps(bundle, indent=2) + "\n")
    print(json.dumps({"status": "prepared", "bundle": str(out / "bundle.json")}))


def classify_output(backend, text, returncode, timed_out, negative=False, expected_cases=4, exact=True):
    """Require guest evidence and exact decoded exit status, including RTL padding."""
    if timed_out:
        return False
    if negative:
        if backend == "spike":
            decoded = returncode == 22
        else:
            codes = []
            for line in text.splitlines():
                if "Assertion failed:" in line and "exit code =" in line:
                    value = line.partition("exit code =")[2].partition(")")[0].strip()
                    if value.isdecimal():
                        codes.append(int(value))
            decoded = returncode == 255 and codes == [22]
        return decoded and "TENSORIR_NUMERICAL_FAILURE" in text
    marker = f"TENSORIR_PASS cases={expected_cases} exact={int(exact)} guards=1 primitives=1"
    passed = returncode == 0 and marker in text.splitlines()
    return passed and (backend == "spike" or "Verilog $finish" in text)


def parse_cycle_samples(text, *, case_count, repeats, cycle_limit=None):
    """Require every expected guest interval, without interpreting host wall time."""
    samples = {}
    for line in text.splitlines():
        if not line.startswith("DSE_CYCLES "):
            continue
        fields = dict(field.split("=", 1) for field in line.split()[1:])
        if set(fields) != {"stage", "sample", "cycles"}:
            raise ValueError("Malformed guest cycle sample")
        try:
            stage, sample, cycles = (int(fields[key]) for key in ("stage", "sample", "cycles"))
        except ValueError as error:
            raise ValueError("Noninteger guest cycle sample") from error
        key = stage, sample
        if key in samples or not 0 <= stage < case_count or not 0 <= sample < repeats:
            raise ValueError("Duplicate or unexpected guest cycle sample")
        if cycles <= 0 or (cycle_limit is not None and cycles > cycle_limit):
            raise ValueError("Invalid guest cycle interval")
        samples[key] = cycles
    if len(samples) != case_count * repeats:
        raise ValueError("Missing guest cycle samples")
    return [[samples[(stage, sample)] for sample in range(repeats)] for stage in range(case_count)]


def run(args):
    from memory_model import bind_memory_model, confirm_memory_model, runtime_environment

    out = args.bundle.resolve(strict=True)
    bundle = json.loads((out / "bundle.json").read_text())
    result = out / (args.name + "-result.json")
    if result.exists():
        raise ValueError("Refusing to overwrite an execution receipt")
    label = "bad-oracle" if args.negative else "valid"
    elf = bundle["ELFs"][label]
    bindings = list(bundle["bindings"]) + [elf]
    env = dict(os.environ)
    memory = None
    if args.backend == "spike" and getattr(args, "memory_variant_receipt", None):
        raise ValueError("Memory variants apply only to Verilator")
    if args.backend == "spike":
        from verify_exported_graph import guest_layout
        _, guest_map = guest_layout(bundle.get("guest_memory_bytes", 256 << 20))
        receipt = json.loads(args.simulator_receipt.read_text())
        bindings.extend(
            [
                identity(args.simulator_receipt),
                {"path": receipt["plugin_path"], "sha256": receipt["plugin_sha256"]},
                {"path": receipt["spike_path"], "sha256": receipt["spike_sha256"]},
            ]
        )
        for key, value in receipt["runtime_environment"].items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = str(value)
        command = [
            receipt["spike_path"],
            "--extlib=" + receipt["plugin_path"],
            "--extension=gemmini",
            guest_map,
            elf["path"],
        ]
    else:
        receipt = json.loads(args.hardware_receipt.read_text())
        memory = bind_memory_model(args.hardware_receipt,
                                   getattr(args, "memory_variant_receipt", None),
                                   getattr(args, "firesim_memory_artifact", None),
                                   getattr(args, "memory_pin_registry", None))
        simulator = memory["simulator"]
        bindings.extend(memory["bindings"])
        env = runtime_environment(memory, env)
        command = [
            simulator["path"],
            "+permissive",
            "+max-cycles=" + str(args.cycles),
            "+loadmem=" + elf["path"],
            "+permissive-off",
            elf["path"],
        ]
    for binding in bindings:
        if identity(binding["path"])["sha256"] != binding["sha256"]:
            raise ValueError("Execution input changed: " + binding["path"])

    def limits():
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
        resource.setrlimit(resource.RLIMIT_FSIZE, (16 << 20, 16 << 20))
        if hasattr(os, "sched_getaffinity"):
            os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})

    start, timed_out = time.monotonic(), False
    with (out / (args.name + ".log")).open("wb") as log:
        child = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, env=env, preexec_fn=limits
        )
        try:
            code = child.wait(timeout=args.seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            child.kill()
            code = child.wait()
    text = (out / (args.name + ".log")).read_text(errors="replace")
    passed = classify_output(
        args.backend, text, code, timed_out, args.negative, bundle["case_count"],
        exact=not any(bundle.get("comparison", {}).get(key, 0) for key in ("rtol", "atol")),
    )
    confirmation = confirm_memory_model(memory, text) if memory is not None else None
    if confirmation is not None and not confirmation["model_confirmed"]:
        passed = False
    timing = None
    timing_error = None
    repeats = bundle.get("timing_repeats", 0)
    if passed and not args.negative and repeats:
        try:
            samples = parse_cycle_samples(
                text, case_count=bundle["case_count"], repeats=repeats,
                cycle_limit=args.cycles if args.backend == "verilator" else None,
            )
            timing = {
                "metric": "target_cycles" if args.backend == "verilator" else "functional_counter",
                "usable_for_dse": args.backend == "verilator",
                "scope": ("selected_rtl_" + memory["kind"]) if memory else "spike_functional_only",
                "samples_by_stage": samples,
                "warmups": 1,
                "cache_policy": "same buffers after one warmup; no inter-repeat cache flush",
                "interval": "model_run including admission and completion; excludes startup, oracle and HTIF",
            }
        except ValueError as error:
            passed, timing_error = False, str(error)
    record = {
        "status": "passed" if passed else "failed_or_incomplete",
        "backend": args.backend,
        "negative": args.negative,
        "command": command,
        "returncode": code,
        "timed_out": timed_out,
        "wall_seconds": time.monotonic() - start,
        "cycle_limit": args.cycles if args.backend == "verilator" else None,
        "ELF": elf,
        "bindings": bindings,
        "log": identity(out / (args.name + ".log")),
        "functional_only": True,
        "performance_qualified": False,
        "timing": timing,
        "timing_error": timing_error,
        "memory_model": memory,
        "memory_confirmation": confirmation,
    }
    result.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": record["status"], "receipt": str(result)}))
    if not passed:
        raise SystemExit(1)


def validate_run_bounds(seconds, cycles):
    """Explicit bounded allowance for model-sized chunks; defaults stay small."""
    if not 1 <= seconds <= 3600 or not 1 <= cycles <= 100_000_000:
        raise ValueError("Use a bounded run: seconds [1,3600], cycles [1,100000000]")


def main():
    from operator_contracts import REQUESTED_OPERATORS

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    build = sub.add_parser("prepare")
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--adapter-receipt", type=Path, required=True)
    build.add_argument("--riscv-gcc", type=Path, required=True)
    build.add_argument("--timing-repeats", type=int, default=0)
    build.add_argument("--rtol", type=float, default=0.0)
    build.add_argument("--atol", type=float, default=0.0)
    build.add_argument("--libm", action="store_true", help="Bind and link the selected GCC math libraries")
    build.add_argument(
        "--cases",
        nargs="+",
        default=["small", "full", "tail", "batched"],
        choices=[
            "small",
            "full",
            "tail",
            "batched",
            "conv",
            "reuse",
            "add_i32",
            "copy_i8",
            "copy_i32",
            "widen",
            "permute",
            "repeat",
            "fill_i8",
            "fill_i32",
            "fill_widen",
            "add_scalar",
            "sum",
            "sum_full",
            "sum_scalar",
            "sum_axes",
            "sum_keepdims",
            "bias_prefix",
            "bias_matrix",
            "bias_column",
            "bias_scalar",
            "bias_wrap",
            "relu_i8",
            "relu_tail",
            "copy_f32",
            "repeat_f32",
            "fill_f32",
            "reflect1d",
            "reflect2d",
            "reflect3d",
            "edge2d",
            "edge3d",
            "constant_i8",
            "constant_i32",
            "constant_f32",
            "constant_zero",
            "constant_nan",
            "constant_crop",
            "cat_i8",
            "cat_i32",
            "cat_f32",
            "scatter_i8",
            "scatter_i32",
            "scatter_f32",
            "select_i32",
            "select_f32",
            "sum64",
            "sum64_axes",
            "sum64_scalar",
            "sum32_export",
            "sum8_export",
            "view_diagonal_i8",
            "view_as_strided_i32",
            "view_select_f32",
            "view_permute_i8",
            "view_split_i32",
            "view_squeeze_i32",
            "view_squeeze_dims_f32",
            "view_unsqueeze_i8",
            "view_reshape_f32",
            "view_slice_i8",
            "view_noop_i32",
            "view_scalar_f32",
            "view_alias_i32",
            *[
                "aten_" + name
                for name in (
                    "mm",
                    "bmm",
                    "addmm",
                    "add",
                    "scalar_add",
                    "subtract",
                    "multiply",
                    "divide",
                    "alpha",
                    "int64_add",
                    "int64_mm",
                )
            ],
            *[
                f"creation_{kind}_{dtype}_{mode}"
                for kind in ("full", "full_like")
                for dtype in ("int8", "int32", "float32")
                for mode in ("native", "folded")
            ],
            *[f"pool_avg{dim}d" for dim in (1, 2, 3)],
            *[f"pool_adaptive{dim}d" for dim in (1, 2, 3)],
            "pool_crop2d",
            "pool_override2d",
            "pool_long2d",
            *["requested_" + name.replace(".", "_") for name in REQUESTED_OPERATORS],
            *[
                "overlap_" + kind
                for kind in (
                    "average",
                    "override",
                    "adaptive",
                    "integer",
                    "holes",
                    "col2im",
                    "boolean",
                )
            ],
            *[
                "convolution_" + kind
                for kind in (
                    "native1d",
                    "native2d",
                    "native3d",
                    "wide",
                    "transpose_wide",
                    "backward",
                    "transpose_backward",
                )
            ],
        ],
    )
    execute = sub.add_parser("run")
    execute.add_argument("--bundle", type=Path, required=True)
    execute.add_argument("--backend", choices=("spike", "verilator"), required=True)
    execute.add_argument("--simulator-receipt", type=Path)
    execute.add_argument("--hardware-receipt", type=Path)
    from memory_model import add_memory_arguments
    add_memory_arguments(execute)
    execute.add_argument("--name", required=True)
    execute.add_argument("--negative", action="store_true")
    execute.add_argument("--seconds", type=int, default=60)
    execute.add_argument("--cycles", type=int, default=300000)
    args = parser.parse_args()
    if args.action == "run":
        try:
            validate_run_bounds(args.seconds, args.cycles)
        except ValueError as error:
            parser.error(str(error))
        if args.backend == "spike" and args.simulator_receipt is None:
            parser.error("Spike requires --simulator-receipt")
        if args.backend == "verilator" and args.hardware_receipt is None:
            parser.error("Verilator requires --hardware-receipt")
        run(args)
    else:
        prepare(args)


if __name__ == "__main__":
    main()
