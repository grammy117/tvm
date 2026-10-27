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
"""Independent queued-DMA checks for structural Gemmini TensorIR scheduling."""

import ctypes
from pathlib import Path
import tempfile
import unittest

import numpy as np
import tvm
from tvm import relax, te, tir
from tvm.relax.backend.contrib.gemmini import prepare_gemmini_graph
from tvm.relax.backend.contrib.gemmini_tensorir import (
    contraction_block,
    lower_tensorir_schedule,
    make_tensorir_schedule,
)
from tvm.relax.backend.contrib.gemmini_sketch import default_schedule


# All commands are delayed until a fence. CPU writes between command issue and
# DMA execution therefore expose a missing staging-buffer completion boundary.
EMULATOR = r"""
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
static int8_t sp[16384][16];
static int32_t acc[1024][16];
static unsigned char loaded[16384][16], initialized[1024][16];
struct command { int op; const void* src; void* dst; uint32_t a,b,c,m,n,k; int64_t stride; int add; uint32_t block; };
static struct command queue[65536];
static unsigned count, faults, commands, fences;
static int wrong, reverse_stores;
struct pending_store { void* dst; uint32_t bits; unsigned width; };
static struct pending_store stores[65536];
static unsigned store_count;
static void check(int condition) { if (!condition) ++faults; }
static void push(struct command x) { if(count>=65536) { ++faults; return; } queue[count++]=x; ++commands; }
static void store(void* dst,const void* src,unsigned width) {
  if(!reverse_stores) { memcpy(dst,src,width);return; }
  if(store_count>=65536) { ++faults;return; }
  struct pending_store* s=&stores[store_count++];s->dst=dst;s->width=width;
  memcpy(&s->bits,src,width);
}
static void flush(void) {
  ++fences;
  for(unsigned t=0;t<count;++t) {
    struct command x=queue[t];
    check(x.m>0 && x.m<=16 && x.n>0 && (x.n<=16 || x.op==11));
    if(x.op==0) {
      check(x.c+x.m<=1024);
      for(unsigned i=0;i<x.m;++i) for(unsigned j=0;j<x.n;++j) { acc[x.c+i][j]=0; initialized[x.c+i][j]=1; }
    } else if(x.op==1) {
      check(x.a+x.m<=16384 && (x.stride==0 || x.stride>=x.n));
      const int8_t* p=x.src;
      for(unsigned i=0;i<x.m;++i) for(unsigned j=0;j<x.n;++j) { sp[x.a+i][j]=p[i*x.stride+j]; loaded[x.a+i][j]=1; }
    } else if(x.op==2) {
      check(x.a+x.m<=16384 && x.b+x.k<=16384 && x.c+x.m<=1024 && x.k>0 && x.k<=16);
      for(unsigned i=0;i<x.m;++i) for(unsigned j=0;j<x.n;++j) {
        int64_t part=0;
        for(unsigned k=0;k<x.k;++k) { check(loaded[x.a+i][k] && loaded[x.b+k][j]); part+=(int32_t)sp[x.a+i][k]*sp[x.b+k][j]; }
        check(part>=-524288 && part<=524287 && (!x.add || initialized[x.c+i][j]));
        uint32_t bits=(x.add?(uint32_t)acc[x.c+i][j]:0u)+(uint32_t)part+(uint32_t)wrong;
        memcpy(&acc[x.c+i][j],&bits,sizeof(bits));initialized[x.c+i][j]=1;
      }
    } else if(x.op==11) {
      /* Multi-block load: DIM-column block b lands at row a + b*block. */
      check(x.stride==0 || x.stride>=x.n);
      check(x.n>0 && x.n<=64 && x.m>0 && x.m<=16);
      const int8_t* p=x.src;
      for(unsigned b=0;b*16<x.n;++b) for(unsigned i=0;i<x.m;++i) for(unsigned j=0;j<16 && b*16+j<x.n;++j) {
        check(x.a+b*x.block+i<16384);
        sp[x.a+b*x.block+i][j]=p[i*x.stride+b*16+j]; loaded[x.a+b*x.block+i][j]=1;
      }
    } else if(x.op==3) {
      check(x.c+x.m<=1024 && x.stride>=x.n);
      for(unsigned i=0;i<x.m;++i) for(unsigned j=0;j<x.n;++j) {
        check(initialized[x.c+i][j]);
        store((unsigned char*)x.dst+4*(i*x.stride+j), &acc[x.c+i][j], 4);
      }
    } else if(x.op==4 || x.op==5) {
      check(x.c+x.m<=1024 && (x.stride==0 || x.stride>=x.n));
      for(unsigned i=0;i<x.m;++i) for(unsigned j=0;j<x.n;++j) {
        if(x.op==5)check(initialized[x.c+i][j]);
        uint32_t bits;
        memcpy(&bits,(const unsigned char*)x.src+4*(i*x.stride+j),4);
        bits += x.op==5?(uint32_t)acc[x.c+i][j]:0;
        memcpy(&acc[x.c+i][j],&bits,sizeof(bits));initialized[x.c+i][j]=1;
      }
    } else if(x.op==12) {
      check(x.c+x.m<=1024 && (x.stride==0 || x.stride>=x.n));
      union {uint32_t bits;float value;} scale={.bits=x.a};
      const int8_t* p=x.src;
      for(unsigned i=0;i<x.m;++i) for(unsigned j=0;j<x.n;++j) {
        if(x.add)check(initialized[x.c+i][j]);
        float rounded=nearbyintf((float)p[i*x.stride+j]*scale.value);
        int32_t v=rounded>127?127:rounded < -128?-128:(int32_t)rounded;
        acc[x.c+i][j]=(x.add?acc[x.c+i][j]:0)+v;
        initialized[x.c+i][j]=1;
      }
    } else if(x.op==13) {
      check(x.c+x.k*x.k<=1024 && (x.k==2 || x.k==3));
      int8_t maximum=-128;
      for(unsigned i=0;i<x.k*x.k;++i) {
        check(initialized[x.c+i][0] && acc[x.c+i][0]>=-128 && acc[x.c+i][0]<=127);
        if(acc[x.c+i][0]>maximum)maximum=acc[x.c+i][0];
      }
      store(x.dst,&maximum,1);
    } else if(x.op==6) {
      check(x.c+x.m<=1024 && (x.stride==0 || x.stride>=x.n));
      const int8_t* p=x.src;
      for(unsigned i=0;i<x.m;++i) for(unsigned j=0;j<x.n;++j) { acc[x.c+i][j]=p[i*x.stride+j];initialized[x.c+i][j]=1; }
    } else if(x.op==7) {
      check(x.a+x.m<=16384 && x.stride>=x.n);
      int8_t* p=x.dst;
      for(unsigned i=0;i<x.m;++i) for(unsigned j=0;j<x.n;++j) { check(loaded[x.a+i][j]);store(&p[i*x.stride+j],&sp[x.a+i][j],1); }
    } else if(x.op==8) {
      check(x.c+x.m<=1024 && x.stride>=x.n);
      int8_t* p=x.dst;
      for(unsigned i=0;i<x.m;++i) for(unsigned j=0;j<x.n;++j) {
        int32_t v=acc[x.c+i][j];
        check(initialized[x.c+i][j] && v>=-128 && v<=127);
        int8_t value=wrong?v:(v<0?0:v);store(&p[i*x.stride+j],&value,1);
      }
    } else if(x.op==9) {
      check(x.c+x.m<=1024 && x.stride>=x.n);
      union {uint32_t bits;float value;} scale={.bits=x.a};
      int8_t* p=x.dst;
      for(unsigned i=0;i<x.m;++i) for(unsigned j=0;j<x.n;++j) {
        check(initialized[x.c+i][j]);
        float rounded=nearbyintf((float)acc[x.c+i][j]*scale.value);
        int32_t value=rounded>127?127:rounded < -128?-128:(int32_t)rounded;
        if(x.b && value<0)value=0;
        int8_t output=value;store(&p[i*x.stride+j],&output,1);
      }
    }
  }
  for(unsigned i=store_count;i>0;--i) {
    struct pending_store s=stores[i-1];memcpy(s.dst,&s.bits,s.width);
  }
  count=store_count=0;
}
void emulator_reset(void) { count=store_count=faults=commands=fences=0; wrong=reverse_stores=0; memset(loaded,0,sizeof(loaded));memset(initialized,0,sizeof(initialized)); }
void emulator_wrong(int x) { wrong=x; }
void emulator_reverse_stores(int x) { reverse_stores=x; }
unsigned emulator_faults(void) { return faults; }
unsigned emulator_commands(void) { return commands; }
void tvm_gemmini_dma_scaled_in(const int8_t* p,uint32_t c,uint32_t m,uint32_t n,int64_t s,uint32_t scale,int32_t add,int32_t wait) {
  push((struct command){.op=12,.src=p,.c=c,.m=m,.n=n,.stride=s,.a=scale,.add=add});if(wait)flush();
}
void tvm_gemmini_dma_pool_out(int8_t* p,uint32_t c,uint32_t size,int32_t wait) {
  push((struct command){.op=13,.dst=p,.c=c,.k=size,.m=1,.n=1});if(wait)flush();
}
unsigned emulator_fences(void) { return fences; }
unsigned emulator_pending(void) { return count; }
int32_t tvm_gemmini_validate_tensor_output(void* c,int64_t cz) {
  if(!c || cz<=0 || (uintptr_t)c % _Alignof(int32_t))return -1;
  if((uint64_t)cz>UINTPTR_MAX-(uintptr_t)c)return -2;
  return 0;
}
int32_t tvm_gemmini_validate_tensor_buffers(const void* a,int64_t az,const void* b,int64_t bz,void* c,int64_t cz) {
  if(!a || !b || !c || az<=0 || bz<=0 || cz<=0) return -1;
  uintptr_t aa=(uintptr_t)a,bb=(uintptr_t)b,cc=(uintptr_t)c;
  if((cc<aa+az && aa<cc+cz)||(cc<bb+bz && bb<cc+cz))return -3;
  return 0;
}
void tvm_gemmini_begin(int64_t a,int64_t b,int64_t c) { (void)a;(void)b;(void)c;check(count==0); }
void tvm_gemmini_end(void) { flush(); }
void tvm_gemmini_zero_acc(uint32_t c,uint32_t m,uint32_t n) { push((struct command){.op=0,.c=c,.m=m,.n=n}); }
void tvm_gemmini_dma_a(const int8_t* p,uint32_t a,uint32_t m,uint32_t n,int64_t stride,int32_t wait) {
  push((struct command){.op=1,.src=p,.a=a,.m=m,.n=n,.stride=stride});if(wait)flush();
}
void tvm_gemmini_dma_b(const int8_t* p,uint32_t a,uint32_t m,uint32_t n,int64_t stride,int32_t wait) { tvm_gemmini_dma_a(p,a,m,n,stride,wait); }
void tvm_gemmini_compute(uint32_t a,uint32_t b,uint32_t c,uint32_t m,uint32_t n,uint32_t k,int32_t add) {
  check(add==0 || add==1);push((struct command){.op=2,.a=a,.b=b,.c=c,.m=m,.n=n,.k=k,.add=add});
}
void tvm_gemmini_compute_ws(uint32_t a,uint32_t b,uint32_t c,uint32_t m,uint32_t n,uint32_t k,int32_t add) {
  tvm_gemmini_compute(a,b,c,m,n,k,add);
}
void tvm_gemmini_dma_a_blocks(const int8_t* p,uint32_t a,uint32_t m,uint32_t n,int64_t stride,uint32_t block,int32_t wait) {
  push((struct command){.op=11,.src=p,.a=a,.m=m,.n=n,.stride=stride,.block=block});if(wait)flush();
}
void tvm_gemmini_dma_b_blocks(const int8_t* p,uint32_t a,uint32_t m,uint32_t n,int64_t stride,uint32_t block,int32_t wait) {
  tvm_gemmini_dma_a_blocks(p,a,m,n,stride,block,wait);
}
void tvm_gemmini_dma_out(int32_t* p,uint32_t c,uint32_t m,uint32_t n,int64_t stride,int32_t wait) {
  push((struct command){.op=3,.dst=p,.c=c,.m=m,.n=n,.stride=stride});if(wait)flush();
}
void tvm_gemmini_dma_scaled_out(int8_t* p,uint32_t c,uint32_t m,uint32_t n,int64_t stride,uint32_t scale,int32_t relu,int32_t wait) {
  push((struct command){.op=9,.dst=p,.c=c,.m=m,.n=n,.stride=stride,.a=scale,.b=relu});if(wait)flush();
}
void tvm_gemmini_dma_acc_write(const int32_t* p,uint32_t c,uint32_t m,uint32_t n,int64_t stride,int32_t wait) {
  push((struct command){.op=4,.src=p,.c=c,.m=m,.n=n,.stride=stride});if(wait)flush();
}
void tvm_gemmini_dma_acc_add(const int32_t* p,uint32_t c,uint32_t m,uint32_t n,int64_t stride,int32_t wait) {
  push((struct command){.op=5,.src=p,.c=c,.m=m,.n=n,.stride=stride});if(wait)flush();
}
void tvm_gemmini_dma_widen(const int8_t* p,uint32_t c,uint32_t m,uint32_t n,int64_t stride,int32_t wait) {
  push((struct command){.op=6,.src=p,.c=c,.m=m,.n=n,.stride=stride});if(wait)flush();
}
void tvm_gemmini_dma_spad_out(int8_t* p,uint32_t a,uint32_t m,uint32_t n,int64_t stride,int32_t wait) {
  push((struct command){.op=7,.dst=p,.a=a,.m=m,.n=n,.stride=stride});if(wait)flush();
}
void tvm_gemmini_dma_relu_out(int8_t* p,uint32_t c,uint32_t m,uint32_t n,int64_t stride,int32_t wait) {
  push((struct command){.op=8,.dst=p,.c=c,.m=m,.n=n,.stride=stride});if(wait)flush();
}
"""


def workload(m, n, k, name="arbitrary_reduction", dtype="int8"):
    a = te.placeholder((m, k), dtype, "left")
    b = te.placeholder((k, n), dtype, "right")
    r = te.reduce_axis((0, k), "reduce_axis")
    c = te.compute(
        (m, n),
        lambda i, j: te.sum(a[i, r].astype("int32") * b[r, j].astype("int32"), axis=r),
        name=name,
    )
    return te.create_prim_func([a, b, c])


class GemminiTensorIRTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gemmini-tensorir-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.emulator = self.root / "emulator.c"
        self.emulator.write_text(EMULATOR)
        self.index = 0

    def build(self, mod):
        built = tvm.build(mod, "llvm")
        self.index += 1
        path = self.root / f"kernel-{self.index}.so"
        built.export_library(str(path), addons=[str(self.emulator)], cc="cc", options=["-std=c11"])
        control = ctypes.CDLL(str(path))
        control.emulator_reset()
        return tvm.runtime.load_module(str(path)), control

    def test_shapes_orders_and_delayed_dma_match_independent_integer_reference(self):
        for shape in ((1, 1, 1), (3, 4, 5), (16, 16, 16), (32, 32, 32), (17, 19, 33)):
            for order in ("ij", "ji"):
                with self.subTest(shape=shape, order=order):
                    m, n, k = shape
                    scheduled = make_tensorir_schedule(workload(m, n, k), order=order)
                    runtime, control = self.build(scheduled.scheduled_mod)
                    a = ((np.arange(m * k).reshape(m, k) * 13) % 256 - 128).astype("int8")
                    b = ((np.arange(k * n).reshape(k, n) * 7) % 256 - 128).astype("int8")
                    for left in (a, np.full_like(a, -128), np.zeros_like(a)):
                        output = tvm.nd.array(np.full((m, n), 123, "int32"))
                        av, bv = tvm.nd.array(left), tvm.nd.array(b)
                        control.emulator_reset()
                        runtime(av, bv, output)
                        np.testing.assert_array_equal(
                            output.numpy(), left.astype("int64") @ b.astype("int64")
                        )
                        np.testing.assert_array_equal(av.numpy(), left)
                        np.testing.assert_array_equal(bv.numpy(), b)
                        self.assertEqual(control.emulator_faults(), 0)
                        self.assertEqual(control.emulator_pending(), 0)
                        self.assertGreater(control.emulator_commands(), 0)
                    self.assertLessEqual(scheduled.metadata["spad_rows"], 16384)
                    self.assertLessEqual(scheduled.metadata["acc_rows"], 1024)

    def test_trace_replays_from_original_untiled_workload(self):
        original = workload(17, 19, 33, "names_do_not_select_the_schedule")
        result = default_schedule(original)
        replay = tir.Schedule(original, debug_mask="all")
        result.trace.apply_to_schedule(replay, remove_postproc=True)
        tvm.ir.assert_structural_equal(result.mod, replay.mod)
        kinds = {item.kind.name for item in result.trace.insts}
        self.assertTrue(
            {
                "SamplePerfectTile",
                "Split",
                "Reorder",
                "CacheRead",
                "CacheWrite",
                "ComputeAt",
                "DecomposeReduction",
                "Tensorize",
            }
            <= kinds
        )
        other = default_schedule(workload(32, 32, 33), order="ji")
        self.assertNotEqual(str(default_schedule(workload(32, 32, 33)).trace), str(other.trace))

    def test_matrix_cache_groups_reuse_dma_and_preserve_tails(self):
        for m, n, k in ((64, 64, 32), (17, 19, 33), (3, 4, 5)):
            a = ((np.arange(m * k).reshape(m, k) * 13) % 256 - 128).astype("int8")
            b = ((np.arange(k * n).reshape(k, n) * 7) % 256 - 128).astype("int8")
            commands = {}
            for row_tiles, col_tiles in ((1, 1), (2, 1), (1, 2), (2, 2), (4, 4)):
                for order in ("ij", "ji"):
                    try:
                        result = make_tensorir_schedule(
                            workload(m, n, k), order=order, row_tiles=row_tiles, col_tiles=col_tiles
                        )
                    except ValueError:
                        # Groups must divide the tile counts; odd shapes keep unit groups.
                        self.assertNotEqual(m, 64)
                        continue
                    runtime, control = self.build(result.scheduled_mod)
                    output = tvm.nd.empty((m, n), "int32")
                    runtime(tvm.nd.array(a), tvm.nd.array(b), output)
                    np.testing.assert_array_equal(
                        output.numpy(), a.astype("int64") @ b.astype("int64")
                    )
                    self.assertEqual(control.emulator_faults(), 0)
                    commands[(row_tiles, col_tiles)] = control.emulator_commands()
                    if m == 64:
                        self.assertEqual(result.metadata["acc_rows"], 16 * row_tiles * col_tiles)
            if m == 64:
                # Larger reuse groups move fewer operand tiles and issue fewer commands.
                self.assertLess(commands[(4, 4)], commands[(1, 1)])

    def test_missing_physical_sram_row_alignment_is_rejected(self):
        from tvm.relax.backend.contrib.gemmini_memory import lower_memory

        schedule = default_schedule(workload(3, 4, 5))

        def erase_alignment(node):
            if isinstance(node, tir.Block) and "buffer_dim_align" in node.annotations:
                return tir.Block(
                    node.iter_vars,
                    node.reads,
                    node.writes,
                    node.name_hint,
                    node.body,
                    node.init,
                    node.alloc_buffers,
                    node.match_buffers,
                    {
                        key: value
                        for key, value in node.annotations.items()
                        if key != "buffer_dim_align"
                    },
                )
            return None

        original = schedule.mod["main"]
        broken = original.with_body(
            tir.stmt_functor.ir_transform(original.body, None, erase_alignment)
        )
        with self.assertRaisesRegex(ValueError, "physical row stride"):
            lower_memory(tvm.IRModule({"main": broken}))

    def test_convolution_layout_scheduling_replay_and_independent_oracle(self):
        # Ordinary LegalizeOps supplies the sliding-window reduction. There is
        # no Gemmini-specific Relax convolution or prepacked im2col operator.
        for shape, kernel, stride, dilation, padding in (
            ((1, 2, 6), (3, 2, 3), (2,), (1,), (1, 1)),
            ((1, 2, 5, 5), (3, 2, 3, 3), (1, 1), (1, 1), (1, 1, 1, 1)),
            ((2, 2, 4, 5), (3, 2, 2, 3), (2, 1), (1, 2), (1, 0, 0, 1)),
            ((1, 1, 3, 3, 3), (2, 1, 2, 2, 2), (1, 1, 1), (1, 1, 1), (0, 0, 0, 0, 0, 0)),
        ):
            dimensions = len(stride)
            a = relax.Var("data", relax.TensorStructInfo(shape, "int8"))
            b = relax.Var("filter", relax.TensorStructInfo(kernel, "int8"))
            builder = relax.BlockBuilder()
            with builder.function("main", [a, b]):
                with builder.dataflow():
                    result = builder.emit_output(
                        getattr(relax.op.nn, f"conv{dimensions}d")(
                            a,
                            b,
                            strides=stride,
                            dilation=dilation,
                            padding=padding,
                            out_dtype="int32",
                        )
                    )
                builder.emit_func_output(result)
            legalized = relax.transform.LegalizeOps()(builder.get())
            func = next(f for f in legalized.functions.values() if isinstance(f, tir.PrimFunc))
            func = func.with_attr("global_symbol", "main")
            av = ((np.arange(np.prod(shape)).reshape(shape) * 17) % 256 - 128).astype("int8")
            bv = ((np.arange(np.prod(kernel)).reshape(kernel) * 11) % 256 - 128).astype("int8")
            output_shape = tuple(
                (
                    shape[d + 2]
                    + padding[d]
                    + padding[d + dimensions]
                    - dilation[d] * (kernel[d + 2] - 1)
                    - 1
                )
                // stride[d]
                + 1
                for d in range(dimensions)
            )
            expected = np.zeros((shape[0], kernel[0], *output_shape), "int64")
            for n, oc, *position in np.ndindex(expected.shape):
                for ic, *tap in np.ndindex(kernel[1:]):
                    source = tuple(
                        position[d] * stride[d] + tap[d] * dilation[d] - padding[d]
                        for d in range(dimensions)
                    )
                    if all(0 <= source[d] < shape[d + 2] for d in range(dimensions)):
                        expected[(n, oc, *position)] += int(av[(n, ic, *source)]) * int(
                            bv[(oc, ic, *tap)]
                        )
            for order in ("ij", "ji"):
                scheduled = make_tensorir_schedule(func, order=order)
                replay = tir.Schedule(func, debug_mask="all")
                scheduled.trace.apply_to_schedule(replay, remove_postproc=True)
                kinds = {item.kind.name for item in scheduled.trace.insts}
                self.assertTrue(
                    {"TransformBlockLayout", "TransformLayout", "ReIndex", "ComputeAt", "Tensorize"}
                    <= kinds
                )
                runtime, control = self.build(scheduled.scheduled_mod)
                output = tvm.nd.empty(expected.shape, "int32")
                runtime(tvm.nd.array(av), tvm.nd.array(bv), output)
                np.testing.assert_array_equal(output.numpy(), expected)
                self.assertEqual(control.emulator_faults(), 0)
            report = []
            prepared = prepare_gemmini_graph(builder.get(), report=report)
            self.assertEqual(sum(row["placement"] == "gemmini" for row in report), 1, report)
            self.assertEqual(
                len([f for f in prepared.functions.values() if isinstance(f, tir.PrimFunc)]), 1
            )

    def test_normal_relax_fusion_batched_matmul_bias_relu_and_cpu_fallback(self):
        for batch in (False, True):
            a_shape, b_shape = ((2, 3, 5), (2, 5, 4)) if batch else ((3, 5), (5, 4))
            a = relax.Var("input_a", relax.TensorStructInfo(a_shape, "int8"))
            b = relax.Var("input_b", relax.TensorStructInfo(b_shape, "int8"))
            bias = relax.Var("bias", relax.TensorStructInfo((4,), "int32"))
            builder = relax.BlockBuilder()
            with builder.function("main", [a, b, bias]):
                with builder.dataflow():
                    product = builder.emit(relax.op.matmul(a, b, out_dtype="int32"))
                    result = builder.emit_output(relax.op.nn.relu(relax.op.add(product, bias)))
                builder.emit_func_output(result)
            original = builder.get()
            for optimize in (False, True):
                report = []
                mod = prepare_gemmini_graph(original, optimize=optimize, report=report)
                self.assertEqual(sum(row["placement"] == "gemmini" for row in report), 1)
                if optimize:
                    self.assertEqual(
                        len([f for f in mod.functions.values() if isinstance(f, tir.PrimFunc)]), 1
                    )
                self.index += 1
                path = self.root / f"graph-{self.index}.so"
                executable = relax.build(mod, "llvm")
                executable.export_library(
                    str(path), addons=[str(self.emulator)], cc="cc", options=["-std=c11"]
                )
                control = ctypes.CDLL(str(path))
                vm = relax.VirtualMachine(tvm.runtime.load_module(str(path)), tvm.cpu())
                av = (np.arange(np.prod(a_shape)).reshape(a_shape) - 10).astype("int8")
                bv = (np.arange(np.prod(b_shape)).reshape(b_shape) - 5).astype("int8")
                dv = np.array([-5, 2, 3, 4], "int32")
                control.emulator_reset()
                output = vm["main"](tvm.nd.array(av), tvm.nd.array(bv), tvm.nd.array(dv)).numpy()
                np.testing.assert_array_equal(
                    output, np.maximum(av.astype("int64") @ bv.astype("int64") + dv, 0)
                )
                self.assertGreater(control.emulator_commands(), 0)
                self.assertEqual(control.emulator_faults(), 0)
                self.assertEqual(control.emulator_pending(), 0)

        cpu = relax.Var("float_input", relax.TensorStructInfo((3, 4), "float32"))
        builder = relax.BlockBuilder()
        with builder.function("main", [cpu]):
            builder.emit_func_output(builder.emit(relax.op.nn.relu(cpu)))
        report = []
        mod = prepare_gemmini_graph(builder.get(), report=report)
        self.assertTrue(report and all(row["placement"] == "cpu" for row in report))
        vm = relax.VirtualMachine(relax.build(mod, "llvm"), tvm.cpu())
        data = np.arange(12, dtype="float32").reshape(3, 4) - 6
        np.testing.assert_array_equal(vm["main"](tvm.nd.array(data)).numpy(), np.maximum(data, 0))

    def test_refuses_unsupported_mathematics_and_arithmetic(self):
        self.assertFalse(contraction_block(workload(2, 3, 4, dtype="uint8")))
        self.assertFalse(contraction_block(workload(1, 1, 131072)))
        original = workload(3, 4, 5)

        def change(node):
            if isinstance(node, tir.Mul) and node.dtype == "int32":
                return node + 1
            return None

        changed = original.with_body(tir.stmt_functor.ir_transform(original.body, None, change))
        self.assertFalse(contraction_block(changed))
        with self.assertRaises(ValueError):
            make_tensorir_schedule(changed)

    def test_wrong_primitive_fails_independent_oracle(self):
        scheduled = make_tensorir_schedule(workload(3, 4, 5))
        runtime, control = self.build(scheduled.scheduled_mod)
        control.emulator_wrong(1)
        output = tvm.nd.empty((3, 4), "int32")
        runtime(
            tvm.nd.array(np.ones((3, 5), "int8")), tvm.nd.array(np.ones((5, 4), "int8")), output
        )
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(output.numpy(), np.full((3, 4), 5))

    def test_fused_scalar_input_and_boolean_output_use_byte_storage(self):
        a = relax.Var("a", relax.TensorStructInfo((3, 5), "int8"))
        b = relax.Var("b", relax.TensorStructInfo((5, 4), "int8"))
        threshold = relax.Var("threshold", relax.TensorStructInfo((), "float32"))
        builder = relax.BlockBuilder()
        with builder.function("main", [a, b, threshold]):
            with builder.dataflow():
                product = builder.emit(relax.op.matmul(a, b, out_dtype="int32"))
                result = builder.emit_output(
                    relax.op.greater(relax.op.astype(product, "float32"), threshold)
                )
            builder.emit_func_output(result)
        report = []
        mod = prepare_gemmini_graph(builder.get(), report=report)
        self.assertEqual(sum(row["placement"] == "gemmini" for row in report), 1, report)
        self.assertEqual(len([f for f in mod.functions.values() if isinstance(f, tir.PrimFunc)]), 1)
        executable = relax.build(mod, "llvm")
        path = self.root / "scalar-bool.so"
        executable.export_library(
            str(path), addons=[str(self.emulator)], cc="cc", options=["-std=c11"]
        )
        control = ctypes.CDLL(str(path))
        control.emulator_reset()
        vm = relax.VirtualMachine(tvm.runtime.load_module(str(path)), tvm.cpu())
        av = np.arange(15, dtype="int8").reshape(3, 5) - 7
        bv = np.arange(20, dtype="int8").reshape(5, 4) - 10
        output = vm["main"](
            tvm.nd.array(av), tvm.nd.array(bv), tvm.nd.array(np.array(0.5, "float32"))
        )
        np.testing.assert_array_equal(
            output.numpy(), (av.astype("int64") @ bv.astype("int64")) > 0.5
        )
        self.assertEqual(control.emulator_faults(), 0)

    def test_standard_metaschedule_rules_features_and_tensorization_postprocessing(self):
        from tvm import meta_schedule as ms
        from tvm.relax.backend.contrib.gemmini_meta_schedule import gemmini_space_generator
        from tvm.relax.backend.contrib.gemmini_target import GemminiCapabilities

        func = workload(17, 19, 33)
        caps = GemminiCapabilities()
        context = ms.TuneContext(
            mod=func,
            target=caps.target(),
            space_generator=gemmini_space_generator(caps),
            num_threads=1,
        )
        designs = context.generate_design_space()
        self.assertEqual(len({str(design.trace) for design in designs}), len(designs))
        candidates = [ms.search_strategy.MeasureCandidate(design, []) for design in designs]
        features = ms.feature_extractor.PerStoreFeature().extract_from(context, candidates)
        self.assertEqual(len(features), len(designs))
        self.assertTrue(
            all(np.isfinite(feature.numpy()).all() and feature.numpy().size for feature in features)
        )
        a = ((np.arange(17 * 33).reshape(17, 33) * 13) % 256 - 128).astype("int8")
        b = ((np.arange(33 * 19).reshape(33, 19) * 7) % 256 - 128).astype("int8")
        native = 0
        for design in designs:
            design.enter_postproc()
            self.assertTrue(
                all(postproc.apply(design) for postproc in context.space_generator.postprocs)
            )
            if "tvm_gemmini" not in design.mod.script():
                continue  # CPU placement of the same function
            native += 1
            lowered = lower_tensorir_schedule(func, design)
            runtime, control = self.build(lowered.scheduled_mod)
            output = tvm.nd.empty((17, 19), "int32")
            runtime(tvm.nd.array(a), tvm.nd.array(b), output)
            np.testing.assert_array_equal(output.numpy(), a.astype("int64") @ b.astype("int64"))
            self.assertEqual(control.emulator_faults(), 0)
        self.assertGreater(native, 1)
        self.assertGreater(len(designs), native)

    def test_missing_staging_wait_is_detected_by_queued_dma(self):
        # A padded 16-deep chunk stages the operand tails per tile on the host.
        result = make_tensorir_schedule(workload(17, 19, 33), reduction_chunk=16)
        func = result.scheduled_mod["main"]

        def remove_wait(node):
            if isinstance(node, tir.Call) and node.op == tvm.ir.Op.get("tir.call_extern"):
                if node.args[0].value in (
                    "tvm_gemmini_dma_a",
                    "tvm_gemmini_dma_b",
                    "tvm_gemmini_dma_out",
                ):
                    return tir.call_extern(
                        "void", node.args[0].value, *node.args[1:-1], tir.IntImm("int32", 0)
                    )
            return None

        bad = func.with_body(tir.stmt_functor.ir_transform(func.body, None, remove_wait))
        runtime, _ = self.build(tvm.IRModule({"main": bad}))
        # Distinct inputs prevent an uninitialized staging read from matching
        # a preceding positive test's recycled allocation by accident.
        rng = np.random.default_rng(6379)
        a = rng.integers(-128, 128, (17, 33), dtype="int8")
        b = rng.integers(-128, 128, (33, 19), dtype="int8")
        output = tvm.nd.array(np.full((17, 19), 0x6D6D6D6D, "int32"))
        runtime(tvm.nd.array(a), tvm.nd.array(b), output)
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(output.numpy(), a.astype("int64") @ b.astype("int64"))

    def test_database_selection_retains_chosen_trace_and_resource_refusals(self):
        from tvm import meta_schedule as ms
        from tvm.relax.backend.contrib.gemmini_meta_schedule import gemmini_space_generator
        from tvm.relax.backend.contrib.gemmini_target import GemminiCapabilities
        from tvm.relax.backend.contrib.gemmini_tensorir import ScheduleGemminiTensorIR

        func = workload(32, 32, 16)
        caps = GemminiCapabilities()
        chosen = default_schedule(func, order="ji", row_tiles=2, col_tiles=2)
        database = ms.database.MemoryDatabase()
        record_workload = database.commit_workload(tvm.IRModule({"main": func}))
        # Diagnostic selection test only: these invented costs are never target
        # measurements or evidence of a performance winner.
        database.commit_tuning_record(
            ms.database.TuningRecord(chosen.trace, record_workload, [1.0], caps.target(), [])
        )
        report = []
        with caps.target(), database:
            mod = relax.transform.MetaScheduleApplyDatabase()(tvm.IRModule({"main": func}))
        mod = ScheduleGemminiTensorIR(report=report)(mod)
        self.assertEqual(report[0]["origin"], "TVM MetaSchedule database")
        self.assertEqual(report[0]["placement"], "gemmini")
        expected = lower_tensorir_schedule(func, chosen).scheduled_mod["main"]
        tvm.ir.assert_structural_equal(mod["main"].body, expected.body)
        with self.assertRaisesRegex(ValueError, "SRAM capacity"):
            make_tensorir_schedule(func, capabilities=GemminiCapabilities(acc_rows=15))
        fallback_report = []
        fallback = ScheduleGemminiTensorIR(
            capabilities=GemminiCapabilities(acc_rows=15), report=fallback_report
        )(tvm.IRModule({"main": func}))
        tvm.ir.assert_structural_equal(fallback["main"], func)
        self.assertEqual(fallback_report[0]["placement"], "cpu")
        with self.assertRaisesRegex(ValueError, "target key"):
            ms.TuneContext(
                mod=func, target="llvm", space_generator=gemmini_space_generator()
            ).generate_design_space()


if __name__ == "__main__":
    unittest.main()
