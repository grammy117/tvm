/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "matmul.h"

#include <stddef.h>
#include <stdint.h>

#ifndef TVM_GEMMINI_FORBID_HW_LOOPS
#define TVM_GEMMINI_FORBID_HW_LOOPS 0
#endif
#if TVM_GEMMINI_FORBID_HW_LOOPS != 1
#error "The OS C-library adapter requires explicit TVM_GEMMINI_FORBID_HW_LOOPS=1"
#endif
#if !defined(TVM_GEMMINI_PE_OUTPUT_BITS) || TVM_GEMMINI_PE_OUTPUT_BITS != 20
#error "Bind the source-verified 20-bit PE output using TVM_GEMMINI_PE_OUTPUT_BITS=20"
#endif
#if !defined(__riscv) || __riscv_xlen != 64
#error "The Gemmini adapter requires an RV64 target"
#endif

/* Load the selected generated header before the library. Do not change its bytes.
 * The pinned library parses normalization branches even when hardware support is
 * absent. This parsing fallback enables no normalization feature. Readout uses
 * either NO_ACTIVATION or the separately admitted exact INT8 ReLU path.
 */
#include "include/gemmini_params.h"
#ifdef HAS_NORMALIZATIONS
#error "This Gemmini adapter does not admit a normalization configuration"
#endif
#ifndef NORM_STAT_IDS
#define NORM_STAT_IDS 2
#endif
#include "include/gemmini.h"

#if defined(ELEM_T_IS_FLOAT) || defined(ELEM_T_IS_LOWPREC_FLOAT)
#error "The Gemmini adapter requires signed integer operands"
#endif
#if !defined(HAS_MVIN_SCALE) || !defined(ACC_SCALE_T_IS_FLOAT)
#error "The Gemmini adapter requires floating-point identity input/output scales"
#endif
#ifndef ACC_READ_FULL_WIDTH
#error "The Gemmini adapter requires full-width accumulator readout"
#endif

_Static_assert(CHAR_BIT == 8 && sizeof(elem_t) == 1 && (elem_t)-1 < 0, "elem_t must be signed int8");
_Static_assert(sizeof(acc_t) == 4 && (acc_t)-1 < 0, "acc_t must be signed int32");
_Static_assert(_Generic((scale_t)0, float: 1, default: 0), "scale_t must be float");
_Static_assert(_Generic((acc_scale_t)0, float: 1, default: 0), "acc_scale_t must be float");
_Static_assert(sizeof(size_t) == 8 && sizeof(uintptr_t) == 8, "RV64 pointer/size ABI required");
_Static_assert(DIM == 16, "the bounded OS schedule requires DIM=16");
_Static_assert(BANK_NUM * BANK_ROWS >= 2 * DIM && ACC_ROWS >= DIM, "insufficient resources for an OS tile");
_Static_assert(ADDR_LEN == 32, "the primitive ABI requires 32-bit local addresses");
_Static_assert(BANK_NUM * BANK_ROWS == 16384 && ACC_ROWS == 1024, "the primitive row allocation contract requires the pinned capacities");

/* Conservative byte envelopes include stride padding. Callers must own and map
 * the entire envelope; this validates representability, not allocation or DMA
 * accessibility. The selected runtime must serialize use of the accelerator.
 */
static int span(const void* data, uint64_t rows, uint64_t cols, uint64_t stride, size_t element_bytes, uintptr_t* begin, uintptr_t* end) {
  uint64_t max_elements = (uint64_t)PTRDIFF_MAX / element_bytes;
  if (data == NULL || stride < cols || cols > max_elements || rows - 1 > (max_elements - cols) / stride) {
    return 0;
  }
  uint64_t bytes = ((rows - 1) * stride + cols) * element_bytes;
  *begin = (uintptr_t)data;
  if (bytes > UINTPTR_MAX - *begin) {
    return 0;
  }
  *end = *begin + bytes;
  return 1;
}

int32_t tvm_gemmini_validate_matmul_i8_i32(const int8_t* a, const int8_t* b, int32_t* c, int64_t m, int64_t n, int64_t k,
                                         int64_t a_stride, int64_t b_stride, int64_t c_stride) {
  if (m <= 0 || n <= 0 || k <= 0 || k > 131071 || a_stride <= 0 || b_stride <= 0 || c_stride <= 0 || (uintptr_t)c % _Alignof(int32_t) != 0) {
    return -1;
  }
  /* CONFIG_ST packs its byte stride into 32 bits; null D still gets configured.
   * Apply the same conservative limit to input DMA strides.
   */
  if ((uint64_t)a_stride > UINT32_MAX || (uint64_t)b_stride > UINT32_MAX ||
      (uint64_t)c_stride > UINT32_MAX / sizeof(acc_t) || (uint64_t)n > UINT32_MAX / sizeof(acc_t)) {
    return -2;
  }
  uintptr_t a_begin, a_end, b_begin, b_end, c_begin, c_end;
  if (!span(a, m, k, a_stride, sizeof(elem_t), &a_begin, &a_end) ||
      !span(b, k, n, b_stride, sizeof(elem_t), &b_begin, &b_end) ||
      !span(c, m, n, c_stride, sizeof(acc_t), &c_begin, &c_end)) {
    return -2;
  }
  if ((c_begin < a_end && a_begin < c_end) || (c_begin < b_end && b_begin < c_end)) {
    return -3;
  }
  if (MVIN_SCALE_IDENTITY != 1.0f || ACC_SCALE_IDENTITY != 1.0f) {
    return -4;
  }
  return 0;
}

/* Match the vendor OS helper's accumulator addressing. The caller supplies row
 * allocation only; this wrapper owns the source-derived instruction flags.
 */
static uint32_t accumulator_address(uint32_t row, int32_t accumulate) {
  return (1U << (ADDR_LEN - 1)) | (1U << (ADDR_LEN - 3)) | ((uint32_t)accumulate << (ADDR_LEN - 2)) | row;
}

/* Configuration and weight-stationary state shadowed by the adapter. Each
 * command stream keeps its configuration until the next config command of the
 * same kind, so an identical reconfiguration is redundant. The shadow is
 * invalidated by begin() and by the vendor tiled-matmul path, which issue
 * their own configuration commands.
 */
static struct {
  int valid_ld[3];
  uint64_t ld_stride[3];
  uint32_t ld_block[3];
  int ld_shrunk[3];
  int valid_st;
  uint64_t st_stride;
  int st_act;
  uint32_t st_scale;
  int valid_ex;
  int ex_dataflow;
  /* Weights currently held by the array in weight-stationary mode. */
  int valid_ws;
  uint32_t ws_row, ws_rows, ws_cols;
} shadow;

static void invalidate_shadow(void) {
  for (int i = 0; i < 3; ++i) shadow.valid_ld[i] = 0;
  shadow.valid_st = 0;
  shadow.valid_ex = 0;
  shadow.valid_ws = 0;
}

static void config_ld(int id, uint64_t stride, int shrunk, uint32_t block) {
  if (shadow.valid_ld[id] && shadow.ld_stride[id] == stride && shadow.ld_shrunk[id] == shrunk &&
      shadow.ld_block[id] == block) {
    return;
  }
  gemmini_extended4_config_ld(stride, MVIN_SCALE_IDENTITY, shrunk, block, id);
  shadow.valid_ld[id] = 1;
  shadow.ld_stride[id] = stride;
  shadow.ld_shrunk[id] = shrunk;
  shadow.ld_block[id] = block;
}

static void config_st(uint64_t stride, int act, float scale) {
  union { uint32_t bits; float value; } s = {.value = scale};
  if (shadow.valid_st && shadow.st_stride == stride && shadow.st_act == act && shadow.st_scale == s.bits) {
    return;
  }
  gemmini_extended_config_st(stride, act, scale);
  shadow.valid_st = 1;
  shadow.st_stride = stride;
  shadow.st_act = act;
  shadow.st_scale = s.bits;
}

static void config_ex(int dataflow) {
  if (shadow.valid_ex && shadow.ex_dataflow == dataflow) {
    return;
  }
  gemmini_config_ex(dataflow, NO_ACTIVATION, 0);
  shadow.valid_ex = 1;
  shadow.ex_dataflow = dataflow;
  shadow.valid_ws = 0;
}

/* A scratchpad load into rows still feeding preloaded weights replaces the
 * source, not the array copy; keep reuse only for disjoint destinations. */
static void scratchpad_written(uint32_t row, uint32_t rows) {
  if (shadow.valid_ws && row < shadow.ws_row + shadow.ws_rows && shadow.ws_row < row + rows) {
    shadow.valid_ws = 0;
  }
}

void tvm_gemmini_begin(int64_t a_stride, int64_t b_stride, int64_t c_stride) {
  /* Flush stale translations once. CPU fences do not establish DMA coherence;
   * exclusive ownership and the platform completion contract remain required.
   */
  asm volatile("fence rw,rw" ::: "memory");
  gemmini_flush(0);
  invalidate_shadow();
  config_ex(OUTPUT_STATIONARY);
  config_st((uint64_t)c_stride * sizeof(acc_t), NO_ACTIVATION, ACC_SCALE_IDENTITY);
  config_ld(0, (uint64_t)a_stride, false, DIM);
  config_ld(1, (uint64_t)b_stride, false, DIM);
}

void tvm_gemmini_load_a(const int8_t* src, uint32_t spadrow, uint32_t rows, uint32_t cols) {
  scratchpad_written(spadrow, rows);
  asm volatile("" ::: "memory");
  gemmini_extended_mvin(src, spadrow, cols, rows);
  asm volatile("" ::: "memory");
}

void tvm_gemmini_load_b(const int8_t* src, uint32_t spadrow, uint32_t rows, uint32_t cols) {
  scratchpad_written(spadrow, rows);
  asm volatile("" ::: "memory");
  gemmini_extended_mvin2(src, spadrow, cols, rows);
  asm volatile("" ::: "memory");
}

void tvm_gemmini_compute(uint32_t arow, uint32_t brow, uint32_t crow, uint32_t m, uint32_t n, uint32_t k, int32_t accumulate) {
  /* Restart the PE reduction for every chunk; accumulation across global K
   * happens in int32 SRAM, preserving the 20-bit PE bound for <=16 products.
   * accumulate=0 overwrites the accumulator rows instead of adding to them.
   */
  config_ex(OUTPUT_STATIONARY);
  gemmini_extended_preload(GARBAGE_ADDR, accumulator_address(crow, accumulate), DIM, DIM, n, m);
  gemmini_extended_compute_preloaded(arow, brow, k, m, n, k);
}

void tvm_gemmini_compute_ws(uint32_t arow, uint32_t brow, uint32_t crow, uint32_t m, uint32_t n, uint32_t k, int32_t accumulate) {
  /* Weight-stationary form of the same contraction. Consecutive calls with the
   * same B rectangle stream new A rows through weights already in the array,
   * as the vendor WS loop does with GARBAGE preloads and accumulated computes.
   */
  config_ex(WEIGHT_STATIONARY);
  uint32_t out = accumulator_address(crow, accumulate);
  if (shadow.valid_ws && shadow.ws_row == brow && shadow.ws_rows == k && shadow.ws_cols == n) {
    gemmini_extended_preload(GARBAGE_ADDR, out, n, k, n, m);
    gemmini_extended_compute_accumulated(arow, GARBAGE_ADDR, k, m, DIM, DIM);
  } else {
    gemmini_extended_preload(brow, out, n, k, n, m);
    gemmini_extended_compute_preloaded(arow, GARBAGE_ADDR, k, m, DIM, DIM);
    shadow.valid_ws = 1;
    shadow.ws_row = brow;
    shadow.ws_rows = k;
    shadow.ws_cols = n;
  }
}

void tvm_gemmini_store(int32_t* dst, uint32_t crow, uint32_t rows, uint32_t cols) {
  asm volatile("" ::: "memory");
  gemmini_extended_mvout(dst, accumulator_address(crow, 1), cols, rows);
  asm volatile("" ::: "memory");
}

void tvm_gemmini_end(void) {
  gemmini_fence();
  asm volatile("fence rw,rw" ::: "memory");
}

int32_t tvm_gemmini_validate_tensor_output(void* c, int64_t c_bytes) {
  if (!c || c_bytes <= 0 || (uintptr_t)c % _Alignof(acc_t)) return -1;
  if ((uint64_t)c_bytes > UINTPTR_MAX - (uintptr_t)c) return -2;
  return 0;
}

int32_t tvm_gemmini_validate_tensor_buffers(const void* a, int64_t a_bytes, const void* b, int64_t b_bytes,
                                          void* c, int64_t c_bytes) {
  if (!a || !b || !c || a_bytes <= 0 || b_bytes <= 0 || c_bytes <= 0 ||
      (uintptr_t)c % _Alignof(acc_t)) return -1;
  uintptr_t aa = (uintptr_t)a, bb = (uintptr_t)b, cc = (uintptr_t)c;
  if ((uint64_t)a_bytes > UINTPTR_MAX-aa || (uint64_t)b_bytes > UINTPTR_MAX-bb ||
      (uint64_t)c_bytes > UINTPTR_MAX-cc) return -2;
  if ((cc < aa + a_bytes && aa < cc + c_bytes) ||
      (cc < bb + b_bytes && bb < cc + c_bytes)) return -3;
  return 0;
}

void tvm_gemmini_zero_acc(uint32_t row, uint32_t rows, uint32_t cols) {
  /* LoadController marks vaddr=0 as all_zeros: no host zero buffer is read.
   * Channel 2 uses full-width accumulator DMA, independently of A/B loads. */
  config_ld(2, DIM * sizeof(acc_t), false, DIM);
  gemmini_extended_mvin3(NULL, (1U << (ADDR_LEN - 1)) | row, cols, rows);
}

void tvm_gemmini_dma_a(const int8_t* src, uint32_t row, uint32_t rows, uint32_t cols, int64_t stride, int32_t wait) {
  config_ld(0, (uint64_t)stride, false, DIM);
  tvm_gemmini_load_a(src, row, rows, cols);
  if (wait) tvm_gemmini_end();
}

void tvm_gemmini_dma_b(const int8_t* src, uint32_t row, uint32_t rows, uint32_t cols, int64_t stride, int32_t wait) {
  config_ld(1, (uint64_t)stride, false, DIM);
  tvm_gemmini_load_b(src, row, rows, cols);
  if (wait) tvm_gemmini_end();
}

/* One mvin of up to MAX_BLOCK_LEN DIM-column blocks; block b lands at row
 * + b*block_rows, so consecutive tiles of a tile-major cache load together. */
void tvm_gemmini_dma_a_blocks(const int8_t* src, uint32_t row, uint32_t rows, uint32_t cols, int64_t stride,
                              uint32_t block_rows, int32_t wait) {
  config_ld(0, (uint64_t)stride, false, block_rows);
  scratchpad_written(row, block_rows * ((cols + DIM - 1) / DIM));
  asm volatile("" ::: "memory");
  gemmini_extended_mvin(src, row, cols, rows);
  asm volatile("" ::: "memory");
  if (wait) tvm_gemmini_end();
}

void tvm_gemmini_dma_b_blocks(const int8_t* src, uint32_t row, uint32_t rows, uint32_t cols, int64_t stride,
                              uint32_t block_rows, int32_t wait) {
  config_ld(1, (uint64_t)stride, false, block_rows);
  scratchpad_written(row, block_rows * ((cols + DIM - 1) / DIM));
  asm volatile("" ::: "memory");
  gemmini_extended_mvin2(src, row, cols, rows);
  asm volatile("" ::: "memory");
  if (wait) tvm_gemmini_end();
}

void tvm_gemmini_dma_out(int32_t* dst, uint32_t row, uint32_t rows, uint32_t cols, int64_t stride, int32_t wait) {
  config_st((uint64_t)stride * sizeof(acc_t), NO_ACTIVATION, ACC_SCALE_IDENTITY);
  tvm_gemmini_store(dst, row, rows, cols);
  if (wait) tvm_gemmini_end();
}

void tvm_gemmini_dma_acc_write(const int32_t* src, uint32_t row, uint32_t rows, uint32_t cols, int64_t stride, int32_t wait) {
  config_ld(2, (uint64_t)stride * sizeof(acc_t), false, DIM);
  asm volatile("" ::: "memory");
  gemmini_extended_mvin3(src, accumulator_address(row, 0), cols, rows);
  asm volatile("" ::: "memory");
  if (wait) tvm_gemmini_end();
}

void tvm_gemmini_dma_acc_add(const int32_t* src, uint32_t row, uint32_t rows, uint32_t cols, int64_t stride, int32_t wait) {
  config_ld(2, (uint64_t)stride * sizeof(acc_t), false, DIM);
  asm volatile("" ::: "memory");
  gemmini_extended_mvin3(src, accumulator_address(row, 1), cols, rows);
  asm volatile("" ::: "memory");
  if (wait) tvm_gemmini_end();
}

void tvm_gemmini_dma_widen(const int8_t* src, uint32_t row, uint32_t rows, uint32_t cols, int64_t stride, int32_t wait) {
  config_ld(2, (uint64_t)stride, true, DIM);
  asm volatile("" ::: "memory");
  gemmini_extended_mvin3(src, accumulator_address(row, 0), cols, rows);
  asm volatile("" ::: "memory");
  if (wait) tvm_gemmini_end();
}

void tvm_gemmini_dma_spad_out(int8_t* dst, uint32_t row, uint32_t rows, uint32_t cols, int64_t stride, int32_t wait) {
  config_st((uint64_t)stride, NO_ACTIVATION, ACC_SCALE_IDENTITY);
  asm volatile("" ::: "memory");
  gemmini_extended_mvout(dst, row, cols, rows);
  asm volatile("" ::: "memory");
  if (wait) tvm_gemmini_end();
}

void tvm_gemmini_dma_relu_out(int8_t* dst, uint32_t row, uint32_t rows, uint32_t cols, int64_t stride, int32_t wait) {
  /* Clear read_full_acc_row for scaled/narrow readout. Callers establish that
   * every source value was widened from INT8; identity scaling is then exact.
   */
  uint32_t address = accumulator_address(row, 0) & ~(1U << (ADDR_LEN - 3));
  config_st((uint64_t)stride, RELU, ACC_SCALE_IDENTITY);
  asm volatile("" ::: "memory");
  gemmini_extended_mvout(dst, address, cols, rows);
  asm volatile("" ::: "memory");
  if (wait) tvm_gemmini_end();
}

void tvm_gemmini_dma_scaled_out(int8_t* dst, uint32_t row, uint32_t rows, uint32_t cols,
                               int64_t stride, uint32_t scale_bits, int32_t relu, int32_t wait) {
  /* Compiler admission proves the source RNE/saturation expression for every
   * INT32 input and supplies a positive finite tensor-wide FP32 scale. */
  union { uint32_t bits; float value; } scale = {.bits = scale_bits};
  uint32_t address = accumulator_address(row, 0) & ~(1U << (ADDR_LEN - 3));
  config_st((uint64_t)stride, relu ? RELU : NO_ACTIVATION, scale.value);
  asm volatile("" ::: "memory");
  gemmini_extended_mvout(dst, address, cols, rows);
  asm volatile("" ::: "memory");
  if (wait) tvm_gemmini_end();
}

int32_t tvm_gemmini_matmul_i8_i32(const int8_t* a, const int8_t* b, int32_t* c, int64_t m, int64_t n, int64_t k,
                                int64_t a_stride, int64_t b_stride, int64_t c_stride) {
  int32_t admission = tvm_gemmini_validate_matmul_i8_i32(a, b, c, m, n, k, a_stride, b_stride, c_stride);
  if (admission != 0) {
    return admission;
  }

  /* Exclusive accelerator ownership is a caller obligation. Flush stale address
   * translations before this invocation; the runtime must make buffers DMA
   * accessible and provide the platform's completion/coherence contract.
   * The vendor fence lacks a compiler memory clobber. These barriers order CPU
   * accesses around its device fence; they cannot establish platform coherence.
   */
  asm volatile("fence rw,rw" ::: "memory");
  gemmini_flush(0);
  /* Fixed tile_K=1 bounds each PE reduction to 16 products: at most 262144 in
   * magnitude, within the source-verified signed 20-bit PE output. The library
   * accumulates successive K tiles in int32 SRAM. Auto tiling can exceed this
   * PE bound; do not substitute tiled_matmul_auto here.
   */
  tiled_matmul((size_t)m, (size_t)n, (size_t)k, a, b, NULL, c, (size_t)a_stride, (size_t)b_stride, (size_t)n, (size_t)c_stride,
               MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, 0, NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
               false, 1, 1, 1, false, false, true, false, 0, OS);
  invalidate_shadow();
  asm volatile("fence rw,rw" ::: "memory");
  return 0;
}
