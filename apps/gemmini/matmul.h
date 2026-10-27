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
 * See the License for the specific language governing permissions
 * and limitations under the License.
 */
#ifndef TVM_APPS_GEMMINI_MATMUL_H_
#define TVM_APPS_GEMMINI_MATMUL_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* C[m,n] = A[m,k] * B[k,n], signed exact arithmetic, no bias or scaling.
 * Strides count elements, with row-major nonoverlapping output storage.
 * Return 0 only after all output writes are complete and visible to the caller.
 * Preserve A/B. Nonzero signals failure; callers must not consume partial output.
 * LowerGemminiMatmul emits positive dimensions, k <= 131071, compact row strides.
 * Hardware/library implementations must independently enforce their capabilities.
 */
int32_t tvm_gemmini_matmul_i8_i32(const int8_t* a, const int8_t* b, int32_t* c, int64_t m, int64_t n, int64_t k, int64_t a_stride, int64_t b_stride, int64_t c_stride);

/* CPU-only whole-operation admission. Call before issuing any primitive.
 * The return values and memory contract match tvm_gemmini_matmul_i8_i32.
 */
int32_t tvm_gemmini_validate_matmul_i8_i32(const int8_t* a, const int8_t* b, int32_t* c, int64_t m, int64_t n, int64_t k, int64_t a_stride, int64_t b_stride, int64_t c_stride);

/* Trusted compiler-generated primitive contract: validation above passed; the
 * caller owns the accelerator exclusively from begin through end. TVM owns all
 * tile loops, pointer offsets, local allocation, reuse, and reduction ordering.
 * Begin uses the validated strides; every DMA pointer is a proven subview of
 * the validated A/B/C envelopes. Strides count elements. Local addresses count
 * DIM-element rows within 16384 scratchpad / 1024 accumulator rows. Rows of each
 * tile are consecutive. The legacy generator reserves 16-row slots; TensorIR
 * may pack shorter tiles at other integral row addresses.
 * A/B tile slots are disjoint. Every tile dimension is in [1,16], and accumulate
 * is exactly 0 (overwrite) or 1 (add to initialized int32 accumulator SRAM).
 * Each compute starts a fresh <=16-product PE reduction. The compiler proves
 * the whole INT32 accumulator range, or supplies explicit modulo-32 updates.
 * General exact INT8 matmul has K <=131071;
 * signed INT8 sums against ones can use K <=16777216. Primitive dimensions
 * remain <=16 in both cases; the larger bound does not widen the hardware.
 * A/B and any DMA source staging buffers remain immutable until end. Store is
 * asynchronous: read C only after end. These functions do no runtime admission.
 * The runtime must supply DMA mapping, completion, and platform coherence.
 */
void tvm_gemmini_begin(int64_t a_stride, int64_t b_stride, int64_t c_stride);
void tvm_gemmini_load_a(const int8_t* src, uint32_t spadrow, uint32_t rows, uint32_t cols);
void tvm_gemmini_load_b(const int8_t* src, uint32_t spadrow, uint32_t rows, uint32_t cols);
void tvm_gemmini_compute(uint32_t arow, uint32_t brow, uint32_t crow, uint32_t m, uint32_t n, uint32_t k, int32_t accumulate);
/* Same contract in weight-stationary mode. Consecutive calls naming the same
 * B rows and shape reuse the weights preloaded into the array. */
void tvm_gemmini_compute_ws(uint32_t arow, uint32_t brow, uint32_t crow, uint32_t m, uint32_t n, uint32_t k, int32_t accumulate);
void tvm_gemmini_store(int32_t* dst, uint32_t crow, uint32_t rows, uint32_t cols);
void tvm_gemmini_end(void);

/* TensorIR memory-scope lowering. Strides count host elements. A nonzero wait
 * protects CPU-private staging buffers whose next use can precede end(). */
void tvm_gemmini_zero_acc(uint32_t row, uint32_t rows, uint32_t cols);
void tvm_gemmini_dma_a(const int8_t* src, uint32_t row, uint32_t rows, uint32_t cols, int64_t stride, int32_t wait);
void tvm_gemmini_dma_b(const int8_t* src, uint32_t row, uint32_t rows, uint32_t cols, int64_t stride, int32_t wait);
/* Multi-block loads: cols <= MAX_BLOCK_LEN*DIM; DIM-column block b is written
 * at row + b*block_rows. */
void tvm_gemmini_dma_a_blocks(const int8_t* src, uint32_t row, uint32_t rows, uint32_t cols, int64_t stride,
                              uint32_t block_rows, int32_t wait);
void tvm_gemmini_dma_b_blocks(const int8_t* src, uint32_t row, uint32_t rows, uint32_t cols, int64_t stride,
                              uint32_t block_rows, int32_t wait);
void tvm_gemmini_dma_out(int32_t* dst, uint32_t row, uint32_t rows, uint32_t cols, int64_t stride, int32_t wait);
/* Full-width overwrite/add preserve INT32 bit patterns; addition wraps modulo
 * 2^32, as the accumulator SRAM adder does. No scaler is applied. Widen uses
 * signed INT8 input and full INT32 destination rows with identity input scale. */
void tvm_gemmini_dma_acc_write(const int32_t* src, uint32_t row, uint32_t rows, uint32_t cols, int64_t stride, int32_t wait);
void tvm_gemmini_dma_acc_add(const int32_t* src, uint32_t row, uint32_t rows, uint32_t cols, int64_t stride, int32_t wait);
void tvm_gemmini_dma_widen(const int8_t* src, uint32_t row, uint32_t rows, uint32_t cols, int64_t stride, int32_t wait);
/* Signed-byte load scaling, RNE/saturate each operand before INT32 addition.
 * Proven scales are in (0,1]; accumulate is 0 (overwrite) or 1 (add). */
void tvm_gemmini_dma_scaled_in(const int8_t* src, uint32_t row, uint32_t rows, uint32_t cols,
                              int64_t stride, uint32_t scale_bits, int32_t accumulate, int32_t wait);
void tvm_gemmini_dma_spad_out(int8_t* dst, uint32_t row, uint32_t rows, uint32_t cols, int64_t stride, int32_t wait);
/* Identity-scale narrow ReLU. Every read element must have been widened from
 * INT8, so no saturation or rounding changes a positive value. */
void tvm_gemmini_dma_relu_out(int8_t* dst, uint32_t row, uint32_t rows, uint32_t cols, int64_t stride, int32_t wait);
void tvm_gemmini_dma_scaled_out(int8_t* dst, uint32_t row, uint32_t rows, uint32_t cols,
                               int64_t stride, uint32_t scale_bits, int32_t relu, int32_t wait);
/* Values-only 2x2/3x3 signed-byte max; size*size complete widened input rows,
 * one channel, no physical padding, one output byte. */
void tvm_gemmini_dma_pool_out(int8_t* dst, uint32_t row, uint32_t size, int32_t wait);
/* Admission for a literal fill with no external source buffers. */
int32_t tvm_gemmini_validate_tensor_output(void* c, int64_t c_bytes);
int32_t tvm_gemmini_validate_tensor_buffers(const void* a, int64_t a_bytes, const void* b, int64_t b_bytes,
                                          void* c, int64_t c_bytes);

#ifdef __cplusplus
}
#endif
#endif  /* TVM_APPS_GEMMINI_MATMUL_H_ */
