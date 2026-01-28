/**
 * quantize.h - CPU and GPU Quantization/Dequantization Functions
 *
 * This file provides reference implementations of quantization functions.
 * These are designed to be educational and easy to understand, while being
 * compatible with llama.cpp's format.
 */

#ifndef QUANTIZE_H
#define QUANTIZE_H

#include "quant_types.h"
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>

// ============================================================================
// CPU Quantization Functions (Reference Implementations)
// ============================================================================

/**
 * Quantize float array to Q4_0 format (CPU reference)
 *
 * @param src  Source float array
 * @param dst  Destination Q4_0 block array
 * @param k    Number of elements (must be multiple of QK4_0)
 *
 * Algorithm:
 * 1. For each block of 32 elements:
 *    a. Find max absolute value (amax)
 *    b. Compute scale: d = amax / 7.0f
 *    c. Quantize each value: q = round(x / d) + 8, clamped to [0, 15]
 *    d. Pack two 4-bit values per byte
 */
inline void quantize_row_q4_0_ref(const float* src, block_q4_0* dst, int64_t k) {
    const int nb = k / QK4_0;  // Number of blocks

    for (int i = 0; i < nb; i++) {
        const float* block_src = src + i * QK4_0;
        block_q4_0& block_dst = dst[i];

        // Step 1: Find max absolute value
        float amax = 0.0f;
        for (int j = 0; j < QK4_0; j++) {
            amax = std::max(amax, std::abs(block_src[j]));
        }

        // Step 2: Compute scale
        // Map to range [-8, 7] * d, so d = amax / 7.0f
        const float d = amax / 7.0f;
        block_dst.d = __float2half(d);

        // Step 3: Quantize and pack
        const float id = (d > 0) ? 1.0f / d : 0.0f;

        // llama.cpp packs values: qs[j] contains q[j] (low) and q[j+16] (high)
        for (int j = 0; j < QK4_0 / 2; j++) {
            // Quantize two values
            int q0 = (int)roundf(block_src[j] * id) + 8;
            int q1 = (int)roundf(block_src[j + QK4_0/2] * id) + 8;

            // Clamp to [0, 15]
            q0 = std::max(0, std::min(15, q0));
            q1 = std::max(0, std::min(15, q1));

            // Pack: low nibble = q0, high nibble = q1
            block_dst.qs[j] = pack_q4_0(q0, q1);
        }
    }
}

/**
 * Dequantize Q4_0 format to float array (CPU reference)
 *
 * @param src  Source Q4_0 block array
 * @param dst  Destination float array
 * @param k    Number of elements
 *
 * Algorithm:
 * For each packed byte:
 *   x_low  = (low_nibble - 8) * d
 *   x_high = (high_nibble - 8) * d
 */
inline void dequantize_row_q4_0(const block_q4_0* src, float* dst, int64_t k) {
    const int nb = k / QK4_0;

    for (int i = 0; i < nb; i++) {
        const block_q4_0& block = src[i];
        float* block_dst = dst + i * QK4_0;

        const float d = __half2float(block.d);

        for (int j = 0; j < QK4_0 / 2; j++) {
            int q0 = get_q4_0_low(block.qs[j]);
            int q1 = get_q4_0_high(block.qs[j]);

            // Dequantize: x = (q - 8) * d
            block_dst[j] = (q0 - 8) * d;
            block_dst[j + QK4_0/2] = (q1 - 8) * d;
        }
    }
}

/**
 * Quantize float array to Q8_0 format (CPU reference)
 *
 * @param src  Source float array
 * @param dst  Destination Q8_0 block array
 * @param k    Number of elements (must be multiple of QK8_0)
 */
inline void quantize_row_q8_0_ref(const float* src, block_q8_0* dst, int64_t k) {
    const int nb = k / QK8_0;

    for (int i = 0; i < nb; i++) {
        const float* block_src = src + i * QK8_0;
        block_q8_0& block_dst = dst[i];

        // Find max absolute value
        float amax = 0.0f;
        for (int j = 0; j < QK8_0; j++) {
            amax = std::max(amax, std::abs(block_src[j]));
        }

        // Compute scale: d = amax / 127.0f
        const float d = amax / 127.0f;
        block_dst.d = __float2half(d);

        // Quantize
        const float id = (d > 0) ? 1.0f / d : 0.0f;
        for (int j = 0; j < QK8_0; j++) {
            int q = (int)roundf(block_src[j] * id);
            block_dst.qs[j] = (int8_t)std::max(-128, std::min(127, q));
        }
    }
}

/**
 * Dequantize Q8_0 format to float array (CPU reference)
 */
inline void dequantize_row_q8_0(const block_q8_0* src, float* dst, int64_t k) {
    const int nb = k / QK8_0;

    for (int i = 0; i < nb; i++) {
        const block_q8_0& block = src[i];
        float* block_dst = dst + i * QK8_0;

        const float d = __half2float(block.d);

        for (int j = 0; j < QK8_0; j++) {
            block_dst[j] = block.qs[j] * d;
        }
    }
}

/**
 * Quantize float array to Q8_1 format (CPU reference)
 *
 * Q8_1 is designed for activation quantization and includes a sum field
 * for compensation when paired with Q4_0 weights.
 *
 * @param src  Source float array
 * @param dst  Destination Q8_1 block array
 * @param k    Number of elements (must be multiple of QK8_1)
 */
inline void quantize_row_q8_1_ref(const float* src, block_q8_1* dst, int64_t k) {
    const int nb = k / QK8_1;

    for (int i = 0; i < nb; i++) {
        const float* block_src = src + i * QK8_1;
        block_q8_1& block_dst = dst[i];

        // Find max absolute value AND compute sum
        float amax = 0.0f;
        float sum = 0.0f;
        for (int j = 0; j < QK8_1; j++) {
            amax = std::max(amax, std::abs(block_src[j]));
            sum += block_src[j];  // Sum of original values
        }

        // Compute scale
        const float d = amax / 127.0f;

        // Store scale and sum as half2
        block_dst.ds = make_half2(__float2half(d), __float2half(sum));

        // Quantize
        const float id = (d > 0) ? 1.0f / d : 0.0f;
        for (int j = 0; j < QK8_1; j++) {
            int q = (int)roundf(block_src[j] * id);
            block_dst.qs[j] = (int8_t)std::max(-128, std::min(127, q));
        }
    }
}

/**
 * Dequantize Q8_1 format to float array (CPU reference)
 */
inline void dequantize_row_q8_1(const block_q8_1* src, float* dst, int64_t k) {
    const int nb = k / QK8_1;

    for (int i = 0; i < nb; i++) {
        const block_q8_1& block = src[i];
        float* block_dst = dst + i * QK8_1;

        const float d = __half2float(__low2half(block.ds));

        for (int j = 0; j < QK8_1; j++) {
            block_dst[j] = block.qs[j] * d;
        }
    }
}

// ============================================================================
// GPU Quantization Kernels
// ============================================================================

/**
 * CUDA kernel for Q4_0 quantization
 * Each thread processes one block of 32 elements
 */
__global__ void quantize_q4_0_kernel(
    const float* __restrict__ x,
    block_q4_0* __restrict__ y,
    int64_t k)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int nb = k / QK4_0;

    if (i >= nb) return;

    const float* xi = x + i * QK4_0;
    block_q4_0& yi = y[i];

    // Find max absolute value (could be optimized with warp reduction)
    float amax = 0.0f;
    #pragma unroll
    for (int j = 0; j < QK4_0; j++) {
        amax = fmaxf(amax, fabsf(xi[j]));
    }

    // Compute scale
    const float d = amax / 7.0f;
    yi.d = __float2half(d);

    // Quantize and pack
    const float id = (d > 0) ? 1.0f / d : 0.0f;

    #pragma unroll
    for (int j = 0; j < QK4_0 / 2; j++) {
        int q0 = __float2int_rn(xi[j] * id) + 8;
        int q1 = __float2int_rn(xi[j + QK4_0/2] * id) + 8;

        q0 = max(0, min(15, q0));
        q1 = max(0, min(15, q1));

        yi.qs[j] = (uint8_t)((q1 << 4) | (q0 & 0x0F));
    }
}

/**
 * CUDA kernel for Q8_0 quantization
 */
__global__ void quantize_q8_0_kernel(
    const float* __restrict__ x,
    block_q8_0* __restrict__ y,
    int64_t k)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int nb = k / QK8_0;

    if (i >= nb) return;

    const float* xi = x + i * QK8_0;
    block_q8_0& yi = y[i];

    // Find max absolute value
    float amax = 0.0f;
    #pragma unroll
    for (int j = 0; j < QK8_0; j++) {
        amax = fmaxf(amax, fabsf(xi[j]));
    }

    // Compute scale
    const float d = amax / 127.0f;
    yi.d = __float2half(d);

    // Quantize
    const float id = (d > 0) ? 1.0f / d : 0.0f;

    #pragma unroll
    for (int j = 0; j < QK8_0; j++) {
        yi.qs[j] = (int8_t)max(-128, min(127, __float2int_rn(xi[j] * id)));
    }
}

/**
 * CUDA kernel for Q8_1 quantization (with sum compensation)
 *
 * This is the critical kernel for activation quantization.
 * The sum field is essential for accurate Q4_0 × Q8_1 computation.
 */
__global__ void quantize_q8_1_kernel(
    const float* __restrict__ x,
    block_q8_1* __restrict__ y,
    int64_t k)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int nb = k / QK8_1;

    if (i >= nb) return;

    const float* xi = x + i * QK8_1;
    block_q8_1& yi = y[i];

    // Find max absolute value AND compute sum
    float amax = 0.0f;
    float sum = 0.0f;

    #pragma unroll
    for (int j = 0; j < QK8_1; j++) {
        float v = xi[j];
        amax = fmaxf(amax, fabsf(v));
        sum += v;  // Sum of original float values
    }

    // Compute scale and store both scale and sum
    const float d = amax / 127.0f;
    yi.ds = make_half2(__float2half(d), __float2half(sum));

    // Quantize
    const float id = (d > 0) ? 1.0f / d : 0.0f;

    #pragma unroll
    for (int j = 0; j < QK8_1; j++) {
        yi.qs[j] = (int8_t)max(-128, min(127, __float2int_rn(xi[j] * id)));
    }
}

// ============================================================================
// Host Wrapper Functions
// ============================================================================

inline void quantize_q4_0_cuda(const float* x, block_q4_0* y, int64_t k,
                                cudaStream_t stream = 0)
{
    const int nb = k / QK4_0;
    const int block_size = 256;
    const int grid_size = (nb + block_size - 1) / block_size;
    quantize_q4_0_kernel<<<grid_size, block_size, 0, stream>>>(x, y, k);
}

inline void quantize_q8_0_cuda(const float* x, block_q8_0* y, int64_t k,
                                cudaStream_t stream = 0)
{
    const int nb = k / QK8_0;
    const int block_size = 256;
    const int grid_size = (nb + block_size - 1) / block_size;
    quantize_q8_0_kernel<<<grid_size, block_size, 0, stream>>>(x, y, k);
}

inline void quantize_q8_1_cuda(const float* x, block_q8_1* y, int64_t k,
                                cudaStream_t stream = 0)
{
    const int nb = k / QK8_1;
    const int block_size = 256;
    const int grid_size = (nb + block_size - 1) / block_size;
    quantize_q8_1_kernel<<<grid_size, block_size, 0, stream>>>(x, y, k);
}

#endif // QUANTIZE_H
