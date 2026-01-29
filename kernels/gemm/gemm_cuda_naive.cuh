/**
 * gemm_cuda_naive.cuh - Naive CUDA GEMM Implementations
 *
 * This file contains the most basic CUDA GEMM implementations.
 * They are simple to understand but not optimized for performance.
 *
 * These kernels serve as:
 * 1. Educational examples to understand the basic algorithm
 * 2. Correctness baselines for testing optimized versions
 * 3. Starting points for incremental optimization
 */

#ifndef GEMM_CUDA_NAIVE_CUH
#define GEMM_CUDA_NAIVE_CUH

#include "quant_types.h"
#include <cuda_runtime.h>

// ============================================================================
// Level 0: Naive FP32 GEMM
// ============================================================================
/**
 * Naive FP32 GEMM kernel
 *
 * Each thread computes ONE element of the output matrix.
 * This is the simplest possible implementation.
 *
 * Performance characteristics:
 * - O(K) global memory reads per thread
 * - No data reuse between threads
 * - Very low arithmetic intensity
 */
__global__ void gemm_fp32_naive_kernel(
    const float* __restrict__ A,  // [M, K]
    const float* __restrict__ B,  // [N, K]
    float* __restrict__ C,        // [M, N]
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[col * K + k];
    }

    C[row * N + col] = sum;
}

// ============================================================================
// Level 1: Naive W4A16 GEMM (4-bit weight, FP32 activation)
// ============================================================================
/**
 * Naive W4A16 GEMM kernel
 *
 * Each thread computes one output element.
 * Dequantization is done on-the-fly within the kernel.
 *
 * This demonstrates:
 * - Basic block structure understanding
 * - 4-bit unpacking
 * - Offset handling (-8)
 */
__global__ void gemm_w4a16_naive_kernel(
    const float* __restrict__ A,       // [M, K] FP32 activation
    const block_q4_0* __restrict__ B,  // [N, K/32] Q4_0 weights
    float* __restrict__ C,             // [M, N] output
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    const int nb = K / QK4_0;  // blocks per row

    // Iterate over all blocks
    for (int b = 0; b < nb; b++) {
        // Get weight block for this output column
        const block_q4_0& block = B[col * nb + b];
        const float d = __half2float(block.d);

        // Process 32 elements in this block
        // Note: qs[k] contains q[k] (low) and q[k+16] (high)
        #pragma unroll
        for (int k = 0; k < QK4_0 / 2; k++) {
            // Unpack 4-bit values
            uint8_t packed = block.qs[k];
            int q0 = (packed & 0x0F) - 8;  // Low nibble, apply offset
            int q1 = (packed >> 4) - 8;    // High nibble, apply offset

            // Dequantize
            float w0 = q0 * d;
            float w1 = q1 * d;

            // Multiply with activation
            int k_idx = b * QK4_0;
            sum += A[row * K + k_idx + k] * w0;
            sum += A[row * K + k_idx + k + 16] * w1;
        }
    }

    C[row * N + col] = sum;
}

// ============================================================================
// Level 2: Naive W8A16 GEMM (8-bit weight, FP32 activation)
// ============================================================================
/**
 * Naive W8A16 GEMM kernel
 *
 * Similar to W4A16 but simpler since Q8_0 values don't need unpacking.
 */
__global__ void gemm_w8a16_naive_kernel(
    const float* __restrict__ A,       // [M, K]
    const block_q8_0* __restrict__ B,  // [N, K/32]
    float* __restrict__ C,             // [M, N]
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    const int nb = K / QK8_0;

    for (int b = 0; b < nb; b++) {
        const block_q8_0& block = B[col * nb + b];
        const float d = __half2float(block.d);

        #pragma unroll
        for (int k = 0; k < QK8_0; k++) {
            float w = block.qs[k] * d;
            sum += A[row * K + b * QK8_0 + k] * w;
        }
    }

    C[row * N + col] = sum;
}

// ============================================================================
// Level 3: Naive W4A8 GEMM (4-bit weight, 8-bit activation with compensation)
// ============================================================================
/**
 * Naive W4A8 GEMM kernel
 *
 * This is the key kernel that demonstrates the compensation formula:
 *   result = d_w × (d_a × sumi - 8 × s_a)
 *
 * The compensation is needed because Q4_0 stores values as [0,15] instead of [-8,7].
 * Rather than subtracting 8 for each weight value, we compute the compensation
 * once per block using the pre-computed sum in Q8_1.
 */
__global__ void gemm_w4a8_naive_kernel(
    const block_q8_1* __restrict__ A,  // [M, K/32] Q8_1 activation
    const block_q4_0* __restrict__ B,  // [N, K/32] Q4_0 weight
    float* __restrict__ C,             // [M, N] output
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    const int nb = K / QK4_0;

    for (int b = 0; b < nb; b++) {
        // Get activation block
        const block_q8_1& block_a = A[row * nb + b];
        const half2 ds_a = block_a.ds;
        const float d_a = __half2float(__low2half(ds_a));   // scale
        const float s_a = __half2float(__high2half(ds_a));  // sum compensation

        // Get weight block
        const block_q4_0& block_w = B[col * nb + b];
        const float d_w = __half2float(block_w.d);

        // Compute integer dot product
        // IMPORTANT: Do NOT subtract 8 from weight values here!
        // The offset is handled via the compensation formula.
        int32_t sumi = 0;

        #pragma unroll
        for (int k = 0; k < QK4_0 / 2; k++) {
            uint8_t packed = block_w.qs[k];
            int q_w0 = (packed & 0x0F);  // [0, 15], no offset!
            int q_w1 = (packed >> 4);    // [0, 15], no offset!

            int8_t q_a0 = block_a.qs[k];
            int8_t q_a1 = block_a.qs[k + 16];

            sumi += (int32_t)q_a0 * q_w0;
            sumi += (int32_t)q_a1 * q_w1;
        }

        // Apply compensation formula
        // This is equivalent to: d_a × d_w × Σ q_a × (q_w - 8)
        sum += d_w * (d_a * sumi - 8.0f * s_a);
    }

    C[row * N + col] = sum;
}

// ============================================================================
// Level 4: Naive W8A8 GEMM (8-bit weight, 8-bit activation)
// ============================================================================
/**
 * Naive W8A8 GEMM kernel
 *
 * Q8_0 weights don't have an offset, so no compensation is needed.
 * This is simpler than W4A8.
 */
__global__ void gemm_w8a8_naive_kernel(
    const block_q8_1* __restrict__ A,  // [M, K/32]
    const block_q8_0* __restrict__ B,  // [N, K/32]
    float* __restrict__ C,             // [M, N]
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    const int nb = K / QK8_0;

    for (int b = 0; b < nb; b++) {
        const block_q8_1& block_a = A[row * nb + b];
        const float d_a = __half2float(__low2half(block_a.ds));

        const block_q8_0& block_w = B[col * nb + b];
        const float d_w = __half2float(block_w.d);

        int32_t sumi = 0;
        #pragma unroll
        for (int k = 0; k < QK8_0; k++) {
            sumi += (int32_t)block_a.qs[k] * (int32_t)block_w.qs[k];
        }

        sum += sumi * d_a * d_w;
    }

    C[row * N + col] = sum;
}

// ============================================================================
// Host Wrapper Functions
// ============================================================================

// Default block dimensions for naive kernels
#define NAIVE_BLOCK_DIM 16

inline void gemm_fp32_naive(
    const float* A, const float* B, float* C,
    int M, int N, int K, cudaStream_t stream = 0)
{
    dim3 block(NAIVE_BLOCK_DIM, NAIVE_BLOCK_DIM);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    gemm_fp32_naive_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

inline void gemm_w4a16_naive(
    const float* A, const block_q4_0* B, float* C,
    int M, int N, int K, cudaStream_t stream = 0)
{
    dim3 block(NAIVE_BLOCK_DIM, NAIVE_BLOCK_DIM);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    gemm_w4a16_naive_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

inline void gemm_w8a16_naive(
    const float* A, const block_q8_0* B, float* C,
    int M, int N, int K, cudaStream_t stream = 0)
{
    dim3 block(NAIVE_BLOCK_DIM, NAIVE_BLOCK_DIM);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    gemm_w8a16_naive_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

inline void gemm_w4a8_naive(
    const block_q8_1* A, const block_q4_0* B, float* C,
    int M, int N, int K, cudaStream_t stream = 0)
{
    dim3 block(NAIVE_BLOCK_DIM, NAIVE_BLOCK_DIM);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    gemm_w4a8_naive_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

inline void gemm_w8a8_naive(
    const block_q8_1* A, const block_q8_0* B, float* C,
    int M, int N, int K, cudaStream_t stream = 0)
{
    dim3 block(NAIVE_BLOCK_DIM, NAIVE_BLOCK_DIM);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    gemm_w8a8_naive_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

#endif // GEMM_CUDA_NAIVE_CUH
