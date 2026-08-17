/**
 * @file gemm_quant_formats.cuh
 * @brief CUDA kernels for quantized GEMM operations (Q4_0 x Q8_1)
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============================================================================
// Quantization Block Structures
// ============================================================================

#define QK4_0 32
#define QK8_1 32
#define BLOCK_Q4_0_BYTES 18
#define BLOCK_Q8_1_BYTES 36

/**
 * Q4_0 block structure (18 bytes for 32 values)
 * - d: scale (fp16, 2 bytes)
 * - qs: packed 4-bit values (16 bytes)
 */
struct block_q4_0 {
    half d;           // scale
    uint8_t qs[16];   // packed 4-bit values
};

/**
 * Q8_1 block structure (36 bytes for 32 values)
 * - ds: scale and sum as half2 (4 bytes)
 *   - ds.x = scale
 *   - ds.y = sum (precomputed sum of quantized values * scale)
 * - qs: int8 values (32 bytes)
 */
struct block_q8_1 {
    half2 ds;         // ds.x = scale, ds.y = sum
    int8_t qs[32];    // quantized values
};

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Compute dot product of one Q4_0 block and one Q8_1 block
 *
 * Formula: result = d_w * (d_a * sumi - 8.0f * s_a)
 *
 * Where:
 * - d_w = Q4_0 scale
 * - d_a = Q8_1 scale
 * - s_a = Q8_1 sum (precomputed)
 * - sumi = integer dot product of packed q4 values and q8 values
 */
__device__ __forceinline__ float vec_dot_q4_0_q8_1(
    const block_q4_0& w_block,
    const block_q8_1& a_block
) {
    // Get scales
    float d_w = __half2float(w_block.d);
    float d_a = __half2float(a_block.ds.x);
    float s_a = __half2float(a_block.ds.y);

    // Compute integer dot product
    int sumi = 0;

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        uint8_t packed = w_block.qs[i];

        // Low nibble corresponds to a_block.qs[i]
        int q0 = packed & 0x0F;
        sumi += q0 * a_block.qs[i];

        // High nibble corresponds to a_block.qs[i + 16]
        int q1 = (packed >> 4) & 0x0F;
        sumi += q1 * a_block.qs[i + 16];
    }

    // Apply compensation formula
    // Q4_0 stores values with +8 offset, so we subtract 8 * sum
    return d_w * (d_a * sumi - 8.0f * s_a);
}

// ============================================================================
// GEMM Kernel: Q4_0 weights x Q8_1 activations
// ============================================================================

/**
 * Naive GEMM kernel for Q4_0 weights and Q8_1 activations
 *
 * Computes: C[M, N] = W[M, K] @ A[N, K]^T
 *
 * Where:
 * - W (weight) is stored as Q4_0 blocks: [M, K/32, 18]
 * - A (activation) is stored as Q8_1 blocks: [N, K/32, 36]
 * - C (output) is [M, N] float32
 *
 * Note: In the test framework, M = output_features (N in spec),
 *       and N = batch_size (M in spec)
 */
__global__ void gemm_q4_0_q8_1_kernel(
    const block_q4_0* __restrict__ weight,     // [M, num_blocks]
    const block_q8_1* __restrict__ activation, // [N, num_blocks]
    float* __restrict__ output,                // [M, N]
    int M,
    int N,
    int K
) {
    // Thread mapping: one thread computes one output element
    int m = blockIdx.y * blockDim.y + threadIdx.y;  // Row (0 to M-1)
    int n = blockIdx.x * blockDim.x + threadIdx.x;  // Col (0 to N-1)

    if (m >= M || n >= N) return;

    int num_blocks = K / QK4_0;

    float acc = 0.0f;

    // Iterate over blocks in K dimension
    for (int b = 0; b < num_blocks; b++) {
        // Weight: row m, block b
        const block_q4_0& w_block = weight[m * num_blocks + b];

        // Activation: row n, block b
        const block_q8_1& a_block = activation[n * num_blocks + b];

        // Accumulate dot product
        acc += vec_dot_q4_0_q8_1(w_block, a_block);
    }

    // Write output
    output[m * N + n] = acc;
}

/**
 * Launch wrapper for the GEMM kernel
 */
inline void gemm_q4_0_q8_1(
    const block_q4_0* weight,
    const block_q8_1* activation,
    float* output,
    int M, int N, int K
) {
    dim3 blockDim(32, 32);
    dim3 gridDim(
        (N + blockDim.x - 1) / blockDim.x,
        (M + blockDim.y - 1) / blockDim.y
    );

    gemm_q4_0_q8_1_kernel<<<gridDim, blockDim>>>(
        weight, activation, output, M, N, K
    );

    cudaDeviceSynchronize();
}
