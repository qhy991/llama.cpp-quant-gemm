/**
 * gemm_cuda_tiled.cuh - Tiled CUDA GEMM Implementations
 *
 * This file contains optimized GEMM implementations using shared memory tiling.
 *
 * Key optimization: Shared Memory Tiling
 * =======================================
 * The naive kernel has poor memory efficiency because each thread independently
 * loads data from global memory, resulting in massive redundant reads.
 *
 * Tiling works by:
 * 1. Loading a tile of data into shared memory cooperatively
 * 2. All threads in the block compute using this shared tile
 * 3. Moving to the next tile
 *
 * Data Reuse Analysis:
 * - Naive: Each A element read M×N times, each B element read M×N times
 * - Tiled: Each A element read M×N/TILE_Y times, each B element read M×N/TILE_X times
 * - Reuse factor: TILE_X × TILE_Y
 */

#ifndef GEMM_CUDA_TILED_CUH
#define GEMM_CUDA_TILED_CUH

#include "quant_types.h"
#include <cuda_runtime.h>

// Tile dimensions
#define TILE_M 32  // Rows per tile
#define TILE_N 32  // Cols per tile
#define TILE_K 32  // K dimension per tile (matches block size)

// ============================================================================
// Level 1: Tiled FP32 GEMM
// ============================================================================
/**
 * Tiled FP32 GEMM kernel
 *
 * Each thread block computes a TILE_M × TILE_N output tile.
 * Data is loaded into shared memory for reuse.
 */
__global__ void gemm_fp32_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_N][TILE_K];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Output position
    int row = by * TILE_M + ty;
    int col = bx * TILE_N + tx;

    float sum = 0.0f;

    // Iterate over tiles
    for (int t = 0; t < (K + TILE_K - 1) / TILE_K; t++) {
        // Cooperative loading into shared memory
        int k_idx = t * TILE_K + tx;

        // Load A tile
        if (row < M && k_idx < K) {
            As[ty][tx] = A[row * K + k_idx];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load B tile (transposed access)
        int col_k = t * TILE_K + ty;
        if (col < N && col_k < K) {
            Bs[tx][ty] = B[col * K + col_k];
        } else {
            Bs[tx][ty] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            sum += As[ty][k] * Bs[tx][k];
        }

        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ============================================================================
// Level 2: Tiled W4A16 GEMM
// ============================================================================
/**
 * Tiled W4A16 GEMM kernel
 *
 * Loads Q4_0 weights into shared memory after dequantization.
 * This trades compute (dequant) for memory bandwidth (shared memory reuse).
 */
__global__ void gemm_w4a16_tiled_kernel(
    const float* __restrict__ A,
    const block_q4_0* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_N][TILE_K];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_M + ty;
    int col = bx * TILE_N + tx;

    float sum = 0.0f;
    const int nb = K / QK4_0;

    // Process in tiles of 32 elements (one block)
    for (int t = 0; t < nb; t++) {
        int k_base = t * QK4_0;

        // Load A tile (each thread loads one element)
        // We load QK4_0 elements per row
        for (int kk = 0; kk < QK4_0; kk += TILE_K) {
            int k_idx = k_base + kk + tx;
            if (row < M && k_idx < K) {
                As[ty][tx] = A[row * K + k_idx];
            } else {
                As[ty][tx] = 0.0f;
            }

            // Load and dequantize B tile
            // Each thread dequantizes and loads elements for its column
            if (col < N) {
                const block_q4_0& block = B[col * nb + t];
                const float d = __half2float(block.d);

                // Figure out which element within the block
                int k_in_block = kk + ty;
                if (k_in_block < QK4_0 / 2) {
                    uint8_t packed = block.qs[k_in_block];
                    int q = (packed & 0x0F) - 8;
                    Bs[tx][ty] = q * d;
                } else if (k_in_block < QK4_0) {
                    uint8_t packed = block.qs[k_in_block - QK4_0/2];
                    int q = (packed >> 4) - 8;
                    Bs[tx][ty] = q * d;
                } else {
                    Bs[tx][ty] = 0.0f;
                }
            } else {
                Bs[tx][ty] = 0.0f;
            }

            __syncthreads();

            // Compute partial sum
            #pragma unroll
            for (int k = 0; k < TILE_K; k++) {
                sum += As[ty][k] * Bs[tx][k];
            }

            __syncthreads();
        }
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ============================================================================
// Level 3: Tiled W4A8 GEMM with Block-level Processing
// ============================================================================
/**
 * Tiled W4A8 GEMM kernel
 *
 * This version processes one quantization block (32 elements) per tile iteration.
 * It accumulates integer dot products and applies compensation at the end.
 */
__global__ void gemm_w4a8_tiled_kernel(
    const block_q8_1* __restrict__ A,
    const block_q4_0* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    // Shared memory for quantized values
    __shared__ int8_t As_q[TILE_M][QK4_0];      // Activation quants
    __shared__ float As_d[TILE_M];              // Activation scales
    __shared__ float As_s[TILE_M];              // Activation sums
    __shared__ uint8_t Bs_q[TILE_N][QK4_0/2];   // Weight quants (packed)
    __shared__ float Bs_d[TILE_N];              // Weight scales

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_M + ty;
    int col = bx * TILE_N + tx;

    float sum = 0.0f;
    const int nb = K / QK4_0;

    // Process each block
    for (int b = 0; b < nb; b++) {
        // Load activation block
        if (row < M) {
            const block_q8_1& block_a = A[row * nb + b];
            As_d[ty] = __half2float(__low2half(block_a.ds));
            As_s[ty] = __half2float(__high2half(block_a.ds));

            // Load quantized values (each thread loads one element)
            if (tx < QK4_0) {
                As_q[ty][tx] = block_a.qs[tx];
            }
        }

        // Load weight block
        if (col < N) {
            const block_q4_0& block_w = B[col * nb + b];
            Bs_d[tx] = __half2float(block_w.d);

            // Load packed values
            if (ty < QK4_0/2) {
                Bs_q[tx][ty] = block_w.qs[ty];
            }
        }

        __syncthreads();

        // Compute integer dot product
        if (row < M && col < N) {
            int32_t sumi = 0;

            #pragma unroll
            for (int k = 0; k < QK4_0/2; k++) {
                uint8_t packed = Bs_q[tx][k];
                int q_w0 = (packed & 0x0F);
                int q_w1 = (packed >> 4);

                sumi += (int32_t)As_q[ty][k] * q_w0;
                sumi += (int32_t)As_q[ty][k + 16] * q_w1;
            }

            // Apply compensation formula
            float d_w = Bs_d[tx];
            float d_a = As_d[ty];
            float s_a = As_s[ty];
            sum += d_w * (d_a * sumi - 8.0f * s_a);
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ============================================================================
// Host Wrapper Functions
// ============================================================================

inline void gemm_fp32_tiled(
    const float* A, const float* B, float* C,
    int M, int N, int K, cudaStream_t stream = 0)
{
    dim3 block(TILE_N, TILE_M);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    gemm_fp32_tiled_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

inline void gemm_w4a16_tiled(
    const float* A, const block_q4_0* B, float* C,
    int M, int N, int K, cudaStream_t stream = 0)
{
    dim3 block(TILE_N, TILE_M);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    gemm_w4a16_tiled_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

inline void gemm_w4a8_tiled(
    const block_q8_1* A, const block_q4_0* B, float* C,
    int M, int N, int K, cudaStream_t stream = 0)
{
    dim3 block(TILE_N, TILE_M);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    gemm_w4a8_tiled_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

#endif // GEMM_CUDA_TILED_CUH
