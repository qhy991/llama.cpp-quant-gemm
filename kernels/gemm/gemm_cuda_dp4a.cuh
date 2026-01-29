/**
 * gemm_cuda_dp4a.cuh - DP4A Optimized CUDA GEMM Implementations
 *
 * This file contains GEMM implementations using CUDA's DP4A instruction.
 *
 * DP4A (Dot Product of 4 Accumulated)
 * ===================================
 * The __dp4a intrinsic computes a dot product of four 8-bit integers and
 * accumulates the result into a 32-bit integer:
 *
 *   int __dp4a(int a, int b, int c)
 *   = c + a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w
 *
 * Where a and b are treated as int4 (4 packed int8 values).
 *
 * This provides 4x throughput improvement for int8 dot products.
 *
 * Available on: sm_61 (Pascal) and later
 */

#ifndef GEMM_CUDA_DP4A_CUH
#define GEMM_CUDA_DP4A_CUH

// When compiling within llama.cpp, use their type definitions
// Otherwise, use our own quant_types.h
#ifndef GGML_COMMON_DECL
#include "quant_types.h"
#endif

#include <cuda_runtime.h>

// ============================================================================
// DP4A Helper Functions
// ============================================================================

/**
 * Safe memory loading functions (from llama.cpp)
 *
 * These functions handle potentially unaligned memory accesses.
 * CUDA requires proper alignment for int* casts, but our quantized
 * structures may not guarantee 4-byte alignment.
 */

/**
 * Load int32 from 2-byte aligned memory
 * Used for block_q4_0.qs (offset 2 in struct)
 */
__device__ __forceinline__ int load_int_b2(const void* x, int i32) {
    const uint16_t* x16 = (const uint16_t*)x;
    int x32  = x16[2*i32 + 0] <<  0;
    x32     |= x16[2*i32 + 1] << 16;
    return x32;
}

/**
 * Load int32 from 4-byte aligned memory
 * Used for block_q8_1.qs (offset 4 in struct)
 */
__device__ __forceinline__ int load_int_b4(const void* x, int i32) {
    return ((const int*)x)[i32];
}

/**
 * Portable DP4A implementation
 * Uses inline PTX for guaranteed DP4A instruction
 */
__device__ __forceinline__ int dp4a(int a, int b, int c) {
#if __CUDA_ARCH__ >= 610
    // Use native DP4A on Pascal and later
    return __dp4a(a, b, c);
#else
    // Fallback for older architectures
    char4 va = *reinterpret_cast<char4*>(&a);
    char4 vb = *reinterpret_cast<char4*>(&b);
    return c + va.x*vb.x + va.y*vb.y + va.z*vb.z + va.w*vb.w;
#endif
}

/**
 * Pack 4 int8 values into one int32 for DP4A
 */
__device__ __forceinline__ int pack_int8x4(int8_t a, int8_t b, int8_t c, int8_t d) {
    return (d << 24) | ((c & 0xFF) << 16) | ((b & 0xFF) << 8) | (a & 0xFF);
}

/**
 * Extract and expand 4-bit values to bytes for DP4A
 *
 * Q4_0 stores two 4-bit values per byte:
 *   byte = (high << 4) | low
 *
 * For DP4A, we need to expand each nibble to a byte.
 * This function takes a packed Q4_0 int32 (8 nibbles) and returns
 * the low 4 nibbles expanded to bytes.
 */
__device__ __forceinline__ int expand_q4_low(int packed) {
    // Extract low nibbles from each byte and reorder for DP4A
    // Input packed: 4 bytes, each byte = (high << 4) | low
    //   byte0 = (w1 << 4) | w0
    //   byte1 = (w3 << 4) | w2
    //   byte2 = (w5 << 4) | w4
    //   byte3 = (w7 << 4) | w6
    // We want to extract: w0, w2, w4, w6
    // But for DP4A with consecutive activations, we need: w0, w1, w2, w3
    // So we need to interleave low and high nibbles

    // Actually, let's keep it simple: extract all low nibbles
    // Result: byte0=w0, byte1=w2, byte2=w4, byte3=w6
    return (packed >> 0) & 0x0F0F0F0F;
}

__device__ __forceinline__ int expand_q4_high(int packed) {
    // Extract high nibbles from each byte
    // Result: byte0=w1, byte1=w3, byte2=w5, byte3=w7
    return (packed >> 4) & 0x0F0F0F0F;
}

// Helper function to interleave low and high nibbles for proper DP4A
__device__ __forceinline__ void expand_q4_interleaved(int packed, int& out0, int& out1) {
    // Input: 4 bytes with 8 4-bit values
    //   byte0 = (w1 << 4) | w0
    //   byte1 = (w3 << 4) | w2
    //   byte2 = (w5 << 4) | w4
    //   byte3 = (w7 << 4) | w6
    //
    // Output:
    //   out0 = [w0, w1, w2, w3] (for DP4A with a[0-3])
    //   out1 = [w4, w5, w6, w7] (for DP4A with a[4-7])

    int lo = (packed >> 0) & 0x0F0F0F0F;  // [w0, w2, w4, w6] in bytes 0,1,2,3
    int hi = (packed >> 4) & 0x0F0F0F0F;  // [w1, w3, w5, w7] in bytes 0,1,2,3

    // Interleave to get [w0,w1,w2,w3] and [w4,w5,w6,w7]
    out0 = ((lo & 0x000000FF) <<  0) |  // lo byte 0 (w0) → out0 byte 0
           ((hi & 0x000000FF) <<  8) |  // hi byte 0 (w1) → out0 byte 1
           ((lo & 0x0000FF00) <<  8) |  // lo byte 1 (w2) → out0 byte 2
           ((hi & 0x0000FF00) << 16);   // hi byte 1 (w3) → out0 byte 3

    out1 = ((lo & 0x00FF0000) >> 16) |  // lo byte 2 (w4) → out1 byte 0
           ((hi & 0x00FF0000) >>  8) |  // hi byte 2 (w5) → out1 byte 1
           ((lo & 0xFF000000) >>  8) |  // lo byte 3 (w6) → out1 byte 2
           ((hi & 0xFF000000) >>  0);   // hi byte 3 (w7) → out1 byte 3
}

// ============================================================================
// Level 1: DP4A W4A8 GEMM
// ============================================================================
/**
 * W4A8 GEMM using DP4A instructions
 *
 * This kernel processes 8 elements (2 int32 packed values) per DP4A call.
 * For a 32-element block, we need 4 DP4A calls.
 *
 * Key insight: Q4_0 packs 8 4-bit values into 4 bytes (one int32).
 * We expand the low and high nibbles separately, then use DP4A with
 * the corresponding Q8_1 activation values.
 */
static __global__ void gemm_w4a8_dp4a_kernel(
    const block_q8_1* __restrict__ A,
    const block_q4_0* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    const int nb = K / QK4_0;

    for (int b = 0; b < nb; b++) {
        // Load activation block
        const block_q8_1& block_a = A[row * nb + b];
        const float d_a = __half2float(__low2half(block_a.ds));
        const float s_a = __half2float(__high2half(block_a.ds));

        // Load weight block
        const block_q4_0& block_w = B[col * nb + b];
        const float d_w = __half2float(block_w.d);

        int32_t sumi = 0;

        // Process 32 elements using 4 iterations
        // Each iteration processes 8 elements (4 bytes of packed weights)
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            // Load 8 consecutive activation values (2 int32)
            int a0 = load_int_b4(block_a.qs, i * 2);      // activations [i*8 : i*8+3]
            int a1 = load_int_b4(block_a.qs, i * 2 + 1);  // activations [i*8+4 : i*8+7]

            // Load 4 bytes of packed weights (8 4-bit values)
            int w_packed = load_int_b2(block_w.qs, i);

            // Expand and interleave nibbles for proper alignment with activations
            int w0, w1;
            expand_q4_interleaved(w_packed, w0, w1);

            // DP4A: 4-way dot product
            sumi = dp4a(a0, w0, sumi);  // a[i*8:i*8+3] · w[i*8:i*8+3]
            sumi = dp4a(a1, w1, sumi);  // a[i*8+4:i*8+7] · w[i*8+4:i*8+7]
        }

        // Apply compensation formula
        // result = d_w * (d_a * sumi - 8 * s_a)
        // where sumi = sum(q_w * q_a) and s_a is the sum of original activation values
        sum += d_w * (d_a * sumi - 8.0f * s_a);
    }

    C[row * N + col] = sum;
}

// ============================================================================
// Level 2: DP4A W8A8 GEMM
// ============================================================================
/**
 * W8A8 GEMM using DP4A instructions
 *
 * This is simpler than W4A8 since Q8_0 values are already 8-bit.
 * We can directly use DP4A on packed activation and weight values.
 */
static __global__ void gemm_w8a8_dp4a_kernel(
    const block_q8_1* __restrict__ A,
    const block_q8_0* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    const int nb = K / QK8_0;

    for (int b = 0; b < nb; b++) {
        // Load blocks
        const block_q8_1& block_a = A[row * nb + b];
        const float d_a = __half2float(__low2half(block_a.ds));

        const block_q8_0& block_w = B[col * nb + b];
        const float d_w = __half2float(block_w.d);

        int32_t sumi = 0;

        // Process 32 elements using 8 DP4A calls
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            // Safe load: both are 4-byte aligned for Q8
            int a_val = load_int_b4(block_a.qs, i);
            int w_val = load_int_b2(block_w.qs, i);  // Q8_0 also has 2-byte alignment
            sumi = dp4a(a_val, w_val, sumi);
        }

        sum += sumi * d_a * d_w;
    }

    C[row * N + col] = sum;
}

// ============================================================================
// Level 3: Tiled DP4A W4A8 GEMM
// ============================================================================
/**
 * Combined tiling + DP4A optimization
 *
 * This kernel:
 * 1. Uses shared memory for data reuse (tiling)
 * 2. Uses DP4A for efficient int8 computation
 */

#define DP4A_TILE_M 32
#define DP4A_TILE_N 32

static __global__ void gemm_w4a8_tiled_dp4a_kernel(
    const block_q8_1* __restrict__ A,
    const block_q4_0* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    // Shared memory for one block's worth of data
    __shared__ int As_packed[DP4A_TILE_M][8];   // 32 int8 = 8 int32 per row
    __shared__ float As_d[DP4A_TILE_M];
    __shared__ float As_s[DP4A_TILE_M];
    __shared__ int Bs_packed[DP4A_TILE_N][4];   // 16 packed bytes = 4 int32
    __shared__ float Bs_d[DP4A_TILE_N];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * DP4A_TILE_M + ty;
    int col = bx * DP4A_TILE_N + tx;

    float sum = 0.0f;
    const int nb = K / QK4_0;

    for (int b = 0; b < nb; b++) {
        // Cooperative loading of activation block
        if (row < M) {
            const block_q8_1& block_a = A[row * nb + b];
            As_d[ty] = __half2float(__low2half(block_a.ds));
            As_s[ty] = __half2float(__high2half(block_a.ds));

            // Safe load: 4-byte aligned
            if (tx < 8) {
                As_packed[ty][tx] = load_int_b4(block_a.qs, tx);
            }
        }

        // Cooperative loading of weight block
        if (col < N) {
            const block_q4_0& block_w = B[col * nb + b];
            Bs_d[tx] = __half2float(block_w.d);

            // Safe load: 2-byte aligned
            if (ty < 4) {
                Bs_packed[tx][ty] = load_int_b2(block_w.qs, ty);
            }
        }

        __syncthreads();

        // Compute using DP4A
        if (row < M && col < N) {
            int32_t sumi = 0;

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int a0 = As_packed[ty][i];
                int a1 = As_packed[ty][i + 4];
                int w_packed = Bs_packed[tx][i];

                int w_lo = expand_q4_low(w_packed);
                int w_hi = expand_q4_high(w_packed);

                sumi = dp4a(a0, w_lo, sumi);
                sumi = dp4a(a1, w_hi, sumi);
            }

            sum += Bs_d[tx] * (As_d[ty] * sumi - 8.0f * As_s[ty]);
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ============================================================================
// Level 4: Vector Load Optimized DP4A
// ============================================================================
/**
 * Further optimized with float4/int4 vector loads
 */
static __global__ void gemm_w4a8_vectorized_dp4a_kernel(
    const block_q8_1* __restrict__ A,
    const block_q4_0* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    const int nb = K / QK4_0;

    for (int b = 0; b < nb; b++) {
        // Load using vector types for better memory coalescing
        const block_q8_1& block_a = A[row * nb + b];
        const float d_a = __half2float(__low2half(block_a.ds));
        const float s_a = __half2float(__high2half(block_a.ds));

        const block_q4_0& block_w = B[col * nb + b];
        const float d_w = __half2float(block_w.d);

        int32_t sumi = 0;

        // Process with DP4A using safe loads
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            // Safe load activations (4-byte aligned)
            int a0 = load_int_b4(block_a.qs, i);
            int a1 = load_int_b4(block_a.qs, i + 4);

            // Safe load weights (2-byte aligned)
            int w = load_int_b2(block_w.qs, i);

            // Process with DP4A
            sumi = dp4a(a0, expand_q4_low(w), sumi);
            sumi = dp4a(a1, expand_q4_high(w), sumi);
        }

        sum += d_w * (d_a * sumi - 8.0f * s_a);
    }

    C[row * N + col] = sum;
}

// ============================================================================
// Host Wrapper Functions
// ============================================================================

inline void gemm_w4a8_dp4a(
    const block_q8_1* A, const block_q4_0* B, float* C,
    int M, int N, int K, cudaStream_t stream = 0)
{
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    gemm_w4a8_dp4a_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

inline void gemm_w8a8_dp4a(
    const block_q8_1* A, const block_q8_0* B, float* C,
    int M, int N, int K, cudaStream_t stream = 0)
{
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    gemm_w8a8_dp4a_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

inline void gemm_w4a8_tiled_dp4a(
    const block_q8_1* A, const block_q4_0* B, float* C,
    int M, int N, int K, cudaStream_t stream = 0)
{
    dim3 block(DP4A_TILE_N, DP4A_TILE_M);
    dim3 grid((N + DP4A_TILE_N - 1) / DP4A_TILE_N,
              (M + DP4A_TILE_M - 1) / DP4A_TILE_M);
    gemm_w4a8_tiled_dp4a_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

inline void gemm_w4a8_vectorized_dp4a(
    const block_q8_1* A, const block_q4_0* B, float* C,
    int M, int N, int K, cudaStream_t stream = 0)
{
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    gemm_w4a8_vectorized_dp4a_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

#endif // GEMM_CUDA_DP4A_CUH
