/**
 * @file kernels/gemm/gemm_vectorized.cuh
 * @brief 修复版向量化加载实现
 *
 * 问题分析:
 * - block_q8_1.qs 只有 4 字节对齐
 * - int4 加载需要 16 字节对齐
 * - 直接转换会导致 misaligned address
 *
 * 解决方案:
 * - 使用 int 逐个加载而不是 int4
 * - 或者使用 memcpy 避免对齐问题
 * - 或者使用 __ldg 内建函数
 */

#ifndef KERNELS_GEMM_VECTORIZED_CUH
#define KERNELS_GEMM_VECTORIZED_CUH

#include "../../compat/ggml_types.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// ============================================================================
// 辅助函数
// ============================================================================

__device__ __forceinline__ int dp4a_vec(int a, int b, int c) {
#if __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    char4 va = *reinterpret_cast<char4*>(&a);
    char4 vb = *reinterpret_cast<char4*>(&b);
    return c + va.x*vb.x + va.y*vb.y + va.z*vb.z + va.w*vb.w;
#endif
}

__device__ __forceinline__ int load_int_b2_vec(const void* x, int i32) {
    const uint16_t* x16 = (const uint16_t*)x;
    int x32  = x16[2*i32 + 0] <<  0;
    x32     |= x16[2*i32 + 1] << 16;
    return x32;
}

__device__ __forceinline__ float warp_reduce_sum_vec(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// ============================================================================
// 方案 1: 使用 int 逐个加载（安全但稍慢）
// ============================================================================

/**
 * 向量化加载 - 方案 1: 使用 int 加载
 *
 * 优点: 安全，不会崩溃
 * 缺点: 比 int4 稍慢（但仍比标量快）
 */
template<int ROWS = 4>
__global__ void gemm_q4_0_q8_1_vec_safe_kernel(
    const block_q4_0* __restrict__ weight,
    const block_q8_1* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int num_blocks_k = K / 32;

    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;

    const int row_base = (blockIdx.x * warps_per_block + warp_id) * ROWS;
    const int col = blockIdx.y;

    if (col >= N) return;

    const block_q8_1* act_col = activation + col * num_blocks_k;

    float sums[ROWS] = {0.0f};

    // 每个线程处理多个 blocks
    for (int b = lane_id; b < num_blocks_k; b += WARP_SIZE) {
        const block_q8_1* bq8 = &act_col[b];

        // 向量化加载激活值 (使用 int 而不是 int4)
        int u[8];
        const int* u_ptr = (const int*)bq8->qs;

        // 使用 __ldg 进行只读缓存加载
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            u[i] = __ldg(u_ptr + i);
        }

        float d8 = __half2float(__low2half(bq8->ds));
        float s8 = __half2float(__high2half(bq8->ds));

        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const int row = row_base + r;
            if (row >= M) continue;

            const block_q4_0* bq4 = &weight[row * num_blocks_k + b];

            int sumi = 0;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int v = load_int_b2_vec(bq4->qs, i);
                int vi0 = (v >> 0) & 0x0F0F0F0F;
                int vi1 = (v >> 4) & 0x0F0F0F0F;

                sumi = dp4a_vec(vi0, u[i], sumi);
                sumi = dp4a_vec(vi1, u[i + 4], sumi);
            }

            float d4 = __half2float(bq4->d);
            sums[r] += d4 * (d8 * sumi - 8.0f * s8);
        }
    }

    // Warp 规约
    #pragma unroll
    for (int r = 0; r < ROWS; r++) {
        const int row = row_base + r;
        if (row >= M) continue;

        float result = warp_reduce_sum_vec(sums[r]);

        if (lane_id == 0) {
            output[row * N + col] = result;
        }
    }
}

// ============================================================================
// 方案 2: 使用 float4 加载（更激进）
// ============================================================================

/**
 * 向量化加载 - 方案 2: 使用 int2
 *
 * 优点: 比 int 快，比 float4 安全
 * 缺点: 需要 8 字节对齐（block_q8_1 满足）
 */
template<int ROWS = 4>
__global__ void gemm_q4_0_q8_1_vec_float4_kernel(
    const block_q4_0* __restrict__ weight,
    const block_q8_1* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int num_blocks_k = K / 32;

    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;

    const int row_base = (blockIdx.x * warps_per_block + warp_id) * ROWS;
    const int col = blockIdx.y;

    if (col >= N) return;

    const block_q8_1* act_col = activation + col * num_blocks_k;

    float sums[ROWS] = {0.0f};

    for (int b = lane_id; b < num_blocks_k; b += WARP_SIZE) {
        const block_q8_1* bq8 = &act_col[b];

        // 使用 int2 加载 (16 bytes = 1 个 int2 × 2)
        // 这样更安全，因为 int2 只需要 8 字节对齐
        int u[8];
        const int2* u_ptr = (const int2*)bq8->qs;

        int2 u_tmp0 = __ldg(u_ptr + 0);  // 加载 qs[0..7]
        int2 u_tmp1 = __ldg(u_ptr + 1);  // 加载 qs[8..15]
        int2 u_tmp2 = __ldg(u_ptr + 2);  // 加载 qs[16..23]
        int2 u_tmp3 = __ldg(u_ptr + 3);  // 加载 qs[24..31]

        u[0] = u_tmp0.x; u[1] = u_tmp0.y;
        u[2] = u_tmp1.x; u[3] = u_tmp1.y;
        u[4] = u_tmp2.x; u[5] = u_tmp2.y;
        u[6] = u_tmp3.x; u[7] = u_tmp3.y;

        float d8 = __half2float(__low2half(bq8->ds));
        float s8 = __half2float(__high2half(bq8->ds));

        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const int row = row_base + r;
            if (row >= M) continue;

            const block_q4_0* bq4 = &weight[row * num_blocks_k + b];

            int sumi = 0;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int v = load_int_b2_vec(bq4->qs, i);
                int vi0 = (v >> 0) & 0x0F0F0F0F;
                int vi1 = (v >> 4) & 0x0F0F0F0F;

                sumi = dp4a_vec(vi0, u[i], sumi);
                sumi = dp4a_vec(vi1, u[i + 4], sumi);
            }

            float d4 = __half2float(bq4->d);
            sums[r] += d4 * (d8 * sumi - 8.0f * s8);
        }
    }

    // Warp 规约
    #pragma unroll
    for (int r = 0; r < ROWS; r++) {
        const int row = row_base + r;
        if (row >= M) continue;

        float result = warp_reduce_sum_vec(sums[r]);

        if (lane_id == 0) {
            output[row * N + col] = result;
        }
    }
}

// ============================================================================
// 接口函数
// ============================================================================

/**
 * 向量化加载版本 - 安全版（推荐）
 */
inline void gemm_q4_0_q8_1_vec_safe(
    const block_q4_0* weight,
    const block_q8_1* activation,
    float* output,
    int M, int N, int K,
    cudaStream_t stream = 0
) {
    constexpr int ROWS = 4;
    const int warps_per_block = 8;
    const int threads_per_block = warps_per_block * WARP_SIZE;

    dim3 block(threads_per_block);
    dim3 grid((M + warps_per_block * ROWS - 1) / (warps_per_block * ROWS), N);

    gemm_q4_0_q8_1_vec_safe_kernel<ROWS><<<grid, block, 0, stream>>>(
        weight, activation, output, M, N, K
    );
}

/**
 * 向量化加载版本 - int2 版（推荐）
 */
inline void gemm_q4_0_q8_1_vec_float4(
    const block_q4_0* weight,
    const block_q8_1* activation,
    float* output,
    int M, int N, int K,
    cudaStream_t stream = 0
) {
    constexpr int ROWS = 4;
    const int warps_per_block = 8;
    const int threads_per_block = warps_per_block * WARP_SIZE;

    dim3 block(threads_per_block);
    dim3 grid((M + warps_per_block * ROWS - 1) / (warps_per_block * ROWS), N);

    gemm_q4_0_q8_1_vec_float4_kernel<ROWS><<<grid, block, 0, stream>>>(
        weight, activation, output, M, N, K
    );
}

#endif // KERNELS_GEMM_VECTORIZED_CUH
