/**
 * @file kernels/gemm/gemm_async_copy.cuh
 * @brief Async Copy + Double Buffering 优化版本
 *
 * 关键技术:
 * 1. 使用 cp.async 异步拷贝数据到 shared memory
 * 2. Double buffering: 两个 shared memory 缓冲区交替使用
 * 3. Pipeline: 在计算 tile N 的同时加载 tile N+1
 */

#ifndef KERNELS_GEMM_ASYNC_COPY_CUH
#define KERNELS_GEMM_ASYNC_COPY_CUH

#include "../../compat/ggml_types.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>

// ============================================================================
// 辅助函数 (复用 warp_optimized.cuh 中的定义)
// ============================================================================

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

__device__ __forceinline__ int dp4a_async(int a, int b, int c) {
#if __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    char4 va = *reinterpret_cast<char4*>(&a);
    char4 vb = *reinterpret_cast<char4*>(&b);
    return c + va.x*vb.x + va.y*vb.y + va.z*vb.z + va.w*vb.w;
#endif
}

__device__ __forceinline__ int load_int_b2_async(const void* x, int i32) {
    const uint16_t* x16 = (const uint16_t*)x;
    int x32  = x16[2*i32 + 0] <<  0;
    x32     |= x16[2*i32 + 1] << 16;
    return x32;
}

__device__ __forceinline__ int load_int_b4_async(const void* x, int i32) {
    return ((const int*)x)[i32];
}

__device__ __forceinline__ float warp_reduce_sum_async(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// ============================================================================
// Async Copy + Double Buffering 2D Tile Kernel
// ============================================================================

/**
 * 使用 double buffering 的 2D Tile 版本（简化版，不使用 cp.async）
 *
 * 通过 double buffering 隐藏内存延迟
 */
template<int TILE_M = 64, int TILE_N = 4, int ROWS = 4>
__global__ void gemm_q4_0_q8_1_async_kernel(
    const block_q4_0* __restrict__ weight,
    const block_q8_1* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int num_blocks_k = K / 32;

    // 当前 block 处理的输出 tile
    const int tile_m_start = blockIdx.x * TILE_M;
    const int tile_n_start = blockIdx.y * TILE_N;

    // Double buffering: 两个 shared memory 缓冲区
    extern __shared__ char smem[];
    constexpr int TILE_K = 128;
    block_q8_1* s_activation[2];
    s_activation[0] = (block_q8_1*)smem;
    s_activation[1] = (block_q8_1*)(smem + TILE_N * TILE_K * sizeof(block_q8_1));

    // Warp 和 lane 信息
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;

    // 当前 warp 处理的行
    const int warp_m_start = tile_m_start + warp_id * ROWS;

    // 累加器
    float acc[ROWS][TILE_N];
    #pragma unroll
    for (int r = 0; r < ROWS; r++) {
        #pragma unroll
        for (int c = 0; c < TILE_N; c++) {
            acc[r][c] = 0.0f;
        }
    }

    // K 维度分块
    const int num_k_tiles = (num_blocks_k + TILE_K - 1) / TILE_K;

    // 预加载第一个 tile
    int current_buffer = 0;

    if (num_k_tiles > 0) {
        const int k_size = min(TILE_K, num_blocks_k);
        const int total_blocks = TILE_N * k_size;

        for (int idx = threadIdx.x; idx < total_blocks; idx += blockDim.x) {
            const int c = idx / k_size;
            const int k = idx % k_size;
            const int col = tile_n_start + c;

            if (col < N) {
                s_activation[0][c * TILE_K + k] = activation[col * num_blocks_k + k];
            }
        }
        __syncthreads();
    }

    // 主循环
    for (int tile = 0; tile < num_k_tiles; tile++) {
        const int k_start = tile * TILE_K;
        const int k_size = min(TILE_K, num_blocks_k - k_start);

        // 异步加载下一个 tile
        const int next_buffer = 1 - current_buffer;
        if (tile + 1 < num_k_tiles) {
            const int next_k_start = (tile + 1) * TILE_K;
            const int next_k_size = min(TILE_K, num_blocks_k - next_k_start);
            const int total_blocks = TILE_N * next_k_size;

            // 使用部分线程加载，其他线程可以开始计算
            if (threadIdx.x < 128) {  // 只用部分线程加载
                for (int idx = threadIdx.x; idx < total_blocks; idx += 128) {
                    const int c = idx / next_k_size;
                    const int k = idx % next_k_size;
                    const int col = tile_n_start + c;

                    if (col < N) {
                        s_activation[next_buffer][c * TILE_K + k] =
                            activation[col * num_blocks_k + next_k_start + k];
                    }
                }
            }
        }

        // 计算当前 tile
        for (int k = lane_id; k < k_size; k += WARP_SIZE) {
            // 预加载激活值
            int u[TILE_N][8];
            float d8[TILE_N], s8[TILE_N];

            #pragma unroll
            for (int c = 0; c < TILE_N; c++) {
                const int col = tile_n_start + c;
                if (col < N) {
                    const block_q8_1* bq8 = &s_activation[current_buffer][c * TILE_K + k];

                    #pragma unroll
                    for (int i = 0; i < 8; i++) {
                        u[c][i] = load_int_b4_async(bq8->qs, i);
                    }
                    d8[c] = __half2float(__low2half(bq8->ds));
                    s8[c] = __half2float(__high2half(bq8->ds));
                }
            }

            // 处理 ROWS 行
            #pragma unroll
            for (int r = 0; r < ROWS; r++) {
                const int row = warp_m_start + r;
                if (row >= M) continue;

                const block_q4_0* bq4 = &weight[row * num_blocks_k + k_start + k];

                int v[4];
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    v[i] = load_int_b2_async(bq4->qs, i);
                }
                float d4 = __half2float(bq4->d);

                #pragma unroll
                for (int c = 0; c < TILE_N; c++) {
                    const int col = tile_n_start + c;
                    if (col >= N) continue;

                    int sumi = 0;
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {
                        int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
                        int vi1 = (v[i] >> 4) & 0x0F0F0F0F;

                        sumi = dp4a_async(vi0, u[c][i], sumi);
                        sumi = dp4a_async(vi1, u[c][i + 4], sumi);
                    }

                    acc[r][c] += d4 * (d8[c] * sumi - 8.0f * s8[c]);
                }
            }
        }

        // 同步等待加载完成
        __syncthreads();

        // 切换缓冲区
        current_buffer = next_buffer;
    }

    // Warp 规约并写回
    #pragma unroll
    for (int r = 0; r < ROWS; r++) {
        const int row = warp_m_start + r;
        if (row >= M) continue;

        #pragma unroll
        for (int c = 0; c < TILE_N; c++) {
            const int col = tile_n_start + c;
            if (col >= N) continue;

            float sum = warp_reduce_sum_async(acc[r][c]);

            if (lane_id == 0) {
                output[row * N + col] = sum;
            }
        }
    }
}

/**
 * Async Copy 版本接口
 */
inline void gemm_q4_0_q8_1_async(
    const block_q4_0* weight,
    const block_q8_1* activation,
    float* output,
    int M, int N, int K,
    cudaStream_t stream = 0
) {
    constexpr int TILE_M = 64;
    constexpr int TILE_N = 4;
    constexpr int TILE_K = 128;
    constexpr int ROWS = 4;

    const int warps_per_block = TILE_M / ROWS;  // 16 warps
    const int threads_per_block = warps_per_block * WARP_SIZE;  // 512 threads

    // Double buffering: 需要 2 倍的 shared memory
    const size_t smem_size = 2 * TILE_N * TILE_K * sizeof(block_q8_1);

    dim3 block(threads_per_block);
    dim3 grid((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);

    gemm_q4_0_q8_1_async_kernel<TILE_M, TILE_N, ROWS>
        <<<grid, block, smem_size, stream>>>(
        weight, activation, output, M, N, K
    );
}

#endif // KERNELS_GEMM_ASYNC_COPY_CUH
