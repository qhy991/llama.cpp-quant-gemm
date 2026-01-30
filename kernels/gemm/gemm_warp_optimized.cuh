/**
 * @file kernels/gemm/gemm_warp_optimized.cuh
 * @brief Warp-level 优化的量化 GEMM 实现
 *
 * 优化策略:
 * 1. 一个 warp (32 threads) 协作计算输出元素
 * 2. 每个线程负责部分 K 维度的 blocks
 * 3. 使用 warp shuffle 进行规约
 * 4. 每个 warp 处理多行以提高并行度
 */

#ifndef KERNELS_GEMM_WARP_OPTIMIZED_CUH
#define KERNELS_GEMM_WARP_OPTIMIZED_CUH

#include "../../compat/ggml_types.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============================================================================
// 常量定义
// ============================================================================

#define WARP_SIZE 32

// 每个 warp 处理的行数 (可调参数)
#define ROWS_PER_WARP 4

// ============================================================================
// 辅助函数
// ============================================================================

__device__ __forceinline__ int dp4a_opt(int a, int b, int c) {
#if __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    char4 va = *reinterpret_cast<char4*>(&a);
    char4 vb = *reinterpret_cast<char4*>(&b);
    return c + va.x*vb.x + va.y*vb.y + va.z*vb.z + va.w*vb.w;
#endif
}

__device__ __forceinline__ int load_int_b2_opt(const void* x, int i32) {
    const uint16_t* x16 = (const uint16_t*)x;
    int x32  = x16[2*i32 + 0] <<  0;
    x32     |= x16[2*i32 + 1] << 16;
    return x32;
}

__device__ __forceinline__ int load_int_b4_opt(const void* x, int i32) {
    return ((const int*)x)[i32];
}

// Warp 规约: 将 warp 内所有线程的值求和
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// ============================================================================
// Q4_0 × Q8_1 单 block 点积 (内联版本，供 warp kernel 使用)
// ============================================================================

__device__ __forceinline__ float vec_dot_q4_0_q8_1_opt(
    const block_q4_0* __restrict__ bq4,
    const block_q8_1* __restrict__ bq8
) {
    int sumi = 0;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int v = load_int_b2_opt(bq4->qs, i);
        int vi0 = (v >> 0) & 0x0F0F0F0F;
        int vi1 = (v >> 4) & 0x0F0F0F0F;

        int u0 = load_int_b4_opt(bq8->qs, i);
        int u1 = load_int_b4_opt(bq8->qs, i + 4);

        sumi = dp4a_opt(vi0, u0, sumi);
        sumi = dp4a_opt(vi1, u1, sumi);
    }

    float d4 = __half2float(bq4->d);
    float d8 = __half2float(__low2half(bq8->ds));
    float s8 = __half2float(__high2half(bq8->ds));

    return d4 * (d8 * sumi - 8.0f * s8);
}

// ============================================================================
// Warp-level 优化 GEMM Kernel: Q4_0 × Q8_1
// ============================================================================
/**
 * 每个 warp 计算 ROWS_PER_WARP 行输出
 *
 * 布局:
 * - gridDim.x = (M + ROWS_PER_WARP - 1) / ROWS_PER_WARP
 * - gridDim.y = N
 * - blockDim.x = WARP_SIZE * warps_per_block
 *
 * 每个 warp 内:
 * - 32 个线程分担 K/32 个 blocks
 * - 每个线程计算部分 blocks 的点积
 * - 最后使用 warp shuffle 规约
 */
__global__ void gemm_q4_0_q8_1_warp_kernel(
    const block_q4_0* __restrict__ weight,
    const block_q8_1* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int num_blocks_k = K / 32;  // K 维度的 block 数量

    // 计算 warp 和 lane ID
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;

    // 计算这个 warp 负责的起始行
    const int global_warp_id = blockIdx.x * warps_per_block + warp_id;
    const int row_start = global_warp_id * ROWS_PER_WARP;

    // 当前处理的列 (N 维度)
    const int col = blockIdx.y;

    if (col >= N) return;

    // 激活值的基地址 (对于当前列)
    const block_q8_1* act_base = activation + col * num_blocks_k;

    // 每个线程处理的 blocks 数量
    const int blocks_per_thread = (num_blocks_k + WARP_SIZE - 1) / WARP_SIZE;

    // 处理 ROWS_PER_WARP 行
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; r++) {
        const int row = row_start + r;
        if (row >= M) continue;

        // 权重的基地址 (对于当前行)
        const block_q4_0* weight_base = weight + row * num_blocks_k;

        // 每个线程计算部分 blocks 的点积
        float thread_sum = 0.0f;

        #pragma unroll 4
        for (int t = 0; t < blocks_per_thread; t++) {
            const int block_idx = lane_id + t * WARP_SIZE;
            if (block_idx < num_blocks_k) {
                thread_sum += vec_dot_q4_0_q8_1_opt(
                    &weight_base[block_idx],
                    &act_base[block_idx]
                );
            }
        }

        // Warp 内规约
        float warp_sum = warp_reduce_sum(thread_sum);

        // Lane 0 写入结果
        if (lane_id == 0) {
            output[row * N + col] = warp_sum;
        }
    }
}

// ============================================================================
// 更激进的优化: 每个 warp 处理一行，使用向量化加载
// ============================================================================

/**
 * 简化版: 每个 warp 处理一行
 * 适用于 K 较大的情况
 */
__global__ void gemm_q4_0_q8_1_warp_v2_kernel(
    const block_q4_0* __restrict__ weight,
    const block_q8_1* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int num_blocks_k = K / 32;

    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;

    // 每个 warp 处理一行
    const int row = blockIdx.x * warps_per_block + warp_id;
    const int col = blockIdx.y;

    if (row >= M || col >= N) return;

    const block_q4_0* weight_row = weight + row * num_blocks_k;
    const block_q8_1* act_col = activation + col * num_blocks_k;

    float thread_sum = 0.0f;

    // 每个线程处理多个 blocks，stride = WARP_SIZE
    for (int b = lane_id; b < num_blocks_k; b += WARP_SIZE) {
        thread_sum += vec_dot_q4_0_q8_1_opt(&weight_row[b], &act_col[b]);
    }

    // Warp 规约
    float result = warp_reduce_sum(thread_sum);

    if (lane_id == 0) {
        output[row * N + col] = result;
    }
}

// ============================================================================
// 双缓冲 + 预取优化版本
// ============================================================================

/**
 * 使用寄存器预取来隐藏内存延迟
 */
__global__ void gemm_q4_0_q8_1_warp_prefetch_kernel(
    const block_q4_0* __restrict__ weight,
    const block_q8_1* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int num_blocks_k = K / 32;

    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;

    const int row = blockIdx.x * warps_per_block + warp_id;
    const int col = blockIdx.y;

    if (row >= M || col >= N) return;

    const block_q4_0* weight_row = weight + row * num_blocks_k;
    const block_q8_1* act_col = activation + col * num_blocks_k;

    float thread_sum = 0.0f;

    // 双缓冲: 预取下一个 block 的数据
    const int stride = WARP_SIZE;
    const int num_iters = (num_blocks_k + stride - 1) / stride;

    for (int iter = 0; iter < num_iters; iter++) {
        const int b = lane_id + iter * stride;

        if (b < num_blocks_k) {
            // 加载当前 block
            const block_q4_0* bq4 = &weight_row[b];
            const block_q8_1* bq8 = &act_col[b];

            // 计算点积
            int sumi = 0;

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int v = load_int_b2_opt(bq4->qs, i);
                int vi0 = (v >> 0) & 0x0F0F0F0F;
                int vi1 = (v >> 4) & 0x0F0F0F0F;

                int u0 = load_int_b4_opt(bq8->qs, i);
                int u1 = load_int_b4_opt(bq8->qs, i + 4);

                sumi = dp4a_opt(vi0, u0, sumi);
                sumi = dp4a_opt(vi1, u1, sumi);
            }

            float d4 = __half2float(bq4->d);
            float d8 = __half2float(__low2half(bq8->ds));
            float s8 = __half2float(__high2half(bq8->ds));

            thread_sum += d4 * (d8 * sumi - 8.0f * s8);
        }
    }

    // Warp 规约
    float result = warp_reduce_sum(thread_sum);

    if (lane_id == 0) {
        output[row * N + col] = result;
    }
}

// ============================================================================
// 多行并行版本 (针对 N 很小的情况优化)
// ============================================================================

/**
 * 当 N 很小 (如 N=1 或 N=2) 时，同时处理多行以提高 GPU 利用率
 * 每个 warp 同时处理 4 行，每个线程在 K 维度上协作
 */
template<int ROWS = 4>
__global__ void gemm_q4_0_q8_1_warp_multirow_kernel(
    const block_q4_0* __restrict__ weight,
    const block_q8_1* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int num_blocks_k = K / 32;

    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;

    // 每个 warp 处理 ROWS 行
    const int row_base = (blockIdx.x * warps_per_block + warp_id) * ROWS;
    const int col = blockIdx.y;

    if (col >= N) return;

    const block_q8_1* act_col = activation + col * num_blocks_k;

    // 每行一个累加器
    float sums[ROWS] = {0.0f};

    // 遍历 K 维度的 blocks
    for (int b = lane_id; b < num_blocks_k; b += WARP_SIZE) {
        // 预加载激活值 (所有行共享)
        const block_q8_1* bq8 = &act_col[b];

        // 预加载激活值到寄存器
        int u[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            u[i] = load_int_b4_opt(bq8->qs, i);
        }
        float d8 = __half2float(__low2half(bq8->ds));
        float s8 = __half2float(__high2half(bq8->ds));

        // 处理每一行
        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const int row = row_base + r;
            if (row >= M) continue;

            const block_q4_0* bq4 = &weight[row * num_blocks_k + b];

            int sumi = 0;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int v = load_int_b2_opt(bq4->qs, i);
                int vi0 = (v >> 0) & 0x0F0F0F0F;
                int vi1 = (v >> 4) & 0x0F0F0F0F;

                sumi = dp4a_opt(vi0, u[i], sumi);
                sumi = dp4a_opt(vi1, u[i + 4], sumi);
            }

            float d4 = __half2float(bq4->d);
            sums[r] += d4 * (d8 * sumi - 8.0f * s8);
        }
    }

    // Warp 规约并写入结果
    #pragma unroll
    for (int r = 0; r < ROWS; r++) {
        const int row = row_base + r;
        if (row >= M) continue;

        float result = warp_reduce_sum(sums[r]);

        if (lane_id == 0) {
            output[row * N + col] = result;
        }
    }
}

// ============================================================================
// 接口函数
// ============================================================================

/**
 * Warp 优化版 Q4_0 × Q8_1 GEMM (基础版)
 */
inline void gemm_q4_0_q8_1_warp(
    const block_q4_0* weight,
    const block_q8_1* activation,
    float* output,
    int M, int N, int K,
    cudaStream_t stream = 0
) {
    const int warps_per_block = 8;  // 每个 block 8 个 warp = 256 threads
    const int threads_per_block = warps_per_block * WARP_SIZE;

    dim3 block(threads_per_block);
    dim3 grid((M + warps_per_block * ROWS_PER_WARP - 1) / (warps_per_block * ROWS_PER_WARP), N);

    gemm_q4_0_q8_1_warp_kernel<<<grid, block, 0, stream>>>(
        weight, activation, output, M, N, K
    );
}

/**
 * Warp 优化版 Q4_0 × Q8_1 GEMM (V2: 每 warp 一行)
 */
inline void gemm_q4_0_q8_1_warp_v2(
    const block_q4_0* weight,
    const block_q8_1* activation,
    float* output,
    int M, int N, int K,
    cudaStream_t stream = 0
) {
    const int warps_per_block = 8;
    const int threads_per_block = warps_per_block * WARP_SIZE;

    dim3 block(threads_per_block);
    dim3 grid((M + warps_per_block - 1) / warps_per_block, N);

    gemm_q4_0_q8_1_warp_v2_kernel<<<grid, block, 0, stream>>>(
        weight, activation, output, M, N, K
    );
}

/**
 * Warp 优化版 Q4_0 × Q8_1 GEMM (预取版)
 */
inline void gemm_q4_0_q8_1_warp_prefetch(
    const block_q4_0* weight,
    const block_q8_1* activation,
    float* output,
    int M, int N, int K,
    cudaStream_t stream = 0
) {
    const int warps_per_block = 8;
    const int threads_per_block = warps_per_block * WARP_SIZE;

    dim3 block(threads_per_block);
    dim3 grid((M + warps_per_block - 1) / warps_per_block, N);

    gemm_q4_0_q8_1_warp_prefetch_kernel<<<grid, block, 0, stream>>>(
        weight, activation, output, M, N, K
    );
}

/**
 * Warp 优化版 Q4_0 × Q8_1 GEMM (多行并行版)
 */
inline void gemm_q4_0_q8_1_warp_multirow(
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

    gemm_q4_0_q8_1_warp_multirow_kernel<ROWS><<<grid, block, 0, stream>>>(
        weight, activation, output, M, N, K
    );
}

// ============================================================================
// Shared Memory 优化版本
// ============================================================================

/**
 * 使用 shared memory 缓存激活值
 * 减少全局内存访问次数
 */
__global__ void gemm_q4_0_q8_1_smem_kernel(
    const block_q4_0* __restrict__ weight,
    const block_q8_1* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int num_blocks_k = K / 32;

    // Shared memory 缓存激活值
    // 每个 block_q8_1 = 36 bytes
    extern __shared__ char smem[];
    block_q8_1* s_activation = (block_q8_1*)smem;

    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;

    const int col = blockIdx.y;
    if (col >= N) return;

    // 协作加载激活值到 shared memory
    // 每个线程加载部分数据
    const block_q8_1* act_col = activation + col * num_blocks_k;

    // 分块处理 K 维度
    constexpr int TILE_K = 128;  // 每次处理 128 个 blocks
    const int num_tiles = (num_blocks_k + TILE_K - 1) / TILE_K;

    // 每个 warp 处理多行
    constexpr int ROWS = 4;
    const int row_base = (blockIdx.x * warps_per_block + warp_id) * ROWS;

    float sums[ROWS] = {0.0f};

    for (int tile = 0; tile < num_tiles; tile++) {
        const int tile_start = tile * TILE_K;
        const int tile_size = min(TILE_K, num_blocks_k - tile_start);

        // 协作加载激活值到 shared memory
        for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
            s_activation[i] = act_col[tile_start + i];
        }
        __syncthreads();

        // 处理每一行
        for (int b = lane_id; b < tile_size; b += WARP_SIZE) {
            const block_q8_1* bq8 = &s_activation[b];

            // 预加载激活值
            int u[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                u[i] = load_int_b4_opt(bq8->qs, i);
            }
            float d8 = __half2float(__low2half(bq8->ds));
            float s8 = __half2float(__high2half(bq8->ds));

            #pragma unroll
            for (int r = 0; r < ROWS; r++) {
                const int row = row_base + r;
                if (row >= M) continue;

                const block_q4_0* bq4 = &weight[row * num_blocks_k + tile_start + b];

                int sumi = 0;
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    int v = load_int_b2_opt(bq4->qs, i);
                    int vi0 = (v >> 0) & 0x0F0F0F0F;
                    int vi1 = (v >> 4) & 0x0F0F0F0F;

                    sumi = dp4a_opt(vi0, u[i], sumi);
                    sumi = dp4a_opt(vi1, u[i + 4], sumi);
                }

                float d4 = __half2float(bq4->d);
                sums[r] += d4 * (d8 * sumi - 8.0f * s8);
            }
        }
        __syncthreads();
    }

    // Warp 规约并写入
    #pragma unroll
    for (int r = 0; r < ROWS; r++) {
        const int row = row_base + r;
        if (row >= M) continue;

        float result = warp_reduce_sum(sums[r]);

        if (lane_id == 0) {
            output[row * N + col] = result;
        }
    }
}

/**
 * Shared memory 优化版接口
 */
inline void gemm_q4_0_q8_1_smem(
    const block_q4_0* weight,
    const block_q8_1* activation,
    float* output,
    int M, int N, int K,
    cudaStream_t stream = 0
) {
    constexpr int ROWS = 4;
    constexpr int TILE_K = 128;
    const int warps_per_block = 8;
    const int threads_per_block = warps_per_block * WARP_SIZE;
    const size_t smem_size = TILE_K * sizeof(block_q8_1);

    dim3 block(threads_per_block);
    dim3 grid((M + warps_per_block * ROWS - 1) / (warps_per_block * ROWS), N);

    gemm_q4_0_q8_1_smem_kernel<<<grid, block, smem_size, stream>>>(
        weight, activation, output, M, N, K
    );
}

// ============================================================================
// 更激进的优化: 增大 ROWS 和优化内存访问模式
// ============================================================================

/**
 * 优化版 multirow: 每个 warp 处理更多行 (8 行)
 */
template<int ROWS = 8>
__global__ void gemm_q4_0_q8_1_warp_multirow8_kernel(
    const block_q4_0* __restrict__ weight,
    const block_q8_1* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int num_blocks_k = K / 32;

    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;

    // 每个 warp 处理 ROWS 行
    const int row_base = (blockIdx.x * warps_per_block + warp_id) * ROWS;
    const int col = blockIdx.y;

    if (col >= N) return;

    const block_q8_1* act_col = activation + col * num_blocks_k;

    // 每行一个累加器
    float sums[ROWS] = {0.0f};

    // 遍历 K 维度的 blocks
    for (int b = lane_id; b < num_blocks_k; b += WARP_SIZE) {
        // 预加载激活值 (所有行共享)
        const block_q8_1* bq8 = &act_col[b];

        // 预加载激活值到寄存器
        int u[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            u[i] = load_int_b4_opt(bq8->qs, i);
        }
        float d8 = __half2float(__low2half(bq8->ds));
        float s8 = __half2float(__high2half(bq8->ds));

        // 处理每一行
        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const int row = row_base + r;
            if (row >= M) continue;

            const block_q4_0* bq4 = &weight[row * num_blocks_k + b];

            int sumi = 0;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int v = load_int_b2_opt(bq4->qs, i);
                int vi0 = (v >> 0) & 0x0F0F0F0F;
                int vi1 = (v >> 4) & 0x0F0F0F0F;

                sumi = dp4a_opt(vi0, u[i], sumi);
                sumi = dp4a_opt(vi1, u[i + 4], sumi);
            }

            float d4 = __half2float(bq4->d);
            sums[r] += d4 * (d8 * sumi - 8.0f * s8);
        }
    }

    // Warp 规约并写入结果
    #pragma unroll
    for (int r = 0; r < ROWS; r++) {
        const int row = row_base + r;
        if (row >= M) continue;

        float result = warp_reduce_sum(sums[r]);

        if (lane_id == 0) {
            output[row * N + col] = result;
        }
    }
}

/**
 * 8 行版本接口
 */
inline void gemm_q4_0_q8_1_warp_multirow8(
    const block_q4_0* weight,
    const block_q8_1* activation,
    float* output,
    int M, int N, int K,
    cudaStream_t stream = 0
) {
    constexpr int ROWS = 8;
    const int warps_per_block = 8;
    const int threads_per_block = warps_per_block * WARP_SIZE;

    dim3 block(threads_per_block);
    dim3 grid((M + warps_per_block * ROWS - 1) / (warps_per_block * ROWS), N);

    gemm_q4_0_q8_1_warp_multirow8_kernel<ROWS><<<grid, block, 0, stream>>>(
        weight, activation, output, M, N, K
    );
}

// ============================================================================
// 优化版: 调整 shared memory tile size
// ============================================================================

/**
 * Shared memory 优化版 - 更大的 tile (256)
 */
__global__ void gemm_q4_0_q8_1_smem_large_kernel(
    const block_q4_0* __restrict__ weight,
    const block_q8_1* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int num_blocks_k = K / 32;

    // Shared memory 缓存激活值
    extern __shared__ char smem[];
    block_q8_1* s_activation = (block_q8_1*)smem;

    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;

    const int col = blockIdx.y;
    if (col >= N) return;

    // 协作加载激活值到 shared memory
    const block_q8_1* act_col = activation + col * num_blocks_k;

    // 分块处理 K 维度
    constexpr int TILE_K = 256;  // 更大的 tile
    const int num_tiles = (num_blocks_k + TILE_K - 1) / TILE_K;

    // 每个 warp 处理多行
    constexpr int ROWS = 4;
    const int row_base = (blockIdx.x * warps_per_block + warp_id) * ROWS;

    float sums[ROWS] = {0.0f};

    for (int tile = 0; tile < num_tiles; tile++) {
        const int tile_start = tile * TILE_K;
        const int tile_size = min(TILE_K, num_blocks_k - tile_start);

        // 协作加载激活值到 shared memory
        for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
            s_activation[i] = act_col[tile_start + i];
        }
        __syncthreads();

        // 处理每一行
        for (int b = lane_id; b < tile_size; b += WARP_SIZE) {
            const block_q8_1* bq8 = &s_activation[b];

            // 预加载激活值
            int u[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                u[i] = load_int_b4_opt(bq8->qs, i);
            }
            float d8 = __half2float(__low2half(bq8->ds));
            float s8 = __half2float(__high2half(bq8->ds));

            #pragma unroll
            for (int r = 0; r < ROWS; r++) {
                const int row = row_base + r;
                if (row >= M) continue;

                const block_q4_0* bq4 = &weight[row * num_blocks_k + tile_start + b];

                int sumi = 0;
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    int v = load_int_b2_opt(bq4->qs, i);
                    int vi0 = (v >> 0) & 0x0F0F0F0F;
                    int vi1 = (v >> 4) & 0x0F0F0F0F;

                    sumi = dp4a_opt(vi0, u[i], sumi);
                    sumi = dp4a_opt(vi1, u[i + 4], sumi);
                }

                float d4 = __half2float(bq4->d);
                sums[r] += d4 * (d8 * sumi - 8.0f * s8);
            }
        }
        __syncthreads();
    }

    // Warp 规约并写入
    #pragma unroll
    for (int r = 0; r < ROWS; r++) {
        const int row = row_base + r;
        if (row >= M) continue;

        float result = warp_reduce_sum(sums[r]);

        if (lane_id == 0) {
            output[row * N + col] = result;
        }
    }
}

/**
 * Large tile shared memory 版本接口
 */
inline void gemm_q4_0_q8_1_smem_large(
    const block_q4_0* weight,
    const block_q8_1* activation,
    float* output,
    int M, int N, int K,
    cudaStream_t stream = 0
) {
    constexpr int ROWS = 4;
    constexpr int TILE_K = 256;
    const int warps_per_block = 8;
    const int threads_per_block = warps_per_block * WARP_SIZE;
    const size_t smem_size = TILE_K * sizeof(block_q8_1);

    dim3 block(threads_per_block);
    dim3 grid((M + warps_per_block * ROWS - 1) / (warps_per_block * ROWS), N);

    gemm_q4_0_q8_1_smem_large_kernel<<<grid, block, smem_size, stream>>>(
        weight, activation, output, M, N, K
    );
}

// ============================================================================
// 向量化加载 + Warp 优化版本
// ============================================================================

/**
 * 使用向量化加载 - 修复版
 * 直接使用原始算法，只是用向量化加载来减少指令数
 */
__global__ void gemm_q4_0_q8_1_vec_kernel(
    const block_q4_0* __restrict__ weight,
    const block_q8_1* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int num_blocks_k = K / 32;

    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;

    constexpr int ROWS = 4;
    const int row_base = (blockIdx.x * warps_per_block + warp_id) * ROWS;
    const int col = blockIdx.y;

    if (col >= N) return;

    const block_q8_1* act_col = activation + col * num_blocks_k;

    float sums[ROWS] = {0.0f};

    // 每个线程处理多个 blocks
    for (int b = lane_id; b < num_blocks_k; b += WARP_SIZE) {
        const block_q8_1* bq8 = &act_col[b];

        // 预加载激活值到寄存器 (使用向量化加载)
        int u[8];
        const int4* u_vec = (const int4*)bq8->qs;
        int4 u_tmp0 = u_vec[0];
        int4 u_tmp1 = u_vec[1];
        u[0] = u_tmp0.x; u[1] = u_tmp0.y; u[2] = u_tmp0.z; u[3] = u_tmp0.w;
        u[4] = u_tmp1.x; u[5] = u_tmp1.y; u[6] = u_tmp1.z; u[7] = u_tmp1.w;

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
                int v = load_int_b2_opt(bq4->qs, i);
                int vi0 = (v >> 0) & 0x0F0F0F0F;
                int vi1 = (v >> 4) & 0x0F0F0F0F;

                sumi = dp4a_opt(vi0, u[i], sumi);
                sumi = dp4a_opt(vi1, u[i + 4], sumi);
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

        float result = warp_reduce_sum(sums[r]);

        if (lane_id == 0) {
            output[row * N + col] = result;
        }
    }
}

/**
 * 向量化加载版接口
 */
inline void gemm_q4_0_q8_1_vec(
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

    gemm_q4_0_q8_1_vec_kernel<<<grid, block, 0, stream>>>(
        weight, activation, output, M, N, K
    );
}

// ============================================================================
// 2D Tile 优化版本 - 每个 block 处理多行多列
// ============================================================================

/**
 * 2D Tile 版本: 每个 block 处理 TILE_M x TILE_N 的输出 tile
 * 关键优化: 激活值加载一次，复用 TILE_M 次
 *
 * 内存布局:
 * - Shared memory: TILE_N 列 × TILE_K blocks 的激活值
 * - 每个 warp 处理 ROWS 行 × TILE_N 列
 */
template<int TILE_M = 64, int TILE_N = 4, int ROWS = 4>
__global__ void gemm_q4_0_q8_1_tile2d_kernel(
    const block_q4_0* __restrict__ weight,
    const block_q8_1* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    const int num_blocks_k = K / 32;

    // 当前 block 处理的输出 tile
    const int tile_m_start = blockIdx.x * TILE_M;
    const int tile_n_start = blockIdx.y * TILE_N;

    // Shared memory: 存储 TILE_N 列的激活值
    extern __shared__ char smem[];
    block_q8_1* s_activation = (block_q8_1*)smem;

    // Warp 和 lane 信息
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;

    // 当前 warp 处理的行
    const int warp_m_start = tile_m_start + warp_id * ROWS;

    // 累加器: ROWS 行 × TILE_N 列
    float acc[ROWS][TILE_N];
    #pragma unroll
    for (int r = 0; r < ROWS; r++) {
        #pragma unroll
        for (int c = 0; c < TILE_N; c++) {
            acc[r][c] = 0.0f;
        }
    }

    // 分块处理 K 维度
    constexpr int TILE_K = 128;
    const int num_k_tiles = (num_blocks_k + TILE_K - 1) / TILE_K;

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        const int k_start = k_tile * TILE_K;
        const int k_size = min(TILE_K, num_blocks_k - k_start);

        // 协作加载激活值到 shared memory
        // 每个线程加载部分数据
        const int total_blocks = TILE_N * k_size;
        for (int idx = threadIdx.x; idx < total_blocks; idx += blockDim.x) {
            const int c = idx / k_size;
            const int k = idx % k_size;
            const int col = tile_n_start + c;

            if (col < N) {
                s_activation[c * TILE_K + k] = activation[col * num_blocks_k + k_start + k];
            }
        }
        __syncthreads();

        // 每个 warp 计算 ROWS_PER_WARP × TILE_N 的输出
        // 在 K 维度上并行
        for (int k = lane_id; k < k_size; k += WARP_SIZE) {
            // 预加载激活值 (TILE_N 列)
            int u[TILE_N][8];
            float d8[TILE_N], s8[TILE_N];

            #pragma unroll
            for (int c = 0; c < TILE_N; c++) {
                const int col = tile_n_start + c;
                if (col < N) {
                    const block_q8_1* bq8 = &s_activation[c * TILE_K + k];

                    #pragma unroll
                    for (int i = 0; i < 8; i++) {
                        u[c][i] = load_int_b4_opt(bq8->qs, i);
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

                // 加载权重
                int v[4];
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    v[i] = load_int_b2_opt(bq4->qs, i);
                }
                float d4 = __half2float(bq4->d);

                // 计算与 TILE_N 列的点积
                #pragma unroll
                for (int c = 0; c < TILE_N; c++) {
                    const int col = tile_n_start + c;
                    if (col >= N) continue;

                    int sumi = 0;
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {
                        int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
                        int vi1 = (v[i] >> 4) & 0x0F0F0F0F;

                        sumi = dp4a_opt(vi0, u[c][i], sumi);
                        sumi = dp4a_opt(vi1, u[c][i + 4], sumi);
                    }

                    acc[r][c] += d4 * (d8[c] * sumi - 8.0f * s8[c]);
                }
            }
        }
        __syncthreads();
    }

    // Warp 规约并写回结果
    #pragma unroll
    for (int r = 0; r < ROWS; r++) {
        const int row = warp_m_start + r;
        if (row >= M) continue;

        #pragma unroll
        for (int c = 0; c < TILE_N; c++) {
            const int col = tile_n_start + c;
            if (col >= N) continue;

            // Warp 规约
            float sum = warp_reduce_sum(acc[r][c]);

            if (lane_id == 0) {
                output[row * N + col] = sum;
            }
        }
    }
}

/**
 * 2D Tile 版本接口
 */
inline void gemm_q4_0_q8_1_tile2d(
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
    const size_t smem_size = TILE_N * TILE_K * sizeof(block_q8_1);

    dim3 block(threads_per_block);
    dim3 grid((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);

    gemm_q4_0_q8_1_tile2d_kernel<TILE_M, TILE_N, ROWS>
        <<<grid, block, smem_size, stream>>>(
        weight, activation, output, M, N, K
    );
}

/**
 * 2D Tile 版本接口 - N=8
 */
inline void gemm_q4_0_q8_1_tile2d_n8(
    const block_q4_0* weight,
    const block_q8_1* activation,
    float* output,
    int M, int N, int K,
    cudaStream_t stream = 0
) {
    constexpr int TILE_M = 64;
    constexpr int TILE_N = 8;
    constexpr int TILE_K = 128;
    constexpr int ROWS = 4;

    const int warps_per_block = TILE_M / ROWS;
    const int threads_per_block = warps_per_block * WARP_SIZE;
    const size_t smem_size = TILE_N * TILE_K * sizeof(block_q8_1);

    dim3 block(threads_per_block);
    dim3 grid((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);

    gemm_q4_0_q8_1_tile2d_kernel<TILE_M, TILE_N, ROWS>
        <<<grid, block, smem_size, stream>>>(
        weight, activation, output, M, N, K
    );
}

/**
 * 2D Tile 版本接口 - 更大的 K tile (256)
 */
inline void gemm_q4_0_q8_1_tile2d_k256(
    const block_q4_0* weight,
    const block_q8_1* activation,
    float* output,
    int M, int N, int K,
    cudaStream_t stream = 0
) {
    constexpr int TILE_M = 64;
    constexpr int TILE_N = 4;
    constexpr int TILE_K = 256;
    constexpr int ROWS = 4;

    const int warps_per_block = TILE_M / ROWS;
    const int threads_per_block = warps_per_block * WARP_SIZE;
    const size_t smem_size = TILE_N * TILE_K * sizeof(block_q8_1);

    dim3 block(threads_per_block);
    dim3 grid((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);

    gemm_q4_0_q8_1_tile2d_kernel<TILE_M, TILE_N, ROWS>
        <<<grid, block, smem_size, stream>>>(
        weight, activation, output, M, N, K
    );
}

/**
 * 2D Tile 版本接口 - 8 rows per warp
 */
inline void gemm_q4_0_q8_1_tile2d_r8(
    const block_q4_0* weight,
    const block_q8_1* activation,
    float* output,
    int M, int N, int K,
    cudaStream_t stream = 0
) {
    constexpr int TILE_M = 64;
    constexpr int TILE_N = 4;
    constexpr int TILE_K = 128;
    constexpr int ROWS = 8;

    const int warps_per_block = TILE_M / ROWS;  // 8 warps
    const int threads_per_block = warps_per_block * WARP_SIZE;
    const size_t smem_size = TILE_N * TILE_K * sizeof(block_q8_1);

    dim3 block(threads_per_block);
    dim3 grid((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);

    gemm_q4_0_q8_1_tile2d_kernel<TILE_M, TILE_N, ROWS>
        <<<grid, block, smem_size, stream>>>(
        weight, activation, output, M, N, K
    );
}

/**
 * 2D Tile 版本接口 - 更大的 M tile
 */
inline void gemm_q4_0_q8_1_tile2d_large(
    const block_q4_0* weight,
    const block_q8_1* activation,
    float* output,
    int M, int N, int K,
    cudaStream_t stream = 0
) {
    constexpr int TILE_M = 128;
    constexpr int TILE_N = 4;
    constexpr int TILE_K = 128;
    constexpr int ROWS = 4;

    const int warps_per_block = TILE_M / ROWS;  // 32 warps = 1024 threads
    const int threads_per_block = warps_per_block * WARP_SIZE;
    const size_t smem_size = TILE_N * TILE_K * sizeof(block_q8_1);

    dim3 block(threads_per_block);
    dim3 grid((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);

    gemm_q4_0_q8_1_tile2d_kernel<TILE_M, TILE_N, ROWS>
        <<<grid, block, smem_size, stream>>>(
        weight, activation, output, M, N, K
    );
}

#endif // KERNELS_GEMM_WARP_OPTIMIZED_CUH
