/**
 * @file kernels/gemm/gemm_fused.cuh
 * @brief Level 4: Kernel 融合优化
 *
 * 目标: 将 FP16 激活量化融合到 GEMM kernel 中
 *
 * 当前流程:
 *   1. quantize_fp16_to_q8_1(fp16_activation, q8_1_activation)  // 单独 kernel
 *   2. gemm_q4_0_q8_1(weight, q8_1_activation, output)          // GEMM kernel
 *
 * 融合后流程:
 *   1. gemm_q4_0_fp16_fused(weight, fp16_activation, output)    // 一个 kernel
 *      - 在 shared memory 中量化
 *      - 立即用于 GEMM 计算
 *
 * 优势:
 *   - 减少一次全局内存读写
 *   - 提高 L2 cache 命中率
 *   - 减少 kernel 启动开销
 *
 * 预期提升: 1.2x (20%)
 */

#ifndef KERNELS_GEMM_FUSED_CUH
#define KERNELS_GEMM_FUSED_CUH

#include "../../compat/ggml_types.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// ============================================================================
// 辅助函数
// ============================================================================

__device__ __forceinline__ int dp4a_fused(int a, int b, int c) {
#if __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    char4 va = *reinterpret_cast<char4*>(&a);
    char4 vb = *reinterpret_cast<char4*>(&b);
    return c + va.x*vb.x + va.y*vb.y + va.z*vb.z + va.w*vb.w;
#endif
}

__device__ __forceinline__ int load_int_b2_fused(const void* x, int i32) {
    const uint16_t* x16 = (const uint16_t*)x;
    int x32  = x16[2*i32 + 0] <<  0;
    x32     |= x16[2*i32 + 1] << 16;
    return x32;
}

__device__ __forceinline__ float warp_reduce_sum_fused(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// ============================================================================
// FP16 → Q8_1 量化 (在 shared memory 中)
// ============================================================================

/**
 * 将 FP16 激活量化为 Q8_1 格式
 *
 * Q8_1 量化公式:
 *   d = max(|x|) / 127.0
 *   s = sum(x)
 *   q = round(x / d), 限制到 [-127, 127]
 */
__device__ void quantize_fp16_to_q8_1_smem(
    const half* fp16_data,      // 输入: 32 个 FP16 值
    block_q8_1* q8_1_block,     // 输出: 1 个 Q8_1 block
    int tid                      // 线程 ID (用于协作)
) {
    // 使用 shared memory 进行规约
    __shared__ float s_max[32];
    __shared__ float s_sum[32];

    // 每个线程处理一个元素
    if (tid < 32) {
        float val = __half2float(fp16_data[tid]);
        s_max[tid] = fabsf(val);
        s_sum[tid] = val;
    }
    __syncthreads();

    // 规约求最大值和求和
    if (tid < 16) {
        s_max[tid] = fmaxf(s_max[tid], s_max[tid + 16]);
        s_sum[tid] += s_sum[tid + 16];
    }
    __syncthreads();

    if (tid < 8) {
        s_max[tid] = fmaxf(s_max[tid], s_max[tid + 8]);
        s_sum[tid] += s_sum[tid + 8];
    }
    __syncthreads();

    if (tid < 4) {
        s_max[tid] = fmaxf(s_max[tid], s_max[tid + 4]);
        s_sum[tid] += s_sum[tid + 4];
    }
    __syncthreads();

    if (tid < 2) {
        s_max[tid] = fmaxf(s_max[tid], s_max[tid + 2]);
        s_sum[tid] += s_sum[tid + 2];
    }
    __syncthreads();

    float max_val, sum_val;
    if (tid == 0) {
        max_val = fmaxf(s_max[0], s_max[1]);
        sum_val = s_sum[0] + s_sum[1];

        // 计算缩放因子
        float d = max_val / 127.0f;
        float id = (d != 0.0f) ? 1.0f / d : 0.0f;

        // 存储 d 和 s
        q8_1_block->ds = __halves2half2(__float2half(d), __float2half(sum_val));
    }
    __syncthreads();

    // 所有线程读取 d
    float d = __half2float(__low2half(q8_1_block->ds));
    float id = (d != 0.0f) ? 1.0f / d : 0.0f;

    // 量化
    if (tid < 32) {
        float val = __half2float(fp16_data[tid]);
        int8_t q = (int8_t)roundf(val * id);
        q = max(-127, min(127, (int)q));
        q8_1_block->qs[tid] = q;
    }
}

// ============================================================================
// 融合 Kernel: Q4_0 × FP16 (在 kernel 内量化为 Q8_1)
// ============================================================================

/**
 * 融合版 GEMM: 在 kernel 内将 FP16 激活量化为 Q8_1
 *
 * 流程:
 * 1. 加载 FP16 激活到 shared memory
 * 2. 协作量化为 Q8_1
 * 3. 立即用于 GEMM 计算
 */
template<int TILE_M = 64, int TILE_N = 4, int ROWS = 4>
__global__ void gemm_q4_0_fp16_fused_kernel(
    const block_q4_0* __restrict__ weight,
    const half* __restrict__ fp16_activation,  // FP16 激活
    float* __restrict__ output,
    int M, int N, int K
) {
    const int num_blocks_k = K / 32;

    // 当前 block 处理的输出 tile
    const int tile_m_start = blockIdx.x * TILE_M;
    const int tile_n_start = blockIdx.y * TILE_N;

    // Shared memory
    extern __shared__ char smem[];
    half* s_fp16 = (half*)smem;  // FP16 激活缓存
    block_q8_1* s_q8_1 = (block_q8_1*)(smem + TILE_N * 128 * 32 * sizeof(half));  // Q8_1 缓存

    // Warp 信息
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
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
    constexpr int TILE_K = 128;
    const int num_k_tiles = (num_blocks_k + TILE_K - 1) / TILE_K;

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        const int k_start = k_tile * TILE_K;
        const int k_size = min(TILE_K, num_blocks_k - k_start);

        // 步骤 1: 协作加载 FP16 激活到 shared memory
        for (int c = 0; c < TILE_N; c++) {
            const int col = tile_n_start + c;
            if (col >= N) continue;

            const half* fp16_col = fp16_activation + col * K + k_start * 32;

            for (int b = threadIdx.x; b < k_size * 32; b += blockDim.x) {
                s_fp16[c * TILE_K * 32 + b] = fp16_col[b];
            }
        }
        __syncthreads();

        // 步骤 2: 协作量化为 Q8_1
        for (int c = 0; c < TILE_N; c++) {
            const int col = tile_n_start + c;
            if (col >= N) continue;

            for (int b = warp_id; b < k_size; b += (blockDim.x / WARP_SIZE)) {
                quantize_fp16_to_q8_1_smem(
                    s_fp16 + c * TILE_K * 32 + b * 32,
                    &s_q8_1[c * TILE_K + b],
                    lane_id
                );
            }
        }
        __syncthreads();

        // 步骤 3: GEMM 计算
        for (int k = lane_id; k < k_size; k += WARP_SIZE) {
            // 预加载激活值
            int u[TILE_N][8];
            float d8[TILE_N], s8[TILE_N];

            #pragma unroll
            for (int c = 0; c < TILE_N; c++) {
                const int col = tile_n_start + c;
                if (col < N) {
                    const block_q8_1* bq8 = &s_q8_1[c * TILE_K + k];

                    #pragma unroll
                    for (int i = 0; i < 8; i++) {
                        u[c][i] = ((const int*)bq8->qs)[i];
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
                    v[i] = load_int_b2_fused(bq4->qs, i);
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

                        sumi = dp4a_fused(vi0, u[c][i], sumi);
                        sumi = dp4a_fused(vi1, u[c][i + 4], sumi);
                    }

                    acc[r][c] += d4 * (d8[c] * sumi - 8.0f * s8[c]);
                }
            }
        }
        __syncthreads();
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

            float sum = warp_reduce_sum_fused(acc[r][c]);

            if (lane_id == 0) {
                output[row * N + col] = sum;
            }
        }
    }
}

// ============================================================================
// 接口函数
// ============================================================================

/**
 * 融合版 GEMM 接口
 */
inline void gemm_q4_0_fp16_fused(
    const block_q4_0* weight,
    const half* fp16_activation,
    float* output,
    int M, int N, int K,
    cudaStream_t stream = 0
) {
    constexpr int TILE_M = 64;
    constexpr int TILE_N = 4;
    constexpr int TILE_K = 128;
    constexpr int ROWS = 4;

    const int warps_per_block = TILE_M / ROWS;
    const int threads_per_block = warps_per_block * WARP_SIZE;

    // Shared memory: FP16 + Q8_1
    const size_t smem_fp16 = TILE_N * TILE_K * 32 * sizeof(half);
    const size_t smem_q8_1 = TILE_N * TILE_K * sizeof(block_q8_1);
    const size_t smem_size = smem_fp16 + smem_q8_1;

    dim3 block(threads_per_block);
    dim3 grid((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);

    gemm_q4_0_fp16_fused_kernel<TILE_M, TILE_N, ROWS>
        <<<grid, block, smem_size, stream>>>(
        weight, fp16_activation, output, M, N, K
    );
}

#endif // KERNELS_GEMM_FUSED_CUH
