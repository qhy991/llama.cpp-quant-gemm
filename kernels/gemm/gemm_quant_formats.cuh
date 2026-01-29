/**
 * @file kernels/gemm/gemm_quant_formats.cuh
 * @brief 多种量化格式的 GEMM 实现
 *
 * 支持的格式:
 * - Q4_0 × Q8_1: 4-bit 对称量化权重 × 8-bit 激活
 * - Q4_1 × Q8_1: 4-bit 非对称量化权重 × 8-bit 激活
 * - Q5_0 × Q8_1: 5-bit 对称量化权重 × 8-bit 激活
 * - Q5_1 × Q8_1: 5-bit 非对称量化权重 × 8-bit 激活
 * - Q8_0 × Q8_1: 8-bit 对称量化权重 × 8-bit 激活
 *
 * 所有实现与 llama.cpp 的 vecdotq.cuh 算法一致
 */

#ifndef KERNELS_GEMM_QUANT_FORMATS_CUH
#define KERNELS_GEMM_QUANT_FORMATS_CUH

#include "../../compat/ggml_types.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============================================================================
// 常量定义
// ============================================================================

#define QK4_0 32
#define QK4_1 32
#define QK5_0 32
#define QK5_1 32
#define QK8_0 32
#define QK8_1 32

// ============================================================================
// DP4A 辅助函数
// ============================================================================

__device__ __forceinline__ int dp4a(int a, int b, int c) {
#if __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    char4 va = *reinterpret_cast<char4*>(&a);
    char4 vb = *reinterpret_cast<char4*>(&b);
    return c + va.x*vb.x + va.y*vb.y + va.z*vb.z + va.w*vb.w;
#endif
}

// 从 2 字节对齐内存加载 int32
__device__ __forceinline__ int load_int_b2(const void* x, int i32) {
    const uint16_t* x16 = (const uint16_t*)x;
    int x32  = x16[2*i32 + 0] <<  0;
    x32     |= x16[2*i32 + 1] << 16;
    return x32;
}

// 从 4 字节对齐内存加载 int32
__device__ __forceinline__ int load_int_b4(const void* x, int i32) {
    return ((const int*)x)[i32];
}

// ============================================================================
// Q4_0 × Q8_1 点积实现 (与 llama.cpp vec_dot_q4_0_q8_1_impl 一致)
// ============================================================================
/**
 * Q4_0 格式:
 *   - d: half 缩放因子
 *   - qs[16]: 32 个 4-bit 值打包 (每字节 2 个), 范围 [0,15], 偏移 8
 *
 * 反量化: x = (q - 8) * d
 *
 * 点积公式:
 *   result = d_w * (d_a * sumi - 8 * s_a)
 */
__device__ __forceinline__ float vec_dot_q4_0_q8_1(
    const block_q4_0* __restrict__ bq4,
    const block_q8_1* __restrict__ bq8
) {
    int sumi = 0;

    // 加载 Q4_0 量化值 (4 个 int32 = 32 个 4-bit 值)
    // Q4_0 布局: qs[i] 的低 nibble = x[i], 高 nibble = x[i+16]
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int v = load_int_b2(bq4->qs, i);
        int vi0 = (v >> 0) & 0x0F0F0F0F;  // 低 nibbles: x[i*4+0:3]
        int vi1 = (v >> 4) & 0x0F0F0F0F;  // 高 nibbles: x[i*4+16:19]

        // 加载 Q8_1 量化值
        int u0 = load_int_b4(bq8->qs, i);      // x[i*4+0:3]
        int u1 = load_int_b4(bq8->qs, i + 4);  // x[i*4+16:19]

        // DP4A 点积
        sumi = dp4a(vi0, u0, sumi);
        sumi = dp4a(vi1, u1, sumi);
    }

    // 获取缩放因子
    float d4 = __half2float(bq4->d);
    float d8 = __half2float(__low2half(bq8->ds));
    float s8 = __half2float(__high2half(bq8->ds));

    // 补偿公式: d4 * (d8 * sumi - 8 * s8)
    return d4 * (d8 * sumi - 8.0f * s8);
}

// ============================================================================
// Q4_1 × Q8_1 点积实现 (与 llama.cpp vec_dot_q4_1_q8_1_impl 一致)
// ============================================================================
/**
 * Q4_1 格式 (非对称量化):
 *   - d: half 缩放因子
 *   - m: half 最小值 (偏移)
 *   - qs[16]: 32 个 4-bit 值打包, 范围 [0,15]
 *
 * 反量化: x = q * d + m
 *
 * 点积公式:
 *   result = d4*d8 * sumi + m4*s8
 */
__device__ __forceinline__ float vec_dot_q4_1_q8_1(
    const block_q4_1* __restrict__ bq4,
    const block_q8_1* __restrict__ bq8
) {
    int sumi = 0;

    // Q4_1 布局: qs[i] 的低 nibble = x[i], 高 nibble = x[i+16]
    // Q8_1 布局: qs[i] = x[i] (顺序存储)
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int v = load_int_b2(bq4->qs, i);
        int vi0 = (v >> 0) & 0x0F0F0F0F;  // x[i*4+0:3]
        int vi1 = (v >> 4) & 0x0F0F0F0F;  // x[i*4+16:19]

        int u0 = load_int_b4(bq8->qs, i);      // x[i*4+0:3]
        int u1 = load_int_b4(bq8->qs, i + 4);  // x[i*4+16:19]

        sumi = dp4a(vi0, u0, sumi);
        sumi = dp4a(vi1, u1, sumi);
    }

    // Q4_1 有两个参数: d (缩放) 和 m (偏移)
    float d4 = __half2float(bq4->d);
    float m4 = __half2float(bq4->m);
    float d8 = __half2float(__low2half(bq8->ds));
    float s8 = __half2float(__high2half(bq8->ds));

    // 公式: d4*d8*sumi + m4*s8
    // 注意: m4*s8 需要除以 (QI8_1 / (vdr * QR4_1)) = 32/(4*2) = 4
    return d4 * d8 * sumi + m4 * s8 / 4.0f;
}

// ============================================================================
// Q5_0 × Q8_1 点积实现 (与 llama.cpp vec_dot_q5_0_q8_1_impl 一致)
// ============================================================================
/**
 * Q5_0 格式:
 *   - d: half 缩放因子
 *   - qh[4]: 第 5 bit (32 个值的高位)
 *   - qs[16]: 低 4 bits
 *
 * 反量化: x = (q - 16) * d, 其中 q 是 5-bit 值 [0,31]
 */
__device__ __forceinline__ float vec_dot_q5_0_q8_1(
    const block_q5_0* __restrict__ bq5,
    const block_q8_1* __restrict__ bq8
) {
    int sumi = 0;

    // qh 包含 32 个第 5 bit
    uint32_t qh;
    memcpy(&qh, bq5->qh, sizeof(qh));

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int vl = load_int_b2(bq5->qs, i);

        // 提取低 4 bits
        int vi0 = (vl >> 0) & 0x0F0F0F0F;
        int vi1 = (vl >> 4) & 0x0F0F0F0F;

        // 添加第 5 bit (从 qh 中提取)
        // qh 布局: bits 0-15 对应 x[0..15], bits 16-31 对应 x[16..31]
        int bit_offset_0 = i * 4;      // vi0 对应 x[i*4..i*4+3]
        int bit_offset_1 = 16 + i * 4; // vi1 对应 x[16+i*4..16+i*4+3]

        vi0 |= ((qh >> (bit_offset_0 + 0)) & 1) << 4;
        vi0 |= ((qh >> (bit_offset_0 + 1)) & 1) << 12;
        vi0 |= ((qh >> (bit_offset_0 + 2)) & 1) << 20;
        vi0 |= ((qh >> (bit_offset_0 + 3)) & 1) << 28;

        vi1 |= ((qh >> (bit_offset_1 + 0)) & 1) << 4;
        vi1 |= ((qh >> (bit_offset_1 + 1)) & 1) << 12;
        vi1 |= ((qh >> (bit_offset_1 + 2)) & 1) << 20;
        vi1 |= ((qh >> (bit_offset_1 + 3)) & 1) << 28;

        int u0 = load_int_b4(bq8->qs, i);      // x[i*4+0:3]
        int u1 = load_int_b4(bq8->qs, i + 4);  // x[i*4+16:19]

        sumi = dp4a(vi0, u0, sumi);
        sumi = dp4a(vi1, u1, sumi);
    }

    float d5 = __half2float(bq5->d);
    float d8 = __half2float(__low2half(bq8->ds));
    float s8 = __half2float(__high2half(bq8->ds));

    // 补偿公式: d5 * (d8 * sumi - 16 * s8)
    return d5 * (d8 * sumi - 16.0f * s8);
}

// ============================================================================
// Q5_1 × Q8_1 点积实现 (与 llama.cpp vec_dot_q5_1_q8_1_impl 一致)
// ============================================================================
/**
 * Q5_1 格式 (非对称):
 *   - d: half 缩放因子
 *   - m: half 最小值
 *   - qh[4]: 第 5 bit
 *   - qs[16]: 低 4 bits
 *
 * 反量化: x = q * d + m
 */
__device__ __forceinline__ float vec_dot_q5_1_q8_1(
    const block_q5_1* __restrict__ bq5,
    const block_q8_1* __restrict__ bq8
) {
    int sumi = 0;

    uint32_t qh;
    memcpy(&qh, bq5->qh, sizeof(qh));

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int vl = load_int_b2(bq5->qs, i);

        int vi0 = (vl >> 0) & 0x0F0F0F0F;
        int vi1 = (vl >> 4) & 0x0F0F0F0F;

        // 添加第 5 bit (从 qh 中提取)
        // qh 布局: bits 0-15 对应 x[0..15], bits 16-31 对应 x[16..31]
        int bit_offset_0 = i * 4;      // vi0 对应 x[i*4..i*4+3]
        int bit_offset_1 = 16 + i * 4; // vi1 对应 x[16+i*4..16+i*4+3]

        vi0 |= ((qh >> (bit_offset_0 + 0)) & 1) << 4;
        vi0 |= ((qh >> (bit_offset_0 + 1)) & 1) << 12;
        vi0 |= ((qh >> (bit_offset_0 + 2)) & 1) << 20;
        vi0 |= ((qh >> (bit_offset_0 + 3)) & 1) << 28;

        vi1 |= ((qh >> (bit_offset_1 + 0)) & 1) << 4;
        vi1 |= ((qh >> (bit_offset_1 + 1)) & 1) << 12;
        vi1 |= ((qh >> (bit_offset_1 + 2)) & 1) << 20;
        vi1 |= ((qh >> (bit_offset_1 + 3)) & 1) << 28;

        int u0 = load_int_b4(bq8->qs, i);      // x[i*4+0:3]
        int u1 = load_int_b4(bq8->qs, i + 4);  // x[i*4+16:19]

        sumi = dp4a(vi0, u0, sumi);
        sumi = dp4a(vi1, u1, sumi);
    }

    float d5 = __half2float(bq5->d);
    float m5 = __half2float(bq5->m);
    float d8 = __half2float(__low2half(bq8->ds));
    float s8 = __half2float(__high2half(bq8->ds));

    // 公式: d5*d8*sumi + m5*s8/4
    return d5 * d8 * sumi + m5 * s8 / 4.0f;
}

// ============================================================================
// Q8_0 × Q8_1 点积实现 (与 llama.cpp vec_dot_q8_0_q8_1_impl 一致)
// ============================================================================
/**
 * Q8_0 格式:
 *   - d: half 缩放因子
 *   - qs[32]: 32 个 int8 量化值
 *
 * 反量化: x = q * d
 */
__device__ __forceinline__ float vec_dot_q8_0_q8_1(
    const block_q8_0* __restrict__ bq8_0,
    const block_q8_1* __restrict__ bq8_1
) {
    int sumi = 0;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int v = load_int_b4(bq8_0->qs, i);
        int u = load_int_b4(bq8_1->qs, i);
        sumi = dp4a(v, u, sumi);
    }

    float d0 = __half2float(bq8_0->d);
    float d1 = __half2float(__low2half(bq8_1->ds));

    return d0 * d1 * sumi;
}

// ============================================================================
// GEMM Kernel 模板
// ============================================================================

/**
 * 通用量化 GEMM Kernel
 *
 * C[M,N] = A[M,K] × B[N,K]^T
 *
 * 模板参数:
 * - BlockW: 权重块类型 (block_q4_0, block_q4_1, etc.)
 * - BlockA: 激活块类型 (block_q8_1)
 * - dot_fn: 点积函数
 */
template<typename BlockW, typename BlockA,
         float (*dot_fn)(const BlockW*, const BlockA*)>
__global__ void gemm_quant_kernel(
    const BlockW* __restrict__ weight,
    const BlockA* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;

    if (m >= M || n >= N) return;

    const int num_blocks = K / 32;  // 每个块 32 个元素
    float sum = 0.0f;

    for (int b = 0; b < num_blocks; b++) {
        sum += dot_fn(&weight[m * num_blocks + b],
                      &activation[n * num_blocks + b]);
    }

    output[m * N + n] = sum;
}

// ============================================================================
// GEMM 接口函数
// ============================================================================

/**
 * Q4_0 × Q8_1 GEMM
 */
inline void gemm_q4_0_q8_1(
    const block_q4_0* weight,
    const block_q8_1* activation,
    float* output,
    int M, int N, int K,
    cudaStream_t stream = 0
) {
    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x,
              (N + block.y - 1) / block.y);

    gemm_quant_kernel<block_q4_0, block_q8_1, vec_dot_q4_0_q8_1>
        <<<grid, block, 0, stream>>>(weight, activation, output, M, N, K);
}

/**
 * Q4_1 × Q8_1 GEMM
 */
inline void gemm_q4_1_q8_1(
    const block_q4_1* weight,
    const block_q8_1* activation,
    float* output,
    int M, int N, int K,
    cudaStream_t stream = 0
) {
    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x,
              (N + block.y - 1) / block.y);

    gemm_quant_kernel<block_q4_1, block_q8_1, vec_dot_q4_1_q8_1>
        <<<grid, block, 0, stream>>>(weight, activation, output, M, N, K);
}

/**
 * Q5_0 × Q8_1 GEMM
 */
inline void gemm_q5_0_q8_1(
    const block_q5_0* weight,
    const block_q8_1* activation,
    float* output,
    int M, int N, int K,
    cudaStream_t stream = 0
) {
    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x,
              (N + block.y - 1) / block.y);

    gemm_quant_kernel<block_q5_0, block_q8_1, vec_dot_q5_0_q8_1>
        <<<grid, block, 0, stream>>>(weight, activation, output, M, N, K);
}

/**
 * Q5_1 × Q8_1 GEMM
 */
inline void gemm_q5_1_q8_1(
    const block_q5_1* weight,
    const block_q8_1* activation,
    float* output,
    int M, int N, int K,
    cudaStream_t stream = 0
) {
    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x,
              (N + block.y - 1) / block.y);

    gemm_quant_kernel<block_q5_1, block_q8_1, vec_dot_q5_1_q8_1>
        <<<grid, block, 0, stream>>>(weight, activation, output, M, N, K);
}

/**
 * Q8_0 × Q8_1 GEMM
 */
inline void gemm_q8_0_q8_1(
    const block_q8_0* weight,
    const block_q8_1* activation,
    float* output,
    int M, int N, int K,
    cudaStream_t stream = 0
) {
    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x,
              (N + block.y - 1) / block.y);

    gemm_quant_kernel<block_q8_0, block_q8_1, vec_dot_q8_0_q8_1>
        <<<grid, block, 0, stream>>>(weight, activation, output, M, N, K);
}

#endif // KERNELS_GEMM_QUANT_FORMATS_CUH
