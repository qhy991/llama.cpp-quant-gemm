/**
 * @file kernels/activation/silu.cuh
 * @brief SiLU (Swish) 激活函数实现
 *
 * SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
 *
 * 这是 LLaMA 模型中使用的主要激活函数。
 */

#ifndef KERNELS_ACTIVATION_SILU_CUH
#define KERNELS_ACTIVATION_SILU_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============================================================================
// CPU 参考实现
// ============================================================================

/**
 * SiLU CPU 参考实现 (FP32)
 */
inline void silu_cpu_f32(const float* x, float* y, int n) {
    for (int i = 0; i < n; i++) {
        y[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

// ============================================================================
// GPU Kernel 实现
// ============================================================================

/**
 * SiLU CUDA Kernel (FP32)
 *
 * 每个线程处理一个元素
 */
__global__ void silu_f32_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        y[idx] = val / (1.0f + expf(-val));
    }
}

/**
 * SiLU CUDA Kernel (FP32) - 向量化版本
 *
 * 每个线程处理 4 个元素
 */
__global__ void silu_f32_vec4_kernel(
    const float4* __restrict__ x,
    float4* __restrict__ y,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n4 = n / 4;

    if (idx < n4) {
        float4 val = x[idx];
        float4 result;
        result.x = val.x / (1.0f + expf(-val.x));
        result.y = val.y / (1.0f + expf(-val.y));
        result.z = val.z / (1.0f + expf(-val.z));
        result.w = val.w / (1.0f + expf(-val.w));
        y[idx] = result;
    }
}

/**
 * SiLU CUDA Kernel (FP16)
 *
 * 使用 half2 处理 FP16 数据
 */
__global__ void silu_f16_kernel(
    const half* __restrict__ x,
    half* __restrict__ y,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = __half2float(x[idx]);
        y[idx] = __float2half(val / (1.0f + expf(-val)));
    }
}

/**
 * SiLU + 元素乘法融合 Kernel
 *
 * 用于 LLaMA FFN: silu(x) * gate
 * 这是 llama.cpp 中 ggml_silu_inplace 的等效实现
 */
__global__ void silu_mul_f32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gate,
    float* __restrict__ y,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        float silu = val / (1.0f + expf(-val));
        y[idx] = silu * gate[idx];
    }
}

// ============================================================================
// 接口函数
// ============================================================================

/**
 * SiLU 前向传播 (FP32)
 *
 * @param x     输入张量
 * @param y     输出张量
 * @param n     元素数量
 * @param stream CUDA 流
 */
inline void silu_forward_f32(
    const float* x,
    float* y,
    int n,
    cudaStream_t stream = 0
) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    silu_f32_kernel<<<grid_size, block_size, 0, stream>>>(x, y, n);
}

/**
 * SiLU 前向传播 - 向量化版本 (FP32)
 *
 * 要求 n 是 4 的倍数且数据对齐
 */
inline void silu_forward_f32_vec4(
    const float* x,
    float* y,
    int n,
    cudaStream_t stream = 0
) {
    const int block_size = 256;
    const int n4 = n / 4;
    const int grid_size = (n4 + block_size - 1) / block_size;

    silu_f32_vec4_kernel<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<const float4*>(x),
        reinterpret_cast<float4*>(y),
        n
    );
}

/**
 * SiLU + 乘法融合 (FP32)
 *
 * 计算: y = silu(x) * gate
 */
inline void silu_mul_forward_f32(
    const float* x,
    const float* gate,
    float* y,
    int n,
    cudaStream_t stream = 0
) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    silu_mul_f32_kernel<<<grid_size, block_size, 0, stream>>>(x, gate, y, n);
}

#endif // KERNELS_ACTIVATION_SILU_CUH
