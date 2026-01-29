/**
 * @file kernels/activation/gelu.cuh
 * @brief GELU 激活函数实现
 *
 * GELU(x) = x * Φ(x) = x * 0.5 * (1 + erf(x/√2))
 *
 * 快速近似: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
 *
 * 用于 GPT 系列模型，某些 LLaMA 变体也使用。
 */

#ifndef KERNELS_ACTIVATION_GELU_CUH
#define KERNELS_ACTIVATION_GELU_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

// 常量
#define GELU_COEF_A     0.044715f
#define SQRT_2_OVER_PI  0.7978845608028654f  // sqrt(2/π)

// ============================================================================
// CPU 参考实现
// ============================================================================

/**
 * GELU 精确实现 (使用 erf)
 */
inline void gelu_cpu_f32(const float* x, float* y, int n) {
    for (int i = 0; i < n; i++) {
        y[i] = 0.5f * x[i] * (1.0f + erff(x[i] / sqrtf(2.0f)));
    }
}

/**
 * GELU 快速近似实现 (tanh 近似)
 * 这是 llama.cpp 中使用的版本
 */
inline void gelu_quick_cpu_f32(const float* x, float* y, int n) {
    for (int i = 0; i < n; i++) {
        float v = x[i];
        y[i] = 0.5f * v * (1.0f + tanhf(SQRT_2_OVER_PI * v * (1.0f + GELU_COEF_A * v * v)));
    }
}

// ============================================================================
// GPU Kernel 实现
// ============================================================================

/**
 * GELU 精确 Kernel
 */
__global__ void gelu_f32_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = x[idx];
        y[idx] = 0.5f * v * (1.0f + erff(v * 0.7071067811865476f));  // 1/√2
    }
}

/**
 * GELU 快速近似 Kernel (llama.cpp 兼容)
 */
__global__ void gelu_quick_f32_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = x[idx];
        y[idx] = 0.5f * v * (1.0f + tanhf(SQRT_2_OVER_PI * v * (1.0f + GELU_COEF_A * v * v)));
    }
}

/**
 * GELU 向量化 Kernel
 */
__global__ void gelu_quick_f32_vec4_kernel(
    const float4* __restrict__ x,
    float4* __restrict__ y,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n4 = n / 4;

    if (idx < n4) {
        float4 val = x[idx];
        float4 result;

        #define GELU_QUICK(v) (0.5f * (v) * (1.0f + tanhf(SQRT_2_OVER_PI * (v) * (1.0f + GELU_COEF_A * (v) * (v)))))

        result.x = GELU_QUICK(val.x);
        result.y = GELU_QUICK(val.y);
        result.z = GELU_QUICK(val.z);
        result.w = GELU_QUICK(val.w);

        #undef GELU_QUICK

        y[idx] = result;
    }
}

// ============================================================================
// 接口函数
// ============================================================================

inline void gelu_forward_f32(
    const float* x,
    float* y,
    int n,
    cudaStream_t stream = 0
) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    gelu_f32_kernel<<<grid_size, block_size, 0, stream>>>(x, y, n);
}

inline void gelu_quick_forward_f32(
    const float* x,
    float* y,
    int n,
    cudaStream_t stream = 0
) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    gelu_quick_f32_kernel<<<grid_size, block_size, 0, stream>>>(x, y, n);
}

inline void gelu_quick_forward_f32_vec4(
    const float* x,
    float* y,
    int n,
    cudaStream_t stream = 0
) {
    const int block_size = 256;
    const int n4 = n / 4;
    const int grid_size = (n4 + block_size - 1) / block_size;
    gelu_quick_f32_vec4_kernel<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<const float4*>(x),
        reinterpret_cast<float4*>(y),
        n
    );
}

#endif // KERNELS_ACTIVATION_GELU_CUH
