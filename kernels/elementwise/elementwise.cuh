/**
 * @file kernels/elementwise/elementwise.cuh
 * @brief 元素级操作实现
 *
 * 包含:
 * - Add: 元素加法
 * - Mul: 元素乘法
 * - Scale: 标量乘法
 * - Add + Scale: 残差连接
 */

#ifndef KERNELS_ELEMENTWISE_ELEMENTWISE_CUH
#define KERNELS_ELEMENTWISE_ELEMENTWISE_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============================================================================
// CPU 参考实现
// ============================================================================

inline void add_cpu_f32(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

inline void mul_cpu_f32(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] * b[i];
    }
}

inline void scale_cpu_f32(const float* a, float s, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] * s;
    }
}

inline void add_scale_cpu_f32(const float* a, const float* b, float s, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i] * s;
    }
}

// ============================================================================
// GPU Kernel 实现
// ============================================================================

// --- 基础版本 ---

__global__ void add_f32_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void mul_f32_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

__global__ void scale_f32_kernel(
    const float* __restrict__ a,
    float s,
    float* __restrict__ c,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * s;
    }
}

/**
 * 残差连接: c = a + b * scale
 */
__global__ void add_scale_f32_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float s,
    float* __restrict__ c,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx] * s;
    }
}

// --- 向量化版本 ---

__global__ void add_f32_vec4_kernel(
    const float4* __restrict__ a,
    const float4* __restrict__ b,
    float4* __restrict__ c,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n4 = n / 4;
    if (idx < n4) {
        float4 va = a[idx];
        float4 vb = b[idx];
        float4 vc;
        vc.x = va.x + vb.x;
        vc.y = va.y + vb.y;
        vc.z = va.z + vb.z;
        vc.w = va.w + vb.w;
        c[idx] = vc;
    }
}

__global__ void mul_f32_vec4_kernel(
    const float4* __restrict__ a,
    const float4* __restrict__ b,
    float4* __restrict__ c,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n4 = n / 4;
    if (idx < n4) {
        float4 va = a[idx];
        float4 vb = b[idx];
        float4 vc;
        vc.x = va.x * vb.x;
        vc.y = va.y * vb.y;
        vc.z = va.z * vb.z;
        vc.w = va.w * vb.w;
        c[idx] = vc;
    }
}

__global__ void scale_f32_vec4_kernel(
    const float4* __restrict__ a,
    float s,
    float4* __restrict__ c,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n4 = n / 4;
    if (idx < n4) {
        float4 va = a[idx];
        float4 vc;
        vc.x = va.x * s;
        vc.y = va.y * s;
        vc.z = va.z * s;
        vc.w = va.w * s;
        c[idx] = vc;
    }
}

// --- 原地版本 ---

__global__ void add_inplace_f32_kernel(
    float* __restrict__ a,
    const float* __restrict__ b,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] += b[idx];
    }
}

__global__ void mul_inplace_f32_kernel(
    float* __restrict__ a,
    const float* __restrict__ b,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] *= b[idx];
    }
}

__global__ void scale_inplace_f32_kernel(
    float* __restrict__ a,
    float s,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] *= s;
    }
}

// --- FP16 版本 ---

__global__ void add_f16_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = __hadd(a[idx], b[idx]);
    }
}

__global__ void mul_f16_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = __hmul(a[idx], b[idx]);
    }
}

// ============================================================================
// 接口函数
// ============================================================================

inline void add_forward_f32(
    const float* a,
    const float* b,
    float* c,
    int n,
    cudaStream_t stream = 0
) {
    const int block_size = 256;

    if (n % 4 == 0 && n >= 256) {
        const int grid_size = ((n / 4) + block_size - 1) / block_size;
        add_f32_vec4_kernel<<<grid_size, block_size, 0, stream>>>(
            reinterpret_cast<const float4*>(a),
            reinterpret_cast<const float4*>(b),
            reinterpret_cast<float4*>(c),
            n
        );
    } else {
        const int grid_size = (n + block_size - 1) / block_size;
        add_f32_kernel<<<grid_size, block_size, 0, stream>>>(a, b, c, n);
    }
}

inline void mul_forward_f32(
    const float* a,
    const float* b,
    float* c,
    int n,
    cudaStream_t stream = 0
) {
    const int block_size = 256;

    if (n % 4 == 0 && n >= 256) {
        const int grid_size = ((n / 4) + block_size - 1) / block_size;
        mul_f32_vec4_kernel<<<grid_size, block_size, 0, stream>>>(
            reinterpret_cast<const float4*>(a),
            reinterpret_cast<const float4*>(b),
            reinterpret_cast<float4*>(c),
            n
        );
    } else {
        const int grid_size = (n + block_size - 1) / block_size;
        mul_f32_kernel<<<grid_size, block_size, 0, stream>>>(a, b, c, n);
    }
}

inline void scale_forward_f32(
    const float* a,
    float s,
    float* c,
    int n,
    cudaStream_t stream = 0
) {
    const int block_size = 256;

    if (n % 4 == 0 && n >= 256) {
        const int grid_size = ((n / 4) + block_size - 1) / block_size;
        scale_f32_vec4_kernel<<<grid_size, block_size, 0, stream>>>(
            reinterpret_cast<const float4*>(a),
            s,
            reinterpret_cast<float4*>(c),
            n
        );
    } else {
        const int grid_size = (n + block_size - 1) / block_size;
        scale_f32_kernel<<<grid_size, block_size, 0, stream>>>(a, s, c, n);
    }
}

inline void add_scale_forward_f32(
    const float* a,
    const float* b,
    float s,
    float* c,
    int n,
    cudaStream_t stream = 0
) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    add_scale_f32_kernel<<<grid_size, block_size, 0, stream>>>(a, b, s, c, n);
}

inline void add_inplace_f32(
    float* a,
    const float* b,
    int n,
    cudaStream_t stream = 0
) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    add_inplace_f32_kernel<<<grid_size, block_size, 0, stream>>>(a, b, n);
}

inline void scale_inplace_f32(
    float* a,
    float s,
    int n,
    cudaStream_t stream = 0
) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    scale_inplace_f32_kernel<<<grid_size, block_size, 0, stream>>>(a, s, n);
}

#endif // KERNELS_ELEMENTWISE_ELEMENTWISE_CUH
