/**
 * @file kernels/normalization/rms_norm.cuh
 * @brief RMS Normalization 实现
 *
 * RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
 *
 * LLaMA 模型使用 RMSNorm 代替 LayerNorm，
 * 因为它计算更简单且效果相当。
 */

#ifndef KERNELS_NORMALIZATION_RMS_NORM_CUH
#define KERNELS_NORMALIZATION_RMS_NORM_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

// ============================================================================
// CPU 参考实现
// ============================================================================

/**
 * RMS Normalization CPU 参考实现
 *
 * @param x      输入 [n_rows, n_cols]
 * @param weight 权重 [n_cols]
 * @param y      输出 [n_rows, n_cols]
 * @param n_rows 行数
 * @param n_cols 列数 (hidden_size)
 * @param eps    数值稳定性常数
 */
inline void rms_norm_cpu_f32(
    const float* x,
    const float* weight,
    float* y,
    int n_rows,
    int n_cols,
    float eps = 1e-5f
) {
    for (int row = 0; row < n_rows; row++) {
        const float* x_row = x + row * n_cols;
        float* y_row = y + row * n_cols;

        // 计算平方和
        double sum_sq = 0.0;
        for (int i = 0; i < n_cols; i++) {
            sum_sq += (double)x_row[i] * x_row[i];
        }

        // 计算 RMS
        float rms = sqrtf((float)(sum_sq / n_cols) + eps);
        float inv_rms = 1.0f / rms;

        // 归一化并应用权重
        for (int i = 0; i < n_cols; i++) {
            y_row[i] = x_row[i] * inv_rms * weight[i];
        }
    }
}

// ============================================================================
// GPU Kernel 实现
// ============================================================================

/**
 * Warp 级别归约求和
 */
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * Block 级别归约求和
 */
__device__ float block_reduce_sum(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;

    if (wid == 0) {
        val = warp_reduce_sum(val);
    }

    return val;
}

/**
 * RMS Norm Kernel - 基础版本
 *
 * 每个 block 处理一行
 */
__global__ void rms_norm_f32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ y,
    int n_cols,
    float eps
) {
    int row = blockIdx.x;
    const float* x_row = x + row * n_cols;
    float* y_row = y + row * n_cols;

    // 并行计算平方和
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < n_cols; i += blockDim.x) {
        float val = x_row[i];
        sum_sq += val * val;
    }

    // Block 归约
    sum_sq = block_reduce_sum(sum_sq);

    // 计算 RMS 的倒数
    __shared__ float inv_rms;
    if (threadIdx.x == 0) {
        float rms = sqrtf(sum_sq / n_cols + eps);
        inv_rms = 1.0f / rms;
    }
    __syncthreads();

    // 归一化并应用权重
    for (int i = threadIdx.x; i < n_cols; i += blockDim.x) {
        y_row[i] = x_row[i] * inv_rms * weight[i];
    }
}

/**
 * RMS Norm Kernel - 向量化版本
 *
 * 使用 float4 加载提高内存带宽利用率
 */
__global__ void rms_norm_f32_vec4_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ y,
    int n_cols,
    float eps
) {
    int row = blockIdx.x;
    const float* x_row = x + row * n_cols;
    float* y_row = y + row * n_cols;
    int n_cols_4 = n_cols / 4;

    const float4* x_vec = reinterpret_cast<const float4*>(x_row);
    const float4* w_vec = reinterpret_cast<const float4*>(weight);
    float4* y_vec = reinterpret_cast<float4*>(y_row);

    // 并行计算平方和
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < n_cols_4; i += blockDim.x) {
        float4 val = x_vec[i];
        sum_sq += val.x * val.x + val.y * val.y +
                  val.z * val.z + val.w * val.w;
    }

    // Block 归约
    sum_sq = block_reduce_sum(sum_sq);

    // 计算 RMS 的倒数
    __shared__ float inv_rms;
    if (threadIdx.x == 0) {
        float rms = sqrtf(sum_sq / n_cols + eps);
        inv_rms = 1.0f / rms;
    }
    __syncthreads();

    // 归一化并应用权重
    for (int i = threadIdx.x; i < n_cols_4; i += blockDim.x) {
        float4 val = x_vec[i];
        float4 w = w_vec[i];
        float4 result;
        result.x = val.x * inv_rms * w.x;
        result.y = val.y * inv_rms * w.y;
        result.z = val.z * inv_rms * w.z;
        result.w = val.w * inv_rms * w.w;
        y_vec[i] = result;
    }
}

/**
 * RMS Norm Kernel - FP16 输入 / FP32 输出
 *
 * 在推理时，输入可能是 FP16，但归一化使用 FP32 以保证精度
 */
__global__ void rms_norm_f16_f32_kernel(
    const half* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ y,
    int n_cols,
    float eps
) {
    int row = blockIdx.x;
    const half* x_row = x + row * n_cols;
    float* y_row = y + row * n_cols;

    // 并行计算平方和
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < n_cols; i += blockDim.x) {
        float val = __half2float(x_row[i]);
        sum_sq += val * val;
    }

    // Block 归约
    sum_sq = block_reduce_sum(sum_sq);

    // 计算 RMS 的倒数
    __shared__ float inv_rms;
    if (threadIdx.x == 0) {
        float rms = sqrtf(sum_sq / n_cols + eps);
        inv_rms = 1.0f / rms;
    }
    __syncthreads();

    // 归一化并应用权重
    for (int i = threadIdx.x; i < n_cols; i += blockDim.x) {
        float val = __half2float(x_row[i]);
        y_row[i] = val * inv_rms * weight[i];
    }
}

// ============================================================================
// 接口函数
// ============================================================================

/**
 * RMS Normalization 前向传播 (FP32)
 *
 * @param x      输入张量 [n_rows, n_cols]
 * @param weight 权重 [n_cols]
 * @param y      输出张量 [n_rows, n_cols]
 * @param n_rows 行数 (batch_size * seq_len)
 * @param n_cols 列数 (hidden_size)
 * @param eps    数值稳定性常数
 * @param stream CUDA 流
 */
inline void rms_norm_forward_f32(
    const float* x,
    const float* weight,
    float* y,
    int n_rows,
    int n_cols,
    float eps = 1e-5f,
    cudaStream_t stream = 0
) {
    // 每行一个 block，每个 block 256 线程
    int block_size = min(256, (n_cols + 31) / 32 * 32);
    int grid_size = n_rows;

    // 选择向量化版本或基础版本
    if (n_cols % 4 == 0 && n_cols >= 64) {
        rms_norm_f32_vec4_kernel<<<grid_size, block_size, 0, stream>>>(
            x, weight, y, n_cols, eps
        );
    } else {
        rms_norm_f32_kernel<<<grid_size, block_size, 0, stream>>>(
            x, weight, y, n_cols, eps
        );
    }
}

/**
 * RMS Normalization 前向传播 (FP16 -> FP32)
 */
inline void rms_norm_forward_f16_f32(
    const half* x,
    const float* weight,
    float* y,
    int n_rows,
    int n_cols,
    float eps = 1e-5f,
    cudaStream_t stream = 0
) {
    int block_size = min(256, (n_cols + 31) / 32 * 32);
    int grid_size = n_rows;

    rms_norm_f16_f32_kernel<<<grid_size, block_size, 0, stream>>>(
        x, weight, y, n_cols, eps
    );
}

#endif // KERNELS_NORMALIZATION_RMS_NORM_CUH
