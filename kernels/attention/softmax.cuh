/**
 * @file kernels/attention/softmax.cuh
 * @brief Softmax 实现
 *
 * Softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))
 *
 * 在注意力机制中用于计算注意力权重。
 */

#ifndef KERNELS_ATTENTION_SOFTMAX_CUH
#define KERNELS_ATTENTION_SOFTMAX_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>
#include <math.h>

// ============================================================================
// CPU 参考实现
// ============================================================================

/**
 * Softmax CPU 参考实现
 *
 * @param x      输入 [n_rows, n_cols]
 * @param y      输出 [n_rows, n_cols]
 * @param n_rows 行数
 * @param n_cols 列数 (softmax 维度)
 * @param scale  可选缩放因子 (通常是 1/√d_k)
 */
inline void softmax_cpu_f32(
    const float* x,
    float* y,
    int n_rows,
    int n_cols,
    float scale = 1.0f
) {
    for (int row = 0; row < n_rows; row++) {
        const float* x_row = x + row * n_cols;
        float* y_row = y + row * n_cols;

        // 1. 找最大值 (数值稳定性)
        float max_val = -FLT_MAX;
        for (int i = 0; i < n_cols; i++) {
            float v = x_row[i] * scale;
            if (v > max_val) max_val = v;
        }

        // 2. 计算 exp 并求和
        float sum = 0.0f;
        for (int i = 0; i < n_cols; i++) {
            y_row[i] = expf(x_row[i] * scale - max_val);
            sum += y_row[i];
        }

        // 3. 归一化
        float inv_sum = 1.0f / sum;
        for (int i = 0; i < n_cols; i++) {
            y_row[i] *= inv_sum;
        }
    }
}

// ============================================================================
// GPU Kernel 辅助函数
// ============================================================================

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ float block_reduce_max(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_max(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : -FLT_MAX;

    if (wid == 0) val = warp_reduce_max(val);

    return val;
}

__device__ float block_reduce_sum_softmax(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_sum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;

    if (wid == 0) val = warp_reduce_sum(val);

    return val;
}

// ============================================================================
// GPU Kernel 实现
// ============================================================================

/**
 * Softmax Kernel - 每行一个 block
 *
 * 适用于序列长度较长的情况
 */
__global__ void softmax_f32_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int n_cols,
    float scale
) {
    int row = blockIdx.x;
    const float* x_row = x + row * n_cols;
    float* y_row = y + row * n_cols;

    // 1. 并行找最大值
    float max_val = -FLT_MAX;
    for (int i = threadIdx.x; i < n_cols; i += blockDim.x) {
        max_val = fmaxf(max_val, x_row[i] * scale);
    }
    max_val = block_reduce_max(max_val);

    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = max_val;
    __syncthreads();
    max_val = s_max;

    // 2. 并行计算 exp 并求和
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n_cols; i += blockDim.x) {
        float v = expf(x_row[i] * scale - max_val);
        y_row[i] = v;
        sum += v;
    }
    sum = block_reduce_sum_softmax(sum);

    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = sum;
    __syncthreads();
    sum = s_sum;

    // 3. 并行归一化
    float inv_sum = 1.0f / sum;
    for (int i = threadIdx.x; i < n_cols; i += blockDim.x) {
        y_row[i] *= inv_sum;
    }
}

/**
 * Softmax Kernel - 小序列优化 (单 warp)
 *
 * 适用于 n_cols <= 32 的情况
 */
__global__ void softmax_f32_small_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int n_rows,
    int n_cols,
    float scale
) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= n_rows) return;

    int lane = threadIdx.x;
    const float* x_row = x + row * n_cols;
    float* y_row = y + row * n_cols;

    // 每个线程处理一个元素
    float val = (lane < n_cols) ? x_row[lane] * scale : -FLT_MAX;

    // Warp 归约找最大值
    float max_val = warp_reduce_max(val);

    // 计算 exp
    float exp_val = (lane < n_cols) ? expf(val - max_val) : 0.0f;

    // Warp 归约求和
    float sum = warp_reduce_sum(exp_val);

    // 写出结果
    if (lane < n_cols) {
        y_row[lane] = exp_val / sum;
    }
}

/**
 * Causal Softmax Kernel (带掩码)
 *
 * 用于自回归模型，mask 掉未来 token
 */
__global__ void softmax_causal_f32_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int n_cols,
    int pos,      // 当前位置
    float scale
) {
    int row = blockIdx.x;
    const float* x_row = x + row * n_cols;
    float* y_row = y + row * n_cols;

    // 计算有效长度 (causal mask)
    int valid_len = min(pos + 1, n_cols);

    // 1. 并行找最大值 (只考虑有效位置)
    float max_val = -FLT_MAX;
    for (int i = threadIdx.x; i < valid_len; i += blockDim.x) {
        max_val = fmaxf(max_val, x_row[i] * scale);
    }
    max_val = block_reduce_max(max_val);

    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = max_val;
    __syncthreads();
    max_val = s_max;

    // 2. 计算 exp 并求和
    float sum = 0.0f;
    for (int i = threadIdx.x; i < valid_len; i += blockDim.x) {
        float v = expf(x_row[i] * scale - max_val);
        y_row[i] = v;
        sum += v;
    }
    sum = block_reduce_sum_softmax(sum);

    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = sum;
    __syncthreads();
    sum = s_sum;

    // 3. 归一化有效位置，mask 位置置零
    float inv_sum = 1.0f / sum;
    for (int i = threadIdx.x; i < n_cols; i += blockDim.x) {
        if (i < valid_len) {
            y_row[i] *= inv_sum;
        } else {
            y_row[i] = 0.0f;
        }
    }
}

// ============================================================================
// 接口函数
// ============================================================================

/**
 * Softmax 前向传播
 *
 * @param x      输入 [n_rows, n_cols]
 * @param y      输出 [n_rows, n_cols]
 * @param n_rows 行数
 * @param n_cols 列数 (softmax 维度)
 * @param scale  缩放因子 (默认 1.0)
 * @param stream CUDA 流
 */
inline void softmax_forward_f32(
    const float* x,
    float* y,
    int n_rows,
    int n_cols,
    float scale = 1.0f,
    cudaStream_t stream = 0
) {
    if (n_cols <= 32) {
        // 小序列：多行并行
        dim3 block(32, 8);
        dim3 grid((n_rows + 7) / 8);
        softmax_f32_small_kernel<<<grid, block, 0, stream>>>(
            x, y, n_rows, n_cols, scale
        );
    } else {
        // 大序列：每行一个 block
        int block_size = min(256, (n_cols + 31) / 32 * 32);
        softmax_f32_kernel<<<n_rows, block_size, 0, stream>>>(
            x, y, n_cols, scale
        );
    }
}

/**
 * Causal Softmax 前向传播 (带掩码)
 */
inline void softmax_causal_forward_f32(
    const float* x,
    float* y,
    int n_rows,
    int n_cols,
    int pos,
    float scale = 1.0f,
    cudaStream_t stream = 0
) {
    int block_size = min(256, (n_cols + 31) / 32 * 32);
    softmax_causal_f32_kernel<<<n_rows, block_size, 0, stream>>>(
        x, y, n_cols, pos, scale
    );
}

#endif // KERNELS_ATTENTION_SOFTMAX_CUH
