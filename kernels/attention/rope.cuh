/**
 * @file kernels/attention/rope.cuh
 * @brief RoPE (Rotary Position Embedding) 实现
 *
 * RoPE 通过旋转对位置信息进行编码:
 *   q' = q * cos(θ) + rotate(q) * sin(θ)
 *
 * 其中 θ = pos * freq, freq = 1 / (base^(2i/d))
 *
 * 默认 base = 10000 (LLaMA)
 *
 * 参考: https://arxiv.org/abs/2104.09864
 */

#ifndef KERNELS_ATTENTION_ROPE_CUH
#define KERNELS_ATTENTION_ROPE_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

// 默认参数
#define ROPE_DEFAULT_BASE     10000.0f
#define ROPE_DEFAULT_FREQ_MAX 1.0f

// ============================================================================
// CPU 参考实现
// ============================================================================

/**
 * 计算 RoPE 频率
 */
inline void rope_compute_freqs(
    float* freqs_cos,
    float* freqs_sin,
    int dim,
    int pos,
    float base = ROPE_DEFAULT_BASE,
    float freq_scale = 1.0f
) {
    for (int i = 0; i < dim / 2; i++) {
        float freq = 1.0f / powf(base, (float)(2 * i) / dim);
        float theta = pos * freq * freq_scale;
        freqs_cos[i] = cosf(theta);
        freqs_sin[i] = sinf(theta);
    }
}

/**
 * RoPE 前向 CPU 参考实现
 *
 * @param x         输入/输出 [n_heads, head_dim]
 * @param n_heads   头数
 * @param head_dim  每头维度
 * @param pos       当前位置
 * @param base      RoPE 基数 (默认 10000)
 * @param freq_scale 频率缩放因子 (默认 1.0)
 */
inline void rope_cpu_f32(
    float* x,
    int n_heads,
    int head_dim,
    int pos,
    float base = ROPE_DEFAULT_BASE,
    float freq_scale = 1.0f
) {
    for (int h = 0; h < n_heads; h++) {
        float* head = x + h * head_dim;

        for (int i = 0; i < head_dim / 2; i++) {
            float freq = 1.0f / powf(base, (float)(2 * i) / head_dim);
            float theta = pos * freq * freq_scale;
            float cos_theta = cosf(theta);
            float sin_theta = sinf(theta);

            float x0 = head[i];
            float x1 = head[i + head_dim / 2];

            head[i] = x0 * cos_theta - x1 * sin_theta;
            head[i + head_dim / 2] = x0 * sin_theta + x1 * cos_theta;
        }
    }
}

/**
 * RoPE (交错布局) CPU 参考实现
 *
 * 某些模型使用交错布局: [x0, x1, x2, x3, ...] -> [(x0,x1), (x2,x3), ...]
 */
inline void rope_interleaved_cpu_f32(
    float* x,
    int n_heads,
    int head_dim,
    int pos,
    float base = ROPE_DEFAULT_BASE,
    float freq_scale = 1.0f
) {
    for (int h = 0; h < n_heads; h++) {
        float* head = x + h * head_dim;

        for (int i = 0; i < head_dim / 2; i++) {
            float freq = 1.0f / powf(base, (float)(2 * i) / head_dim);
            float theta = pos * freq * freq_scale;
            float cos_theta = cosf(theta);
            float sin_theta = sinf(theta);

            float x0 = head[2 * i];
            float x1 = head[2 * i + 1];

            head[2 * i] = x0 * cos_theta - x1 * sin_theta;
            head[2 * i + 1] = x0 * sin_theta + x1 * cos_theta;
        }
    }
}

// ============================================================================
// GPU Kernel 实现
// ============================================================================

/**
 * RoPE Kernel - 标准布局
 *
 * 输入布局: [batch, seq, n_heads, head_dim]
 * 处理单个 token 位置
 */
__global__ void rope_f32_kernel(
    float* __restrict__ x,
    int n_heads,
    int head_dim,
    int pos,
    float base,
    float freq_scale
) {
    int h = blockIdx.x;  // head index
    int i = threadIdx.x; // position in half of head_dim

    if (h >= n_heads || i >= head_dim / 2) return;

    float* head = x + h * head_dim;

    // 计算频率和角度
    float freq = 1.0f / powf(base, (float)(2 * i) / head_dim);
    float theta = pos * freq * freq_scale;
    float cos_theta = cosf(theta);
    float sin_theta = sinf(theta);

    // 加载
    float x0 = head[i];
    float x1 = head[i + head_dim / 2];

    // 旋转
    head[i] = x0 * cos_theta - x1 * sin_theta;
    head[i + head_dim / 2] = x0 * sin_theta + x1 * cos_theta;
}

/**
 * RoPE Kernel - 交错布局
 */
__global__ void rope_interleaved_f32_kernel(
    float* __restrict__ x,
    int n_heads,
    int head_dim,
    int pos,
    float base,
    float freq_scale
) {
    int h = blockIdx.x;
    int i = threadIdx.x;

    if (h >= n_heads || i >= head_dim / 2) return;

    float* head = x + h * head_dim;

    float freq = 1.0f / powf(base, (float)(2 * i) / head_dim);
    float theta = pos * freq * freq_scale;
    float cos_theta = cosf(theta);
    float sin_theta = sinf(theta);

    float x0 = head[2 * i];
    float x1 = head[2 * i + 1];

    head[2 * i] = x0 * cos_theta - x1 * sin_theta;
    head[2 * i + 1] = x0 * sin_theta + x1 * cos_theta;
}

/**
 * RoPE Kernel - 批量处理多个位置
 *
 * 处理整个序列
 */
__global__ void rope_batch_f32_kernel(
    float* __restrict__ x,
    int seq_len,
    int n_heads,
    int head_dim,
    int start_pos,  // 起始位置
    float base,
    float freq_scale
) {
    int seq = blockIdx.y;        // sequence position
    int h = blockIdx.x;          // head index
    int i = threadIdx.x;         // position in half of head_dim

    if (h >= n_heads || i >= head_dim / 2 || seq >= seq_len) return;

    int pos = start_pos + seq;
    float* head = x + (seq * n_heads + h) * head_dim;

    float freq = 1.0f / powf(base, (float)(2 * i) / head_dim);
    float theta = pos * freq * freq_scale;
    float cos_theta = cosf(theta);
    float sin_theta = sinf(theta);

    float x0 = head[i];
    float x1 = head[i + head_dim / 2];

    head[i] = x0 * cos_theta - x1 * sin_theta;
    head[i + head_dim / 2] = x0 * sin_theta + x1 * cos_theta;
}

/**
 * 预计算 RoPE cos/sin 表
 */
__global__ void rope_precompute_freqs_kernel(
    float* __restrict__ cos_cache,
    float* __restrict__ sin_cache,
    int max_seq_len,
    int head_dim,
    float base,
    float freq_scale
) {
    int pos = blockIdx.x;
    int i = threadIdx.x;

    if (pos >= max_seq_len || i >= head_dim / 2) return;

    float freq = 1.0f / powf(base, (float)(2 * i) / head_dim);
    float theta = pos * freq * freq_scale;

    cos_cache[pos * (head_dim / 2) + i] = cosf(theta);
    sin_cache[pos * (head_dim / 2) + i] = sinf(theta);
}

/**
 * 使用预计算表的 RoPE Kernel
 */
__global__ void rope_with_cache_f32_kernel(
    float* __restrict__ x,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    int n_heads,
    int head_dim,
    int pos
) {
    int h = blockIdx.x;
    int i = threadIdx.x;

    if (h >= n_heads || i >= head_dim / 2) return;

    float* head = x + h * head_dim;

    float cos_theta = cos_cache[pos * (head_dim / 2) + i];
    float sin_theta = sin_cache[pos * (head_dim / 2) + i];

    float x0 = head[i];
    float x1 = head[i + head_dim / 2];

    head[i] = x0 * cos_theta - x1 * sin_theta;
    head[i + head_dim / 2] = x0 * sin_theta + x1 * cos_theta;
}

// ============================================================================
// 接口函数
// ============================================================================

/**
 * RoPE 前向传播 (单位置)
 */
inline void rope_forward_f32(
    float* x,
    int n_heads,
    int head_dim,
    int pos,
    float base = ROPE_DEFAULT_BASE,
    float freq_scale = 1.0f,
    cudaStream_t stream = 0
) {
    int block_size = head_dim / 2;
    rope_f32_kernel<<<n_heads, block_size, 0, stream>>>(
        x, n_heads, head_dim, pos, base, freq_scale
    );
}

/**
 * RoPE 前向传播 (交错布局)
 */
inline void rope_interleaved_forward_f32(
    float* x,
    int n_heads,
    int head_dim,
    int pos,
    float base = ROPE_DEFAULT_BASE,
    float freq_scale = 1.0f,
    cudaStream_t stream = 0
) {
    int block_size = head_dim / 2;
    rope_interleaved_f32_kernel<<<n_heads, block_size, 0, stream>>>(
        x, n_heads, head_dim, pos, base, freq_scale
    );
}

/**
 * RoPE 前向传播 (批量)
 */
inline void rope_batch_forward_f32(
    float* x,
    int seq_len,
    int n_heads,
    int head_dim,
    int start_pos = 0,
    float base = ROPE_DEFAULT_BASE,
    float freq_scale = 1.0f,
    cudaStream_t stream = 0
) {
    dim3 grid(n_heads, seq_len);
    int block_size = head_dim / 2;
    rope_batch_f32_kernel<<<grid, block_size, 0, stream>>>(
        x, seq_len, n_heads, head_dim, start_pos, base, freq_scale
    );
}

/**
 * 预计算 RoPE 缓存
 */
inline void rope_precompute_freqs(
    float* cos_cache,
    float* sin_cache,
    int max_seq_len,
    int head_dim,
    float base = ROPE_DEFAULT_BASE,
    float freq_scale = 1.0f,
    cudaStream_t stream = 0
) {
    int block_size = head_dim / 2;
    rope_precompute_freqs_kernel<<<max_seq_len, block_size, 0, stream>>>(
        cos_cache, sin_cache, max_seq_len, head_dim, base, freq_scale
    );
}

/**
 * RoPE 前向传播 (使用缓存)
 */
inline void rope_with_cache_forward_f32(
    float* x,
    const float* cos_cache,
    const float* sin_cache,
    int n_heads,
    int head_dim,
    int pos,
    cudaStream_t stream = 0
) {
    int block_size = head_dim / 2;
    rope_with_cache_f32_kernel<<<n_heads, block_size, 0, stream>>>(
        x, cos_cache, sin_cache, n_heads, head_dim, pos
    );
}

#endif // KERNELS_ATTENTION_ROPE_CUH
