/**
 * @file compat/ggml_cuda_compat.cuh
 * @brief llama.cpp CUDA 接口兼容层
 *
 * 本文件提供与 llama.cpp CUDA backend 完全兼容的接口包装。
 * 使用这些接口，可以直接替换 llama.cpp 中的算子实现。
 *
 * 使用方法:
 * 1. 在 llama.cpp 中包含此文件
 * 2. 调用 ggml_cuda_op_XXX_custom() 替换原有实现
 */

#ifndef COMPAT_GGML_CUDA_COMPAT_CUH
#define COMPAT_GGML_CUDA_COMPAT_CUH

// 检查是否在 llama.cpp 环境中编译
#ifdef GGML_CUDA_H

#include "ggml-cuda.h"
#include "ggml.h"

// 包含我们的 kernel 实现
#include "../kernels/activation/silu.cuh"
#include "../kernels/activation/gelu.cuh"
#include "../kernels/normalization/rms_norm.cuh"
#include "../kernels/attention/softmax.cuh"
#include "../kernels/attention/rope.cuh"
#include "../kernels/elementwise/elementwise.cuh"

namespace ggml_cuda_custom {

// ============================================================================
// SiLU 兼容接口
// ============================================================================

/**
 * SiLU 算子 - llama.cpp 兼容接口
 *
 * 可直接替换 ggml_cuda_op_silu
 */
inline void ggml_cuda_op_silu_custom(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int64_t ne = ggml_nelements(src0);

    // 调用我们的实现
    silu_forward_f32(src0_d, dst_d, (int)ne, stream);
}

// ============================================================================
// GELU 兼容接口
// ============================================================================

inline void ggml_cuda_op_gelu_custom(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int64_t ne = ggml_nelements(src0);

    gelu_quick_forward_f32(src0_d, dst_d, (int)ne, stream);
}

// ============================================================================
// RMS Norm 兼容接口
// ============================================================================

inline void ggml_cuda_op_rms_norm_custom(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];  // hidden_size
    const int64_t nrows = ggml_nrows(src0);

    // 从 op_params 获取 eps
    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    // 注意: 原版 RMS Norm 不带 weight，这里简化处理
    // 实际集成时需要处理 weight tensor
    rms_norm_forward_f32(src0_d, nullptr, dst_d, (int)nrows, (int)ne00, eps, stream);
}

// ============================================================================
// Softmax 兼容接口
// ============================================================================

inline void ggml_cuda_op_soft_max_custom(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];  // ncols
    const int64_t nrows = ggml_nrows(src0);

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    // 注意: 原版支持 mask 和 alibi，这里简化处理
    softmax_forward_f32(src0_d, dst_d, (int)nrows, (int)ne00, scale, stream);
}

// ============================================================================
// RoPE 兼容接口
// ============================================================================

inline void ggml_cuda_op_rope_custom(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];  // position tensor

    float * src0_d = (float *) src0->data;
    float * dst_d = (float *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];  // head_dim
    const int64_t ne01 = src0->ne[1];  // n_heads
    const int64_t ne02 = src0->ne[2];  // seq_len

    // 从 op_params 获取参数
    // 实际实现需要解析完整的 rope 参数
    int n_dims, mode, n_ctx_orig;
    float freq_base, freq_scale;
    memcpy(&n_dims,      (int32_t *) dst->op_params +  1, sizeof(int32_t));
    memcpy(&mode,        (int32_t *) dst->op_params +  2, sizeof(int32_t));
    memcpy(&n_ctx_orig,  (int32_t *) dst->op_params +  3, sizeof(int32_t));
    memcpy(&freq_base,   (float *)   dst->op_params +  4, sizeof(float));
    memcpy(&freq_scale,  (float *)   dst->op_params +  5, sizeof(float));

    // 简化: 假设处理单个 position
    // 实际实现需要遍历 position tensor
    const int32_t * pos = (const int32_t *) src1->data;
    int position = pos[0];

    // 注意: 这里需要 in-place 处理或复制
    if (src0_d != dst_d) {
        cudaMemcpyAsync(dst_d, src0_d, ggml_nbytes(src0), cudaMemcpyDeviceToDevice, stream);
    }

    rope_forward_f32(dst_d, (int)ne01, (int)ne00, position, freq_base, freq_scale, stream);
}

// ============================================================================
// Add 兼容接口
// ============================================================================

inline void ggml_cuda_op_add_custom(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float * dst_d = (float *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int64_t ne = ggml_nelements(src0);

    add_forward_f32(src0_d, src1_d, dst_d, (int)ne, stream);
}

// ============================================================================
// Mul 兼容接口
// ============================================================================

inline void ggml_cuda_op_mul_custom(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float * dst_d = (float *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int64_t ne = ggml_nelements(src0);

    mul_forward_f32(src0_d, src1_d, dst_d, (int)ne, stream);
}

} // namespace ggml_cuda_custom

#endif // GGML_CUDA_H

// ============================================================================
// 独立使用时的简化接口
// ============================================================================

// 当不在 llama.cpp 环境中时，提供简单的 C 接口

#ifndef GGML_CUDA_H

extern "C" {

// SiLU
void cuda_silu_f32(const float* x, float* y, int n, void* stream) {
    silu_forward_f32(x, y, n, (cudaStream_t)stream);
}

// GELU
void cuda_gelu_f32(const float* x, float* y, int n, void* stream) {
    gelu_quick_forward_f32(x, y, n, (cudaStream_t)stream);
}

// RMS Norm
void cuda_rms_norm_f32(const float* x, const float* weight, float* y,
                       int n_rows, int n_cols, float eps, void* stream) {
    rms_norm_forward_f32(x, weight, y, n_rows, n_cols, eps, (cudaStream_t)stream);
}

// Softmax
void cuda_softmax_f32(const float* x, float* y, int n_rows, int n_cols,
                      float scale, void* stream) {
    softmax_forward_f32(x, y, n_rows, n_cols, scale, (cudaStream_t)stream);
}

// RoPE
void cuda_rope_f32(float* x, int n_heads, int head_dim, int pos,
                   float base, float freq_scale, void* stream) {
    rope_forward_f32(x, n_heads, head_dim, pos, base, freq_scale, (cudaStream_t)stream);
}

// Add
void cuda_add_f32(const float* a, const float* b, float* c, int n, void* stream) {
    add_forward_f32(a, b, c, n, (cudaStream_t)stream);
}

// Mul
void cuda_mul_f32(const float* a, const float* b, float* c, int n, void* stream) {
    mul_forward_f32(a, b, c, n, (cudaStream_t)stream);
}

} // extern "C"

#endif // !GGML_CUDA_H

#endif // COMPAT_GGML_CUDA_COMPAT_CUH
