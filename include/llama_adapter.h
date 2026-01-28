/**
 * llama_adapter.h - Adapter Layer for llama.cpp Integration
 *
 * This file provides adapter functions to bridge our quantized GEMM
 * implementations with llama.cpp's ggml_tensor interface.
 *
 * Purpose:
 * 1. Enable direct comparison with llama.cpp's implementations
 * 2. Allow using our kernels within llama.cpp's computation graph
 * 3. Provide a clean interface for integration
 *
 * Usage:
 *   #include "llama_adapter.h"
 *   #include "ggml.h"  // llama.cpp header
 *
 *   // Use our GEMM with llama.cpp tensors
 *   gemm_w4a8_from_ggml(activation_tensor, weights_tensor, output_tensor);
 */

#ifndef LLAMA_ADAPTER_H
#define LLAMA_ADAPTER_H

#include "quant_types.h"
#include "gemm_cuda_naive.cuh"
#include "gemm_cuda_tiled.cuh"
#include "gemm_cuda_dp4a.cuh"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations (to avoid requiring ggml.h in header)
struct ggml_tensor;

// ============================================================================
// Interface Alignment: ggml_tensor -> Our Interface
// ============================================================================

/**
 * Extract dimensions from ggml_tensor
 *
 * llama.cpp convention:
 *   tensor->ne[0] = columns (K dimension)
 *   tensor->ne[1] = rows (M or N dimension)
 *
 * Our convention:
 *   C[M, N] = A[M, K] × B[N, K]^T
 */
inline void extract_dims_from_tensor(
    const struct ggml_tensor * activation,
    const struct ggml_tensor * weights,
    int * M, int * N, int * K)
{
    // Activation: [K, M] in llama.cpp, we need [M, K]
    *M = activation->ne[1];  // rows
    *K = activation->ne[0];  // cols
    
    // Weights: [K, N] in llama.cpp, we need [N, K]
    *N = weights->ne[1];     // rows
    // K is the same
}

/**
 * W4A8 GEMM using llama.cpp tensors
 *
 * @param activation  Q8_1 activation tensor [K, M]
 * @param weights     Q4_0 weight tensor [K, N]
 * @param output      FP32 output tensor [N, M]
 * @param kernel_type Which kernel to use: "naive", "tiled", "dp4a"
 */
void gemm_w4a8_from_ggml(
    const struct ggml_tensor * activation,
    const struct ggml_tensor * weights,
    struct ggml_tensor * output,
    const char * kernel_type = "naive"
);

/**
 * W4A16 GEMM using llama.cpp tensors
 *
 * @param activation  FP32 activation tensor [K, M]
 * @param weights     Q4_0 weight tensor [K, N]
 * @param output      FP32 output tensor [N, M]
 */
void gemm_w4a16_from_ggml(
    const struct ggml_tensor * activation,
    const struct ggml_tensor * weights,
    struct ggml_tensor * output,
    const char * kernel_type = "naive"
);

/**
 * FP32 GEMM using llama.cpp tensors
 *
 * @param activation  FP32 activation tensor [K, M]
 * @param weights     FP32 weight tensor [K, N]
 * @param output      FP32 output tensor [N, M]
 */
void gemm_fp32_from_ggml(
    const struct ggml_tensor * activation,
    const struct ggml_tensor * weights,
    struct ggml_tensor * output,
    const char * kernel_type = "naive"
);

// ============================================================================
// Helper: Type Validation
// ============================================================================

/**
 * Validate tensor types match expected quantization formats
 */
bool validate_tensor_types(
    const struct ggml_tensor * activation,
    const struct ggml_tensor * weights,
    const struct ggml_tensor * output,
    int expected_activation_type,
    int expected_weight_type,
    int expected_output_type
);

// ============================================================================
// Helper: Data Pointer Extraction
// ============================================================================

/**
 * Extract data pointer from ggml_tensor with type safety
 */
template<typename T>
inline T* get_tensor_data(struct ggml_tensor * tensor) {
    return reinterpret_cast<T*>(tensor->data);
}

template<typename T>
inline const T* get_tensor_data(const struct ggml_tensor * tensor) {
    return reinterpret_cast<const T*>(tensor->data);
}

#ifdef __cplusplus
}
#endif

#endif // LLAMA_ADAPTER_H
