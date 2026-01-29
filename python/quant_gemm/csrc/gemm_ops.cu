/**
 * @file gemm_ops.cu
 * @brief CUDA implementations for quantized GEMM operations
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

// Include the kernel implementations
#include "gemm/gemm_quant_formats.cuh"

// ============================================================================
// Constants
// ============================================================================

#define QK4_0 32
#define QK8_1 32
#define BLOCK_Q4_0_BYTES 18
#define BLOCK_Q8_1_BYTES 36

// ============================================================================
// Quantization Kernels
// ============================================================================

/**
 * Q4_0 Quantization Kernel
 * Each thread processes one block of 32 elements
 */
__global__ void quantize_q4_0_kernel(
    const float* __restrict__ input,
    uint8_t* __restrict__ output,
    int num_blocks
) {
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= num_blocks) return;

    const float* src = input + block_idx * QK4_0;
    uint8_t* dst = output + block_idx * BLOCK_Q4_0_BYTES;

    // Find max absolute value
    float max_abs = 0.0f;
    for (int i = 0; i < QK4_0; i++) {
        float abs_val = fabsf(src[i]);
        if (abs_val > max_abs) max_abs = abs_val;
    }

    // Compute scale
    float scale = max_abs / 7.0f;
    float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

    // Store scale as half
    half* d_ptr = reinterpret_cast<half*>(dst);
    *d_ptr = __float2half(scale);

    // Quantize and pack: qs[i] = (q[i] & 0x0F) | ((q[i+16] & 0x0F) << 4)
    uint8_t* qs = dst + 2;  // Skip the 2-byte scale
    for (int i = 0; i < 16; i++) {
        int8_t q0 = static_cast<int8_t>(roundf(src[i] * inv_scale));
        int8_t q1 = static_cast<int8_t>(roundf(src[i + 16] * inv_scale));

        // Clamp to [-8, 7] then add 8 to get [0, 15]
        q0 = max(-8, min(7, (int)q0)) + 8;
        q1 = max(-8, min(7, (int)q1)) + 8;

        qs[i] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
    }
}

/**
 * Q8_1 Quantization Kernel
 * Each thread processes one block of 32 elements
 */
__global__ void quantize_q8_1_kernel(
    const float* __restrict__ input,
    uint8_t* __restrict__ output,
    int num_blocks
) {
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= num_blocks) return;

    const float* src = input + block_idx * QK8_1;
    uint8_t* dst = output + block_idx * BLOCK_Q8_1_BYTES;

    // Find max absolute value and compute sum
    float max_abs = 0.0f;
    float sum = 0.0f;
    for (int i = 0; i < QK8_1; i++) {
        float val = src[i];
        float abs_val = fabsf(val);
        if (abs_val > max_abs) max_abs = abs_val;
        sum += val;
    }

    // Compute scale
    float scale = max_abs / 127.0f;
    float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

    // Store scale and sum as half2
    half2* ds_ptr = reinterpret_cast<half2*>(dst);
    *ds_ptr = make_half2(__float2half(scale), __float2half(sum));

    // Quantize
    int8_t* qs = reinterpret_cast<int8_t*>(dst + 4);  // Skip the 4-byte half2
    for (int i = 0; i < QK8_1; i++) {
        int q = static_cast<int>(roundf(src[i] * inv_scale));
        qs[i] = static_cast<int8_t>(max(-127, min(127, q)));
    }
}

/**
 * Q4_0 Dequantization Kernel
 */
__global__ void dequantize_q4_0_kernel(
    const uint8_t* __restrict__ input,
    float* __restrict__ output,
    int num_blocks
) {
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= num_blocks) return;

    const uint8_t* src = input + block_idx * BLOCK_Q4_0_BYTES;
    float* dst = output + block_idx * QK4_0;

    // Read scale
    const half* d_ptr = reinterpret_cast<const half*>(src);
    float scale = __half2float(*d_ptr);

    // Dequantize
    const uint8_t* qs = src + 2;
    for (int i = 0; i < 16; i++) {
        uint8_t packed = qs[i];
        int q0 = (packed & 0x0F) - 8;  // Low nibble
        int q1 = ((packed >> 4) & 0x0F) - 8;  // High nibble

        dst[i] = q0 * scale;
        dst[i + 16] = q1 * scale;
    }
}

// ============================================================================
// PyTorch Interface Functions
// ============================================================================

torch::Tensor quantize_q4_0_cuda(torch::Tensor input) {
    // Get dimensions
    auto sizes = input.sizes().vec();
    int64_t K = sizes.back();
    int64_t num_rows = input.numel() / K;
    int64_t num_blocks = num_rows * (K / QK4_0);

    // Create output tensor: [..., K//32, 18]
    sizes.back() = K / QK4_0;
    sizes.push_back(BLOCK_Q4_0_BYTES);

    auto options = torch::TensorOptions()
        .dtype(torch::kUInt8)
        .device(input.device());
    torch::Tensor output = torch::empty(sizes, options);

    // Launch kernel
    int threads = 256;
    int blocks = (num_blocks + threads - 1) / threads;

    quantize_q4_0_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<uint8_t>(),
        num_blocks
    );

    return output;
}

torch::Tensor quantize_q8_1_cuda(torch::Tensor input) {
    // Get dimensions
    auto sizes = input.sizes().vec();
    int64_t K = sizes.back();
    int64_t num_rows = input.numel() / K;
    int64_t num_blocks = num_rows * (K / QK8_1);

    // Create output tensor: [..., K//32, 36]
    sizes.back() = K / QK8_1;
    sizes.push_back(BLOCK_Q8_1_BYTES);

    auto options = torch::TensorOptions()
        .dtype(torch::kUInt8)
        .device(input.device());
    torch::Tensor output = torch::empty(sizes, options);

    // Launch kernel
    int threads = 256;
    int blocks = (num_blocks + threads - 1) / threads;

    quantize_q8_1_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<uint8_t>(),
        num_blocks
    );

    return output;
}

torch::Tensor dequantize_q4_0_cuda(torch::Tensor input, int K) {
    // Get dimensions
    auto sizes = input.sizes().vec();
    sizes.pop_back();  // Remove the 18
    int64_t num_blocks_per_row = sizes.back();
    sizes.back() = K;  // Replace with original K

    int64_t total_blocks = input.numel() / BLOCK_Q4_0_BYTES;

    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(input.device());
    torch::Tensor output = torch::empty(sizes, options);

    // Launch kernel
    int threads = 256;
    int blocks = (total_blocks + threads - 1) / threads;

    dequantize_q4_0_kernel<<<blocks, threads>>>(
        input.data_ptr<uint8_t>(),
        output.data_ptr<float>(),
        total_blocks
    );

    return output;
}

torch::Tensor gemm_q4_0_q8_1_cuda(
    torch::Tensor weight_q,
    torch::Tensor activation_q,
    int M, int N, int K
) {
    // Create output tensor
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(weight_q.device());
    torch::Tensor output = torch::empty({M, N}, options);

    // Get raw pointers
    const block_q4_0* weight_ptr =
        reinterpret_cast<const block_q4_0*>(weight_q.data_ptr<uint8_t>());
    const block_q8_1* activation_ptr =
        reinterpret_cast<const block_q8_1*>(activation_q.data_ptr<uint8_t>());
    float* output_ptr = output.data_ptr<float>();

    // Call the kernel (from gemm_quant_formats.cuh)
    gemm_q4_0_q8_1(weight_ptr, activation_ptr, output_ptr, M, N, K);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel error: ", cudaGetErrorString(err));
    }

    return output;
}
