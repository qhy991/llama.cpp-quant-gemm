/**
 * @file bindings.cpp
 * @brief PyTorch C++ extension bindings for quantized GEMM kernels
 */

#include <torch/extension.h>

// Forward declarations (implemented in gemm_ops.cu)
torch::Tensor quantize_q4_0_cuda(torch::Tensor input);
torch::Tensor quantize_q8_1_cuda(torch::Tensor input);
torch::Tensor dequantize_q4_0_cuda(torch::Tensor input, int K);
torch::Tensor gemm_q4_0_q8_1_cuda(
    torch::Tensor weight_q,
    torch::Tensor activation_q,
    int M, int N, int K
);
torch::Tensor gemm_q4_0_fp32_cuda(
    torch::Tensor weight_q,
    torch::Tensor activation,
    int M, int N, int K
);

// Wrapper functions with input validation
torch::Tensor quantize_q4_0(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() >= 1, "Input must have at least 1 dimension");

    int K = input.size(-1);
    TORCH_CHECK(K % 32 == 0, "Last dimension must be divisible by 32, got ", K);

    return quantize_q4_0_cuda(input);
}

torch::Tensor quantize_q8_1(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() >= 1, "Input must have at least 1 dimension");

    int K = input.size(-1);
    TORCH_CHECK(K % 32 == 0, "Last dimension must be divisible by 32, got ", K);

    return quantize_q8_1_cuda(input);
}

torch::Tensor dequantize_q4_0(torch::Tensor input, int K) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kUInt8, "Input must be uint8");
    TORCH_CHECK(K % 32 == 0, "K must be divisible by 32, got ", K);

    return dequantize_q4_0_cuda(input, K);
}

torch::Tensor gemm_q4_0_q8_1(
    torch::Tensor weight_q,
    torch::Tensor activation_q,
    int M, int N, int K
) {
    TORCH_CHECK(weight_q.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(activation_q.is_cuda(), "Activation must be a CUDA tensor");
    TORCH_CHECK(weight_q.dtype() == torch::kUInt8, "Weight must be uint8");
    TORCH_CHECK(activation_q.dtype() == torch::kUInt8, "Activation must be uint8");

    TORCH_CHECK(K % 32 == 0, "K must be divisible by 32, got ", K);

    int num_blocks = K / 32;
    TORCH_CHECK(weight_q.numel() == M * num_blocks * 18,
                "Weight shape mismatch: expected ", M * num_blocks * 18,
                " elements, got ", weight_q.numel());
    TORCH_CHECK(activation_q.numel() == N * num_blocks * 36,
                "Activation shape mismatch: expected ", N * num_blocks * 36,
                " elements, got ", activation_q.numel());

    return gemm_q4_0_q8_1_cuda(weight_q, activation_q, M, N, K);
}

torch::Tensor gemm_q4_0_fp32(
    torch::Tensor weight_q,
    torch::Tensor activation,
    int M, int N, int K
) {
    TORCH_CHECK(weight_q.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be a CUDA tensor");
    TORCH_CHECK(weight_q.dtype() == torch::kUInt8, "Weight must be uint8");
    TORCH_CHECK(activation.dtype() == torch::kFloat32, "Activation must be float32");

    TORCH_CHECK(K % 32 == 0, "K must be divisible by 32, got ", K);

    // Note: For w4a16 (FP32 activation), test framework passes M, N directly
    // Unlike w4a8 where it swaps them
    int M_batch = M;   // Batch size
    int N_features = N; // Output features

    int num_blocks = K / 32;
    TORCH_CHECK(weight_q.numel() == N_features * num_blocks * 18,
                "Weight shape mismatch: expected ", N_features * num_blocks * 18,
                " elements, got ", weight_q.numel());

    return gemm_q4_0_fp32_cuda(weight_q, activation, M_batch, N_features, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Quantized GEMM CUDA kernels";

    m.def("quantize_q4_0", &quantize_q4_0,
          "Quantize FP32 tensor to Q4_0 format",
          py::arg("input"));

    m.def("quantize_q8_1", &quantize_q8_1,
          "Quantize FP32 tensor to Q8_1 format",
          py::arg("input"));

    m.def("dequantize_q4_0", &dequantize_q4_0,
          "Dequantize Q4_0 tensor to FP32",
          py::arg("input"), py::arg("K"));

    m.def("gemm_q4_0_q8_1", &gemm_q4_0_q8_1,
          "Quantized GEMM with Q4_0 weights and Q8_1 activations",
          py::arg("weight_q"), py::arg("activation_q"),
          py::arg("M"), py::arg("N"), py::arg("K"));

    m.def("gemm_q4_0_fp32", &gemm_q4_0_fp32,
          "Quantized GEMM with Q4_0 weights and FP32 activations",
          py::arg("weight_q"), py::arg("activation"),
          py::arg("M"), py::arg("N"), py::arg("K"));
}
