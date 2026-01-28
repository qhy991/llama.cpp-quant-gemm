/**
 * Step 3: W4A16 Quantized GEMM
 * =============================
 *
 * This step implements Weight-4bit Activation-FP16/32 GEMM.
 * The weights are pre-quantized to Q4_0, activations remain in FP32.
 *
 * Key Concepts:
 * - Dequantize-on-the-fly during GEMM
 * - 4-bit unpacking: (packed & 0x0F) - 8 for low, (packed >> 4) - 8 for high
 * - Memory savings: 4x reduction in weight storage
 *
 * Build:
 *   nvcc -O3 -arch=sm_86 step3_w4a16_gemm.cu -o step3_w4a16_gemm -lcurand
 */

#include <cstdio>
#include <cstdlib>

#include "../include/quant_types.h"
#include "../include/quantize.h"
#include "../include/gemm_reference.h"
#include "../include/gemm_cuda_naive.cuh"
#include "../include/gemm_cuda_tiled.cuh"
#include "../include/test_utils.h"

void run_w4a16_test(int M, int N, int K, const char* name) {
    printf("\n========================================\n");
    printf("Test: W4A16 %s\n", name);
    printf("Dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    printf("========================================\n");

    const int nb = K / QK4_0;

    // Allocate host memory
    float* h_A = new float[M * K];
    float* h_B_fp32 = new float[N * K];
    block_q4_0* h_B_q4 = new block_q4_0[N * nb];
    float* h_C_fp32_ref = new float[M * N];
    float* h_C_q4_ref = new float[M * N];
    float* h_C_naive = new float[M * N];
    float* h_C_tiled = new float[M * N];

    // Initialize with random data
    srand(42);
    for (int i = 0; i < M * K; i++) {
        h_A[i] = 2.0f * (float)rand() / RAND_MAX - 1.0f;
    }
    for (int i = 0; i < N * K; i++) {
        h_B_fp32[i] = 2.0f * (float)rand() / RAND_MAX - 1.0f;
    }

    // Quantize weights to Q4_0
    printf("\n[Step 3.1] Quantize weights to Q4_0...\n");
    for (int row = 0; row < N; row++) {
        quantize_row_q4_0_ref(&h_B_fp32[row * K], &h_B_q4[row * nb], K);
    }

    // Compute FP32 reference (ground truth)
    printf("\n[Step 3.2] FP32 Reference GEMM (ground truth)...\n");
    gemm_fp32_reference(h_A, h_B_fp32, h_C_fp32_ref, M, N, K);

    // Compute W4A16 CPU reference
    printf("\n[Step 3.3] W4A16 CPU Reference GEMM...\n");
    gemm_w4a16_reference(h_A, h_B_q4, h_C_q4_ref, M, N, K);
    compare_results("W4A16 vs FP32", h_C_q4_ref, h_C_fp32_ref, M * N);

    // Allocate device memory
    CudaBuffer<float> d_A(M * K);
    CudaBuffer<block_q4_0> d_B(N * nb);
    CudaBuffer<float> d_C(M * N);

    d_A.copyFromHost(h_A);
    d_B.copyFromHost(h_B_q4);

    // Calculate FLOPS and bytes
    double flops = 2.0 * M * N * K;
    double bytes_q4 = M * K * sizeof(float) + N * nb * sizeof(block_q4_0) + M * N * sizeof(float);

    // Naive W4A16 GEMM
    printf("\n[Step 3.4] W4A16 Naive CUDA GEMM...\n");
    d_C.zero();
    auto naive_result = benchmark_kernel(
        "W4A16 Naive",
        [&]() { gemm_w4a16_naive(d_A.ptr, d_B.ptr, d_C.ptr, M, N, K); },
        5, 20, flops, bytes_q4
    );
    d_C.copyToHost(h_C_naive);
    compare_results("Naive vs CPU", h_C_naive, h_C_q4_ref, M * N);
    printf("Time: %.3f ms, TFLOPS: %.3f\n", naive_result.time_ms, naive_result.tflops);

    // Tiled W4A16 GEMM
    printf("\n[Step 3.5] W4A16 Tiled CUDA GEMM...\n");
    d_C.zero();
    auto tiled_result = benchmark_kernel(
        "W4A16 Tiled",
        [&]() { gemm_w4a16_tiled(d_A.ptr, d_B.ptr, d_C.ptr, M, N, K); },
        5, 20, flops, bytes_q4
    );
    d_C.copyToHost(h_C_tiled);
    compare_results("Tiled vs CPU", h_C_tiled, h_C_q4_ref, M * N);
    printf("Time: %.3f ms, TFLOPS: %.3f\n", tiled_result.time_ms, tiled_result.tflops);

    // Summary
    printf("\n[Performance Summary]\n");
    printf("Quantization Error (NMSE): %.6e\n",
           compute_nmse(h_C_q4_ref, h_C_fp32_ref, M * N));
    printf("Speedup (Tiled vs Naive): %.2fx\n",
           naive_result.time_ms / tiled_result.time_ms);

    // Memory analysis
    float weight_reduction = (float)(N * K * sizeof(float)) / (N * nb * sizeof(block_q4_0));
    printf("Weight Memory Reduction: %.2fx\n", weight_reduction);

    // Cleanup
    delete[] h_A;
    delete[] h_B_fp32;
    delete[] h_B_q4;
    delete[] h_C_fp32_ref;
    delete[] h_C_q4_ref;
    delete[] h_C_naive;
    delete[] h_C_tiled;
}

int main() {
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║     Step 3: W4A16 Quantized GEMM                          ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n");

    print_device_info();

    printf("\nW4A16 means:\n");
    printf("  - Weights: 4-bit quantized (Q4_0)\n");
    printf("  - Activations: 16/32-bit floating point\n");
    printf("\nThis is the simplest quantized GEMM approach.\n");
    printf("Weights are dequantized on-the-fly during computation.\n");

    run_w4a16_test(1, 4096, 4096, "Single Token");
    run_w4a16_test(128, 4096, 4096, "Medium Batch");
    run_w4a16_test(512, 4096, 4096, "Large Batch");

    printf("\n╔═══════════════════════════════════════════════════════════╗\n");
    printf("║     Step 3 Complete!                                      ║\n");
    printf("║     Next: Step 4 - W4A8 Quantized GEMM with Compensation  ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n");

    return 0;
}
