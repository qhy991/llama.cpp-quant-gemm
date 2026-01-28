/**
 * Step 1: FP32 GEMM - The Foundation
 * ===================================
 *
 * This is the first step in our quantized GEMM tutorial.
 * We start with a pure FP32 GEMM implementation to establish:
 * 1. The correctness baseline
 * 2. The performance baseline
 * 3. The interface convention
 *
 * Key Concepts:
 * - GEMM convention: C[M,N] = A[M,K] × B[N,K]^T
 * - Row-major storage
 * - Naive vs optimized implementations
 *
 * Build:
 *   nvcc -O3 -arch=sm_86 step1_fp32_gemm.cu -o step1_fp32_gemm -lcurand
 *
 * Run:
 *   ./step1_fp32_gemm
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../include/quant_types.h"
#include "../include/gemm_reference.h"
#include "../include/gemm_cuda_naive.cuh"
#include "../include/gemm_cuda_tiled.cuh"
#include "../include/test_utils.h"

// Test dimensions (typical LLM shapes)
struct TestConfig {
    int M, N, K;
    const char* name;
};

TestConfig test_configs[] = {
    {1,    4096, 4096, "Single Token (M=1)"},
    {16,   4096, 4096, "Small Batch (M=16)"},
    {128,  4096, 4096, "Medium Batch (M=128)"},
    {512,  4096, 4096, "Large Batch (M=512)"},
    {512,  4096, 14336, "FFN Up (M=512, K=14336)"},
};

void run_test(int M, int N, int K, const char* name) {
    printf("\n========================================\n");
    printf("Test: %s\n", name);
    printf("Dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    printf("========================================\n");

    // Allocate host memory
    float* h_A = new float[M * K];
    float* h_B = new float[N * K];
    float* h_C_ref = new float[M * N];
    float* h_C_naive = new float[M * N];
    float* h_C_tiled = new float[M * N];

    // Initialize with random data
    srand(42);
    for (int i = 0; i < M * K; i++) {
        h_A[i] = (float)rand() / RAND_MAX - 0.5f;
    }
    for (int i = 0; i < N * K; i++) {
        h_B[i] = (float)rand() / RAND_MAX - 0.5f;
    }

    // Compute CPU reference
    printf("\n[Step 1.1] CPU Reference GEMM...\n");
    auto cpu_start = std::chrono::high_resolution_clock::now();
    gemm_fp32_reference(h_A, h_B, h_C_ref, M, N, K);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    float cpu_ms = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();
    printf("CPU time: %.2f ms\n", cpu_ms);

    // Allocate device memory
    CudaBuffer<float> d_A(M * K);
    CudaBuffer<float> d_B(N * K);
    CudaBuffer<float> d_C(M * N);

    d_A.copyFromHost(h_A);
    d_B.copyFromHost(h_B);

    // Calculate theoretical FLOPS
    double flops = 2.0 * M * N * K;  // multiply-add = 2 ops
    double bytes_naive = (M * K + N * K + M * N) * sizeof(float);

    printf("\n[Step 1.2] CUDA Naive GEMM...\n");
    d_C.zero();
    auto naive_result = benchmark_kernel(
        "FP32 Naive",
        [&]() { gemm_fp32_naive(d_A.ptr, d_B.ptr, d_C.ptr, M, N, K); },
        5, 20, flops, bytes_naive
    );
    d_C.copyToHost(h_C_naive);
    compare_results("Naive vs CPU", h_C_naive, h_C_ref, M * N);
    printf("Time: %.3f ms, TFLOPS: %.3f\n", naive_result.time_ms, naive_result.tflops);

    printf("\n[Step 1.3] CUDA Tiled GEMM...\n");
    d_C.zero();
    auto tiled_result = benchmark_kernel(
        "FP32 Tiled",
        [&]() { gemm_fp32_tiled(d_A.ptr, d_B.ptr, d_C.ptr, M, N, K); },
        5, 20, flops, bytes_naive
    );
    d_C.copyToHost(h_C_tiled);
    compare_results("Tiled vs CPU", h_C_tiled, h_C_ref, M * N);
    printf("Time: %.3f ms, TFLOPS: %.3f\n", tiled_result.time_ms, tiled_result.tflops);

    printf("\n[Performance Summary]\n");
    printf("Speedup (Tiled vs Naive): %.2fx\n", naive_result.time_ms / tiled_result.time_ms);
    printf("Speedup (Tiled vs CPU): %.2fx\n", cpu_ms / tiled_result.time_ms);

    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C_ref;
    delete[] h_C_naive;
    delete[] h_C_tiled;
}

int main() {
    // Disable stdout buffering for immediate output
    setvbuf(stdout, NULL, _IONBF, 0);
    
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║     Step 1: FP32 GEMM - The Foundation                    ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n");
    fflush(stdout);

    print_device_info();
    fflush(stdout);

    printf("\nThis step establishes the FP32 baseline.\n");
    printf("We compare:\n");
    printf("  1. CPU reference implementation (ground truth)\n");
    printf("  2. CUDA naive implementation (one thread per output)\n");
    printf("  3. CUDA tiled implementation (shared memory optimization)\n");

    // Run tests
    for (const auto& config : test_configs) {
        run_test(config.M, config.N, config.K, config.name);
    }

    printf("\n╔═══════════════════════════════════════════════════════════╗\n");
    printf("║     Step 1 Complete!                                      ║\n");
    printf("║     Next: Step 2 - Introduction to Quantization           ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n");

    return 0;
}
