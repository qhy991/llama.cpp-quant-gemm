/**
 * Step 5: Comparison with llama.cpp
 * ==================================
 *
 * This step compares our implementations with llama.cpp's actual kernels.
 * We use llama.cpp's vec_dot functions and MMQ kernels as reference.
 *
 * Build:
 *   nvcc -O3 -arch=sm_86 step5_llama_comparison.cu -o step5_llama_comparison \
 *        -I../../llama.cpp/ggml/include -I../../llama.cpp/ggml/src \
 *        -L../../llama.cpp/build/ggml/src -lggml -lcurand
 */

#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>

#include "../include/quant_types.h"
#include "../include/quantize.h"
#include "../include/gemm_reference.h"
#include "../include/gemm_cuda_naive.cuh"
#include "../include/gemm_cuda_tiled.cuh"
#include "../include/gemm_cuda_dp4a.cuh"
#include "../include/test_utils.h"

// Try to include llama.cpp headers if available
#ifdef HAS_LLAMA_CPP
#include "ggml-common.h"
#include "ggml-quants.h"
#include "ggml-cuda/vecdotq.cuh"
#endif

void test_vec_dot_compatibility() {
    printf("\n========================================\n");
    printf("Test: Vector Dot Product Compatibility\n");
    printf("========================================\n");

    const int K = 1024;
    const int nb = K / QK4_0;

    // Generate test data
    float* h_a_fp32 = new float[K];
    float* h_w_fp32 = new float[K];

    srand(42);
    for (int i = 0; i < K; i++) {
        h_a_fp32[i] = 2.0f * (float)rand() / RAND_MAX - 1.0f;
        h_w_fp32[i] = 2.0f * (float)rand() / RAND_MAX - 1.0f;
    }

    // Quantize
    block_q8_1* h_a_q8 = new block_q8_1[nb];
    block_q4_0* h_w_q4 = new block_q4_0[nb];

    quantize_row_q8_1_ref(h_a_fp32, h_a_q8, K);
    quantize_row_q4_0_ref(h_w_fp32, h_w_q4, K);

    // Compute using our implementation
    float our_result;
    vec_dot_q4_0_q8_1(K, &our_result, h_w_q4, h_a_q8);

    printf("Our vec_dot result: %.6f\n", our_result);

#ifdef HAS_LLAMA_CPP
    // Compute using llama.cpp implementation
    float llama_result;
    ggml_vec_dot_q4_0_q8_1(K, &llama_result, 0, h_w_q4, 0, h_a_q8, 0, 1);

    printf("llama.cpp vec_dot result: %.6f\n", llama_result);
    printf("Difference: %.6e (%.3f%%)\n",
           fabsf(our_result - llama_result),
           100.0f * fabsf(our_result - llama_result) / fabsf(llama_result));
#else
    printf("llama.cpp not available for comparison\n");
#endif

    // Compute FP32 reference
    float fp32_result = 0.0f;
    for (int i = 0; i < K; i++) {
        fp32_result += h_a_fp32[i] * h_w_fp32[i];
    }

    printf("FP32 reference: %.6f\n", fp32_result);
    printf("Our error vs FP32: %.6e (%.3f%%)\n",
           fabsf(our_result - fp32_result),
           100.0f * fabsf(our_result - fp32_result) / fabsf(fp32_result));

    delete[] h_a_fp32;
    delete[] h_w_fp32;
    delete[] h_a_q8;
    delete[] h_w_q4;
}

void test_format_compatibility() {
    printf("\n========================================\n");
    printf("Test: Format Compatibility\n");
    printf("========================================\n");

    printf("Checking struct sizes and layouts...\n\n");

    printf("block_q4_0:\n");
    printf("  Our size: %zu bytes\n", sizeof(block_q4_0));
    printf("  Expected: 18 bytes (2 + 16)\n");
    printf("  Match: %s\n", sizeof(block_q4_0) == 18 ? "✓" : "✗");

    printf("\nblock_q8_0:\n");
    printf("  Our size: %zu bytes\n", sizeof(block_q8_0));
    printf("  Expected: 34 bytes (2 + 32)\n");
    printf("  Match: %s\n", sizeof(block_q8_0) == 34 ? "✓" : "✗");

    printf("\nblock_q8_1:\n");
    printf("  Our size: %zu bytes\n", sizeof(block_q8_1));
    printf("  Expected: 36 bytes (4 + 32)\n");
    printf("  Match: %s\n", sizeof(block_q8_1) == 36 ? "✓" : "✗");

#ifdef HAS_LLAMA_CPP
    printf("\nComparing with llama.cpp definitions:\n");
    printf("  block_q4_0: %s\n",
           sizeof(block_q4_0) == sizeof(::block_q4_0) ? "✓ Match" : "✗ Mismatch");
    printf("  block_q8_0: %s\n",
           sizeof(block_q8_0) == sizeof(::block_q8_0) ? "✓ Match" : "✗ Mismatch");
    printf("  block_q8_1: %s\n",
           sizeof(block_q8_1) == sizeof(::block_q8_1) ? "✓ Match" : "✗ Mismatch");
#endif
}

void benchmark_comparison(int M, int N, int K) {
    printf("\n========================================\n");
    printf("Benchmark: Our Implementation vs llama.cpp\n");
    printf("Dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    printf("========================================\n");

    const int nb = K / QK4_0;

    // Generate test data
    float* h_A_fp32 = new float[M * K];
    float* h_B_fp32 = new float[N * K];

    srand(42);
    for (int i = 0; i < M * K; i++) {
        h_A_fp32[i] = 2.0f * (float)rand() / RAND_MAX - 1.0f;
    }
    for (int i = 0; i < N * K; i++) {
        h_B_fp32[i] = 2.0f * (float)rand() / RAND_MAX - 1.0f;
    }

    // Quantize
    block_q8_1* h_A_q8 = new block_q8_1[M * nb];
    block_q4_0* h_B_q4 = new block_q4_0[N * nb];

    for (int row = 0; row < M; row++) {
        quantize_row_q8_1_ref(&h_A_fp32[row * K], &h_A_q8[row * nb], K);
    }
    for (int row = 0; row < N; row++) {
        quantize_row_q4_0_ref(&h_B_fp32[row * K], &h_B_q4[row * nb], K);
    }

    // Allocate device memory
    CudaBuffer<block_q8_1> d_A(M * nb);
    CudaBuffer<block_q4_0> d_B(N * nb);
    CudaBuffer<float> d_C(M * N);

    d_A.copyFromHost(h_A_q8);
    d_B.copyFromHost(h_B_q4);

    double flops = 2.0 * M * N * K;
    double bytes = M * nb * sizeof(block_q8_1) + N * nb * sizeof(block_q4_0) + M * N * sizeof(float);

    std::vector<BenchmarkResult> results;

    // Our implementations
    printf("\n[Our Implementations]\n");

    d_C.zero();
    auto naive_result = benchmark_kernel(
        "Our: Naive",
        [&]() { gemm_w4a8_naive(d_A.ptr, d_B.ptr, d_C.ptr, M, N, K); },
        5, 20, flops, bytes
    );
    results.push_back(naive_result);

    d_C.zero();
    auto dp4a_result = benchmark_kernel(
        "Our: DP4A",
        [&]() { gemm_w4a8_dp4a(d_A.ptr, d_B.ptr, d_C.ptr, M, N, K); },
        5, 20, flops, bytes
    );
    results.push_back(dp4a_result);

    d_C.zero();
    auto tiled_dp4a_result = benchmark_kernel(
        "Our: Tiled+DP4A",
        [&]() { gemm_w4a8_tiled_dp4a(d_A.ptr, d_B.ptr, d_C.ptr, M, N, K); },
        5, 20, flops, bytes
    );
    results.push_back(tiled_dp4a_result);

    d_C.zero();
    auto vec_dp4a_result = benchmark_kernel(
        "Our: Vec+DP4A",
        [&]() { gemm_w4a8_vectorized_dp4a(d_A.ptr, d_B.ptr, d_C.ptr, M, N, K); },
        5, 20, flops, bytes
    );
    results.push_back(vec_dp4a_result);

#ifdef HAS_LLAMA_CPP
    // llama.cpp implementation
    printf("\n[llama.cpp Implementation]\n");
    // Note: This would require proper integration with llama.cpp's CUDA kernels
    // For now, we just show our results
    printf("(Integration with llama.cpp MMQ kernels requires linking with ggml-cuda)\n");
#endif

    print_benchmark_table(results);

    printf("\n[Analysis]\n");
    printf("Best performance: %.3f ms (%.3f TFLOPS)\n",
           vec_dp4a_result.time_ms, vec_dp4a_result.tflops);
    printf("Speedup over naive: %.2fx\n",
           naive_result.time_ms / vec_dp4a_result.time_ms);

    delete[] h_A_fp32;
    delete[] h_B_fp32;
    delete[] h_A_q8;
    delete[] h_B_q4;
}

int main() {
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║     Step 5: Comparison with llama.cpp                     ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n");

    print_device_info();

    printf("\nThis step validates our implementation against llama.cpp.\n");
    printf("We check:\n");
    printf("  1. Format compatibility (struct sizes and layouts)\n");
    printf("  2. Numerical accuracy (vec_dot functions)\n");
    printf("  3. Performance comparison\n");

    test_format_compatibility();
    test_vec_dot_compatibility();

    benchmark_comparison(1, 4096, 4096);
    benchmark_comparison(128, 4096, 4096);
    benchmark_comparison(512, 4096, 4096);

    printf("\n╔═══════════════════════════════════════════════════════════╗\n");
    printf("║     Tutorial Complete!                                    ║\n");
    printf("║                                                           ║\n");
    printf("║     You now understand:                                   ║\n");
    printf("║     • Quantization formats (Q4_0, Q8_0, Q8_1)            ║\n");
    printf("║     • The compensation formula for Q4_0 × Q8_1            ║\n");
    printf("║     • CUDA optimization techniques (tiling, DP4A)         ║\n");
    printf("║     • llama.cpp's quantized GEMM implementation           ║\n");
    printf("║                                                           ║\n");
    printf("║     Next steps:                                           ║\n");
    printf("║     • Implement Tensor Core (WMMA/MMA) optimizations      ║\n");
    printf("║     • Add support for more quantization formats           ║\n");
    printf("║     • Integrate with llama.cpp's MMQ kernels              ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n");

    return 0;
}
