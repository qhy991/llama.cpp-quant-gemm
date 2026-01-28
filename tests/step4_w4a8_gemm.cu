/**
 * Step 4: W4A8 Quantized GEMM with Sum Compensation
 * ==================================================
 *
 * This is the MOST IMPORTANT step in understanding llama.cpp's quantized GEMM.
 *
 * When both weights (Q4_0) and activations (Q8_1) are quantized, we need
 * special handling because Q4_0 stores values in [0,15] instead of [-8,7].
 *
 * The Compensation Formula:
 * -------------------------
 * Q4_0: x_w = (q_w - 8) × d_w
 * Q8_1: x_a = q_a × d_a
 *
 * Dot product:
 *   result = Σ x_a × x_w
 *          = Σ (q_a × d_a) × ((q_w - 8) × d_w)
 *          = d_a × d_w × Σ q_a × q_w - 8 × d_a × d_w × Σ q_a
 *          = d_a × d_w × sumi - 8 × d_w × (d_a × Σ q_a)
 *
 * Since Σ q_a ≈ s_a / d_a (where s_a = Σ x_a stored in Q8_1):
 *   result = d_w × (d_a × sumi - 8 × s_a)
 *
 * This is the formula used in llama.cpp!
 *
 * Build:
 *   nvcc -O3 -arch=sm_86 step4_w4a8_gemm.cu -o step4_w4a8_gemm -lcurand
 */

#include <cstdio>
#include <cstdlib>

#include "../include/quant_types.h"
#include "../include/quantize.h"
#include "../include/gemm_reference.h"
#include "../include/gemm_cuda_naive.cuh"
#include "../include/gemm_cuda_tiled.cuh"
#include "../include/gemm_cuda_dp4a.cuh"
#include "../include/test_utils.h"

/**
 * Demonstrate why compensation is needed
 */
void demonstrate_compensation() {
    printf("\n========================================\n");
    printf("Demonstration: Why Compensation is Needed\n");
    printf("========================================\n");

    const int K = 32;  // One block

    float a_vals[K] = {
        0.5f, 0.3f, -0.2f, 0.1f, 0.4f, -0.5f, 0.2f, 0.3f,
        -0.1f, 0.6f, 0.2f, -0.3f, 0.1f, 0.4f, -0.2f, 0.5f,
        0.3f, -0.4f, 0.2f, 0.1f, -0.3f, 0.5f, 0.2f, -0.1f,
        0.4f, 0.3f, -0.2f, 0.1f, 0.5f, -0.4f, 0.3f, 0.2f
    };

    float w_vals[K] = {
        0.1f, -0.2f, 0.3f, 0.4f, -0.1f, 0.2f, -0.3f, 0.1f,
        0.2f, -0.1f, 0.4f, -0.2f, 0.1f, 0.3f, -0.4f, 0.2f,
        -0.2f, 0.3f, 0.1f, -0.3f, 0.2f, 0.1f, -0.2f, 0.4f,
        0.1f, -0.3f, 0.2f, 0.3f, -0.1f, 0.2f, 0.1f, -0.2f
    };

    // 1. FP32 ground truth
    float fp32_result = 0.0f;
    for (int i = 0; i < K; i++) {
        fp32_result += a_vals[i] * w_vals[i];
    }
    printf("\n1. FP32 Ground Truth: %.6f\n", fp32_result);

    // 2. Quantize
    block_q8_1 q8_block;
    block_q4_0 q4_block;
    quantize_row_q8_1_ref(a_vals, &q8_block, K);
    quantize_row_q4_0_ref(w_vals, &q4_block, K);

    float d_a = __half2float(__low2half(q8_block.ds));
    float s_a = __half2float(__high2half(q8_block.ds));
    float d_w = __half2float(q4_block.d);

    printf("\n2. Quantization Parameters:\n");
    printf("   Q8_1: d_a=%.6f, s_a=%.6f\n", d_a, s_a);
    printf("   Q4_0: d_w=%.6f\n", d_w);

    // 3. Wrong computation (without compensation)
    int32_t sumi = 0;
    for (int k = 0; k < K/2; k++) {
        int q_w0 = (q4_block.qs[k] & 0x0F);
        int q_w1 = (q4_block.qs[k] >> 4);
        sumi += (int32_t)q8_block.qs[k] * q_w0;
        sumi += (int32_t)q8_block.qs[k + 16] * q_w1;
    }

    float wrong_result = sumi * d_a * d_w;
    printf("\n3. Without Compensation (WRONG): %.6f\n", wrong_result);
    printf("   Error: %.6f (%.1f%%)\n",
           fabsf(wrong_result - fp32_result),
           100.0f * fabsf(wrong_result - fp32_result) / fabsf(fp32_result));

    // 4. Correct computation (with compensation)
    float correct_result = d_w * (d_a * sumi - 8.0f * s_a);
    printf("\n4. With Compensation (CORRECT): %.6f\n", correct_result);
    printf("   Error: %.6f (%.1f%%)\n",
           fabsf(correct_result - fp32_result),
           100.0f * fabsf(correct_result - fp32_result) / fabsf(fp32_result));

    // 5. Why does compensation work?
    printf("\n5. Mathematical Explanation:\n");
    printf("   sumi = Σ q_a[i] × q_w[i] = %d\n", sumi);
    printf("   s_a = Σ x_a[i] = %.6f\n", s_a);
    printf("\n   The key insight:\n");
    printf("   Q4_0 stores values in [0,15], not [-8,7]\n");
    printf("   Real weight = (stored - 8) × d_w\n");
    printf("\n   Compensation term: -8 × d_w × Σ q_a\n");
    printf("   Since Σ q_a ≈ s_a / d_a, we get: -8 × s_a\n");
    printf("   Final: d_w × (d_a × sumi - 8 × s_a)\n");
}

void run_w4a8_test(int M, int N, int K, const char* name) {
    printf("\n========================================\n");
    printf("Test: W4A8 %s\n", name);
    printf("Dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    printf("========================================\n");

    const int nb = K / QK4_0;

    // Allocate host memory
    float* h_A_fp32 = new float[M * K];
    float* h_B_fp32 = new float[N * K];
    block_q8_1* h_A_q8 = new block_q8_1[M * nb];
    block_q4_0* h_B_q4 = new block_q4_0[N * nb];
    float* h_C_fp32_ref = new float[M * N];
    float* h_C_q4_ref = new float[M * N];
    float* h_C_naive = new float[M * N];
    float* h_C_tiled = new float[M * N];
    float* h_C_dp4a = new float[M * N];
    float* h_C_tiled_dp4a = new float[M * N];
    float* h_C_vec_dp4a = new float[M * N];

    // Initialize with random data
    srand(42);
    for (int i = 0; i < M * K; i++) {
        h_A_fp32[i] = 2.0f * (float)rand() / RAND_MAX - 1.0f;
    }
    for (int i = 0; i < N * K; i++) {
        h_B_fp32[i] = 2.0f * (float)rand() / RAND_MAX - 1.0f;
    }

    // Quantize
    printf("\n[Step 4.1] Quantize activations (Q8_1) and weights (Q4_0)...\n");
    for (int row = 0; row < M; row++) {
        quantize_row_q8_1_ref(&h_A_fp32[row * K], &h_A_q8[row * nb], K);
    }
    for (int row = 0; row < N; row++) {
        quantize_row_q4_0_ref(&h_B_fp32[row * K], &h_B_q4[row * nb], K);
    }

    // FP32 reference
    printf("\n[Step 4.2] FP32 Reference GEMM...\n");
    gemm_fp32_reference(h_A_fp32, h_B_fp32, h_C_fp32_ref, M, N, K);

    // W4A8 CPU reference
    printf("\n[Step 4.3] W4A8 CPU Reference GEMM (with compensation)...\n");
    gemm_w4a8_reference(h_A_q8, h_B_q4, h_C_q4_ref, M, N, K);
    compare_results("W4A8 vs FP32", h_C_q4_ref, h_C_fp32_ref, M * N);

    // Allocate device memory
    CudaBuffer<block_q8_1> d_A(M * nb);
    CudaBuffer<block_q4_0> d_B(N * nb);
    CudaBuffer<float> d_C(M * N);

    d_A.copyFromHost(h_A_q8);
    d_B.copyFromHost(h_B_q4);

    // Calculate metrics
    double flops = 2.0 * M * N * K;
    double bytes = M * nb * sizeof(block_q8_1) + N * nb * sizeof(block_q4_0) + M * N * sizeof(float);

    std::vector<BenchmarkResult> results;

    // Naive W4A8 GEMM
    printf("\n[Step 4.4] W4A8 Naive CUDA GEMM...\n");
    d_C.zero();
    auto naive_result = benchmark_kernel(
        "W4A8 Naive",
        [&]() { gemm_w4a8_naive(d_A.ptr, d_B.ptr, d_C.ptr, M, N, K); },
        5, 20, flops, bytes
    );
    results.push_back(naive_result);
    d_C.copyToHost(h_C_naive);
    compare_results("Naive vs CPU", h_C_naive, h_C_q4_ref, M * N);

    // Tiled W4A8 GEMM
    printf("\n[Step 4.5] W4A8 Tiled CUDA GEMM...\n");
    d_C.zero();
    auto tiled_result = benchmark_kernel(
        "W4A8 Tiled",
        [&]() { gemm_w4a8_tiled(d_A.ptr, d_B.ptr, d_C.ptr, M, N, K); },
        5, 20, flops, bytes
    );
    results.push_back(tiled_result);
    d_C.copyToHost(h_C_tiled);
    compare_results("Tiled vs CPU", h_C_tiled, h_C_q4_ref, M * N);

    // DP4A W4A8 GEMM
    printf("\n[Step 4.6] W4A8 DP4A CUDA GEMM...\n");
    d_C.zero();
    auto dp4a_result = benchmark_kernel(
        "W4A8 DP4A",
        [&]() { gemm_w4a8_dp4a(d_A.ptr, d_B.ptr, d_C.ptr, M, N, K); },
        5, 20, flops, bytes
    );
    results.push_back(dp4a_result);
    d_C.copyToHost(h_C_dp4a);
    compare_results("DP4A vs CPU", h_C_dp4a, h_C_q4_ref, M * N);

    // Tiled + DP4A W4A8 GEMM
    printf("\n[Step 4.7] W4A8 Tiled+DP4A CUDA GEMM...\n");
    d_C.zero();
    auto tiled_dp4a_result = benchmark_kernel(
        "W4A8 Tiled+DP4A",
        [&]() { gemm_w4a8_tiled_dp4a(d_A.ptr, d_B.ptr, d_C.ptr, M, N, K); },
        5, 20, flops, bytes
    );
    results.push_back(tiled_dp4a_result);
    d_C.copyToHost(h_C_tiled_dp4a);
    compare_results("Tiled+DP4A vs CPU", h_C_tiled_dp4a, h_C_q4_ref, M * N);

    // Vectorized + DP4A W4A8 GEMM
    printf("\n[Step 4.8] W4A8 Vectorized+DP4A CUDA GEMM...\n");
    d_C.zero();
    auto vec_dp4a_result = benchmark_kernel(
        "W4A8 Vec+DP4A",
        [&]() { gemm_w4a8_vectorized_dp4a(d_A.ptr, d_B.ptr, d_C.ptr, M, N, K); },
        5, 20, flops, bytes
    );
    results.push_back(vec_dp4a_result);
    d_C.copyToHost(h_C_vec_dp4a);
    compare_results("Vec+DP4A vs CPU", h_C_vec_dp4a, h_C_q4_ref, M * N);

    // Print summary
    printf("\n[Performance Summary]\n");
    print_benchmark_table(results);

    printf("Quantization Error (NMSE vs FP32): %.6e\n",
           compute_nmse(h_C_q4_ref, h_C_fp32_ref, M * N));

    // Cleanup
    delete[] h_A_fp32;
    delete[] h_B_fp32;
    delete[] h_A_q8;
    delete[] h_B_q4;
    delete[] h_C_fp32_ref;
    delete[] h_C_q4_ref;
    delete[] h_C_naive;
    delete[] h_C_tiled;
    delete[] h_C_dp4a;
    delete[] h_C_tiled_dp4a;
    delete[] h_C_vec_dp4a;
}

int main() {
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║     Step 4: W4A8 Quantized GEMM with Compensation         ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n");

    print_device_info();

    printf("\nW4A8 means:\n");
    printf("  - Weights: 4-bit quantized (Q4_0)\n");
    printf("  - Activations: 8-bit quantized (Q8_1)\n");
    printf("\nThe key is the COMPENSATION FORMULA:\n");
    printf("  result = d_w × (d_a × sumi - 8 × s_a)\n");
    printf("\nThis compensates for Q4_0's [0,15] storage of [-8,7] values.\n");

    demonstrate_compensation();

    run_w4a8_test(1, 4096, 4096, "Single Token");
    run_w4a8_test(128, 4096, 4096, "Medium Batch");
    run_w4a8_test(512, 4096, 4096, "Large Batch");
    run_w4a8_test(512, 4096, 14336, "FFN Up Layer");

    printf("\n╔═══════════════════════════════════════════════════════════╗\n");
    printf("║     Step 4 Complete!                                      ║\n");
    printf("║     Next: Step 5 - Comparison with llama.cpp              ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n");

    return 0;
}
