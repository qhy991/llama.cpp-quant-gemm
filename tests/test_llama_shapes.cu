/**
 * test_llama_shapes.cu
 * =====================
 *
 * Test naive quantized GEMM implementations with Llama model shapes
 * Specifically: M=4096, K=14336, N=1,2,3,4,5,8
 *
 * Build:
 *   nvcc -O3 -arch=sm_120 -I../include test_llama_shapes.cu -o test_llama_shapes
 */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "../include/quant_types.h"
#include "../include/quantize.h"
#include "../include/gemm_reference.h"
#include "../include/gemm_cuda_naive.cuh"
#include "../include/test_utils.h"

void test_w4a16_shape(int M, int N, int K) {
    printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("W4A16 (Q4_0): M=%d, N=%d, K=%d\n", M, N, K);
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    const int nb = K / QK4_0;

    // Allocate host memory
    float* h_A = new float[M * K];
    float* h_B_fp32 = new float[N * K];
    block_q4_0* h_B_q4 = new block_q4_0[N * nb];
    float* h_C_ref = new float[M * N];
    float* h_C_gpu = new float[M * N];

    // Initialize with random data
    srand(42);
    for (int i = 0; i < M * K; i++) {
        h_A[i] = 2.0f * (float)rand() / RAND_MAX - 1.0f;
    }
    for (int i = 0; i < N * K; i++) {
        h_B_fp32[i] = 2.0f * (float)rand() / RAND_MAX - 1.0f;
    }

    // Quantize weights to Q4_0
    for (int row = 0; row < N; row++) {
        quantize_row_q4_0_ref(&h_B_fp32[row * K], &h_B_q4[row * nb], K);
    }

    // Compute CPU reference
    gemm_w4a16_reference(h_A, h_B_q4, h_C_ref, M, N, K);

    // Allocate device memory
    CudaBuffer<float> d_A(M * K);
    CudaBuffer<block_q4_0> d_B(N * nb);
    CudaBuffer<float> d_C(M * N);

    d_A.copyFromHost(h_A);
    d_B.copyFromHost(h_B_q4);

    // Calculate FLOPS and bytes
    double flops = 2.0 * M * N * K;
    double bytes = M * K * sizeof(float) + N * nb * sizeof(block_q4_0) + M * N * sizeof(float);

    // Benchmark naive kernel
    d_C.zero();
    auto result = benchmark_kernel(
        "W4A16 Naive",
        [&]() { gemm_w4a16_naive(d_A.ptr, d_B.ptr, d_C.ptr, M, N, K); },
        5, 50, flops, bytes
    );

    d_C.copyToHost(h_C_gpu);

    // Verify correctness
    float nmse = compute_nmse(h_C_gpu, h_C_ref, M * N);
    bool passed = nmse < 1e-3f;

    printf("Time:     %.2f μs\n", result.time_ms * 1000.0);
    printf("GFLOPS:   %.2f\n", result.tflops * 1000.0);
    printf("TFLOPS:   %.3f\n", result.tflops);
    printf("NMSE:     %.6e\n", nmse);
    printf("Status:   %s\n", passed ? "✅ PASS" : "❌ FAIL");

    // Cleanup
    delete[] h_A;
    delete[] h_B_fp32;
    delete[] h_B_q4;
    delete[] h_C_ref;
    delete[] h_C_gpu;
}

void test_w4a8_shape(int M, int N, int K) {
    printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("W4A8 (Q8_1×Q4_0): M=%d, N=%d, K=%d\n", M, N, K);
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    const int nb_q8 = K / QK8_1;
    const int nb_q4 = K / QK4_0;

    // Allocate host memory
    float* h_A = new float[M * K];
    float* h_B = new float[N * K];
    block_q8_1* h_A_q8 = new block_q8_1[M * nb_q8];  // Activation: Q8_1
    block_q4_0* h_B_q4 = new block_q4_0[N * nb_q4];  // Weight: Q4_0
    float* h_C_ref = new float[M * N];
    float* h_C_gpu = new float[M * N];

    // Initialize with random data
    srand(42);
    for (int i = 0; i < M * K; i++) {
        h_A[i] = 2.0f * (float)rand() / RAND_MAX - 1.0f;
    }
    for (int i = 0; i < N * K; i++) {
        h_B[i] = 2.0f * (float)rand() / RAND_MAX - 1.0f;
    }

    // Quantize: A to Q8_1, B to Q4_0
    for (int row = 0; row < M; row++) {
        quantize_row_q8_1_ref(&h_A[row * K], &h_A_q8[row * nb_q8], K);
    }
    for (int row = 0; row < N; row++) {
        quantize_row_q4_0_ref(&h_B[row * K], &h_B_q4[row * nb_q4], K);
    }

    // Compute CPU reference
    gemm_w4a8_reference(h_A_q8, h_B_q4, h_C_ref, M, N, K);

    // Allocate device memory
    CudaBuffer<block_q8_1> d_A(M * nb_q8);
    CudaBuffer<block_q4_0> d_B(N * nb_q4);
    CudaBuffer<float> d_C(M * N);

    d_A.copyFromHost(h_A_q8);
    d_B.copyFromHost(h_B_q4);

    // Calculate FLOPS and bytes
    double flops = 2.0 * M * N * K;
    double bytes = M * nb_q8 * sizeof(block_q8_1) + N * nb_q4 * sizeof(block_q4_0) + M * N * sizeof(float);

    // Benchmark naive kernel
    d_C.zero();
    auto result = benchmark_kernel(
        "W4A8 Naive",
        [&]() { gemm_w4a8_naive(d_A.ptr, d_B.ptr, d_C.ptr, M, N, K); },
        5, 50, flops, bytes
    );

    d_C.copyToHost(h_C_gpu);

    // Verify correctness
    float nmse = compute_nmse(h_C_gpu, h_C_ref, M * N);
    bool passed = nmse < 1e-3f;

    printf("Time:     %.2f μs\n", result.time_ms * 1000.0);
    printf("GFLOPS:   %.2f\n", result.tflops * 1000.0);
    printf("TFLOPS:   %.3f\n", result.tflops);
    printf("NMSE:     %.6e\n", nmse);
    printf("Status:   %s\n", passed ? "✅ PASS" : "❌ FAIL");

    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_A_q8;
    delete[] h_B_q4;
    delete[] h_C_ref;
    delete[] h_C_gpu;
}

void test_w8a8_shape(int M, int N, int K) {
    printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("W8A8 (Q8_1×Q8_0): M=%d, N=%d, K=%d\n", M, N, K);
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    const int nb_q8_1 = K / QK8_1;
    const int nb_q8_0 = K / QK8_0;

    // Allocate host memory
    float* h_A = new float[M * K];
    float* h_B = new float[N * K];
    block_q8_1* h_A_q8 = new block_q8_1[M * nb_q8_1];  // Activation: Q8_1
    block_q8_0* h_B_q8 = new block_q8_0[N * nb_q8_0];  // Weight: Q8_0
    float* h_C_ref = new float[M * N];
    float* h_C_gpu = new float[M * N];

    // Initialize with random data
    srand(42);
    for (int i = 0; i < M * K; i++) {
        h_A[i] = 2.0f * (float)rand() / RAND_MAX - 1.0f;
    }
    for (int i = 0; i < N * K; i++) {
        h_B[i] = 2.0f * (float)rand() / RAND_MAX - 1.0f;
    }

    // Quantize: A to Q8_1, B to Q8_0
    for (int row = 0; row < M; row++) {
        quantize_row_q8_1_ref(&h_A[row * K], &h_A_q8[row * nb_q8_1], K);
    }
    for (int row = 0; row < N; row++) {
        quantize_row_q8_0_ref(&h_B[row * K], &h_B_q8[row * nb_q8_0], K);
    }

    // Compute CPU reference
    gemm_w8a8_reference(h_A_q8, h_B_q8, h_C_ref, M, N, K);

    // Allocate device memory
    CudaBuffer<block_q8_1> d_A(M * nb_q8_1);
    CudaBuffer<block_q8_0> d_B(N * nb_q8_0);
    CudaBuffer<float> d_C(M * N);

    d_A.copyFromHost(h_A_q8);
    d_B.copyFromHost(h_B_q8);

    // Calculate FLOPS and bytes
    double flops = 2.0 * M * N * K;
    double bytes = M * nb_q8_1 * sizeof(block_q8_1) + N * nb_q8_0 * sizeof(block_q8_0) + M * N * sizeof(float);

    // Benchmark naive kernel
    d_C.zero();
    auto result = benchmark_kernel(
        "W8A8 Naive",
        [&]() { gemm_w8a8_naive(d_A.ptr, d_B.ptr, d_C.ptr, M, N, K); },
        5, 50, flops, bytes
    );

    d_C.copyToHost(h_C_gpu);

    // Verify correctness
    float nmse = compute_nmse(h_C_gpu, h_C_ref, M * N);
    bool passed = nmse < 1e-3f;

    printf("Time:     %.2f μs\n", result.time_ms * 1000.0);
    printf("GFLOPS:   %.2f\n", result.tflops * 1000.0);
    printf("TFLOPS:   %.3f\n", result.tflops);
    printf("NMSE:     %.6e\n", nmse);
    printf("Status:   %s\n", passed ? "✅ PASS" : "❌ FAIL");

    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_A_q8;
    delete[] h_B_q8;
    delete[] h_C_ref;
    delete[] h_C_gpu;
}

int main() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  Naive Quantized GEMM Performance Test                      ║\n");
    printf("║  Llama Model Shapes: M=4096, K=14336, N=1,2,3,4,5,8         ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");

    print_device_info();

    const int M = 4096;
    const int K = 14336;

    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  W4A16 (Q4_0) Tests                                          ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");

    test_w4a16_shape(M, 1, K);
    test_w4a16_shape(M, 2, K);
    test_w4a16_shape(M, 3, K);
    test_w4a16_shape(M, 4, K);
    test_w4a16_shape(M, 5, K);
    test_w4a16_shape(M, 8, K);

    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  W4A8 (Q4_0×Q8_1) Tests                                      ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");

    test_w4a8_shape(M, 1, K);
    test_w4a8_shape(M, 2, K);
    test_w4a8_shape(M, 3, K);
    test_w4a8_shape(M, 4, K);
    test_w4a8_shape(M, 5, K);
    test_w4a8_shape(M, 8, K);

    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  W8A8 (Q8_0×Q8_1) Tests                                      ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");

    test_w8a8_shape(M, 1, K);
    test_w8a8_shape(M, 2, K);
    test_w8a8_shape(M, 3, K);
    test_w8a8_shape(M, 4, K);
    test_w8a8_shape(M, 5, K);
    test_w8a8_shape(M, 8, K);

    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  All Tests Complete!                                         ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");

    return 0;
}
