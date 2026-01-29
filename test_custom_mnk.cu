#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Include the quantization formats header
#include "kernels/gemm/gemm_quant_formats.cuh"

// Simple quantization functions
void quantize_row_q4_0_ref(const float* src, block_q4_0* dst, int64_t k) {
    const int nb = k / QK4_0;
    for (int i = 0; i < nb; i++) {
        float amax = 0.0f;
        for (int j = 0; j < QK4_0; j++) {
            float v = fabsf(src[i * QK4_0 + j]);
            amax = fmaxf(amax, v);
        }
        const float d = amax / 15.0f;
        const float id = d ? 1.0f / d : 0.0f;

        dst[i].d = __float2half(d);

        // Q4_0 layout: qs[i] low nibble = x[i], high nibble = x[i+16]
        for (int j = 0; j < 16; j++) {
            float x0 = src[i * QK4_0 + j] * id + 8.0f;
            float x1 = src[i * QK4_0 + j + 16] * id + 8.0f;
            uint8_t xi0 = (uint8_t)(x0 < 0 ? 0 : (x0 > 15 ? 15 : x0 + 0.5f));
            uint8_t xi1 = (uint8_t)(x1 < 0 ? 0 : (x1 > 15 ? 15 : x1 + 0.5f));
            dst[i].qs[j] = xi0 | (xi1 << 4);
        }
    }
}

void quantize_row_q8_0_ref(const float* src, block_q8_0* dst, int64_t k) {
    const int nb = k / QK8_0;
    for (int i = 0; i < nb; i++) {
        float amax = 0.0f;
        for (int j = 0; j < QK8_0; j++) {
            float v = fabsf(src[i * QK8_0 + j]);
            amax = fmaxf(amax, v);
        }
        const float d = amax / 127.0f;
        const float id = d ? 1.0f / d : 0.0f;

        dst[i].d = __float2half(d);

        for (int j = 0; j < QK8_0; j++) {
            float x = src[i * QK8_0 + j] * id;
            dst[i].qs[j] = (int8_t)(x + (x >= 0 ? 0.5f : -0.5f));
        }
    }
}

void quantize_row_q8_1_ref(const float* src, block_q8_1* dst, int64_t k) {
    const int nb = k / QK8_1;
    for (int i = 0; i < nb; i++) {
        float amax = 0.0f;
        float sum = 0.0f;
        for (int j = 0; j < QK8_1; j++) {
            float v = src[i * QK8_1 + j];
            sum += v;
            amax = fmaxf(amax, fabsf(v));
        }
        const float d = amax / 127.0f;
        const float id = d ? 1.0f / d : 0.0f;

        dst[i].ds = __halves2half2(__float2half(d), __float2half(sum));

        for (int j = 0; j < QK8_1; j++) {
            float x = src[i * QK8_1 + j] * id;
            dst[i].qs[j] = (int8_t)(x + (x >= 0 ? 0.5f : -0.5f));
        }
    }
}

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// CPU reference implementation
void gemm_fp32_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[n * K + k];
            }
            C[m * N + n] = sum;
        }
    }
}

// Calculate error metrics
void calculate_error(const float* ref, const float* test, int size,
                     float* max_err, float* avg_err, float* mse, float* nmse) {
    *max_err = 0.0f;
    *avg_err = 0.0f;
    *mse = 0.0f;

    float ref_sum_sq = 0.0f;

    for (int i = 0; i < size; i++) {
        float diff = fabsf(ref[i] - test[i]);
        float sq_diff = (ref[i] - test[i]) * (ref[i] - test[i]);

        *max_err = fmaxf(*max_err, diff);
        *avg_err += diff;
        *mse += sq_diff;
        ref_sum_sq += ref[i] * ref[i];
    }

    *avg_err /= size;
    *mse /= size;
    *nmse = (*mse) / (ref_sum_sq / size + 1e-10f);
}

// CUDA kernel for Q4_0 × Q8_1 GEMM
__global__ void gemm_q4_0_q8_1_kernel(
    const block_q4_0* __restrict__ A,
    const block_q8_1* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y;

    if (n >= N || m >= M) return;

    int nb = K / QK4_0;

    const block_q4_0* bq4_row = A + m * nb;
    const block_q8_1* bq8_col = B + n * nb;

    float sum = 0.0f;

    for (int i = 0; i < nb; i++) {
        sum += vec_dot_q4_0_q8_1(&bq4_row[i], &bq8_col[i]);
    }

    C[m * N + n] = sum;
}

// CUDA kernel for Q8_0 × Q8_1 GEMM
__global__ void gemm_q8_0_q8_1_kernel(
    const block_q8_0* __restrict__ A,
    const block_q8_1* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y;

    if (n >= N || m >= M) return;

    int nb = K / QK8_0;

    const block_q8_0* bq8_row = A + m * nb;
    const block_q8_1* bq8_col = B + n * nb;

    float sum = 0.0f;

    for (int i = 0; i < nb; i++) {
        sum += vec_dot_q8_0_q8_1(&bq8_row[i], &bq8_col[i]);
    }

    C[m * N + n] = sum;
}

void test_q4_0_q8_1(int M, int N, int K) {
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Test: Q4_0 × Q8_1 GEMM\n");
    printf("  Dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    printf("═══════════════════════════════════════════════════════════════\n");

    // Allocate host memory
    float* h_A = (float*)malloc(M * K * sizeof(float));
    float* h_B = (float*)malloc(N * K * sizeof(float));
    float* h_C_ref = (float*)malloc(M * N * sizeof(float));
    float* h_C_test = (float*)malloc(M * N * sizeof(float));

    // Initialize with random data
    srand(42);
    for (int i = 0; i < M * K; i++) {
        h_A[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    for (int i = 0; i < N * K; i++) {
        h_B[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }

    printf("[1/5] Computing CPU reference...\n");
    gemm_fp32_cpu(h_A, h_B, h_C_ref, M, N, K);

    // Quantize
    printf("[2/5] Quantizing weights (Q4_0)...\n");
    int nb_A = (M * K) / QK4_0;
    block_q4_0* h_A_q4 = (block_q4_0*)malloc(nb_A * sizeof(block_q4_0));
    quantize_row_q4_0_ref(h_A, h_A_q4, M * K);

    printf("[3/5] Quantizing activations (Q8_1)...\n");
    int nb_B = (N * K) / QK8_1;
    block_q8_1* h_B_q8 = (block_q8_1*)malloc(nb_B * sizeof(block_q8_1));
    quantize_row_q8_1_ref(h_B, h_B_q8, N * K);

    // Allocate device memory
    block_q4_0* d_A;
    block_q8_1* d_B;
    float* d_C;

    CUDA_CHECK(cudaMalloc(&d_A, nb_A * sizeof(block_q4_0)));
    CUDA_CHECK(cudaMalloc(&d_B, nb_B * sizeof(block_q8_1)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A_q4, nb_A * sizeof(block_q4_0), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B_q8, nb_B * sizeof(block_q8_1), cudaMemcpyHostToDevice));

    printf("[4/5] Running CUDA kernel...\n");

    // Launch kernel
    dim3 block(32, 1);
    dim3 grid((N + block.x - 1) / block.x, M);

    // Warmup
    for (int i = 0; i < 5; i++) {
        gemm_q4_0_q8_1_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int num_runs = 20;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_runs; i++) {
        gemm_q4_0_q8_1_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    float avg_time = milliseconds / num_runs;

    CUDA_CHECK(cudaMemcpy(h_C_test, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    printf("[5/5] Verifying results...\n");

    // Calculate errors
    float max_err, avg_err, mse, nmse;
    calculate_error(h_C_ref, h_C_test, M * N, &max_err, &avg_err, &mse, &nmse);

    // Print sample values
    printf("\n  Sample Values (first 5):\n");
    printf("  Index  | Reference    | Kernel       | Diff        \n");
    printf("  -------|--------------|--------------|-------------\n");
    for (int i = 0; i < 5 && i < M * N; i++) {
        float diff = h_C_test[i] - h_C_ref[i];
        printf("  %-6d | %12.6f | %12.6f | %+12.6f\n",
               i, h_C_ref[i], h_C_test[i], diff);
    }

    // Print metrics
    printf("\n  Error Metrics:\n");
    printf("    MSE:      %.6e\n", mse);
    printf("    NMSE:     %.6e (%.4f%%)\n", nmse, nmse * 100.0f);
    printf("    Max Err:  %.6e\n", max_err);
    printf("    Avg Err:  %.6e\n", avg_err);

    // Performance
    double flops = 2.0 * M * N * K;
    double tflops = (flops / (avg_time * 1e-3)) / 1e12;

    printf("\n  Performance:\n");
    printf("    Time:     %.4f ms\n", avg_time);
    printf("    TFLOPS:   %.4f\n", tflops);

    // Pass/Fail
    bool passed = nmse < 0.01f;  // 1% threshold
    printf("\n%s (NMSE %.6f %s 0.01)\n",
           passed ? "✅ PASSED" : "❌ FAILED",
           nmse,
           passed ? "<" : ">=");

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C_ref);
    free(h_C_test);
    free(h_A_q4);
    free(h_B_q8);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

void test_q8_0_q8_1(int M, int N, int K) {
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Test: Q8_0 × Q8_1 GEMM\n");
    printf("  Dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    printf("═══════════════════════════════════════════════════════════════\n");

    // Allocate host memory
    float* h_A = (float*)malloc(M * K * sizeof(float));
    float* h_B = (float*)malloc(N * K * sizeof(float));
    float* h_C_ref = (float*)malloc(M * N * sizeof(float));
    float* h_C_test = (float*)malloc(M * N * sizeof(float));

    // Initialize with random data
    srand(42);
    for (int i = 0; i < M * K; i++) {
        h_A[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    for (int i = 0; i < N * K; i++) {
        h_B[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }

    printf("[1/5] Computing CPU reference...\n");
    gemm_fp32_cpu(h_A, h_B, h_C_ref, M, N, K);

    // Quantize
    printf("[2/5] Quantizing weights (Q8_0)...\n");
    int nb_A = (M * K) / QK8_0;
    block_q8_0* h_A_q8 = (block_q8_0*)malloc(nb_A * sizeof(block_q8_0));
    quantize_row_q8_0_ref(h_A, h_A_q8, M * K);

    printf("[3/5] Quantizing activations (Q8_1)...\n");
    int nb_B = (N * K) / QK8_1;
    block_q8_1* h_B_q8 = (block_q8_1*)malloc(nb_B * sizeof(block_q8_1));
    quantize_row_q8_1_ref(h_B, h_B_q8, N * K);

    // Allocate device memory
    block_q8_0* d_A;
    block_q8_1* d_B;
    float* d_C;

    CUDA_CHECK(cudaMalloc(&d_A, nb_A * sizeof(block_q8_0)));
    CUDA_CHECK(cudaMalloc(&d_B, nb_B * sizeof(block_q8_1)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A_q8, nb_A * sizeof(block_q8_0), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B_q8, nb_B * sizeof(block_q8_1), cudaMemcpyHostToDevice));

    printf("[4/5] Running CUDA kernel...\n");

    // Launch kernel
    dim3 block(32, 1);
    dim3 grid((N + block.x - 1) / block.x, M);

    // Warmup
    for (int i = 0; i < 5; i++) {
        gemm_q8_0_q8_1_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int num_runs = 20;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_runs; i++) {
        gemm_q8_0_q8_1_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    float avg_time = milliseconds / num_runs;

    CUDA_CHECK(cudaMemcpy(h_C_test, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    printf("[5/5] Verifying results...\n");

    // Calculate errors
    float max_err, avg_err, mse, nmse;
    calculate_error(h_C_ref, h_C_test, M * N, &max_err, &avg_err, &mse, &nmse);

    // Print sample values
    printf("\n  Sample Values (first 5):\n");
    printf("  Index  | Reference    | Kernel       | Diff        \n");
    printf("  -------|--------------|--------------|-------------\n");
    for (int i = 0; i < 5 && i < M * N; i++) {
        float diff = h_C_test[i] - h_C_ref[i];
        printf("  %-6d | %12.6f | %12.6f | %+12.6f\n",
               i, h_C_ref[i], h_C_test[i], diff);
    }

    // Print metrics
    printf("\n  Error Metrics:\n");
    printf("    MSE:      %.6e\n", mse);
    printf("    NMSE:     %.6e (%.4f%%)\n", nmse, nmse * 100.0f);
    printf("    Max Err:  %.6e\n", max_err);
    printf("    Avg Err:  %.6e\n", avg_err);

    // Performance
    double flops = 2.0 * M * N * K;
    double tflops = (flops / (avg_time * 1e-3)) / 1e12;

    printf("\n  Performance:\n");
    printf("    Time:     %.4f ms\n", avg_time);
    printf("    TFLOPS:   %.4f\n", tflops);

    // Pass/Fail
    bool passed = nmse < 0.01f;  // 1% threshold
    printf("\n%s (NMSE %.6f %s 0.01)\n",
           passed ? "✅ PASSED" : "❌ FAILED",
           nmse,
           passed ? "<" : ">=");

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C_ref);
    free(h_C_test);
    free(h_A_q8);
    free(h_B_q8);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main(int argc, char** argv) {
    // Default dimensions: M=4096, N=2, K=14336
    int M = 4096;
    int N = 2;
    int K = 14336;

    // Parse command line arguments
    if (argc >= 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }

    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║     Custom Dimension GEMM Test                                ║\n");
    printf("║     M=%d, N=%d, K=%d                                  ║\n", M, N, K);
    printf("╚═══════════════════════════════════════════════════════════════╝\n");

    // Get GPU info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("\nGPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);

    // Run tests
    test_q4_0_q8_1(M, N, K);
    test_q8_0_q8_1(M, N, K);

    printf("\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  All tests completed!\n");
    printf("═══════════════════════════════════════════════════════════════\n");

    return 0;
}
