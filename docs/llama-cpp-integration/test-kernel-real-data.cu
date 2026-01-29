/**
 * Simple real data test for custom DP4A kernel
 *
 * This creates a minimal test using CUDA directly to verify the kernel works
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

// Include the quantization types from llama.cpp
#include "../../ggml/src/ggml-common.h"

// Include our custom kernel
#include "../../../quant-gemm-from-scratch/include/gemm_cuda_dp4a.cuh"

// Helper functions
float compute_mse(const float* a, const float* b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum / n;
}

float compute_nmse(const float* a, const float* b, int n) {
    double mse = 0.0;
    double norm = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = a[i] - b[i];
        mse += diff * diff;
        norm += b[i] * b[i];
    }
    return (norm > 0) ? (mse / norm) : 0.0f;
}

// CPU reference implementation (row-major layout to match kernel)
// A: [M, K] activation matrix (row-major)
// B: [N, K] weight matrix (row-major)
// C: [M, N] output matrix (row-major)
void cpu_gemm_fp32(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[n * K + k];  // Row-major
            }
            C[m * N + n] = sum;  // Row-major
        }
    }
}

// Quantize FP32 to Q4_0
void quantize_q4_0(const float* src, block_q4_0* dst, int n) {
    const int block_size = 32;
    const int num_blocks = n / block_size;

    for (int b = 0; b < num_blocks; b++) {
        const float* block_src = src + b * block_size;

        // Find max absolute value
        float max_abs = 0.0f;
        for (int i = 0; i < block_size; i++) {
            float abs_val = fabsf(block_src[i]);
            if (abs_val > max_abs) max_abs = abs_val;
        }

        // Compute scale
        float scale = max_abs / 7.0f;  // 4-bit signed: -8 to 7
        float inv_scale = (scale > 0) ? (1.0f / scale) : 0.0f;

        // Store scale as FP16
        dst[b].d = __float2half(scale);

        // Quantize values
        for (int i = 0; i < 16; i++) {
            int8_t v0 = roundf(block_src[i * 2 + 0] * inv_scale);
            int8_t v1 = roundf(block_src[i * 2 + 1] * inv_scale);

            // Clamp to 4-bit range: -8 to 7
            v0 = (v0 < -8) ? -8 : ((v0 > 7) ? 7 : v0);
            v1 = (v1 < -8) ? -8 : ((v1 > 7) ? 7 : v1);

            // Pack two 4-bit values into one byte
            dst[b].qs[i] = ((v0 + 8) & 0x0F) | (((v1 + 8) & 0x0F) << 4);
        }
    }
}

// Quantize FP32 to Q8_1
void quantize_q8_1(const float* src, block_q8_1* dst, int n) {
    const int block_size = 32;
    const int num_blocks = n / block_size;

    for (int b = 0; b < num_blocks; b++) {
        const float* block_src = src + b * block_size;

        // Find max absolute value
        float max_abs = 0.0f;
        float sum = 0.0f;
        for (int i = 0; i < block_size; i++) {
            float abs_val = fabsf(block_src[i]);
            if (abs_val > max_abs) max_abs = abs_val;
            sum += block_src[i];
        }

        // Compute scale
        float scale = max_abs / 127.0f;
        float inv_scale = (scale > 0) ? (1.0f / scale) : 0.0f;

        // Store scale and sum as half2
        dst[b].ds = make_half2(__float2half(scale), __float2half(sum));

        // Quantize values
        for (int i = 0; i < block_size; i++) {
            int8_t v = roundf(block_src[i] * inv_scale);
            v = (v < -127) ? -127 : ((v > 127) ? 127 : v);
            dst[b].qs[i] = v;
        }
    }
}

int main() {
    printf("=== Custom DP4A Kernel Real Data Test ===\n\n");

    // Test configuration
    const int M = 4;      // Batch size
    const int N = 512;    // Output dimension
    const int K = 1024;   // Hidden dimension (must be multiple of 32)

    printf("Test dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    printf("  Weight: [K=%d, N=%d] Q4_0\n", K, N);
    printf("  Activation: [K=%d, M=%d] Q8_1\n", K, M);
    printf("  Output: [N=%d, M=%d] FP32\n\n", N, M);

    // Initialize random seed
    srand(time(NULL));

    // ========================================================================
    // Step 1: Generate random data
    // ========================================================================
    printf("Step 1: Generating random data...\n");

    float* weight_fp32 = (float*)malloc(K * N * sizeof(float));
    float* activation_fp32 = (float*)malloc(K * M * sizeof(float));
    float* output_cpu = (float*)malloc(N * M * sizeof(float));

    // Generate weight with normal distribution
    for (int i = 0; i < K * N; i++) {
        float u1 = (rand() + 1.0f) / (RAND_MAX + 1.0f);
        float u2 = (rand() + 1.0f) / (RAND_MAX + 1.0f);
        weight_fp32[i] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2) * 0.1f;
    }

    // Generate activation with normal distribution
    for (int i = 0; i < K * M; i++) {
        float u1 = (rand() + 1.0f) / (RAND_MAX + 1.0f);
        float u2 = (rand() + 1.0f) / (RAND_MAX + 1.0f);
        activation_fp32[i] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2) * 0.5f;
    }

    printf("  ✓ Generated %d weight values\n", K * N);
    printf("  ✓ Generated %d activation values\n\n", K * M);

    // ========================================================================
    // Step 2: Compute CPU reference
    // ========================================================================
    printf("Step 2: Computing CPU reference (FP32)...\n");
    cpu_gemm_fp32(activation_fp32, weight_fp32, output_cpu, M, N, K);
    printf("  ✓ CPU reference computed\n\n");

    // ========================================================================
    // Step 3: Quantize data
    // ========================================================================
    printf("Step 3: Quantizing data...\n");

    const int weight_blocks = (K * N) / 32;
    const int activation_blocks = (K * M) / 32;

    block_q4_0* weight_q4 = (block_q4_0*)malloc(weight_blocks * sizeof(block_q4_0));
    block_q8_1* activation_q8 = (block_q8_1*)malloc(activation_blocks * sizeof(block_q8_1));

    quantize_q4_0(weight_fp32, weight_q4, K * N);
    quantize_q8_1(activation_fp32, activation_q8, K * M);

    printf("  ✓ Weight quantized to Q4_0 (%d blocks)\n", weight_blocks);
    printf("  ✓ Activation quantized to Q8_1 (%d blocks)\n\n", activation_blocks);

    // ========================================================================
    // Step 4: Run CUDA kernel
    // ========================================================================
    printf("Step 4: Running custom DP4A kernel on GPU...\n");

    // Allocate GPU memory
    block_q4_0* d_weight;
    block_q8_1* d_activation;
    float* d_output;

    cudaMalloc(&d_weight, weight_blocks * sizeof(block_q4_0));
    cudaMalloc(&d_activation, activation_blocks * sizeof(block_q8_1));
    cudaMalloc(&d_output, N * M * sizeof(float));

    // Copy data to GPU
    cudaMemcpy(d_weight, weight_q4, weight_blocks * sizeof(block_q4_0), cudaMemcpyHostToDevice);
    cudaMemcpy(d_activation, activation_q8, activation_blocks * sizeof(block_q8_1), cudaMemcpyHostToDevice);

    printf("  ✓ Data copied to GPU\n");

    // Launch kernel
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);

    printf("  Launching kernel: grid(%d,%d), block(%d,%d)\n", grid.x, grid.y, block.x, block.y);

    gemm_w4a8_dp4a_kernel<<<grid, block>>>(d_activation, d_weight, d_output, M, N, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "  ✗ Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "  ✗ Kernel execution failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("  ✓ Kernel executed successfully\n");

    // Copy result back
    float* output_cuda = (float*)malloc(N * M * sizeof(float));
    cudaMemcpy(output_cuda, d_output, N * M * sizeof(float), cudaMemcpyDeviceToHost);

    printf("  ✓ Results copied from GPU\n\n");

    // ========================================================================
    // Step 5: Verify results
    // ========================================================================
    printf("Step 5: Verifying results...\n");

    float mse = compute_mse(output_cuda, output_cpu, N * M);
    float nmse = compute_nmse(output_cuda, output_cpu, N * M);

    printf("\n--- Error Metrics ---\n");
    printf("  MSE:  %.6e\n", mse);
    printf("  NMSE: %.6e\n", nmse);

    // Sample values
    printf("\n--- Sample Values (first 10) ---\n");
    printf("  Index | CPU (FP32) | CUDA (Q4_0) | Diff\n");
    printf("  ------|------------|-------------|----------\n");
    for (int i = 0; i < 10 && i < N * M; i++) {
        float diff = output_cuda[i] - output_cpu[i];
        printf("  %5d | %10.6f | %11.6f | %+.6f\n",
               i, output_cpu[i], output_cuda[i], diff);
    }

    // Statistics
    float max_diff = 0.0f;
    float avg_diff = 0.0f;
    for (int i = 0; i < N * M; i++) {
        float diff = fabsf(output_cuda[i] - output_cpu[i]);
        if (diff > max_diff) max_diff = diff;
        avg_diff += diff;
    }
    avg_diff /= (N * M);

    printf("\n--- Statistics ---\n");
    printf("  Max absolute difference: %.6e\n", max_diff);
    printf("  Avg absolute difference: %.6e\n", avg_diff);

    // ========================================================================
    // Step 6: Evaluate
    // ========================================================================
    printf("\n=== Test Results ===\n");

    bool passed = true;
    const float nmse_threshold = 0.01f;  // 1% for Q4_0

    if (nmse < nmse_threshold) {
        printf("✅ NMSE test PASSED (%.6e < %.6e)\n", nmse, nmse_threshold);
    } else {
        printf("❌ NMSE test FAILED (%.6e >= %.6e)\n", nmse, nmse_threshold);
        passed = false;
    }

    float output_sum = 0.0f;
    for (int i = 0; i < N * M; i++) {
        output_sum += fabsf(output_cuda[i]);
    }

    if (output_sum > 1e-6f) {
        printf("✅ Output is non-zero (sum = %.6f)\n", output_sum);
    } else {
        printf("❌ Output is all zeros\n");
        passed = false;
    }

    if (max_diff < 100.0f) {
        printf("✅ Results are in reasonable range\n");
    } else {
        printf("❌ Results are out of reasonable range\n");
        passed = false;
    }

    printf("\n");
    if (passed) {
        printf("🎉 ALL TESTS PASSED!\n");
        printf("   The custom DP4A kernel is working correctly with real data.\n");
    } else {
        printf("❌ SOME TESTS FAILED\n");
    }

    // Cleanup
    free(weight_fp32);
    free(activation_fp32);
    free(output_cpu);
    free(output_cuda);
    free(weight_q4);
    free(activation_q8);
    cudaFree(d_weight);
    cudaFree(d_activation);
    cudaFree(d_output);

    printf("\n=== Test Complete ===\n");
    return passed ? 0 : 1;
}
