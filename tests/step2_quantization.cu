/**
 * Step 2: Introduction to Quantization
 * =====================================
 *
 * This step introduces the quantization concepts and formats.
 * We will:
 * 1. Understand Q4_0, Q8_0, Q8_1 formats
 * 2. Implement and test quantization/dequantization
 * 3. Measure quantization error
 *
 * Key Concepts:
 * - Block quantization (32 elements per block)
 * - Scale factor calculation
 * - 4-bit packing/unpacking
 * - Sum compensation in Q8_1
 *
 * Build:
 *   nvcc -O3 -arch=sm_86 step2_quantization.cu -o step2_quantization -lcurand
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "../include/quant_types.h"
#include "../include/quantize.h"
#include "../include/test_utils.h"

void test_q4_0_quantization() {
    printf("\n========================================\n");
    printf("Test: Q4_0 Quantization\n");
    printf("========================================\n");

    const int K = 1024;  // 32 blocks of 32 elements
    const int nb = K / QK4_0;

    // Allocate memory
    float* h_original = new float[K];
    float* h_reconstructed = new float[K];
    block_q4_0* h_quantized = new block_q4_0[nb];

    // Initialize with random data in [-1, 1]
    srand(42);
    for (int i = 0; i < K; i++) {
        h_original[i] = 2.0f * (float)rand() / RAND_MAX - 1.0f;
    }

    // Test CPU quantization
    printf("\n[CPU Q4_0 Quantization]\n");
    quantize_row_q4_0_ref(h_original, h_quantized, K);
    dequantize_row_q4_0(h_quantized, h_reconstructed, K);

    // Compute error metrics
    float mse = compute_mse(h_reconstructed, h_original, K);
    float nmse = compute_nmse(h_reconstructed, h_original, K);
    float max_err = compute_max_abs_error(h_reconstructed, h_original, K);

    printf("MSE: %.6e\n", mse);
    printf("NMSE: %.6e\n", nmse);
    printf("Max Absolute Error: %.6e\n", max_err);
    printf("Memory Reduction: %.1fx (FP32 -> Q4_0: %.1f bits/element)\n",
           sizeof(float) * QK4_0 / (float)sizeof(block_q4_0),
           8.0f * sizeof(block_q4_0) / QK4_0);

    // Show example block
    printf("\n[Example Block Analysis]\n");
    printf("Block 0:\n");
    printf("  Scale (d): %.6f\n", __half2float(h_quantized[0].d));
    printf("  First 8 values:\n");
    for (int j = 0; j < 4; j++) {
        uint8_t packed = h_quantized[0].qs[j];
        int q0 = (packed & 0x0F);
        int q1 = (packed >> 4);
        printf("    qs[%d]: packed=0x%02x -> q[%d]=%2d (%.4f -> %.4f), q[%d]=%2d (%.4f -> %.4f)\n",
               j, packed,
               j, q0 - 8, h_original[j], h_reconstructed[j],
               j+16, q1 - 8, h_original[j+16], h_reconstructed[j+16]);
    }

    // Test GPU quantization
    printf("\n[GPU Q4_0 Quantization]\n");
    CudaBuffer<float> d_original(K);
    CudaBuffer<block_q4_0> d_quantized(nb);

    d_original.copyFromHost(h_original);

    auto result = benchmark_kernel(
        "Q4_0 GPU Quantize",
        [&]() { quantize_q4_0_cuda(d_original.ptr, d_quantized.ptr, K); },
        10, 100
    );

    printf("Time: %.3f ms\n", result.time_ms);

    // Verify GPU result matches CPU
    block_q4_0* h_gpu_quantized = new block_q4_0[nb];
    d_quantized.copyToHost(h_gpu_quantized);

    bool match = true;
    for (int i = 0; i < nb && match; i++) {
        if (__half2float(h_gpu_quantized[i].d) != __half2float(h_quantized[i].d)) {
            match = false;
        }
        for (int j = 0; j < QK4_0/2 && match; j++) {
            if (h_gpu_quantized[i].qs[j] != h_quantized[i].qs[j]) {
                match = false;
            }
        }
    }
    printf("GPU vs CPU match: %s\n", match ? "PASS" : "FAIL");

    delete[] h_original;
    delete[] h_reconstructed;
    delete[] h_quantized;
    delete[] h_gpu_quantized;
}

void test_q8_0_quantization() {
    printf("\n========================================\n");
    printf("Test: Q8_0 Quantization\n");
    printf("========================================\n");

    const int K = 1024;
    const int nb = K / QK8_0;

    float* h_original = new float[K];
    float* h_reconstructed = new float[K];
    block_q8_0* h_quantized = new block_q8_0[nb];

    srand(42);
    for (int i = 0; i < K; i++) {
        h_original[i] = 2.0f * (float)rand() / RAND_MAX - 1.0f;
    }

    printf("\n[CPU Q8_0 Quantization]\n");
    quantize_row_q8_0_ref(h_original, h_quantized, K);
    dequantize_row_q8_0(h_quantized, h_reconstructed, K);

    float mse = compute_mse(h_reconstructed, h_original, K);
    float nmse = compute_nmse(h_reconstructed, h_original, K);
    float max_err = compute_max_abs_error(h_reconstructed, h_original, K);

    printf("MSE: %.6e\n", mse);
    printf("NMSE: %.6e\n", nmse);
    printf("Max Absolute Error: %.6e\n", max_err);
    printf("Memory Reduction: %.1fx (FP32 -> Q8_0: %.1f bits/element)\n",
           sizeof(float) * QK8_0 / (float)sizeof(block_q8_0),
           8.0f * sizeof(block_q8_0) / QK8_0);

    delete[] h_original;
    delete[] h_reconstructed;
    delete[] h_quantized;
}

void test_q8_1_quantization() {
    printf("\n========================================\n");
    printf("Test: Q8_1 Quantization (with Sum)\n");
    printf("========================================\n");

    const int K = 1024;
    const int nb = K / QK8_1;

    float* h_original = new float[K];
    float* h_reconstructed = new float[K];
    block_q8_1* h_quantized = new block_q8_1[nb];

    srand(42);
    for (int i = 0; i < K; i++) {
        h_original[i] = 2.0f * (float)rand() / RAND_MAX - 1.0f;
    }

    printf("\n[CPU Q8_1 Quantization]\n");
    quantize_row_q8_1_ref(h_original, h_quantized, K);
    dequantize_row_q8_1(h_quantized, h_reconstructed, K);

    float mse = compute_mse(h_reconstructed, h_original, K);
    float nmse = compute_nmse(h_reconstructed, h_original, K);

    printf("MSE: %.6e\n", mse);
    printf("NMSE: %.6e\n", nmse);
    printf("Memory per element: %.1f bits\n", 8.0f * sizeof(block_q8_1) / QK8_1);

    // Analyze the sum field
    printf("\n[Sum Field Analysis]\n");
    printf("The sum field stores the sum of original float values.\n");
    printf("This is crucial for Q4_0 × Q8_1 compensation.\n\n");

    for (int i = 0; i < 3; i++) {
        float actual_sum = 0.0f;
        for (int j = 0; j < QK8_1; j++) {
            actual_sum += h_original[i * QK8_1 + j];
        }
        float stored_sum = __half2float(__high2half(h_quantized[i].ds));
        printf("Block %d: actual_sum=%.4f, stored_sum=%.4f, diff=%.6f\n",
               i, actual_sum, stored_sum, fabsf(actual_sum - stored_sum));
    }

    // Test GPU quantization
    printf("\n[GPU Q8_1 Quantization]\n");
    CudaBuffer<float> d_original(K);
    CudaBuffer<block_q8_1> d_quantized(nb);

    d_original.copyFromHost(h_original);

    auto result = benchmark_kernel(
        "Q8_1 GPU Quantize",
        [&]() { quantize_q8_1_cuda(d_original.ptr, d_quantized.ptr, K); },
        10, 100
    );

    printf("Time: %.3f ms\n", result.time_ms);

    delete[] h_original;
    delete[] h_reconstructed;
    delete[] h_quantized;
}

void test_quantization_distribution() {
    printf("\n========================================\n");
    printf("Test: Quantization Error Distribution\n");
    printf("========================================\n");

    // Test with different input distributions
    const int K = 4096;
    float* h_data = new float[K];
    float* h_recon = new float[K];

    // Normal distribution (typical for NN weights)
    printf("\n[Normal Distribution Input]\n");
    srand(42);
    for (int i = 0; i < K; i++) {
        // Box-Muller transform for normal distribution
        float u1 = (float)rand() / RAND_MAX;
        float u2 = (float)rand() / RAND_MAX;
        h_data[i] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    }

    {
        block_q4_0* q4 = new block_q4_0[K / QK4_0];
        block_q8_0* q8 = new block_q8_0[K / QK8_0];

        quantize_row_q4_0_ref(h_data, q4, K);
        dequantize_row_q4_0(q4, h_recon, K);
        printf("Q4_0 NMSE: %.6e\n", compute_nmse(h_recon, h_data, K));

        quantize_row_q8_0_ref(h_data, q8, K);
        dequantize_row_q8_0(q8, h_recon, K);
        printf("Q8_0 NMSE: %.6e\n", compute_nmse(h_recon, h_data, K));

        delete[] q4;
        delete[] q8;
    }

    // Uniform distribution
    printf("\n[Uniform Distribution Input]\n");
    for (int i = 0; i < K; i++) {
        h_data[i] = 2.0f * (float)rand() / RAND_MAX - 1.0f;
    }

    {
        block_q4_0* q4 = new block_q4_0[K / QK4_0];
        block_q8_0* q8 = new block_q8_0[K / QK8_0];

        quantize_row_q4_0_ref(h_data, q4, K);
        dequantize_row_q4_0(q4, h_recon, K);
        printf("Q4_0 NMSE: %.6e\n", compute_nmse(h_recon, h_data, K));

        quantize_row_q8_0_ref(h_data, q8, K);
        dequantize_row_q8_0(q8, h_recon, K);
        printf("Q8_0 NMSE: %.6e\n", compute_nmse(h_recon, h_data, K));

        delete[] q4;
        delete[] q8;
    }

    delete[] h_data;
    delete[] h_recon;
}

int main() {
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║     Step 2: Introduction to Quantization                  ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n");

    print_device_info();

    printf("\nIn this step, we learn about quantization formats:\n");
    printf("  1. Q4_0: 4-bit weights with scale (4.5 bits/elem)\n");
    printf("  2. Q8_0: 8-bit weights with scale (8.5 bits/elem)\n");
    printf("  3. Q8_1: 8-bit activations with scale AND sum (9 bits/elem)\n");
    printf("\nThe sum in Q8_1 is critical for accurate Q4_0 × Q8_1 computation.\n");

    test_q4_0_quantization();
    test_q8_0_quantization();
    test_q8_1_quantization();
    test_quantization_distribution();

    printf("\n╔═══════════════════════════════════════════════════════════╗\n");
    printf("║     Step 2 Complete!                                      ║\n");
    printf("║     Next: Step 3 - W4A16 Quantized GEMM                   ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n");

    return 0;
}
