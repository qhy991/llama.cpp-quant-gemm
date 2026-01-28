/**
 * gemm_reference.h - Reference GEMM Implementations
 *
 * This file provides CPU reference implementations of GEMM operations.
 * These serve as ground truth for correctness verification.
 *
 * GEMM Convention used throughout this project:
 *   C[M, N] = A[M, K] × B[N, K]^T
 *
 * Where:
 *   A = Activation matrix (row-major)
 *   B = Weight matrix (row-major, but used as column-major via transpose)
 *   C = Output matrix (row-major)
 */

#ifndef GEMM_REFERENCE_H
#define GEMM_REFERENCE_H

#include "quant_types.h"
#include "quantize.h"
#include <cstring>

// ============================================================================
// Step 1: Pure FP32 Reference GEMM (Ground Truth)
// ============================================================================
/**
 * Standard FP32 GEMM: C = A × B^T
 *
 * This is the baseline we compare everything against.
 *
 * @param A  Activation matrix [M, K], row-major
 * @param B  Weight matrix [N, K], row-major (transposed during computation)
 * @param C  Output matrix [M, N], row-major
 * @param M  Number of rows in A and C
 * @param N  Number of rows in B and columns in C
 * @param K  Common dimension (columns in A and B)
 */
inline void gemm_fp32_reference(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K)
{
    // Clear output
    memset(C, 0, M * N * sizeof(float));

    // Standard triple-nested loop GEMM
    // C[i,j] = Σ_k A[i,k] × B[j,k]
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[j * K + k];
            }
            C[i * N + j] = sum;
        }
    }
}

// ============================================================================
// Step 2: W4A16 Reference GEMM (4-bit Weight, FP32 Activation)
// ============================================================================
/**
 * W4A16 GEMM: C = A (FP32) × B (Q4_0)^T
 *
 * This demonstrates the basic dequantize-then-compute approach.
 * Not efficient, but clearly shows the mathematical operations.
 *
 * @param A  Activation matrix [M, K], FP32
 * @param B  Weight matrix [N, K/32], Q4_0 quantized
 * @param C  Output matrix [M, N], FP32
 */
inline void gemm_w4a16_reference(
    const float* A,
    const block_q4_0* B,
    float* C,
    int M, int N, int K)
{
    const int nb = K / QK4_0;  // Number of blocks per row

    memset(C, 0, M * N * sizeof(float));

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;

            // Iterate over blocks
            for (int b = 0; b < nb; b++) {
                const block_q4_0& block = B[j * nb + b];
                const float d = __half2float(block.d);

                // Iterate within block
                for (int k = 0; k < QK4_0 / 2; k++) {
                    // Unpack 4-bit values
                    int q0 = get_q4_0_low(block.qs[k]);
                    int q1 = get_q4_0_high(block.qs[k]);

                    // Dequantize and multiply
                    // Note: Q4_0 values are stored as [0,15], need to subtract 8
                    float w0 = (q0 - 8) * d;
                    float w1 = (q1 - 8) * d;

                    int k_idx = b * QK4_0;
                    sum += A[i * K + k_idx + k] * w0;
                    sum += A[i * K + k_idx + k + QK4_0/2] * w1;
                }
            }

            C[i * N + j] = sum;
        }
    }
}

// ============================================================================
// Step 3: W8A16 Reference GEMM (8-bit Weight, FP32 Activation)
// ============================================================================
/**
 * W8A16 GEMM: C = A (FP32) × B (Q8_0)^T
 */
inline void gemm_w8a16_reference(
    const float* A,
    const block_q8_0* B,
    float* C,
    int M, int N, int K)
{
    const int nb = K / QK8_0;

    memset(C, 0, M * N * sizeof(float));

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;

            for (int b = 0; b < nb; b++) {
                const block_q8_0& block = B[j * nb + b];
                const float d = __half2float(block.d);

                for (int k = 0; k < QK8_0; k++) {
                    float w = block.qs[k] * d;
                    sum += A[i * K + b * QK8_0 + k] * w;
                }
            }

            C[i * N + j] = sum;
        }
    }
}

// ============================================================================
// Step 4: W4A8 Reference GEMM (4-bit Weight, 8-bit Activation with Compensation)
// ============================================================================
/**
 * W4A8 GEMM: C = A (Q8_1) × B (Q4_0)^T
 *
 * This is the most important reference implementation. It shows how the
 * Q8_1 sum field is used to compensate for Q4_0's -8 offset.
 *
 * Mathematical derivation:
 *   x_a = q_a × d_a                    (Q8_1 dequantization)
 *   x_w = (q_w - 8) × d_w              (Q4_0 dequantization)
 *
 *   dot = Σ x_a × x_w
 *       = Σ (q_a × d_a) × ((q_w - 8) × d_w)
 *       = d_a × d_w × Σ q_a × (q_w - 8)
 *       = d_a × d_w × (Σ q_a × q_w - 8 × Σ q_a)
 *
 *   Now, Σ q_a ≈ Σ (x_a / d_a) = s / d_a  (where s = Σ x_a, stored in Q8_1)
 *
 *   dot = d_a × d_w × Σ q_a × q_w - d_a × d_w × 8 × (s / d_a)
 *       = d_a × d_w × sumi - 8 × d_w × s
 *       = d_w × (d_a × sumi - 8 × s)
 *
 * This formula is implemented in llama.cpp's vec_dot_q4_0_q8_1_impl
 */
inline void gemm_w4a8_reference(
    const block_q8_1* A,
    const block_q4_0* B,
    float* C,
    int M, int N, int K)
{
    const int nb = K / QK4_0;  // Number of blocks per row

    memset(C, 0, M * N * sizeof(float));

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;

            for (int b = 0; b < nb; b++) {
                // Get Q8_1 activation block
                const block_q8_1& block_a = A[i * nb + b];
                const float d_a = __half2float(__low2half(block_a.ds));
                const float s_a = __half2float(__high2half(block_a.ds));

                // Get Q4_0 weight block
                const block_q4_0& block_w = B[j * nb + b];
                const float d_w = __half2float(block_w.d);

                // Compute integer dot product (without -8 offset in weights)
                int32_t sumi = 0;
                for (int k = 0; k < QK4_0 / 2; k++) {
                    // Get quantized values
                    int q_w0 = get_q4_0_low(block_w.qs[k]);   // [0, 15]
                    int q_w1 = get_q4_0_high(block_w.qs[k]);  // [0, 15]

                    int8_t q_a0 = block_a.qs[k];
                    int8_t q_a1 = block_a.qs[k + QK4_0/2];

                    // Integer multiply-accumulate (no offset applied here!)
                    sumi += (int32_t)q_a0 * q_w0;
                    sumi += (int32_t)q_a1 * q_w1;
                }

                // Apply the compensation formula:
                // result = d_w × (d_a × sumi - 8 × s_a)
                sum += d_w * (d_a * sumi - 8.0f * s_a);
            }

            C[i * N + j] = sum;
        }
    }
}

// ============================================================================
// Step 5: W8A8 Reference GEMM (8-bit Weight, 8-bit Activation)
// ============================================================================
/**
 * W8A8 GEMM: C = A (Q8_1) × B (Q8_0)^T
 *
 * Q8_0 doesn't have an offset, so no compensation is needed.
 * However, we still need the sum for potential Q4_0 layers in the model.
 */
inline void gemm_w8a8_reference(
    const block_q8_1* A,
    const block_q8_0* B,
    float* C,
    int M, int N, int K)
{
    const int nb = K / QK8_0;

    memset(C, 0, M * N * sizeof(float));

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;

            for (int b = 0; b < nb; b++) {
                const block_q8_1& block_a = A[i * nb + b];
                const float d_a = __half2float(__low2half(block_a.ds));

                const block_q8_0& block_w = B[j * nb + b];
                const float d_w = __half2float(block_w.d);

                // Integer dot product
                int32_t sumi = 0;
                for (int k = 0; k < QK8_0; k++) {
                    sumi += (int32_t)block_a.qs[k] * (int32_t)block_w.qs[k];
                }

                // Simple scaling (no offset compensation needed)
                sum += sumi * d_a * d_w;
            }

            C[i * N + j] = sum;
        }
    }
}

// ============================================================================
// Vector Dot Product Functions (llama.cpp compatible signatures)
// ============================================================================
/**
 * Vector dot product: Q4_0 × Q8_1
 * Compatible with llama.cpp's ggml_vec_dot_q4_0_q8_1
 */
inline void vec_dot_q4_0_q8_1(
    int n,              // Number of elements
    float* s,           // Output (scalar result)
    const void* vx,     // Q4_0 weights
    const void* vy)     // Q8_1 activations
{
    const block_q4_0* x = (const block_q4_0*)vx;
    const block_q8_1* y = (const block_q8_1*)vy;
    const int nb = n / QK4_0;

    float sum = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d_w = __half2float(x[i].d);
        const float d_a = __half2float(__low2half(y[i].ds));
        const float s_a = __half2float(__high2half(y[i].ds));

        int32_t sumi = 0;
        for (int k = 0; k < QK4_0 / 2; k++) {
            int q_w0 = get_q4_0_low(x[i].qs[k]);
            int q_w1 = get_q4_0_high(x[i].qs[k]);

            sumi += (int32_t)y[i].qs[k] * q_w0;
            sumi += (int32_t)y[i].qs[k + QK4_0/2] * q_w1;
        }

        sum += d_w * (d_a * sumi - 8.0f * s_a);
    }

    *s = sum;
}

/**
 * Vector dot product: Q8_0 × Q8_1
 */
inline void vec_dot_q8_0_q8_1(
    int n,
    float* s,
    const void* vx,
    const void* vy)
{
    const block_q8_0* x = (const block_q8_0*)vx;
    const block_q8_1* y = (const block_q8_1*)vy;
    const int nb = n / QK8_0;

    float sum = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d_w = __half2float(x[i].d);
        const float d_a = __half2float(__low2half(y[i].ds));

        int32_t sumi = 0;
        for (int k = 0; k < QK8_0; k++) {
            sumi += (int32_t)x[i].qs[k] * (int32_t)y[i].qs[k];
        }

        sum += sumi * d_w * d_a;
    }

    *s = sum;
}

#endif // GEMM_REFERENCE_H
