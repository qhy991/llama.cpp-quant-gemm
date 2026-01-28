/**
 * quant_types.h - Quantization Type Definitions Compatible with llama.cpp
 *
 * This file defines the quantization block structures that are 100% compatible
 * with llama.cpp's implementation. Understanding these structures is fundamental
 * to implementing quantized GEMM operations.
 *
 * Reference: llama.cpp/ggml/src/ggml-common.h
 */

#ifndef QUANT_TYPES_H
#define QUANT_TYPES_H

// ============================================================================
// IMPORTANT: Prevent duplicate definitions when compiling within llama.cpp
// ============================================================================
// When this header is included from llama.cpp's CUDA files, the types
// block_q4_0, block_q8_0, and block_q8_1 are already defined in ggml-common.h.
// We skip our definitions in that case.
#ifdef GGML_COMMON_DECL
// llama.cpp types are already defined, skip our definitions
#else
// Standalone mode: define our own types

#include <cstdint>
#include <cuda_fp16.h>

// ============================================================================
// Block Size Constants
// ============================================================================
// All quantization formats use 32 elements per block. This is a design choice
// that balances memory efficiency (amortizing scale storage) with quantization
// accuracy (smaller blocks = better accuracy but more overhead).
#define QK4_0 32  // Q4_0: 32 elements per block
#define QK8_0 32  // Q8_0: 32 elements per block
#define QK8_1 32  // Q8_1: 32 elements per block

// ============================================================================
// Q4_0 Format: 4-bit Quantization (Weights)
// ============================================================================
// Q4_0 is the most commonly used 4-bit quantization format in llama.cpp.
// It uses symmetric quantization around zero.
//
// Memory Layout:
// +----------------+--------------------------------+
// | d (2 bytes)    |        qs[16] (16 bytes)       |
// | half scale     | packed 4-bit values            |
// +----------------+--------------------------------+
// Total: 18 bytes for 32 values = 4.5 bits/value
//
// Quantization:
//   d = max(|x|) / 7.0f  (scale factor)
//   q = round(x / d) + 8, clamped to [0, 15]
//
// Dequantization:
//   x = (q - 8) * d
//
// Note: The +8 offset maps signed range [-8, 7] to unsigned [0, 15]
typedef struct {
    half d;              // Scale factor (delta)
    uint8_t qs[QK4_0/2]; // Packed 4-bit quantized values (2 per byte)
} block_q4_0;

static_assert(sizeof(block_q4_0) == 18, "block_q4_0 must be 18 bytes");

// ============================================================================
// Q8_0 Format: 8-bit Quantization (Weights)
// ============================================================================
// Q8_0 provides higher precision than Q4_0 with 8-bit quantization.
// Uses symmetric quantization without offset.
//
// Memory Layout:
// +----------------+--------------------------------+
// | d (2 bytes)    |        qs[32] (32 bytes)       |
// | half scale     | 8-bit signed values            |
// +----------------+--------------------------------+
// Total: 34 bytes for 32 values = 8.5 bits/value
//
// Quantization:
//   d = max(|x|) / 127.0f
//   q = round(x / d), clamped to [-128, 127]
//
// Dequantization:
//   x = q * d
typedef struct {
    half d;           // Scale factor
    int8_t qs[QK8_0]; // 8-bit signed quantized values
} block_q8_0;

static_assert(sizeof(block_q8_0) == 34, "block_q8_0 must be 34 bytes");

// ============================================================================
// Q8_1 Format: 8-bit Quantization (Activations with Sum Compensation)
// ============================================================================
// Q8_1 is specifically designed for activation quantization when paired
// with Q4_0 weights. The key difference from Q8_0 is the sum field.
//
// Memory Layout:
// +------------------+--------------------------------+
// | ds (4 bytes)     |        qs[32] (32 bytes)       |
// | d + s (half2)    | 8-bit signed values            |
// +------------------+--------------------------------+
// Total: 36 bytes for 32 values = 9 bits/value
//
// The sum field stores: s = Σ x[i] (sum of original float values)
//
// Why is sum needed?
// When computing Q4_0 × Q8_1 dot product:
//   result = Σ (q_w - 8) * d_w * q_a * d_a
//          = d_w * d_a * Σ (q_w * q_a) - 8 * d_w * d_a * Σ q_a
//          = d_w * d_a * sumi - 8 * d_w * Σ x_a
//          = d_w * (d_a * sumi - 8 * s)
//
// The -8 offset in Q4_0 creates a bias term that must be compensated.
// Storing the original sum allows accurate compensation.
typedef struct {
    half2 ds;         // d (scale) and s (sum) packed as half2
    int8_t qs[QK8_1]; // 8-bit signed quantized values
} block_q8_1;

static_assert(sizeof(block_q8_1) == 36, "block_q8_1 must be 36 bytes");

// ============================================================================
// Helper Functions for Block Access
// ============================================================================

// Extract low and high 4-bit values from packed byte
__host__ __device__ inline int get_q4_0_low(uint8_t packed) {
    return (packed & 0x0F);
}

__host__ __device__ inline int get_q4_0_high(uint8_t packed) {
    return (packed >> 4);
}

// Pack two 4-bit values into one byte
__host__ __device__ inline uint8_t pack_q4_0(int q0, int q1) {
    return (uint8_t)((q1 << 4) | (q0 & 0x0F));
}

// Get scale from Q8_1 block
__device__ inline float get_q8_1_d(const block_q8_1& block) {
    return __half2float(__low2half(block.ds));
}

// Get sum from Q8_1 block
__device__ inline float get_q8_1_s(const block_q8_1& block) {
    return __half2float(__high2half(block.ds));
}

// Note: make_half2 is already defined in cuda_fp16.hpp

// ============================================================================
// Type Traits for Template Programming
// ============================================================================
template <typename T>
struct quant_traits;

template <>
struct quant_traits<block_q4_0> {
    static constexpr int block_size = QK4_0;
    static constexpr int bytes_per_block = sizeof(block_q4_0);
    static constexpr float bits_per_element = 4.5f;
    static constexpr bool has_sum = false;
};

template <>
struct quant_traits<block_q8_0> {
    static constexpr int block_size = QK8_0;
    static constexpr int bytes_per_block = sizeof(block_q8_0);
    static constexpr float bits_per_element = 8.5f;
    static constexpr bool has_sum = false;
};

template <>
struct quant_traits<block_q8_1> {
    static constexpr int block_size = QK8_1;
    static constexpr int bytes_per_block = sizeof(block_q8_1);
    static constexpr float bits_per_element = 9.0f;
    static constexpr bool has_sum = true;
};

#endif // GGML_COMMON_H (end of standalone mode definitions)

#endif // QUANT_TYPES_H
