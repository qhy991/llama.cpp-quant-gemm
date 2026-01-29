/**
 * @file compat/ggml_types.h
 * @brief llama.cpp 兼容类型定义
 *
 * 本文件定义了与 llama.cpp 完全兼容的数据类型，使得本项目的算子
 * 可以直接替换 llama.cpp 中的对应实现。
 *
 * 类型来源: llama.cpp/ggml/src/ggml-common.h
 * 同步日期: 2026-01-28
 *
 * 重要提示:
 * - 所有类型的大小和布局必须与 llama.cpp 完全一致
 * - 使用 static_assert 验证大小
 * - 修改前请先检查 llama.cpp 的最新版本
 */

#ifndef COMPAT_GGML_TYPES_H
#define COMPAT_GGML_TYPES_H

#pragma once

#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

// ============================================================================
// 块大小常量
// ============================================================================
// 这些常量定义了每个量化块包含的元素数量
// 必须与 llama.cpp 中的定义完全一致

#define QK4_0 32    // Q4_0 块大小: 32 个元素
#define QK4_1 32    // Q4_1 块大小: 32 个元素
#define QK5_0 32    // Q5_0 块大小: 32 个元素
#define QK5_1 32    // Q5_1 块大小: 32 个元素
#define QK8_0 32    // Q8_0 块大小: 32 个元素
#define QK8_1 32    // Q8_1 块大小: 32 个元素

// K-quant 块大小
#define QK_K 256    // K-quant 超级块大小

// ============================================================================
// Q4_0 格式
// ============================================================================
/**
 * Q4_0: 4-bit 对称量化
 *
 * 内存布局:
 * +----------------+--------------------------------+
 * | d (2 bytes)    |        qs[16] (16 bytes)       |
 * | half 缩放因子  | 4-bit 量化值 (每字节2个)       |
 * +----------------+--------------------------------+
 * 总计: 18 bytes 存储 32 个值 = 4.5 bits/value
 *
 * 量化公式:
 *   d = max(|x|) / 7.0
 *   q = round(x / d) + 8, 限制到 [0, 15]
 *
 * 反量化公式:
 *   x = (q - 8) * d
 */
typedef struct {
    half d;                 // 缩放因子 (delta)
    uint8_t qs[QK4_0/2];   // 量化值，每字节打包两个 4-bit 值
} block_q4_0;

static_assert(sizeof(block_q4_0) == 18, "block_q4_0 must be 18 bytes");

// ============================================================================
// Q4_1 格式
// ============================================================================
/**
 * Q4_1: 4-bit 非对称量化
 *
 * 内存布局:
 * +----------------+----------------+------------------------+
 * | d (2 bytes)    | m (2 bytes)    |    qs[16] (16 bytes)   |
 * | half 缩放因子  | half 最小值    | 4-bit 量化值           |
 * +----------------+----------------+------------------------+
 * 总计: 20 bytes 存储 32 个值 = 5 bits/value
 *
 * 量化公式:
 *   m = min(x)
 *   d = (max(x) - min(x)) / 15.0
 *   q = round((x - m) / d), 限制到 [0, 15]
 *
 * 反量化公式:
 *   x = q * d + m
 */
typedef struct {
    half d;                 // 缩放因子
    half m;                 // 最小值 (偏移)
    uint8_t qs[QK4_1/2];   // 量化值
} block_q4_1;

static_assert(sizeof(block_q4_1) == 20, "block_q4_1 must be 20 bytes");

// ============================================================================
// Q5_0 格式
// ============================================================================
/**
 * Q5_0: 5-bit 对称量化
 *
 * 内存布局:
 * +----------------+----------------+------------------------+
 * | d (2 bytes)    | qh[4] (4 bytes)|    qs[16] (16 bytes)   |
 * | half 缩放因子  | 高位           | 低 4-bit               |
 * +----------------+----------------+------------------------+
 * 总计: 22 bytes 存储 32 个值 = 5.5 bits/value
 */
typedef struct {
    half d;                 // 缩放因子
    uint8_t qh[4];         // 第 5 bit
    uint8_t qs[QK5_0/2];   // 低 4 bits
} block_q5_0;

static_assert(sizeof(block_q5_0) == 22, "block_q5_0 must be 22 bytes");

// ============================================================================
// Q5_1 格式
// ============================================================================
/**
 * Q5_1: 5-bit 非对称量化
 */
typedef struct {
    half d;                 // 缩放因子
    half m;                 // 最小值
    uint8_t qh[4];         // 第 5 bit
    uint8_t qs[QK5_1/2];   // 低 4 bits
} block_q5_1;

static_assert(sizeof(block_q5_1) == 24, "block_q5_1 must be 24 bytes");

// ============================================================================
// Q8_0 格式
// ============================================================================
/**
 * Q8_0: 8-bit 对称量化 (用于权重)
 *
 * 内存布局:
 * +----------------+--------------------------------+
 * | d (2 bytes)    |        qs[32] (32 bytes)       |
 * | half 缩放因子  | 8-bit 有符号量化值             |
 * +----------------+--------------------------------+
 * 总计: 34 bytes 存储 32 个值 = 8.5 bits/value
 *
 * 量化公式:
 *   d = max(|x|) / 127.0
 *   q = round(x / d), 限制到 [-128, 127]
 *
 * 反量化公式:
 *   x = q * d
 */
typedef struct {
    half d;                 // 缩放因子
    int8_t qs[QK8_0];      // 8-bit 有符号量化值
} block_q8_0;

static_assert(sizeof(block_q8_0) == 34, "block_q8_0 must be 34 bytes");

// ============================================================================
// Q8_1 格式
// ============================================================================
/**
 * Q8_1: 8-bit 量化带求和 (用于激活)
 *
 * 内存布局:
 * +------------------+--------------------------------+
 * | ds (4 bytes)     |        qs[32] (32 bytes)       |
 * | half2 (d + s)    | 8-bit 有符号量化值             |
 * +------------------+--------------------------------+
 * 总计: 36 bytes 存储 32 个值 = 9 bits/value
 *
 * ds 字段:
 * - d (低半部分): 缩放因子
 * - s (高半部分): 原始值的求和 Σx[i]
 *
 * 为什么需要 s?
 * 在计算 Q4_0 × Q8_1 点积时:
 *   result = Σ (q_w - 8) * d_w * q_a * d_a
 *          = d_w * d_a * Σ (q_w * q_a) - 8 * d_w * d_a * Σ q_a
 *          = d_w * (d_a * sumi - 8 * s)
 *
 * 存储原始值的和 s 允许精确的补偿计算。
 */
typedef struct {
    half2 ds;               // d (缩放) 和 s (求和) 打包为 half2
    int8_t qs[QK8_1];      // 8-bit 有符号量化值
} block_q8_1;

static_assert(sizeof(block_q8_1) == 36, "block_q8_1 must be 36 bytes");

// ============================================================================
// 量化类型枚举
// ============================================================================
/**
 * 量化类型枚举 (与 llama.cpp 的 ggml_type 对应)
 */
enum QuantType {
    QUANT_TYPE_F32  = 0,
    QUANT_TYPE_F16  = 1,
    QUANT_TYPE_Q4_0 = 2,
    QUANT_TYPE_Q4_1 = 3,
    QUANT_TYPE_Q5_0 = 6,
    QUANT_TYPE_Q5_1 = 7,
    QUANT_TYPE_Q8_0 = 8,
    QUANT_TYPE_Q8_1 = 9,
    // K-quants
    QUANT_TYPE_Q2_K = 10,
    QUANT_TYPE_Q3_K = 11,
    QUANT_TYPE_Q4_K = 12,
    QUANT_TYPE_Q5_K = 13,
    QUANT_TYPE_Q6_K = 14,
    QUANT_TYPE_Q8_K = 15,
};

// ============================================================================
// 辅助函数
// ============================================================================

/**
 * 获取指定量化类型的块大小
 */
__host__ __device__ inline int get_block_size(QuantType type) {
    switch (type) {
        case QUANT_TYPE_Q4_0:
        case QUANT_TYPE_Q4_1:
        case QUANT_TYPE_Q5_0:
        case QUANT_TYPE_Q5_1:
        case QUANT_TYPE_Q8_0:
        case QUANT_TYPE_Q8_1:
            return 32;
        case QUANT_TYPE_Q2_K:
        case QUANT_TYPE_Q3_K:
        case QUANT_TYPE_Q4_K:
        case QUANT_TYPE_Q5_K:
        case QUANT_TYPE_Q6_K:
        case QUANT_TYPE_Q8_K:
            return 256;
        default:
            return 1;
    }
}

/**
 * 获取指定量化类型每个块的字节数
 */
__host__ __device__ inline int get_block_bytes(QuantType type) {
    switch (type) {
        case QUANT_TYPE_Q4_0: return sizeof(block_q4_0);
        case QUANT_TYPE_Q4_1: return sizeof(block_q4_1);
        case QUANT_TYPE_Q5_0: return sizeof(block_q5_0);
        case QUANT_TYPE_Q5_1: return sizeof(block_q5_1);
        case QUANT_TYPE_Q8_0: return sizeof(block_q8_0);
        case QUANT_TYPE_Q8_1: return sizeof(block_q8_1);
        default: return 0;
    }
}

/**
 * 获取量化类型的名称
 */
inline const char* get_type_name(QuantType type) {
    switch (type) {
        case QUANT_TYPE_F32:  return "F32";
        case QUANT_TYPE_F16:  return "F16";
        case QUANT_TYPE_Q4_0: return "Q4_0";
        case QUANT_TYPE_Q4_1: return "Q4_1";
        case QUANT_TYPE_Q5_0: return "Q5_0";
        case QUANT_TYPE_Q5_1: return "Q5_1";
        case QUANT_TYPE_Q8_0: return "Q8_0";
        case QUANT_TYPE_Q8_1: return "Q8_1";
        default: return "Unknown";
    }
}

// ============================================================================
// CUDA 辅助宏
// ============================================================================

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define KERNEL_CHECK() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "Kernel error: %s\n", cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
        cudaDeviceSynchronize(); \
    } while(0)

#endif // COMPAT_GGML_TYPES_H
