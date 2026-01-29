# 量化格式 GEMM 详解

本文档详细说明各种量化格式的 GEMM 实现及其与 llama.cpp 的对应关系。

## 支持的量化格式

| 格式 | 位宽 | 类型 | 存储大小/32值 | llama.cpp 对应 |
|------|------|------|---------------|----------------|
| Q4_0 | 4.5 bits | 对称 | 18 bytes | `block_q4_0` |
| Q4_1 | 5 bits | 非对称 | 20 bytes | `block_q4_1` |
| Q5_0 | 5.5 bits | 对称 | 22 bytes | `block_q5_0` |
| Q5_1 | 6 bits | 非对称 | 24 bytes | `block_q5_1` |
| Q8_0 | 8.5 bits | 对称 | 34 bytes | `block_q8_0` |
| Q8_1 | 9 bits | 带求和 | 36 bytes | `block_q8_1` |

## 数据结构对比

### Q4_0 (本项目 vs llama.cpp)

```cpp
// 本项目 (compat/ggml_types.h)
typedef struct {
    half d;              // 缩放因子
    uint8_t qs[16];      // 32 个 4-bit 值
} block_q4_0;            // 18 bytes

// llama.cpp (ggml-common.h)
typedef struct {
    ggml_half d;
    uint8_t qs[QK4_0 / 2];
} block_q4_0;            // 18 bytes

// 完全一致 ✓
```

### Q4_1 (非对称)

```cpp
typedef struct {
    half d;              // 缩放因子
    half m;              // 最小值 (偏移)
    uint8_t qs[16];      // 32 个 4-bit 值
} block_q4_1;            // 20 bytes
```

### Q5_0 (5-bit 对称)

```cpp
typedef struct {
    half d;              // 缩放因子
    uint8_t qh[4];       // 第 5 bit (32 个值)
    uint8_t qs[16];      // 低 4 bits
} block_q5_0;            // 22 bytes
```

### Q8_1 (激活量化，带求和)

```cpp
typedef struct {
    half2 ds;            // d=缩放, s=原值求和
    int8_t qs[32];       // 32 个 int8 值
} block_q8_1;            // 36 bytes
```

## 点积公式

### Q4_0 × Q8_1 (对称)

```
原始公式:
  result = Σ (q_w - 8) * d_w * q_a * d_a

展开:
  = d_w * d_a * Σ (q_w - 8) * q_a
  = d_w * d_a * (Σ q_w * q_a - 8 * Σ q_a)

利用 Σ q_a ≈ s_a / d_a:
  = d_w * (d_a * sumi - 8 * s_a)

其中:
  sumi = Σ q_w * q_a  (使用 DP4A 计算)
  s_a = 原始激活值求和 (存储在 block_q8_1.ds.y)
```

### Q4_1 × Q8_1 (非对称)

```
原始公式:
  result = Σ (q_w * d_w + m_w) * (q_a * d_a)

展开:
  = d_w * d_a * Σ q_w * q_a + m_w * d_a * Σ q_a
  = d_w * d_a * sumi + m_w * s_a / scale_factor

scale_factor = QI8_1 / (vdr * QR4_1) = 32 / (4 * 2) = 4
```

### Q5_0 × Q8_1 (5-bit 对称)

```
与 Q4_0 类似，但偏移为 16:
  result = d_w * (d_a * sumi - 16 * s_a)

第 5 bit 从 qh 字段提取并合并到量化值中。
```

### Q8_0 × Q8_1 (纯点积)

```
最简单的情况，无偏移:
  result = d_w * d_a * Σ q_w * q_a
```

## DP4A 指令优化

所有实现都使用 DP4A 指令加速点积计算:

```cpp
int dp4a(int a, int b, int c) {
    // a, b 被视为 4 个打包的 int8
    char4 va = *(char4*)&a;
    char4 vb = *(char4*)&b;
    return c + va.x*vb.x + va.y*vb.y + va.z*vb.z + va.w*vb.w;
}

// 单指令完成 4 次乘累加
// 比标量实现快 4 倍
```

## llama.cpp 函数对应表

| 本项目函数 | llama.cpp 函数 | 文件 |
|-----------|----------------|------|
| `vec_dot_q4_0_q8_1()` | `vec_dot_q4_0_q8_1_impl<vdr>()` | vecdotq.cuh |
| `vec_dot_q4_1_q8_1()` | `vec_dot_q4_1_q8_1_impl<vdr>()` | vecdotq.cuh |
| `vec_dot_q5_0_q8_1()` | `vec_dot_q5_0_q8_1_impl<vdr>()` | vecdotq.cuh |
| `vec_dot_q5_1_q8_1()` | `vec_dot_q5_1_q8_1_impl<vdr>()` | vecdotq.cuh |
| `vec_dot_q8_0_q8_1()` | `vec_dot_q8_0_q8_1_impl<T,vdr>()` | vecdotq.cuh |

## 使用示例

```cpp
#include "kernels/gemm/gemm_quant_formats.cuh"

// Q4_0 × Q8_1 GEMM
gemm_q4_0_q8_1(d_weight, d_activation, d_output, M, N, K);

// Q8_0 × Q8_1 GEMM (更高精度)
gemm_q8_0_q8_1(d_weight, d_activation, d_output, M, N, K);
```

## 测试

```bash
# 测试所有量化格式
make test-gemm-all

# 输出示例:
# GEMM_Q4_0_Q8_1: PASSED (NMSE 0.89%)
# GEMM_Q4_1_Q8_1: PASSED (NMSE 0.92%)
# GEMM_Q5_0_Q8_1: PASSED (NMSE 0.45%)
# GEMM_Q5_1_Q8_1: PASSED (NMSE 0.48%)
# GEMM_Q8_0_Q8_1: PASSED (NMSE 0.02%)
```

## 精度 vs 压缩率

| 格式 | 压缩率 (vs FP32) | 典型 NMSE |
|------|-----------------|-----------|
| Q8_0 | 3.76x | < 0.1% |
| Q5_1 | 5.33x | ~ 0.5% |
| Q5_0 | 5.82x | ~ 0.5% |
| Q4_1 | 6.4x | ~ 1% |
| Q4_0 | 7.1x | ~ 1% |

选择建议:
- 高精度需求: Q8_0 或 Q5_x
- 平衡选择: Q4_1 (非对称更好处理偏斜分布)
- 极致压缩: Q4_0
