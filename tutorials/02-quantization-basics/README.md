# 教程 02: 量化基础

## 学习目标

- 理解为什么需要量化
- 掌握对称量化和非对称量化
- 理解 Q4_0 和 Q8_1 格式
- 实现简单的量化和反量化

## 为什么需要量化？

### 内存瓶颈

LLM 推理主要受内存带宽限制:

```
模型参数加载时间 = 模型大小 / 内存带宽

例如 LLaMA-7B:
- FP16: 14GB / 500 GB/s = 28ms/token (理论最优)
- Q4_0: 3.5GB / 500 GB/s = 7ms/token

量化带来 4x 加速!
```

### 量化的代价

- 精度损失 (通常可接受)
- 需要额外的计算 (反量化)
- 更复杂的 kernel 实现

## 对称量化

### 原理

```
将 FP32 范围 [-max_abs, +max_abs] 映射到 INT8 范围 [-127, 127]

量化: q = round(x / scale)
反量化: x = q * scale

其中: scale = max(|x|) / 127
```

### 代码实现

```cpp
// 对称量化到 INT8
void quantize_symmetric(const float* x, int8_t* q, float* scale, int n) {
    // 1. 找最大绝对值
    float max_abs = 0.0f;
    for (int i = 0; i < n; i++) {
        if (fabsf(x[i]) > max_abs) {
            max_abs = fabsf(x[i]);
        }
    }

    // 2. 计算缩放因子
    *scale = max_abs / 127.0f;
    float inv_scale = (*scale > 0) ? (127.0f / max_abs) : 0.0f;

    // 3. 量化
    for (int i = 0; i < n; i++) {
        q[i] = (int8_t)roundf(x[i] * inv_scale);
    }
}

// 反量化
void dequantize_symmetric(const int8_t* q, float scale, float* x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = q[i] * scale;
    }
}
```

## 非对称量化

### 原理

```
将 FP32 范围 [min, max] 映射到 UINT8 范围 [0, 255]

量化: q = round((x - min) / scale)
反量化: x = q * scale + min

其中:
  scale = (max - min) / 255
  zero_point = round(-min / scale)
```

非对称量化可以更好地处理分布不对称的数据。

## Q4_0 格式详解

### 内存布局

```cpp
typedef struct {
    half d;              // 缩放因子 (2 bytes)
    uint8_t qs[16];      // 32 个 4-bit 值 (16 bytes)
} block_q4_0;            // 总计 18 bytes

// 每字节打包 2 个 4-bit 值
// qs[i] = (high_nibble << 4) | low_nibble
```

### 量化过程

```cpp
// Q4_0 使用对称量化，范围 [0, 15]，偏移 8
// 实际范围: [-8, 7]

void quantize_q4_0(const float* x, block_q4_0* dst, int n) {
    const int block_size = 32;
    const int num_blocks = n / block_size;

    for (int b = 0; b < num_blocks; b++) {
        const float* block = x + b * block_size;

        // 1. 找最大绝对值
        float max_abs = 0.0f;
        for (int i = 0; i < 32; i++) {
            if (fabsf(block[i]) > max_abs) {
                max_abs = fabsf(block[i]);
            }
        }

        // 2. 计算缩放因子 (映射到 [-7, 7])
        float scale = max_abs / 7.0f;
        dst[b].d = __float2half(scale);

        float inv_scale = (scale > 0) ? (1.0f / scale) : 0.0f;

        // 3. 量化并打包
        for (int i = 0; i < 16; i++) {
            // 量化两个值
            int q0 = (int)roundf(block[i*2 + 0] * inv_scale);
            int q1 = (int)roundf(block[i*2 + 1] * inv_scale);

            // 限制到 [-8, 7] 并加上偏移 8
            q0 = (q0 < -8) ? 0 : ((q0 > 7) ? 15 : (q0 + 8));
            q1 = (q1 < -8) ? 0 : ((q1 > 7) ? 15 : (q1 + 8));

            // 打包到一个字节
            dst[b].qs[i] = (q1 << 4) | q0;
        }
    }
}
```

### 反量化过程

```cpp
void dequantize_q4_0(const block_q4_0* src, float* dst, int n) {
    const int block_size = 32;
    const int num_blocks = n / block_size;

    for (int b = 0; b < num_blocks; b++) {
        float scale = __half2float(src[b].d);
        float* block = dst + b * block_size;

        for (int i = 0; i < 16; i++) {
            uint8_t packed = src[b].qs[i];

            // 解包并减去偏移 8
            int q0 = (packed & 0x0F) - 8;
            int q1 = ((packed >> 4) & 0x0F) - 8;

            // 反量化
            block[i*2 + 0] = q0 * scale;
            block[i*2 + 1] = q1 * scale;
        }
    }
}
```

## Q8_1 格式详解

### 内存布局

```cpp
typedef struct {
    half2 ds;            // d (缩放) + s (求和) (4 bytes)
    int8_t qs[32];       // 32 个 8-bit 值 (32 bytes)
} block_q8_1;            // 总计 36 bytes
```

### 为什么需要求和 s？

在 Q4_0 × Q8_1 点积中:

```
result = Σ (q_w - 8) * d_w * q_a * d_a
       = d_w * d_a * Σ (q_w - 8) * q_a
       = d_w * d_a * (Σ q_w * q_a - 8 * Σ q_a)
```

其中 `Σ q_a` 需要从原始值计算: `Σ q_a ≈ Σ x_a / d_a`

所以存储 `s = Σ x_a` 允许精确计算:

```
result = d_w * (d_a * sumi - 8 * s)
```

### 量化过程

```cpp
void quantize_q8_1(const float* x, block_q8_1* dst, int n) {
    const int block_size = 32;
    const int num_blocks = n / block_size;

    for (int b = 0; b < num_blocks; b++) {
        const float* block = x + b * block_size;

        // 1. 计算最大绝对值和求和
        float max_abs = 0.0f;
        float sum = 0.0f;
        for (int i = 0; i < 32; i++) {
            if (fabsf(block[i]) > max_abs) {
                max_abs = fabsf(block[i]);
            }
            sum += block[i];
        }

        // 2. 计算缩放因子
        float scale = max_abs / 127.0f;
        float inv_scale = (scale > 0) ? (1.0f / scale) : 0.0f;

        // 3. 存储 d 和 s
        dst[b].ds = make_half2(__float2half(scale), __float2half(sum));

        // 4. 量化
        for (int i = 0; i < 32; i++) {
            dst[b].qs[i] = (int8_t)roundf(block[i] * inv_scale);
        }
    }
}
```

## 量化误差分析

### NMSE (Normalized Mean Squared Error)

```cpp
float compute_nmse(const float* original, const float* dequantized, int n) {
    double sum_sq_err = 0.0;
    double sum_sq_ref = 0.0;

    for (int i = 0; i < n; i++) {
        float diff = dequantized[i] - original[i];
        sum_sq_err += diff * diff;
        sum_sq_ref += original[i] * original[i];
    }

    return (float)(sum_sq_err / sum_sq_ref);
}
```

### 典型量化误差

| 格式 | 位宽 | 典型 NMSE |
|------|------|-----------|
| Q8_0 | 8.5 bits/value | < 0.01% |
| Q4_0 | 4.5 bits/value | ~ 1% |
| Q2_K | 2.5 bits/value | ~ 5% |

## 练习

1. 实现 Q4_1 量化 (非对称)
2. 比较 Q4_0 和 Q4_1 在不同数据分布上的误差
3. 实现 Q8_1 的 CUDA 量化 kernel

## 下一步

- [教程 03: 朴素 GEMM](../03-naive-gemm/README.md)
