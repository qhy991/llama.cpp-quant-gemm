# Kernel 实现指南: w4a16_q4_0_fp32

**Family:** quant_gemm
**Version:** 1.0.0

Quantized GEMM with Q4_0 weights and FP32 activations. Simpler than W4A8 as no activation quantization needed.

---

## 概述

本指南描述如何实现 `w4a16_q4_0_fp32` kernel。该 kernel 是量化矩阵乘法(GEMM)的一个变体。

## 目录

1. [Kernel 函数签名](#kernel-函数签名)
2. [输入格式](#输入格式)
3. [输出格式](#输出格式)
4. [数学公式](#数学公式)
5. [Pybind11 集成](#pybind11-集成)
6. [测试验证](#测试验证)
7. [实现检查清单](#实现检查清单)

---

## Kernel 函数签名

### 必须使用的函数名和签名

```cpp
__global__ void gemm_q4_0_fp32(
    const uint8_t* weight,
    const float* activation,
    float* output,
    int M,
    int N,
    int K
)
```

**参数说明:**

- **M**: Batch dimension
  - 默认值: N/A
  - 约束: M >= 1

- **N**: Output features
  - 默认值: 4096
  - 约束: 无

- **K**: Input features
  - 默认值: 4096
  - 约束: K % 32 == 0


## 输入格式

### Weight

**Data Type:** `block_q4_0`

**Format:** Q4_0

4-bit quantization with scale. Each block of 32 values stored in 18 bytes.

**Memory Layout:**
```

Block layout (18 bytes per block, 32 values):
  Bytes 0-1:   scale (fp16)
  Bytes 2-17:  packed 4-bit values (16 bytes = 128 bits = 32 values)

Memory layout: [scale][q0][q1]...[q31] where each qi is 4 bits

```

**Dequantization Formula:**
```cpp
value = (q - 8) * scale
```

**Dequantization Code:**
```cpp

// Q4_0 dequantization
__half scale = *(__half*)&block[0];  // First 2 bytes
uint8_t packed = block[i/2 + 2];     // Data starts at byte 2
uint8_t q = (i % 2 == 0) ? (packed & 0x0F) : (packed >> 4);
float value = (float(q) - 8.0f) * __half2float(scale);

```

**Shape:** `[4096, 128 (computed as K/32), 18]`

Q4_0 quantized weight tensor


### Activation

**Data Type:** `float32`

**Format:** FP32

Standard 32-bit floating point

**Memory Layout:**
```
Standard FP32 (4 bytes per value)
```

**Dequantization Formula:**
```cpp
value = x
```

**Dequantization Code:**
```cpp
// No dequantization needed for FP32
float value = x;
```

**Shape:** `[1, 4096]`

FP32 activation tensor


### 内存布局约定

- 所有输入都是行优先 (row-major) 存储
- K 维度必须是 32 的倍数 (量化 block size)
- 对于量化数据，最后一个维度包含完整的 block 字节数

---

## 输出格式

### Output

**Data Type:** `float32`
**Shape:** `MxN`

Output tensor

**注意:** 输出是行优先存储的 [M, N] 矩阵。

## 数学公式

### 高层公式
```
C[m,n] = sum_k(A[m,k] * dequant(B[n,k]))
```

### 反量化
```cpp
w[i] = (qs[i] - 8) * d
```

---

## ⚠️ 常见错误和陷阱

**请仔细阅读本节以避免常见的实现错误！**

### 🚨 CRITICAL: Q4_0 Packing Format

**Q4_0 使用 SPLIT-BY-16 打包，不是连续对！**

✅ **正确的理解:**
```
byte[0]  = weight[0]  (low nibble) | weight[16] (high nibble)
byte[1]  = weight[1]  (low nibble) | weight[17] (high nibble)
...
byte[15] = weight[15] (low nibble) | weight[31] (high nibble)
```

❌ **错误的理解 (常见错误):**
```
byte[0] = weight[0] (low) | weight[1] (high)  // WRONG!
byte[1] = weight[2] (low) | weight[3] (high)  // WRONG!
```

✅ **正确的解包代码:**
```cpp
for (int i = 0; i < 16; i++) {
    uint8_t packed = data_ptr[i];
    
    // Low nibble -> weight[i]
    uint8_t q0 = packed & 0x0F;
    float w0 = (float(q0) - 8.0f) * scale;
    sum += activation[k_start + i] * w0;
    
    // High nibble -> weight[i + 16]
    uint8_t q1 = packed >> 4;
    float w1 = (float(q1) - 8.0f) * scale;
    sum += activation[k_start + i + 16] * w1;
}
```

**验证方法:**
1. 先测试 quantize -> dequantize 往返
2. 使用简单固定值测试 (weight=0.5, activation=2.0)
3. 确保 NMSE < 0.05

### 🚨 Dimension Conventions

**本 kernel 使用 FP32 activation (w4a16 约定):**

```cpp
// Kernel 计算: C[M, N] = A[M, K] @ W[N, K]^T
// 调用约定: kernel(weight, activation, M, N, K)
// 输出直接是 [M, N]，无需转置
```

### 🚨 Memory and Performance Pitfalls

1. **Integer Overflow**
   ```cpp
   // ❌ WRONG: 可能溢出
   int offset = n * num_blocks * 18;
   
   // ✅ CORRECT: 使用 long long
   long long offset = (long long)(n * num_blocks) * 18;
   ```

2. **Memory Alignment (float4)**
   ```cpp
   // float4 需要 16-byte 对齐
   // 如果地址未对齐，会退化为 4 次单独读取
   // 确保 K 是 4 的倍数且起始地址对齐
   ```

3. **Quantization Offset**
   ```cpp
   // Q4_0: 值偏移 8 (范围 [0,15] -> [-8,7])
   float w = (float(q) - 8.0f) * scale;  // 必须减 8!
   ```

4. **Block Size Assumptions**
   ```cpp
   // 不要硬编码 32，使用常量
   int num_blocks = K / QK4_0;  // QK4_0 = 32
   ```

### ✅ Testing Best Practices

**测试顺序 (从简单到复杂):**

1. **Quantization Roundtrip**
   ```python
   x -> quantize -> dequantize -> x'
   max_error = (x - x').abs().max()
   assert max_error < 1.0  # Q4_0 有显著误差
   ```

2. **Fixed Values**
   ```python
   weight = torch.full((N, K), 0.5)
   activation = torch.full((M, K), 2.0)
   expected_output = K * 0.5 * 2.0 = K
   ```

3. **Different Data Patterns**
   - All zeros
   - All ones
   - Positive only (torch.rand)
   - Mixed signs (torch.randn) ← 最容易暴露 bug

4. **NMSE Thresholds**
   - Q4_0: NMSE < 0.05 (5%)
   - Q8_1: NMSE < 0.01 (1%)
   - FP16: NMSE < 0.001 (0.1%)

### 📚 Reference Implementations

**在实现前，请参考:**

1. **Dequantization Reference**
   - 查看 `dequantize_q4_0_kernel` in gemm_ops.cu
   - 确保你的解包逻辑与之一致

2. **llama.cpp Q4_0 Format**
   - https://github.com/ggerganov/llama.cpp/blob/master/ggml.c
   - 搜索 `dequantize_row_q4_0`

3. **Working Kernels**
   - w4a8_q4_0_q8_1: 参考量化 activation 的实现
   - w4a16_q4_0_fp32: 参考 FP32 activation 的实现


---

## Pybind11 集成

在 `bindings.cpp` 中添加以下声明:

```cpp
// Include header (if separate)
// #include "kernels/kernel.cu"

// Binding declaration
m.def("gemm_q4_0_fp32",
    [](py::array_t<uint8_t>, py::array_t<float>, py::array_t<float>) {
        // TODO: Implement buffer_info extraction and kernel launch
        // See w4a8_q4_0_q8_1 variant for reference
        py::gil_scoped_release release;
        // Launch kernel here
    },
    py::arg("weight", "activation"),
    "Kernel implementation for w4a16_q4_0_fp32"
);
```

**重要提示:**
- 函数名必须与 `spec.json` 中的 `kernel.entry_point` 一致
- 参数顺序必须与 spec 中定义的顺序一致
- 必须释放 GIL (`py::gil_scoped_release`)


---

## 测试框架验证

### 验证流程

```
1. 测试框架生成随机输入数据
2. 调用 reference.py 生成正确输出
3. 调用你的 kernel 生成实际输出
4. 比较两者并计算 NMSE
5. 验证 NMSE 是否 ≤ 0.05
```

### 精度要求

- **指标:** NMSE
- **阈值:** 0.05

**NMSE 计算公式:**
```python
nmse = np.mean((ref - actual) ** 2) / np.mean(ref ** 2)
```

### 测试配置

- `single`: M=1, N=4096, K=4096
- `small_batch`: M=4, N=4096, K=4096
- `medium_batch`: M=128, N=4096, K=4096

### 参考实现

位置: `reference.py:run`

### 验收标准

1. **正确性**: 所有测试配置的 NMSE ≤ 0.05
2. **性能**: 需要达到最低性能目标
3. **稳定性**: 多次运行结果一致


---

## 实现检查清单

### 开始实现前

- [ ] 阅读 `KERNEL_IMPLEMENTATION_GUIDE.md` 了解 GEMM 基础
- [ ] 理解本指南中的所有输入格式和数学公式
- [ ] 阅读 `kernel.cu` 中的参考实现（如果存在）

### 实现步骤

1. [ ] **创建文件**: `operators/quant_gemm/variants/w4a16_q4_0_fp32/kernel.cu`
2. [ ] **实现 kernel 函数**:
   ```cpp
   __global__ void gemm_q4_0_fp32(
    const uint8_t* weight,
    const float* activation,
    float* output,
    int M,
    int N,
    int K
)
   {
       // TODO: 实现 kernel 逻辑
   }
   ```
3. [ ] **添加 pybind11 声明**: 在 `bindings.cpp` 中注册函数
4. [ ] **编译验证**: `python setup.py build_ext --inplace`
5. [ ] **运行测试**: `python test_operator.py w4a16_q4_0_fp32 operators/quant_gemm/variants/w4a16_q4_0_fp32`

### 输入参数

  - weight: block_q4_0, shape=NxK/32x18
  - activation: float32, shape=MxK

### 输出参数

  - output: float32, shape=MxN

### 验收标准

- [ ] 所有测试配置通过
- [ ] NMSE ≤ 0.05
- [ ] 无内存泄漏
- [ ] 代码符合项目规范


---

## 参考资源

- **Variant 目录:** `python/operators/quant_gemm/variants/w4a16_q4_0_fp32`
- **Spec 文件:** `python/operators/quant_gemm/variants/w4a16_q4_0_fp32/spec.json`
- **Kernel 文件:** `python/operators/quant_gemm/variants/w4a16_q4_0_fp32/kernel.cu`
- **Reference 实现:** `python/operators/quant_gemm/variants/w4a16_q4_0_fp32/reference.py:run`

---

## 快速开始

```bash
# 1. 创建 kernel 文件
touch operators/quant_gemm/variants/w4a16_q4_0_fp32/kernel.cu

# 2. 实现 kernel (参考上面的函数签名)

# 3. 在 bindings.cpp 中添加声明

# 4. 编译
python setup.py build_ext --inplace

# 5. 测试
python test_operator.py w4a16_q4_0_fp32 operators/quant_gemm/variants/w4a16_q4_0_fp32
```