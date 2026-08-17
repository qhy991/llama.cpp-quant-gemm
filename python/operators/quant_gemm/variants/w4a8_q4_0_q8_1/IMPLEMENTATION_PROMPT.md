# Kernel 实现指南: w4a8_q4_0_q8_1

**Family:** quant_gemm
**Version:** 1.0.0

Quantized GEMM with Q4_0 weights and Q8_1 activations. Compatible with llama.cpp mmq kernels.

---

## 概述

本指南描述如何实现 `w4a8_q4_0_q8_1` kernel。该 kernel 是量化矩阵乘法(GEMM)的一个变体。

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
__global__ void gemm_q4_0_q8_1(
    const uint8_t* weight,
    const uint8_t* activation,
    float* output,
    int M,
    int N,
    int K
)
```

**参数说明:**

- **M**: Batch dimension (batch_size * seq_len)
  - 默认值: N/A
  - 约束: M >= 1

- **N**: Output features (model hidden_size)
  - 默认值: 4096
  - 约束: 无

- **K**: Input features (must be multiple of 32)
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

Q4_0 quantized weight tensor [output_features, num_blocks, 18 bytes/block]


### Activation

**Data Type:** `block_q8_1`

**Format:** Q8_1

8-bit quantization with scale and min. Each block of 32 values stored in 36 bytes.

**Memory Layout:**
```

Block layout (36 bytes per block, 32 values):
  Bytes 0-1:   scale (fp16)
  Bytes 2-3:   min (fp16)
  Bytes 4-35:  int8 values (32 bytes)

```

**Dequantization Formula:**
```cpp
value = q * scale + min
```

**Dequantization Code:**
```cpp

// Q8_1 dequantization
__half scale = *(__half*)&block[0];
__half min = *(__half*)&block[2];
int8_t q = block[i + 4];
float value = float(q) * __half2float(scale) + __half2float(min);

```

**Shape:** `[1, 128 (computed as K/32), 36]`

Q8_1 quantized activation tensor [batch, num_blocks, 36 bytes/block]


### 内存布局约定

- 所有输入都是行优先 (row-major) 存储
- K 维度必须是 32 的倍数 (量化 block size)
- 对于量化数据，最后一个维度包含完整的 block 字节数

---

## 输出格式

### Output

**Data Type:** `float32`
**Shape:** `MxN`

Output tensor [batch, output_features]

**注意:** 输出是行优先存储的 [M, N] 矩阵。

## 数学公式

### 高层公式
```
result = d_w * (d_a * sumi - 8.0f * s_a)
```

**解释:**

Q4_0 stores values with +8 offset. Compensation term -8*s_a corrects for this.


---

## Pybind11 集成

在 `bindings.cpp` 中添加以下声明:

```cpp
// Include header (if separate)
// #include "kernels/kernel.cu"

// Binding declaration
m.def("gemm_q4_0_q8_1",
    [](py::array_t<uint8_t>, py::array_t<uint8_t>, py::array_t<float>) {
        // TODO: Implement buffer_info extraction and kernel launch
        // See w4a8_q4_0_q8_1 variant for reference
        py::gil_scoped_release release;
        // Launch kernel here
    },
    py::arg("weight", "activation"),
    "Kernel implementation for w4a8_q4_0_q8_1"
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
5. 验证 NMSE 是否 ≤ 0.1
```

### 精度要求

- **指标:** NMSE
- **阈值:** 0.1

**NMSE 计算公式:**
```python
nmse = np.mean((ref - actual) ** 2) / np.mean(ref ** 2)
```

### 测试配置

- `single`: M=1, N=4096, K=4096
- `small_batch`: M=4, N=4096, K=4096
- `medium_batch`: M=128, N=4096, K=4096
- `large_batch`: M=4096, N=4096, K=4096

### 参考实现

位置: `reference.py:run`

### 验收标准

1. **正确性**: 所有测试配置的 NMSE ≤ 0.1
2. **性能**: 需要达到最低性能目标
3. **稳定性**: 多次运行结果一致


---

## 实现检查清单

### 开始实现前

- [ ] 阅读 `KERNEL_IMPLEMENTATION_GUIDE.md` 了解 GEMM 基础
- [ ] 理解本指南中的所有输入格式和数学公式
- [ ] 阅读 `kernel.cu` 中的参考实现（如果存在）

### 实现步骤

1. [ ] **创建文件**: `operators/quant_gemm/variants/w4a8_q4_0_q8_1/kernel.cu`
2. [ ] **实现 kernel 函数**:
   ```cpp
   __global__ void gemm_q4_0_q8_1(
    const uint8_t* weight,
    const uint8_t* activation,
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
5. [ ] **运行测试**: `python test_operator.py w4a8_q4_0_q8_1 operators/quant_gemm/variants/w4a8_q4_0_q8_1`

### 输入参数

  - weight: block_q4_0, shape=NxK/32x18
  - activation: block_q8_1, shape=MxK/32x36

### 输出参数

  - output: float32, shape=MxN

### 验收标准

- [ ] 所有测试配置通过
- [ ] NMSE ≤ 0.1
- [ ] 无内存泄漏
- [ ] 代码符合项目规范


---

## 参考资源

- **Variant 目录:** `operators/quant_gemm/variants/w4a8_q4_0_q8_1`
- **Spec 文件:** `operators/quant_gemm/variants/w4a8_q4_0_q8_1/spec.json`
- **Kernel 文件:** `operators/quant_gemm/variants/w4a8_q4_0_q8_1/kernel.cu`
- **Reference 实现:** `operators/quant_gemm/variants/w4a8_q4_0_q8_1/reference.py:run`

---

## 快速开始

```bash
# 1. 创建 kernel 文件
touch operators/quant_gemm/variants/w4a8_q4_0_q8_1/kernel.cu

# 2. 实现 kernel (参考上面的函数签名)

# 3. 在 bindings.cpp 中添加声明

# 4. 编译
python setup.py build_ext --inplace

# 5. 测试
python test_operator.py w4a8_q4_0_q8_1 operators/quant_gemm/variants/w4a8_q4_0_q8_1
```