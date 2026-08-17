# Kernel Implementation Guide

本文档详细说明如何实现一个符合测试框架要求的 GEMM kernel。以 `w4a16_q4_0_fp32` 为例。

---

## 目录

1. [概述](#概述)
2. [JSON Schema 要求](#json-schema-要求)
3. [Kernel 函数签名](#kernel-函数签名)
4. [输入格式详解](#输入格式详解)
5. [数学公式](#数学公式)
6. [Pybind11 集成](#pybind11-集成)
7. [测试框架验证流程](#测试框架验证流程)
8. [性能要求](#性能要求)
9. [完整实现清单](#完整实现清单)
10. [参考实现](#参考实现)

---

## 概述

### 任务：实现 W4A16 GEMM Kernel

**算子名称**: `w4a16_q4_0_fp32`
**功能**: Q4_0 量化权重 × FP32 激活值的矩阵乘法
**位置**: `operators/quant_gemm/variants/w4a16_q4_0_fp32/`

### 已提供的文件

```
operators/quant_gemm/variants/w4a16_q4_0_fp32/
├── spec.json          ✅ 已提供：算子规格定义
└── reference.py       ✅ 已提供：Python 参考实现（仅用于理解，不用于测试）
```

### 需要实现的文件

```
csrc/
├── gemm_q4_0_fp32.cu  ❌ 需要实现：CUDA kernel
└── bindings.cpp       ❌ 需要添加：Pybind11 绑定
```

---

## JSON Schema 要求

### 完整 spec.json

```json
{
  "name": "w4a16_q4_0_fp32",
  "family": "quant_gemm",
  "version": "1.0.0",
  "description": "Quantized GEMM with Q4_0 weights and FP32 activations",

  "kernel": {
    "file": "kernel.cu",
    "entry_point": "gemm_q4_0_fp32"
  },

  "inputs": {
    "weight": {
      "dtype": "block_q4_0",
      "shape": ["N", "K/32", 18],
      "description": "Q4_0 quantized weight tensor",
      "quantizer": "quantize_q4_0"
    },
    "activation": {
      "dtype": "float32",
      "shape": ["M", "K"],
      "description": "FP32 activation tensor"
    }
  },

  "outputs": {
    "output": {
      "dtype": "float32",
      "shape": ["M", "N"],
      "description": "Output tensor"
    }
  },

  "params": {
    "M": {"type": "int", "description": "Batch dimension", "constraint": "M >= 1"},
    "N": {"type": "int", "default": 4096, "description": "Output features"},
    "K": {"type": "int", "default": 4096, "description": "Input features", "constraint": "K % 32 == 0"}
  },

  "test_configs": [
    {"name": "single", "M": 1, "N": 4096, "K": 4096},
    {"name": "small_batch", "M": 4, "N": 4096, "K": 4096},
    {"name": "medium_batch", "M": 128, "N": 4096, "K": 4096}
  ],

  "accuracy": {
    "metric": "nmse",
    "threshold": 0.05
  }
}
```

### 关键要求解读

| 字段 | 含义 | 要求 |
|------|------|------|
| `kernel.entry_point` | Pybind 函数名 | 必须注册为 `gemm_q4_0_fp32` |
| `inputs.weight.dtype` | 权重类型 | Q4_0 量化格式（18 bytes/block） |
| `inputs.weight.shape` | 权重形状 | `[N, K/32, 18]` |
| `inputs.activation.dtype` | 激活值类型 | FP32（不需要量化） |
| `inputs.activation.shape` | 激活值形状 | `[M, K]` |
| `outputs.output.shape` | 输出形状 | `[M, N]` |
| `accuracy.threshold` | 精度阈值 | NMSE ≤ 0.05 |

---

## Kernel 函数签名

### C++ 函数签名

```cpp
torch::Tensor gemm_q4_0_fp32(
    torch::Tensor weight,      // [N, K/32, 18] uint8, Q4_0 quantized
    torch::Tensor activation,  // [M, K] float32
    int64_t N,                 // Output features
    int64_t M,                 // Batch size
    int64_t K                  // Input features
);
```

### 参数说明

| 参数 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `weight` | `torch::Tensor` | `[N, K/32, 18]` | Q4_0 量化权重，dtype=uint8 |
| `activation` | `torch::Tensor` | `[M, K]` | FP32 激活值 |
| `N` | `int64_t` | - | 输出特征维度（权重行数） |
| `M` | `int64_t` | - | 批次大小（激活值行数） |
| `K` | `int64_t` | - | 输入特征维度（必须是 32 的倍数） |
| **返回值** | `torch::Tensor` | `[M, N]` | FP32 输出 |

### ⚠️ 重要：维度约定

测试框架使用以下约定：

```
数学公式：output[M, N] = activation[M, K] @ weight[N, K]^T
```

但 kernel 接收的参数顺序是 `(weight, activation, N, M, K)`，并且：
- Kernel 内部计算：`C[N, M] = W[N, K] @ A[M, K]^T`
- 测试框架会自动转置输出：`output = kernel(...).T`

**你只需要实现标准的 GEMM，测试框架会处理转置。**

---

## 输入格式详解

### Q4_0 Block 格式

每个 Q4_0 block 包含 32 个 4-bit 量化值，存储在 18 bytes 中：

```
Byte Layout (18 bytes total):
┌─────────┬─────────────────────────────────────┐
│ 0-1     │ 2-17                                │
├─────────┼─────────────────────────────────────┤
│ d (fp16)│ qs[16] (16 bytes, 32 packed 4-bit)  │
└─────────┴─────────────────────────────────────┘

d: scale factor (half precision, 2 bytes)
qs: 32 packed 4-bit values (16 bytes)
    - qs[i] = (q_high << 4) | q_low
    - q_low = qs[i] & 0x0F  (values 0-15)
    - q_high = qs[i] >> 4   (values 0-15)
```

### 反量化公式

```
原始值 = (量化值 - 8) * scale
```

对于 block 中的第 i 个值：
```cpp
float d = __half2float(*(half*)&block[0]);  // 读取 scale
uint8_t q = block[2 + i/2];                  // 读取 packed byte
uint8_t q_val = (i % 2 == 0) ? (q & 0x0F) : (q >> 4);  // 解包
float dequant = (float(q_val) - 8.0f) * d;  // 反量化
```

### Weight Tensor 布局

```
weight: [N, K/32, 18]
        │  │     └─ 每个 block 18 bytes
        │  └─ K/32 个 blocks（每个 block 32 个值）
        └─ N 行（输出特征数）

示例：N=4096, K=4096
  weight.shape = [4096, 128, 18]
  weight.dtype = torch.uint8
  weight.device = cuda
```

### Activation Tensor 布局

```
activation: [M, K]
            │  └─ K 个 FP32 值
            └─ M 行（批次大小）

示例：M=1, K=4096
  activation.shape = [1, 4096]
  activation.dtype = torch.float32
  activation.device = cuda
```

---

## 数学公式

### 高层公式

```
output[m, n] = Σ(k=0 to K-1) activation[m, k] * dequant(weight[n, k])
```

### Block-wise 计算

```
K = num_blocks * 32

output[m, n] = Σ(b=0 to num_blocks-1) Σ(i=0 to 31)
               activation[m, b*32+i] * dequant(weight[n, b, i])
```

其中：
```
dequant(weight[n, b, i]) = (q[i] - 8) * d_b
```

- `d_b`: block b 的 scale factor
- `q[i]`: block b 中第 i 个 4-bit 量化值（0-15）

### 伪代码

```python
for m in range(M):
    for n in range(N):
        acc = 0.0
        for b in range(K // 32):
            # 读取 weight block
            d = weight[n, b, 0:2]  # scale (fp16)
            qs = weight[n, b, 2:18]  # 32 packed 4-bit values

            # 计算 block 内积
            for i in range(32):
                q_val = unpack_4bit(qs, i)  # 0-15
                w_dequant = (q_val - 8) * d
                a_val = activation[m, b*32 + i]
                acc += a_val * w_dequant

        output[m, n] = acc
```

---

## Pybind11 集成

### 1. 在 `csrc/bindings.cpp` 中添加声明

```cpp
// 在文件顶部添加声明
torch::Tensor gemm_q4_0_fp32(
    torch::Tensor weight,
    torch::Tensor activation,
    int64_t N,
    int64_t M,
    int64_t K
);
```

### 2. 在 `PYBIND11_MODULE` 中注册

```cpp
PYBIND11_MODULE(_C, m) {
    // ... 其他绑定 ...

    m.def("gemm_q4_0_fp32", &gemm_q4_0_fp32, "W4A16 GEMM (Q4_0 x FP32)",
          py::arg("weight"),
          py::arg("activation"),
          py::arg("N"),
          py::arg("M"),
          py::arg("K"));
}
```

### 3. 函数命名要求

⚠️ **必须使用 spec.json 中定义的 `entry_point` 名称**

```json
"kernel": {
    "entry_point": "gemm_q4_0_fp32"  // ← 必须匹配
}
```

测试框架会尝试以下名称（按顺序）：
1. `gemm_q4_0_fp32`（来自 entry_point）
2. `gemm_w4a16_q4_0_fp32`（基于 name）
3. `w4a16_q4_0_fp32`（直接使用 name）

**推荐使用第一个名称以确保兼容性。**

---

## 测试框架验证流程

### 测试流程图

```
1. 生成随机 FP32 输入
   ├─ weight_fp32: [N, K] ~ N(0, 1)
   └─ activation_fp32: [M, K] ~ N(0, 1)
          ↓
2. 量化 weight
   └─ weight_q4_0 = quantize_q4_0(weight_fp32)  → [N, K/32, 18]
          ↓
3. 运行 reference (FP32 matmul)
   └─ ref_output = activation_fp32 @ weight_fp32.T  → [M, N]
          ↓
4. 运行你的 kernel
   └─ output = gemm_q4_0_fp32(weight_q4_0, activation_fp32, N, M, K).T  → [M, N]
          ↓
5. 计算误差
   └─ NMSE = MSE(output, ref_output) / Var(ref_output)
          ↓
6. 判断通过
   └─ PASS if NMSE ≤ 0.05
```

### NMSE 计算公式

```python
def compute_nmse(output, reference):
    mse = torch.mean((output - reference) ** 2)
    ref_var = torch.var(reference)
    return mse / ref_var
```

### 测试配置

测试框架会运行以下配置：

| Config | M | N | K | 说明 |
|--------|---|---|---|------|
| single | 1 | 4096 | 4096 | 单样本推理 |
| small_batch | 4 | 4096 | 4096 | 小批次 |
| medium_batch | 128 | 4096 | 4096 | 中等批次 |

### 运行测试

```bash
# 方式 1：使用简单测试工具
python test_operator.py w4a16_q4_0_fp32 operators/quant_gemm/variants/w4a16_q4_0_fp32

# 方式 2：使用完整框架
python test_operators_framework.py

# 方式 3：带 benchmark
python test_operator.py w4a16_q4_0_fp32 operators/quant_gemm/variants/w4a16_q4_0_fp32 --benchmark
```

### 预期输出

```
============================================================
 Testing: w4a16_q4_0_fp32
============================================================
Folder: operators/quant_gemm/variants/w4a16_q4_0_fp32
Module: quant_gemm._C
Device: cuda

Configs: 3

------------------------------------------------------------
 Correctness Tests
------------------------------------------------------------
[PASS] single: nmse=2.3456e-03 (threshold=0.05)
[PASS] small_batch: nmse=2.4123e-03 (threshold=0.05)
[PASS] medium_batch: nmse=2.3987e-03 (threshold=0.05)

Results: 3 passed, 0 failed
============================================================
```

---

## 性能要求

### 最低性能目标

基于 RTX 5070 Laptop GPU 的参考性能（来自 w4a8 kernel）：

| Config | M | N | K | 目标性能 |
|--------|---|---|---|----------|
| single | 1 | 4096 | 4096 | > 300 GFLOPS |
| small_batch | 4 | 4096 | 4096 | > 700 GFLOPS |
| medium_batch | 128 | 4096 | 4096 | > 1000 GFLOPS |

### 性能优化建议

1. **使用 Tensor Cores**（如果可用）
   - FP16 accumulation
   - WMMA API 或 mma.sync

2. **内存访问优化**
   - Coalesced global memory access
   - Shared memory tiling
   - Register blocking

3. **Block 级优化**
   - 每个 block 处理多个输出元素
   - 向量化加载（float4, uint4）

4. **Warp 级优化**
   - Warp shuffle 用于 reduction
   - 避免 warp divergence

---

## 完整实现清单

### ✅ 实现前检查清单

- [ ] 理解 Q4_0 格式（18 bytes/block）
- [ ] 理解反量化公式：`(q - 8) * d`
- [ ] 理解维度约定：kernel 输出 `[N, M]`，框架转置为 `[M, N]`
- [ ] 准备 CUDA 开发环境

### ✅ 实现步骤

1. **创建 CUDA kernel 文件**
   ```bash
   touch csrc/gemm_q4_0_fp32.cu
   ```

2. **实现 CUDA kernel**
   - [ ] 实现 Q4_0 反量化逻辑
   - [ ] 实现 GEMM 计算
   - [ ] 优化内存访问
   - [ ] 添加边界检查

3. **实现 C++ wrapper**
   - [ ] 输入验证（shape, dtype, device）
   - [ ] 分配输出 tensor
   - [ ] 调用 CUDA kernel
   - [ ] 返回结果

4. **添加 Pybind11 绑定**
   - [ ] 在 `bindings.cpp` 中声明函数
   - [ ] 在 `PYBIND11_MODULE` 中注册
   - [ ] 使用正确的函数名：`gemm_q4_0_fp32`

5. **编译和测试**
   ```bash
   # 编译
   python setup.py build_ext --inplace

   # 测试
   python test_operator.py w4a16_q4_0_fp32 operators/quant_gemm/variants/w4a16_q4_0_fp32
   ```

6. **性能优化**
   - [ ] 运行 benchmark
   - [ ] 使用 Nsight Compute 分析
   - [ ] 优化瓶颈
   - [ ] 验证性能目标

### ✅ 验收标准

- [ ] 所有测试配置通过（NMSE ≤ 0.05）
- [ ] 性能达到目标（> 300 GFLOPS for M=1）
- [ ] 无内存泄漏
- [ ] 无 CUDA 错误
- [ ] 代码可读性良好

---

## 参考实现

### 现有 W4A8 Kernel 参考

可以参考已实现的 `gemm_q4_0_q8_1` kernel：

```bash
# 查看现有实现
cat csrc/gemm_q4_0_q8_1.cu
```

主要区别：
- W4A8: 两个输入都需要反量化
- W4A16: 只有 weight 需要反量化，activation 是 FP32

### 简化的 CUDA Kernel 框架

```cpp
// csrc/gemm_q4_0_fp32.cu

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// CUDA kernel
__global__ void gemm_q4_0_fp32_kernel(
    const uint8_t* __restrict__ weight,  // [N, K/32, 18]
    const float* __restrict__ activation, // [M, K]
    float* __restrict__ output,           // [N, M]
    int64_t N,
    int64_t M,
    int64_t K
) {
    // TODO: 实现 kernel
    // 1. 计算输出位置 (n, m)
    // 2. 遍历 K 维度的所有 blocks
    // 3. 对每个 block：
    //    - 读取 scale (fp16)
    //    - 读取 32 个 4-bit 值
    //    - 反量化并累加
    // 4. 写入输出
}

// C++ wrapper
torch::Tensor gemm_q4_0_fp32(
    torch::Tensor weight,
    torch::Tensor activation,
    int64_t N,
    int64_t M,
    int64_t K
) {
    // 输入验证
    TORCH_CHECK(weight.device().is_cuda(), "weight must be on CUDA");
    TORCH_CHECK(activation.device().is_cuda(), "activation must be on CUDA");
    TORCH_CHECK(weight.dtype() == torch::kUInt8, "weight must be uint8");
    TORCH_CHECK(activation.dtype() == torch::kFloat32, "activation must be float32");

    // 分配输出
    auto output = torch::zeros({N, M},
        torch::TensorOptions().dtype(torch::kFloat32).device(weight.device()));

    // 启动 kernel
    dim3 block(/* TODO */);
    dim3 grid(/* TODO */);

    gemm_q4_0_fp32_kernel<<<grid, block>>>(
        weight.data_ptr<uint8_t>(),
        activation.data_ptr<float>(),
        output.data_ptr<float>(),
        N, M, K
    );

    return output;
}
```

### Python 测试脚本

```python
# test_my_kernel.py
import torch
import quant_gemm._C as _C

# 参数
M, N, K = 1, 4096, 4096

# 生成输入
weight_fp32 = torch.randn(N, K, dtype=torch.float32, device='cuda')
activation = torch.randn(M, K, dtype=torch.float32, device='cuda')

# 量化 weight
weight_q4_0 = _C.quantize_q4_0(weight_fp32)

# 运行 kernel
output = _C.gemm_q4_0_fp32(weight_q4_0, activation, N, M, K)
output = output.T  # 转置为 [M, N]

# 运行 reference
ref_output = torch.matmul(activation, weight_fp32.T)

# 计算误差
mse = torch.mean((output - ref_output) ** 2).item()
ref_var = torch.var(ref_output).item()
nmse = mse / ref_var

print(f"NMSE: {nmse:.6f}")
print(f"PASS" if nmse <= 0.05 else "FAIL")
```

---

## 常见问题

### Q1: 为什么 kernel 输出 `[N, M]` 而不是 `[M, N]`？

**A**: 这是为了优化内存访问。测试框架会自动转置输出。你只需要实现标准的 GEMM。

### Q2: 如何处理 K 不是 32 的倍数？

**A**: spec.json 中有约束 `"constraint": "K % 32 == 0"`，测试框架保证 K 是 32 的倍数。

### Q3: NMSE 阈值为什么是 0.05？

**A**: W4A16 比 W4A8 精度更高（activation 是 FP32），所以阈值更严格（0.05 vs 0.1）。

### Q4: 如何调试 CUDA kernel？

**A**:
```bash
# 使用小配置测试
python test_operator.py w4a16_q4_0_fp32 operators/quant_gemm/variants/w4a16_q4_0_fp32 \
    --config "M=1,N=32,K=128"

# 使用 cuda-gdb
cuda-gdb python
```

### Q5: 性能不达标怎么办？

**A**:
1. 使用 Nsight Compute 分析：`ncu --set full python test_operator.py ...`
2. 检查内存访问模式
3. 增加 occupancy
4. 使用 Tensor Cores

---

## 总结

### 核心要求

1. **函数签名**: `gemm_q4_0_fp32(weight, activation, N, M, K) -> Tensor[N, M]`
2. **输入格式**: weight 是 Q4_0 `[N, K/32, 18]`，activation 是 FP32 `[M, K]`
3. **数学公式**: `output[n,m] = Σ_k activation[m,k] * dequant(weight[n,k])`
4. **反量化**: `dequant(q) = (q - 8) * scale`
5. **精度要求**: NMSE ≤ 0.05
6. **性能要求**: > 300 GFLOPS (M=1, N=4096, K=4096)

### 下一步

1. 阅读本文档
2. 查看 `csrc/gemm_q4_0_q8_1.cu` 参考实现
3. 实现 `csrc/gemm_q4_0_fp32.cu`
4. 添加 Pybind11 绑定
5. 编译测试
6. 优化性能

祝实现顺利！🚀
