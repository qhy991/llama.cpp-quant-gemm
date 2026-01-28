# llama.cpp 接口对齐指南

## 📋 当前状态

### 当前项目的接口

当前项目使用的是**直接CUDA kernel调用**，接口如下：

```cpp
// 当前接口（直接指针）
void gemm_w4a8_naive(
    const block_q8_1* A,      // 激活值 (Q8_1)
    const block_q4_0* B,      // 权重 (Q4_0)
    float* C,                 // 输出 (FP32)
    int M, int N, int K,      // 矩阵维度
    cudaStream_t stream = 0
);
```

### llama.cpp 的接口

llama.cpp 使用 **ggml_tensor** 抽象层，接口如下：

```cpp
// llama.cpp 接口（通过 ggml_tensor）
struct ggml_tensor * ggml_mul_mat(
    struct ggml_context * ctx,
    struct ggml_tensor * a,   // 激活值 tensor
    struct ggml_tensor * b    // 权重 tensor
);
```

## 🔄 接口差异分析

| 特性 | 当前项目 | llama.cpp |
|------|---------|-----------|
| **数据表示** | 直接指针 (`block_q8_1*`, `block_q4_0*`) | `ggml_tensor*` 结构体 |
| **内存管理** | 用户管理 (`CudaBuffer`) | ggml_context 管理 |
| **维度信息** | 显式参数 (M, N, K) | 存储在 tensor 中 (`ne[0]`, `ne[1]`, `ne[2]`) |
| **类型信息** | 编译时类型 (`block_q8_1`, `block_q4_0`) | 运行时类型 (`GGML_TYPE_Q8_1`, `GGML_TYPE_Q4_0`) |
| **调用方式** | 直接 kernel 调用 | 通过计算图 (`ggml_graph_compute`) |

## 🎯 接口对齐方案

### 方案 1: 适配层（推荐）

创建一个适配层，将 llama.cpp 的 `ggml_tensor` 转换为我们的接口：

```cpp
// 适配层：从 ggml_tensor 提取数据并调用我们的 kernel
void gemm_w4a8_from_ggml_tensor(
    const struct ggml_tensor * activation,  // Q8_1 tensor
    const struct ggml_tensor * weights,     // Q4_0 tensor
    struct ggml_tensor * output             // FP32 tensor
) {
    // 1. 提取维度
    int M = activation->ne[1];  // rows
    int K = activation->ne[0];  // cols
    int N = weights->ne[1];     // rows
    
    // 2. 提取数据指针
    const block_q8_1* A = (const block_q8_1*)activation->data;
    const block_q4_0* B = (const block_q4_0*)weights->data;
    float* C = (float*)output->data;
    
    // 3. 调用我们的 kernel
    gemm_w4a8_naive(A, B, C, M, N, K);
}
```

### 方案 2: 包装层

将我们的接口包装成 ggml 操作：

```cpp
// 包装层：将我们的 kernel 包装成 ggml 操作
struct ggml_tensor * ggml_mul_mat_q4_0_q8_1_custom(
    struct ggml_context * ctx,
    struct ggml_tensor * a,  // Q8_1
    struct ggml_tensor * b   // Q4_0
) {
    // 创建输出 tensor
    struct ggml_tensor * result = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, 
        a->ne[1], b->ne[1]  // M, N
    );
    
    // 调用我们的实现
    gemm_w4a8_from_ggml_tensor(a, b, result);
    
    return result;
}
```

### 方案 3: 直接替换 llama.cpp 的 kernel

在 llama.cpp 的 CUDA 后端中，直接使用我们的 kernel 实现：

```cpp
// 在 llama.cpp/ggml/src/ggml-cuda.cu 中
void ggml_cuda_mul_mat_q4_0_q8_1(
    const ggml_tensor * src0,  // Q4_0 weights
    const ggml_tensor * src1,  // Q8_1 activations
    ggml_tensor * dst          // FP32 output
) {
    // 使用我们的实现
    const block_q4_0* B = (const block_q4_0*)src0->data;
    const block_q8_1* A = (const block_q8_1*)src1->data;
    float* C = (float*)dst->data;
    
    int M = src1->ne[1];
    int N = src0->ne[1];
    int K = src0->ne[0];
    
    // 调用我们的 kernel
    gemm_w4a8_naive(A, B, C, M, N, K);
}
```

## 📝 数据结构兼容性

### ✅ 已兼容的部分

1. **量化格式结构体**：
   - `block_q4_0` - 与 llama.cpp 100% 兼容
   - `block_q8_0` - 与 llama.cpp 100% 兼容
   - `block_q8_1` - 与 llama.cpp 100% 兼容

2. **内存布局**：
   - 块大小：32 元素/块
   - 字节对齐：与 llama.cpp 一致

### ⚠️ 需要注意的部分

1. **内存对齐**：
   - llama.cpp 可能使用对齐的内存分配
   - 我们的实现需要处理未对齐的情况（DP4A 错误）

2. **维度约定**：
   - llama.cpp: `tensor->ne[0]` = 列数, `tensor->ne[1]` = 行数
   - 我们的接口: `M` = 行数, `N` = 列数, `K` = 共同维度

3. **内存管理**：
   - llama.cpp 使用 `ggml_context` 管理内存
   - 我们的实现使用 `CudaBuffer` 或原始指针

## 🔧 实现步骤

### Step 1: 创建适配层头文件

创建 `include/llama_adapter.h`：

```cpp
#ifndef LLAMA_ADAPTER_H
#define LLAMA_ADAPTER_H

#include "quant_types.h"
#include "gemm_cuda_naive.cuh"
#include "gemm_cuda_tiled.cuh"
#include "gemm_cuda_dp4a.cuh"

#ifdef __cplusplus
extern "C" {
#endif

// 前向声明
struct ggml_tensor;
struct ggml_context;

// 从 ggml_tensor 调用我们的 GEMM
void gemm_w4a8_from_ggml(
    const struct ggml_tensor * activation,
    const struct ggml_tensor * weights,
    struct ggml_tensor * output
);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_ADAPTER_H
```

### Step 2: 实现适配层

创建 `src/llama_adapter.cu`：

```cpp
#include "llama_adapter.h"
#include "ggml.h"  // 需要 llama.cpp 的头文件

void gemm_w4a8_from_ggml(
    const struct ggml_tensor * activation,
    const struct ggml_tensor * weights,
    struct ggml_tensor * output
) {
    // 验证类型
    assert(activation->type == GGML_TYPE_Q8_1);
    assert(weights->type == GGML_TYPE_Q4_0);
    assert(output->type == GGML_TYPE_F32);
    
    // 提取维度
    int M = activation->ne[1];  // 行数
    int K = activation->ne[0];  // 列数
    int N = weights->ne[1];     // 行数
    
    // 提取数据指针
    const block_q8_1* A = (const block_q8_1*)activation->data;
    const block_q4_0* B = (const block_q4_0*)weights->data;
    float* C = (float*)output->data;
    
    // 调用我们的实现
    gemm_w4a8_naive(A, B, C, M, N, K);
}
```

### Step 3: 创建对比测试

创建 `tests/step5_llama_comparison.cu`：

```cpp
// Step 5: 与 llama.cpp 对比
// 1. 使用相同的输入数据
// 2. 调用 llama.cpp 的算子
// 3. 调用我们的算子
// 4. 对比结果和性能
```

## 📊 接口对齐检查清单

- [ ] 数据结构兼容性验证
- [ ] 内存对齐处理
- [ ] 维度约定转换
- [ ] 类型系统映射
- [ ] 错误处理统一
- [ ] 性能对比测试
- [ ] 正确性验证

## 🎯 下一步

1. **创建适配层代码**（Step 5）
2. **实现对比测试**
3. **性能基准测试**
4. **文档完善**

---

**注意**: 当前项目主要关注**教育性**和**算法正确性**，接口对齐是为了：
1. 验证实现的正确性
2. 与 llama.cpp 进行性能对比
3. 展示如何在实际项目中使用
