# 集成指南

本文档介绍如何将 quant-gemm-from-scratch 的 kernel 集成到 llama.cpp 或其他项目中。

---

## 📋 目录

- [集成方法](#集成方法)
- [嵌入式集成](#嵌入式集成)
- [独立库集成](#独立库集成)
- [端到端验证](#端到端验证)
- [性能对比](#性能对比)

---

## 🎯 集成方法

### 方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **嵌入式集成** | 类型兼容保证 | 需要修改源码 | 开发测试 |
| **独立库集成** | 不修改源码 | 需要适配层 | 生产环境 |
| **替换式集成** | 完全替换原版 | 风险较高 | 性能优化 |

---

## 🔌 嵌入式集成

### 步骤 1: 修改 llama.cpp

```bash
cd /home/haiyan/Agent4Kernel/llama.cpp
```

编辑 `ggml/src/ggml-cuda/mmq.cuh`，在第13行添加：

```cuda
// ============================================================================
// CUSTOM KERNEL: Include our custom gemm_w4a8_dp4a implementation
// ============================================================================
#include "/home/haiyan/Agent4Kernel/quant-gemm-from-scratch/include/gemm_cuda_dp4a.cuh"
```

### 步骤 2: 编译 llama.cpp

```bash
mkdir -p build && cd build
cmake .. -DLLAMA_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
make -j$(nproc)
```

### 步骤 3: 验证集成

```bash
# 运行对比测试
cd /home/haiyan/Agent4Kernel/llama.cpp/tests
nvcc -o mmq_vs_baseline_test mmq_vs_baseline_test.cu \
  -I../ggml/include -I../ggml/src \
  -I/home/haiyan/Agent4Kernel/quant-gemm-from-scratch/include \
  -lcuda -lcudart

./mmq_vs_baseline_test
```

**预期输出**:
```
╔═════════════════════════════════════════════╗
║   llama.cpp MMQ vs Baseline 对比测试        ║
╚═════════════════════════════════════════════╝

[测试: M1_K4096_N4096]
  Baseline:       2.3456 ms
    vs CPU:       ✓ PASS (max_err: 1.23e-05)
  MMQ DP4A:       0.5678 ms
    vs CPU:       ✓ PASS (max_err: 1.45e-05)
  Baseline vs MMQ: ✓ PASS (max_err: 2.34e-06)
  加速比: 4.13x
```

---

## 📦 独立库集成

### 方法 1: 静态库

#### 编译静态库

```bash
cd /home/haiyan/Agent4Kernel/quant-gemm-from-scratch
mkdir -p build && cd build

cmake .. -DBUILD_SHARED_LIBS=OFF
make -j

# 生成 libquant_gemm.a
```

#### 使用静态库

```cpp
// your_project.cu
#include <quant_gemm/gemm_cuda_dp4a.cuh>

int main() {
    // 使用 kernel
    gemm_w4a8_dp4a(A, B, C, M, N, K);
    return 0;
}
```

```bash
# 编译你的项目
nvcc -o your_project your_project.cu \
  -I/path/to/quant-gemm-from-scratch/include \
  -L/path/to/quant-gemm-from-scratch/build \
  -lquant_gemm
```

### 方法 2: Header-Only

将所有实现放在 `.cuh` 文件中，直接 include：

```cpp
#include "quant-gemm-from-scratch/include/gemm_cuda_dp4a.cuh"
```

**优点**: 无需链接库
**缺点**: 编译时间较长

---

## 🔄 替换式集成

### 完全替换 llama.cpp 的 vec_dot 实现

#### 步骤 1: 备份原版

```bash
cd /home/haiyan/Agent4Kernel/llama.cpp/ggml/src/ggml-cuda
cp vecdotq.cuh vecdotq.cuh.backup
```

#### 步骤 2: 修改 vecdotq.cuh

```cuda
// vecdotq.cuh

// 原版实现
#if 0  // 禁用原版
template <int vdr> static __device__ __forceinline__ float vec_dot_q4_0_q8_1_impl(
    const int * v, const int * u, const float & d4, const half2 & ds8) {
    // ... 原版代码
}
#endif

// 使用自定义实现
#include "/path/to/quant-gemm-from-scratch/kernels/gemm/gemm_quant_formats.cuh"

// 适配层
template <int vdr> static __device__ __forceinline__ float vec_dot_q4_0_q8_1_impl(
    const int * v, const int * u, const float & d4, const half2 & ds8) {

    // 转换为我们的格式
    block_q4_0 bq4;
    block_q8_1 bq8;
    // ... 数据转换

    // 调用我们的实现
    return vec_dot_q4_0_q8_1(&bq4, &bq8);
}
```

#### 步骤 3: 测试验证

```bash
# 编译
cd build
make -j

# 运行测试
./bin/llama-cli -m model-Q4_0.gguf -p "Hello" -n 50
```

---

## ✅ 端到端验证

### 真实模型推理测试

#### 准备模型

```bash
# 下载或量化模型
cd /home/haiyan/Agent4Kernel/llama.cpp
./bin/llama-quantize model.gguf model-Q4_0.gguf Q4_0
./bin/llama-quantize model.gguf model-Q4_1.gguf Q4_1
./bin/llama-quantize model.gguf model-Q5_0.gguf Q5_0
./bin/llama-quantize model.gguf model-Q5_1.gguf Q5_1
```

#### 运行推理对比

```bash
# 1. 使用原版 llama.cpp
./bin/llama-cli-original -m model-Q4_0.gguf -p "Hello, world!" -n 50 \
  --seed 42 > original_output.txt

# 2. 使用自定义 kernel
./bin/llama-cli-custom -m model-Q4_0.gguf -p "Hello, world!" -n 50 \
  --seed 42 > custom_output.txt

# 3. 对比输出
diff original_output.txt custom_output.txt
```

#### 数值对比

```python
# compare_outputs.py
import numpy as np

def compare_outputs(file1, file2):
    with open(file1) as f1, open(file2) as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    # 提取 token 概率
    probs1 = extract_probs(lines1)
    probs2 = extract_probs(lines2)

    # 计算差异
    diff = np.abs(probs1 - probs2)
    print(f"Max diff: {diff.max():.6f}")
    print(f"Mean diff: {diff.mean():.6f}")

    return diff.max() < 1e-4  # 阈值

if __name__ == "__main__":
    result = compare_outputs("original_output.txt", "custom_output.txt")
    print("✓ PASS" if result else "✗ FAIL")
```

---

## ⚡ 性能对比

### Benchmark 测试

```bash
# 创建 benchmark 脚本
cat > benchmark.sh << 'EOF'
#!/bin/bash

MODEL="model-Q4_0.gguf"
PROMPT="Once upon a time"
N_TOKENS=100

echo "=== Original llama.cpp ==="
time ./bin/llama-cli-original -m $MODEL -p "$PROMPT" -n $N_TOKENS

echo ""
echo "=== Custom kernel ==="
time ./bin/llama-cli-custom -m $MODEL -p "$PROMPT" -n $N_TOKENS
EOF

chmod +x benchmark.sh
./benchmark.sh
```

### 使用 Nsight Systems 分析

```bash
# 分析原版
nsys profile -o original ./bin/llama-cli-original -m model-Q4_0.gguf -p "Hello" -n 50

# 分析自定义版本
nsys profile -o custom ./bin/llama-cli-custom -m model-Q4_0.gguf -p "Hello" -n 50

# 对比
nsys-ui original.nsys-rep custom.nsys-rep
```

---

## 🔍 集成验证清单

### 编译时验证

- [ ] 无编译错误
- [ ] 无链接错误
- [ ] 无类型不匹配警告

### 功能验证

- [ ] 单元测试通过
- [ ] 集成测试通过
- [ ] 输出数值正确（误差 < 1%）

### 性能验证

- [ ] 性能不低于原版
- [ ] 内存使用合理
- [ ] 无内存泄漏

### 端到端验证

- [ ] 真实模型推理成功
- [ ] 输出与原版一致
- [ ] 无崩溃或错误

---

## 🐛 常见集成问题

### 问题 1: 类型不匹配

**错误**:
```
error: no matching function for call to 'vec_dot_q4_0_q8_1'
```

**解决**:
```cuda
// 确保使用相同的类型定义
#include "ggml-cuda/common.cuh"  // llama.cpp 的类型

// 或者使用条件编译
#ifdef GGML_COMMON_DECL
    // 使用 llama.cpp 的类型
#else
    // 使用我们自己的类型
    #include "quant_types.h"
#endif
```

### 问题 2: 符号重定义

**错误**:
```
multiple definition of `gemm_w4a8_dp4a_kernel'
```

**解决**:
```cuda
// 使用 static 关键字
static __global__ void gemm_w4a8_dp4a_kernel(...)
```

### 问题 3: 路径问题

**错误**:
```
fatal error: gemm_cuda_dp4a.cuh: No such file or directory
```

**解决**:
```bash
# 使用绝对路径
#include "/home/haiyan/Agent4Kernel/quant-gemm-from-scratch/include/gemm_cuda_dp4a.cuh"

# 或者添加到 include 路径
nvcc -I/home/haiyan/Agent4Kernel/quant-gemm-from-scratch/include ...
```

---

## 📊 集成测试报告

查看详细的集成测试结果：

- [集成测试报告](../reports/INTEGRATION_TEST_REPORT.md)
- [最终测试报告](../reports/FINAL_TEST_REPORT.md)

---

## 🔗 相关文档

- [测试指南](TESTING_GUIDE.md) - 如何运行测试
- [测试方法分析](../analysis/TESTING_METHOD_ANALYSIS.md) - 测试方法论
- [接口对齐](../INTERFACE_ALIGNMENT_STATUS.md) - 接口兼容性

---

**最后更新**: 2026-01-29
