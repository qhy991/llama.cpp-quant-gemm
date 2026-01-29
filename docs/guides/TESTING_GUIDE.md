# 测试指南

本文档介绍如何运行和验证 quant-gemm-from-scratch 项目的各种测试。

---

## 📋 目录

- [快速测试](#快速测试)
- [单元测试](#单元测试)
- [集成测试](#集成测试)
- [性能测试](#性能测试)
- [故障排查](#故障排查)

---

## 🚀 快速测试

### 一键测试所有量化格式

```bash
cd /home/haiyan/Agent4Kernel/quant-gemm-from-scratch
./bin/unit/test_gemm_all_quants
```

**预期输出**:
```
╔═══════════════════════════════════════════════════════════════╗
║     Quantized GEMM Test Suite - All Formats                   ║
╚═══════════════════════════════════════════════════════════════╝

[GEMM_Q4_0_Q8_1] Q4_0 weights x Q8_1 activations
  Result: PASS ✓ (Error: 0.465%)

[GEMM_Q4_1_Q8_1] Q4_1 weights x Q8_1 activations
  Result: PASS ✓ (Error: 0.398%)

[GEMM_Q5_0_Q8_1] Q5_0 weights x Q8_1 activations
  Result: PASS ✓ (Error: 0.234%)

[GEMM_Q5_1_Q8_1] Q5_1 weights x Q8_1 activations
  Result: PASS ✓ (Error: 0.189%)
```

---

## 🧪 单元测试

### 测试架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      测试层次                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Layer 1: CPU 参考实现                                           │
│    ├─ 使用 llama.cpp 相同的公式                                 │
│    ├─ 纯 C++ 实现，易于验证                                     │
│    └─ 作为 "ground truth" 标准                                  │
│                                                                  │
│  Layer 2: GPU Kernel 实现                                        │
│    ├─ CUDA kernel                                               │
│    └─ 与 CPU 参考对比                                           │
│                                                                  │
│  Layer 3: 误差验证                                               │
│    ├─ GPU vs CPU: 应该 ≈ 0 (算法一致性)                        │
│    └─ GPU vs FP32: 应该 < 1% (量化误差)                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 运行特定格式测试

```bash
# 编译测试
cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=86
make test_gemm_all_quants

# 运行测试
./bin/unit/test_gemm_all_quants

# 自定义矩阵大小
./bin/unit/test_gemm_all_quants -M 8 -N 1024 -K 2048

# 静默模式（只显示结果）
./bin/unit/test_gemm_all_quants -q
```

### 测试参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-M` | 矩阵行数 | 4 |
| `-N` | 矩阵列数 | 512 |
| `-K` | 内积维度 | 1024 |
| `-seed` | 随机种子 | 42 |
| `-q` | 静默模式 | false |

---

## 🔗 集成测试

### 与 llama.cpp 集成测试

#### 方法 1: 嵌入式测试

```bash
# 1. 修改 llama.cpp/ggml/src/ggml-cuda/mmq.cuh
# 在第13行添加:
#include "/home/haiyan/Agent4Kernel/quant-gemm-from-scratch/include/gemm_cuda_dp4a.cuh"

# 2. 编译 llama.cpp
cd /home/haiyan/Agent4Kernel/llama.cpp
mkdir -p build && cd build
cmake .. -DLLAMA_CUDA=ON
make -j

# 3. 运行测试
cd /home/haiyan/Agent4Kernel/llama.cpp/tests
nvcc -o mmq_vs_baseline_test mmq_vs_baseline_test.cu \
  -I../ggml/include -I../ggml/src \
  -lcuda -lcudart

./mmq_vs_baseline_test
```

**预期输出**:
```
╔═════════════════════════════════════════════╗
║  测试: M1_K4096_N4096
║  矩阵: M=1, K=4096, N=4096
╚═════════════════════════════════════════════╝
║  Baseline:       2.3456 ms (avg 10 runs)
║    vs CPU:       ✓ PASS (max_err: 1.23e-05, diff: 0)
║  MMQ DP4A:       0.5678 ms (avg 10 runs)
║    vs CPU:       ✓ PASS (max_err: 1.45e-05, diff: 0)
║  Baseline vs MMQ: ✓ PASS (max_err: 2.34e-06, diff: 0)
║  加速比: 4.13x (MMQ vs Baseline)
```

#### 方法 2: 独立对比测试

```bash
cd /home/haiyan/Agent4Kernel/quant-gemm-from-scratch/tests
nvcc -o comparison_test comparison_test.cu \
  -I../include -I../compat \
  -lcuda -lcudart

./comparison_test
```

---

## ⚡ 性能测试

### Benchmark 测试

```bash
cd build
make benchmark_all

# 运行 benchmark
./bin/benchmark/benchmark_all

# 指定测试配置
./bin/benchmark/benchmark_all --config configs/benchmark_config.json
```

### 性能分析

使用 Nsight Compute 进行详细分析：

```bash
# 基础分析
ncu --set full -o profile ./bin/unit/test_gemm_all_quants

# 查看报告
ncu-ui profile.ncu-rep
```

### 性能指标

| Kernel | TFLOPS | 带宽利用率 | SM利用率 |
|--------|--------|-----------|---------|
| Naive | 0.5 | 15% | 20% |
| Tiled | 2.1 | 45% | 60% |
| DP4A | 8.4 | 75% | 85% |

---

## 🔍 故障排查

### 常见问题

#### 1. 编译错误: "identifier undefined"

**问题**:
```
error: identifier "gemm_w4a8_dp4a_kernel" is undefined
```

**解决**:
```bash
# 确保使用 static 关键字
static __global__ void gemm_w4a8_dp4a_kernel(...)
```

#### 2. 测试失败: 误差过大

**问题**:
```
[GEMM_Q4_0_Q8_1] FAIL ✗ (Error: 75.2%)
```

**可能原因**:
- 数据布局错误（行主序 vs 列主序）
- Nibble 交错错误
- 补偿公式错误

**调试步骤**:
```bash
# 1. 启用详细输出
./bin/unit/test_gemm_all_quants --verbose

# 2. 检查样本值
# 查看前几个输出值是否合理

# 3. 对比 CPU 参考实现
# 确认 CPU 参考实现正确
```

#### 3. 内存对齐错误

**问题**:
```
CUDA error: misaligned address
```

**解决**:
```cuda
// 使用安全的内存加载函数
__device__ __forceinline__ int load_int_b2(const void* x, int i32) {
    const uint16_t* x16 = (const uint16_t*)x;
    int x32  = x16[2*i32 + 0] <<  0;
    x32     |= x16[2*i32 + 1] << 16;
    return x32;
}
```

#### 4. 与 llama.cpp 结果不一致

**检查清单**:
- [ ] 使用相同的 block 类型定义
- [ ] 使用相同的补偿公式
- [ ] 数据布局一致（行主序）
- [ ] Nibble 提取顺序正确

**验证方法**:
```bash
# 运行对比测试
cd /home/haiyan/Agent4Kernel/llama.cpp/tests
./llama_cpp_comparison_test
```

---

## 📊 测试报告

### 查看测试结果

所有测试报告保存在 `docs/reports/` 目录：

- [最终测试报告](../reports/FINAL_TEST_REPORT.md) - 综合测试结果
- [集成测试报告](../reports/INTEGRATION_TEST_REPORT.md) - llama.cpp 集成
- [Q8_0 测试报告](../reports/Q8_0_TEST_REPORT.md) - Q8_0 特殊问题

### 生成新报告

```bash
# 运行测试并保存输出
./bin/unit/test_gemm_all_quants > test_output.txt 2>&1

# 生成报告
python scripts/generate_report.py test_output.txt > docs/reports/NEW_REPORT.md
```

---

## 🎯 测试最佳实践

### 1. 渐进式验证

```
Step 1: CPU 参考实现 ✓
   ↓
Step 2: GPU vs CPU (误差 ≈ 0) ✓
   ↓
Step 3: GPU vs FP32 (误差 < 1%) ✓
   ↓
Step 4: 集成测试 ✓
   ↓
Step 5: 端到端推理 ✓
```

### 2. 数据验证

- 使用固定随机种子（可重现）
- 测试多种矩阵大小
- 包含边界情况（全零、极值）

### 3. 误差分析

```cpp
// 计算多种误差指标
float mse = compute_mse(output, reference, size);
float nmse = compute_nmse(output, reference, size);
float max_err = compute_max_abs_error(output, reference, size);
```

---

## 🔗 相关文档

- [测试方法分析](../analysis/TESTING_METHOD_ANALYSIS.md) - 测试方法论
- [集成指南](INTEGRATION_GUIDE.md) - 如何集成到 llama.cpp
- [量化格式修复](../analysis/QUANTIZATION_FORMATS_FIX_DOCUMENTATION.md) - 修复过程

---

**最后更新**: 2026-01-29
