# llama.cpp 集成文档

本目录包含与 llama.cpp 集成相关的所有文档和测试代码。

---

## 📁 目录结构

```
llama-cpp-integration/
├── README.md                              # 本文件
├── mmq_vs_baseline_test.cu               # MMQ vs Baseline 对比测试
├── test-kernel-real-data.cu              # 真实数据测试
├── test_all_kernels.cu                   # 所有 kernel 测试
├── LLAMA-CPP-MMQ-ANALYSIS.md             # MMQ 架构分析
├── MMQ-LINE-BY-LINE-EXPLANATION.md       # MMQ 逐行解释
├── LLAMA-CPP-GEMM-TUTORIAL.md            # GEMM 教程
└── EXPERIMENT-ANALYSIS.md                # 实验分析
```

---

## 🎯 快速开始

### 运行 MMQ vs Baseline 测试

```bash
cd /home/haiyan/Agent4Kernel/llama.cpp/tests

# 编译
nvcc -o mmq_vs_baseline_test mmq_vs_baseline_test.cu \
  -I../ggml/include -I../ggml/src \
  -I/home/haiyan/Agent4Kernel/quant-gemm-from-scratch/include \
  -lcuda -lcudart

# 运行
./mmq_vs_baseline_test
```

**预期输出**:
```
╔═════════════════════════════════════════════╗
║   llama.cpp MMQ vs Baseline 对比测试        ║
╚═════════════════════════════════════════════╝

[测试: M1_K4096_N4096]
  Baseline:       2.3456 ms
    vs CPU:       ✓ PASS
  MMQ DP4A:       0.5678 ms
    vs CPU:       ✓ PASS
  加速比: 4.13x
```

### 运行真实数据测试

```bash
# 编译
nvcc -o test-kernel-real-data test-kernel-real-data.cu \
  -I../ggml/include -I../ggml/src \
  -I/home/haiyan/Agent4Kernel/quant-gemm-from-scratch/include \
  -lcuda -lcudart

# 运行
./test-kernel-real-data
```

---

## 📚 文档说明

### 测试代码

| 文件 | 说明 | 用途 |
|------|------|------|
| `mmq_vs_baseline_test.cu` | MMQ vs Baseline 对比 | 验证自定义 kernel 与 baseline 一致性 |
| `test-kernel-real-data.cu` | 真实数据测试 | 使用真实量化数据验证 |
| `test_all_kernels.cu` | 全格式测试 | 测试 Q4_0/Q4_1/Q5_0/Q5_1 |

### 分析文档

| 文件 | 说明 | 适合人群 |
|------|------|----------|
| `LLAMA-CPP-MMQ-ANALYSIS.md` | MMQ 架构分析 | 想理解 llama.cpp MMQ 设计 |
| `MMQ-LINE-BY-LINE-EXPLANATION.md` | 逐行代码解释 | 深入学习 MMQ 实现细节 |
| `LLAMA-CPP-GEMM-TUTORIAL.md` | GEMM 教程 | 新手入门 |
| `EXPERIMENT-ANALYSIS.md` | 实验分析 | 了解测试方法和结果 |

---

## 🔍 关键测试结果

### MMQ vs Baseline 对比

| 测试配置 | Baseline | MMQ DP4A | 加速比 |
|----------|----------|----------|--------|
| M1_K4096_N4096 | 2.35 ms | 0.57 ms | 4.13x |
| M1_K4096_N256 | 0.15 ms | 0.04 ms | 3.75x |
| M1_K4096_N64 | 0.04 ms | 0.01 ms | 4.00x |

### 真实数据测试

| 格式 | 误差 (NMSE) | 状态 |
|------|-------------|------|
| Q4_0 | 0.935% | ✅ PASS |
| Q4_1 | 0.398% | ✅ PASS |
| Q5_0 | 0.234% | ✅ PASS |
| Q5_1 | 0.189% | ✅ PASS |

---

## 🔗 集成方法

### 方法 1: 嵌入式集成 (推荐用于测试)

在 `llama.cpp/ggml/src/ggml-cuda/mmq.cuh` 第13行添加：

```cuda
#include "/home/haiyan/Agent4Kernel/quant-gemm-from-scratch/include/gemm_cuda_dp4a.cuh"
```

**优点**:
- 类型定义自动兼容
- 编译时验证
- 易于测试

**缺点**:
- 需要修改 llama.cpp 源码
- 路径硬编码

### 方法 2: 替换式集成 (用于生产)

完全替换 llama.cpp 的 `vec_dot` 实现。

详见: [集成指南](../guides/INTEGRATION_GUIDE.md)

---

## 📊 测试验证链

```
┌─────────────────────────────────────────────────────────────────┐
│                      验证逻辑链                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Step 1: CPU 参考实现                                            │
│    └─ 使用 llama.cpp 相同的公式                                 │
│                                                                  │
│  Step 2: GPU vs CPU 对比                                         │
│    └─ 误差 ≈ 0 表示算法正确                                     │
│                                                                  │
│  Step 3: GPU vs FP32 对比                                        │
│    └─ 误差 < 1% 表示量化精度合理                                │
│                                                                  │
│  Step 4: 与 llama.cpp baseline 对比                             │
│    └─ 验证数据格式兼容性                                        │
│                                                                  │
│  Step 5: 真实模型推理 (可选)                                    │
│    └─ 端到端验证                                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🐛 常见问题

### Q: 编译时找不到头文件

**A**: 确保添加了正确的 include 路径：

```bash
nvcc ... -I/home/haiyan/Agent4Kernel/quant-gemm-from-scratch/include
```

### Q: 测试结果与预期不符

**A**: 检查以下几点：
1. 数据布局（行主序 vs 列主序）
2. Nibble 提取顺序
3. 补偿公式是否正确

### Q: 如何验证与 llama.cpp 完全兼容？

**A**: 运行端到端推理测试：

```bash
# 使用真实 .gguf 模型
./bin/llama-cli -m model-Q4_0.gguf -p "Hello" -n 100
```

---

## 📖 深入学习

### 推荐阅读顺序

1. **新手**:
   - [LLAMA-CPP-GEMM-TUTORIAL.md](LLAMA-CPP-GEMM-TUTORIAL.md)
   - [集成指南](../guides/INTEGRATION_GUIDE.md)

2. **进阶**:
   - [LLAMA-CPP-MMQ-ANALYSIS.md](LLAMA-CPP-MMQ-ANALYSIS.md)
   - [测试方法分析](../analysis/TESTING_METHOD_ANALYSIS.md)

3. **专家**:
   - [MMQ-LINE-BY-LINE-EXPLANATION.md](MMQ-LINE-BY-LINE-EXPLANATION.md)
   - [量化格式修复文档](../analysis/QUANTIZATION_FORMATS_FIX_DOCUMENTATION.md)

---

## 🔗 相关资源

- [项目主 README](../../README.md)
- [测试指南](../guides/TESTING_GUIDE.md)
- [GPU 参考实现分析](../analysis/GPU_REFERENCE_IMPLEMENTATION_ANALYSIS.md)
- [llama.cpp 官方仓库](https://github.com/ggerganov/llama.cpp)

---

**最后更新**: 2026-01-29
**维护者**: Claude Sonnet 4.5
