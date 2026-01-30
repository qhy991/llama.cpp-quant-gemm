# Quantized GEMM from Scratch

高性能量化矩阵乘法（GEMM）CUDA 实现，从零开始优化到接近工业级性能。

## 🎯 项目目标

实现 Q4_0 × Q8_1 量化 GEMM，用于 LLM 推理加速，目标性能：775 GFLOPS (llama.cpp 水平)。

## 🏆 性能成果

在 **NVIDIA GeForce RTX 5070 Laptop GPU** 上测试：

| 场景 | 矩阵尺寸 | 性能 | 加速比 | 目标达成率 |
|------|---------|------|--------|-----------|
| **单序列推理** | 4096×2×14336 | **516 GFLOPS** | 4.4x | 66.6% |
| **批处理推理** | 4096×4×14336 | **3346 GFLOPS** | 36.5x | **431.8%** 🚀 |

### 性能对比

```
Naive 实现:        117 GFLOPS  (1.0x)
Warp 优化:         511 GFLOPS  (4.4x)
2D Tiling:         516 GFLOPS  (4.4x)
Double Buffering:  511 GFLOPS  (4.4x, N=2)
                  3346 GFLOPS (28.5x, N=4) 🚀
```

## ✨ 核心特性

- ✅ **高性能**: 单序列 516 GFLOPS，批处理 3346 GFLOPS
- ✅ **高精度**: 99.98%+ 准确率，误差 < 0.5%
- ✅ **多种优化**: Warp 协作、2D Tiling、Double Buffering
- ✅ **完整文档**: 详细的优化报告和性能分析
- ✅ **易于使用**: 简单的 API 接口

## 📁 项目结构

```
quant-gemm-from-scratch/
├── compat/
│   └── ggml_types.h              # 量化数据结构定义
├── kernels/
│   └── gemm/
│       ├── gemm_quant_formats.cuh    # Naive 实现 (基线)
│       ├── gemm_warp_optimized.cuh   # Warp + 2D Tile 优化
│       └── gemm_async_copy.cuh       # Double Buffering 优化
├── tests/
│   ├── benchmark_warp_optimized.cu   # 完整性能测试
│   ├── benchmark_best.cu             # 最佳 kernels 测试
│   ├── test_correctness.cu           # 正确性验证
│   └── profile_kernel.cu             # Profiling 工具
├── docs/
│   ├── warp_optimization_summary.md      # Warp 优化总结
│   ├── 2d_tiling_final_report.md         # 2D Tiling 报告
│   ├── correctness_report.md             # 正确性报告
│   └── final_optimization_report.md      # 最终优化报告
└── README.md                         # 本文档
```

## 🚀 快速开始

### 环境要求

- CUDA Toolkit 11.0+
- NVIDIA GPU (Compute Capability 6.1+)
- C++17 编译器

### 编译

```bash
# 编译性能测试
nvcc -O3 -arch=sm_80 -std=c++17 -o tests/benchmark_best tests/benchmark_best.cu

# 编译正确性测试
nvcc -O3 -arch=sm_80 -std=c++17 -o tests/test_correctness tests/test_correctness.cu
```

**注意**: 将 `sm_80` 替换为你的 GPU 架构

### 运行测试

```bash
# 性能测试
./tests/benchmark_best 4096 2 14336

# 正确性测试
./tests/test_correctness
```

## 📖 使用示例

```cpp
#include "kernels/gemm/gemm_warp_optimized.cuh"

// 调用 GEMM
gemm_q4_0_q8_1_tile2d(
    d_weight, d_activation, d_output,
    M, N, K, stream
);
```

## 📚 详细文档

位于 `docs/` 目录：

1. **warp_optimization_summary.md** - Warp 优化详解
2. **2d_tiling_final_report.md** - 2D Tiling 分析
3. **correctness_report.md** - 正确性验证
4. **final_optimization_report.md** - 完整优化报告

---

**项目状态**: 完成
**最后更新**: 2026-01-30
