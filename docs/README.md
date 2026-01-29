# Documentation Index

本目录包含 quant-gemm-from-scratch 项目的所有文档。

## 📚 文档组织

```
docs/
├── README.md                    # 本文件 - 文档索引
├── guides/                      # 使用指南
│   ├── GETTING_STARTED.md      # 快速开始
│   ├── TESTING_GUIDE.md        # 测试指南
│   ├── INTEGRATION_GUIDE.md    # 集成指南
│   ├── CUDA-GEMM-BENCHMARK-TUTORIAL.md  # CUDA GEMM 性能测试教程
│   └── QUICK-REFERENCE.md      # 快速参考指南
├── analysis/                    # 技术分析
│   ├── GPU_REFERENCE_IMPLEMENTATION_ANALYSIS.md
│   ├── TESTING_METHOD_ANALYSIS.md
│   ├── QUANTIZATION_FORMATS_FIX_DOCUMENTATION.md
│   └── KERNEL_TEST_FRAMEWORK_ANALYSIS.md
├── reports/                     # 测试报告
│   ├── FINAL_TEST_REPORT.md
│   ├── Q8_0_TEST_REPORT.md
│   └── INTEGRATION_TEST_REPORT.md
└── testing/                     # 测试相关
    ├── TEST_METHODOLOGY.md
    ├── VERIFICATION_STRATEGY.md
    └── CUDA-12.8-TEST-LOG.md   # CUDA 12.8 完整测试日志
```

---

## 🚀 快速导航

### 新手入门
- [快速开始](guides/GETTING_STARTED.md) - 5分钟上手
- [项目架构](../PROJECT_ARCHITECTURE.md) - 理解项目结构
- [算子列表](../OPERATOR_LIST.md) - 支持的量化格式

### 性能测试 🆕
- [CUDA GEMM 性能测试教程](guides/CUDA-GEMM-BENCHMARK-TUTORIAL.md) - 完整的性能测试教程
- [快速参考指南](guides/QUICK-REFERENCE.md) - 常用命令速查
- [CUDA 12.8 测试日志](testing/CUDA-12.8-TEST-LOG.md) - 详细测试输出记录

### 测试验证
- [测试指南](guides/TESTING_GUIDE.md) - 如何运行测试
- [测试方法分析](analysis/TESTING_METHOD_ANALYSIS.md) - 测试方法论
- [最终测试报告](reports/FINAL_TEST_REPORT.md) - 测试结果

### 技术深入
- [量化格式修复文档](analysis/QUANTIZATION_FORMATS_FIX_DOCUMENTATION.md) - Q4/Q5 修复过程
- [GPU参考实现分析](analysis/GPU_REFERENCE_IMPLEMENTATION_ANALYSIS.md) - 与 llama.cpp 对比
- [Kernel测试框架分析](analysis/KERNEL_TEST_FRAMEWORK_ANALYSIS.md) - 测试框架设计

### 集成使用
- [集成指南](guides/INTEGRATION_GUIDE.md) - 如何集成到 llama.cpp
- [接口对齐](INTERFACE_ALIGNMENT_STATUS.md) - 接口兼容性
- [集成测试报告](reports/INTEGRATION_TEST_REPORT.md) - 集成测试结果

---

## 📖 按主题浏览

### 量化格式

| 格式 | 状态 | 误差 | 文档 |
|------|------|------|------|
| Q4_0 | ✅ | 0.465% | [修复文档](analysis/QUANTIZATION_FORMATS_FIX_DOCUMENTATION.md#q4_0) |
| Q4_1 | ✅ | 0.398% | [修复文档](analysis/QUANTIZATION_FORMATS_FIX_DOCUMENTATION.md#q4_1) |
| Q5_0 | ✅ | 0.234% | [修复文档](analysis/QUANTIZATION_FORMATS_FIX_DOCUMENTATION.md#q5_0) |
| Q5_1 | ✅ | 0.189% | [修复文档](analysis/QUANTIZATION_FORMATS_FIX_DOCUMENTATION.md#q5_1) |
| Q8_0 | ⚠️ | 对齐问题 | [Q8_0报告](reports/Q8_0_TEST_REPORT.md) |

### 测试方法

| 测试类型 | 文档 | 说明 |
|----------|------|------|
| 单元测试 | [测试指南](guides/TESTING_GUIDE.md#unit-tests) | 独立 kernel 测试 |
| 集成测试 | [集成指南](guides/INTEGRATION_GUIDE.md) | llama.cpp 集成 |
| 端到端测试 | [测试方法分析](analysis/TESTING_METHOD_ANALYSIS.md#end-to-end) | 真实模型推理 |

### 性能优化

| 优化技术 | 文档 | 加速比 |
|----------|------|--------|
| llama.cpp 优化 | [性能测试教程](guides/CUDA-GEMM-BENCHMARK-TUTORIAL.md) | ~110x |
| DP4A | [DP4A教程](../tutorials/06-dp4a-optimization/) | ~4x |
| Tiling | [Tiling教程](../tutorials/03-gemm-tiled/) | ~2x |
| Vectorization | [代码示例](../kernels/gemm/gemm_cuda_dp4a.cuh) | ~1.5x |

---

## 🔍 按问题查找

### "如何开始使用？"
→ [快速开始指南](guides/GETTING_STARTED.md)

### "测试失败了怎么办？"
→ [测试指南 - 故障排查](guides/TESTING_GUIDE.md#troubleshooting)

### "如何集成到 llama.cpp？"
→ [集成指南](guides/INTEGRATION_GUIDE.md)

### "为什么我的结果与 llama.cpp 不一致？"
→ [测试方法分析](analysis/TESTING_METHOD_ANALYSIS.md)

### "Q4/Q5 格式是如何修复的？"
→ [量化格式修复文档](analysis/QUANTIZATION_FORMATS_FIX_DOCUMENTATION.md)

### "GPU 参考实现是什么？"
→ [GPU参考实现分析](analysis/GPU_REFERENCE_IMPLEMENTATION_ANALYSIS.md)

### "如何测试 llama.cpp 的性能？" 🆕
→ [CUDA GEMM 性能测试教程](guides/CUDA-GEMM-BENCHMARK-TUTORIAL.md)

### "llama.cpp 的优化效果如何？" 🆕
→ [CUDA 12.8 测试日志](testing/CUDA-12.8-TEST-LOG.md) - 查看 8.83 TFLOPS 的测试结果

---

## 📝 文档贡献

### 文档类型

- **guides/** - 面向用户的操作指南
- **analysis/** - 技术深入分析
- **reports/** - 测试和实验报告
- **testing/** - 测试方法论

### 文档规范

1. 使用 Markdown 格式
2. 包含清晰的标题层次
3. 代码示例使用语法高亮
4. 包含目录（TOC）
5. 添加交叉引用链接

---

## 🔗 相关资源

- [项目 README](../README.md) - 项目主页
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - 上游项目
- [CUDA 编程指南](https://docs.nvidia.com/cuda/) - NVIDIA 官方文档

---

**最后更新**: 2026-01-29
**维护者**: Claude Sonnet 4.5
