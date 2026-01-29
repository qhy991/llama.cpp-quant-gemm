# 📚 CUDA Kernel 开发教程

欢迎来到 CUDA Kernel 开发教程系列！本教程将带你从零开始，逐步实现 llama.cpp 中使用的各种 CUDA 算子。

---

## 🎯 学习目标

完成本系列教程后，你将能够：

- ✅ 理解 CUDA 编程模型和内存层次结构
- ✅ 从零实现高效的 GEMM (矩阵乘法) kernel
- ✅ 理解和实现各种量化格式 (Q4_0, Q8_0 等)
- ✅ 使用 DP4A 指令优化整数计算
- ✅ 实现常见的神经网络算子 (SiLU, GELU, RMSNorm 等)
- ✅ 将自己的 kernel 集成到 llama.cpp

---

## 📖 教程目录

### 入门篇

| 教程 | 主题 | 难度 | 状态 |
|------|------|------|------|
| [00-introduction](./00-introduction/) | 项目介绍与环境配置 | ⭐ | ✅ |
| [01-cuda-basics](./01-cuda-basics/) | CUDA 编程基础 | ⭐ | 📝 |

### 基础篇：GEMM 实现

| 教程 | 主题 | 难度 | 状态 |
|------|------|------|------|
| [02-gemm-naive](./02-gemm-naive/) | 朴素 GEMM 实现 | ⭐⭐ | 📝 |
| [03-gemm-tiled](./03-gemm-tiled/) | Tiled GEMM 优化 | ⭐⭐⭐ | 📝 |

### 进阶篇：量化技术

| 教程 | 主题 | 难度 | 状态 |
|------|------|------|------|
| [04-quantization-basics](./04-quantization-basics/) | 量化基础理论 | ⭐⭐ | 📝 |
| [05-q4-gemm](./05-q4-gemm/) | Q4_0 量化 GEMM | ⭐⭐⭐ | 📝 |
| [06-dp4a-optimization](./06-dp4a-optimization/) | DP4A 指令优化 | ⭐⭐⭐⭐ | ✅ |

### 高级篇：更多算子

| 教程 | 主题 | 难度 | 状态 |
|------|------|------|------|
| 07-activation-functions | 激活函数 (SiLU, GELU) | ⭐⭐ | 📝 |
| 08-normalization | 归一化 (RMSNorm) | ⭐⭐⭐ | 📝 |
| 09-attention | 注意力机制 | ⭐⭐⭐⭐ | 📝 |
| 10-llama-integration | llama.cpp 集成 | ⭐⭐⭐ | 📝 |

**图例**: ✅ 已完成 | 📝 进行中 | ⬜ 计划中

---

## 🚀 快速开始

### 1. 环境要求

```bash
# CUDA Toolkit 12.0+
nvcc --version

# 支持 sm_75+ 的 GPU (Turing/Ampere/Ada/Blackwell)
nvidia-smi
```

### 2. 克隆项目

```bash
cd /home/haiyan/Agent4Kernel
git clone <repo-url> quant-gemm-from-scratch
cd quant-gemm-from-scratch
```

### 3. 运行第一个示例

```bash
cd tutorials/00-introduction
nvcc -o hello hello.cu
./hello
```

---

## 📁 项目结构

```
quant-gemm-from-scratch/
├── compat/                  # llama.cpp 兼容类型
├── kernels/                 # 算子实现
│   ├── gemm/               # 矩阵乘法
│   ├── activation/         # 激活函数
│   └── ...
├── tests/                   # 测试程序
├── tutorials/               # 📚 你在这里！
│   ├── 00-introduction/
│   ├── 01-cuda-basics/
│   └── ...
└── docs/                    # 文档
```

---

## 📈 学习路径

```
                              ┌─────────────────┐
                              │ Flash Attention │
                              │    (高级)       │
                              └────────┬────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
             ┌──────┴──────┐    ┌──────┴──────┐    ┌──────┴──────┐
             │    ROPE     │    │   Softmax   │    │  LayerNorm  │
             │   (中级)    │    │   (中级)    │    │   (中级)    │
             └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
                    │                  │                  │
                    └──────────────────┼──────────────────┘
                                       │
                              ┌────────┴────────┐
                              │   RMSNorm       │
                              │   SiLU/GELU     │
                              │   (基础-中级)   │
                              └────────┬────────┘
                                       │
                              ┌────────┴────────┐
                              │  DP4A 优化      │
                              │  Q4 GEMM        │
                              │   (进阶)        │
                              └────────┬────────┘
                                       │
                              ┌────────┴────────┐
                              │   量化基础      │
                              │  Tiled GEMM     │
                              │   (基础)        │
                              └────────┬────────┘
                                       │
                              ┌────────┴────────┐
                              │   CUDA 基础     │
                              │  Naive GEMM     │
                              │   (入门)        │
                              └─────────────────┘
```

---

## 💡 学习建议

1. **循序渐进**: 按顺序学习，每个教程都基于前面的知识
2. **动手实践**: 不要只看代码，要自己写和调试
3. **做笔记**: 记录你的理解和遇到的问题
4. **查阅文档**: 善用 CUDA 官方文档和 Nsight 工具
5. **对比验证**: 用测试框架验证你的实现

---

## 🔗 参考资源

### 官方文档
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [PTX ISA Reference](https://docs.nvidia.com/cuda/parallel-thread-execution/)

### 相关项目
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [CUTLASS](https://github.com/NVIDIA/cutlass)

### 论文
- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [LLM.int8()](https://arxiv.org/abs/2208.07339)

---

## 🤝 贡献

欢迎贡献新的教程或改进现有内容！请参阅 [CONTRIBUTING.md](../docs/CONTRIBUTING.md)。

---

**Happy Learning! 🎉**
