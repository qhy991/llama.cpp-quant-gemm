# 快速入门指南

本指南帮助你快速开始使用本教程项目。

## 📋 前置要求

### 硬件
- NVIDIA GPU（计算能力 ≥ 6.1）
- 推荐：RTX 3000/4000 系列、A100、H100

### 软件
- CUDA Toolkit ≥ 11.0
- C++17 编译器
- conda 环境（推荐使用 KM-12.8）

## 🚀 5 分钟快速开始

### 1. 检查环境

```bash
# 激活 conda 环境
conda activate KM-12.8

# 检查 CUDA
nvcc --version

# 检查 GPU
nvidia-smi
```

### 2. 编译项目

```bash
cd /home/haiyan/Agent4Kernel/quant-gemm-from-scratch

# 编译所有步骤
make

# 或者使用自动化脚本
./scripts/build_and_test.sh build
```

### 3. 运行第一个测试

```bash
# 运行 Step 1: FP32 GEMM 基准
make step1

# 你应该看到类似输出：
# ╔═══════════════════════════════════════════════════════════╗
# ║     Step 1: FP32 GEMM - The Foundation                    ║
# ╚═══════════════════════════════════════════════════════════╝
# 
# === Device Information ===
# Device: NVIDIA GeForce RTX 5070 Laptop GPU
# ...
```

### 4. 运行所有测试

```bash
# 运行所有 5 个步骤
make test

# 或者使用脚本
./scripts/build_and_test.sh
```

## 📚 学习路径

建议按以下顺序学习：

### Step 1: FP32 GEMM 基准 (30 分钟)
- 理解 GEMM 的基本算法
- 对比 CPU、Naive CUDA、Tiled CUDA
- 建立性能基准

```bash
make step1
```

### Step 2: 量化介绍 (45 分钟)
- 学习 Q4_0、Q8_0、Q8_1 格式
- 理解量化误差
- 掌握 sum 字段的作用

```bash
make step2
```

### Step 3: W4A16 GEMM (30 分钟)
- 实现权重量化 GEMM
- 理解在线反量化
- 观察内存节省

```bash
make step3
```

### Step 4: W4A8 GEMM ⭐ (60 分钟)
- **最重要的一步！**
- 深入理解补偿公式
- 学习 DP4A 优化
- 对比多种优化版本

```bash
make step4
```

### Step 5: llama.cpp 对比 (30 分钟)
- 验证兼容性
- 性能对比
- 了解差距

```bash
make step5
```

## 🔍 深入学习

### 阅读代码

推荐阅读顺序：

1. **类型定义**：`include/quant_types.h`
   - 理解量化格式的数据结构

2. **量化函数**：`include/quantize.h`
   - 学习量化/反量化算法

3. **参考实现**：`include/gemm_reference.h`
   - 理解正确的算法（ground truth）

4. **CUDA 实现**：
   - `include/gemm_cuda_naive.cuh` - 基础实现
   - `include/gemm_cuda_tiled.cuh` - Tiling 优化
   - `include/gemm_cuda_dp4a.cuh` - DP4A 优化

5. **测试代码**：`tests/step*.cu`
   - 看如何使用这些函数
   - 理解测试方法

### 修改和实验

尝试以下实验：

1. **调整 Tile 大小**：
   ```c
   // 在 gemm_cuda_tiled.cuh 中
   #define TILE_M 32  // 尝试 16, 32, 64
   #define TILE_N 32
   #define TILE_K 32
   ```

2. **测试不同矩阵大小**：
   ```c
   // 在 tests/step*.cu 中修改
   TestConfig test_configs[] = {
       {256, 4096, 4096, "Custom Test"},
   };
   ```

3. **添加性能分析**：
   ```bash
   # 使用 nsys 分析
   nsys profile --stats=true ./bin/step4_w4a8_gemm
   ```

## 🐛 常见问题

### Q: 编译时找不到 nvcc

```bash
# 确保激活了正确的 conda 环境
conda activate KM-12.8

# 或者添加 CUDA 到 PATH
export PATH=/usr/local/cuda/bin:$PATH
```

### Q: GPU 架构不匹配

```bash
# 检查你的 GPU 计算能力
nvidia-smi --query-gpu=compute_cap --format=csv

# 使用正确的架构编译
make clean
make CUDA_ARCH=sm_XX  # 替换 XX
```

### Q: 结果不正确

检查以下几点：
1. 是否使用了补偿公式（Step 4）
2. 量化/反量化是否正确
3. 内存布局是否正确

### Q: 性能很低

可能的原因：
1. GPU 架构设置不正确
2. Tile 大小不合适
3. 没有使用 DP4A 优化

## 📖 进一步学习

### 推荐资源

1. **CUDA 编程**：
   - [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
   - [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

2. **量化技术**：
   - [Quantization Survey](https://arxiv.org/abs/2103.13630)
   - [LLM.int8() Paper](https://arxiv.org/abs/2208.07339)

3. **llama.cpp**：
   - [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)
   - 阅读 `ggml/src/ggml-cuda/` 目录下的代码

### 下一步

完成本教程后，你可以：

1. **实现更多优化**：
   - Tensor Core (WMMA/MMA)
   - Stream-K 并行
   - 异步拷贝

2. **支持更多格式**：
   - Q5_0/Q5_1
   - Q6_K
   - MXFP4

3. **集成到实际项目**：
   - 在 llama.cpp 中测试
   - 在自己的推理引擎中使用

## 💡 学习建议

1. **动手实践**：不要只看代码，要运行和修改
2. **理解原理**：重点理解补偿公式的数学推导
3. **性能分析**：使用 nsys/nvprof 分析性能
4. **对比学习**：与 llama.cpp 的实现对比
5. **循序渐进**：按步骤学习，不要跳过

## 🎯 学习目标检查

完成本教程后，你应该能够：

- [ ] 解释 Q4_0、Q8_0、Q8_1 的区别
- [ ] 推导补偿公式
- [ ] 实现基本的量化 GEMM
- [ ] 使用 DP4A 优化
- [ ] 分析量化误差
- [ ] 理解 llama.cpp 的优化技术

---

**祝学习愉快！有问题请查看 README.md 或开 Issue。**
