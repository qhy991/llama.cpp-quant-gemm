# 快速开始指南

本项目是一个独立的 CUDA kernel 开发学习工程，与 llama.cpp 完全兼容。

## 环境要求

- CUDA Toolkit 11.0+
- GPU: NVIDIA Pascal (SM 6.1) 或更新
- GCC 7+ (Linux) 或 MSVC 2019+ (Windows)

## 快速验证

```bash
# 检查 CUDA 安装
nvcc --version

# 检查 GPU
nvidia-smi
```

## 编译和运行

### 运行传统步骤测试

```bash
cd quant-gemm-from-scratch

# 编译所有程序
make

# 运行所有步骤测试
make test

# 运行单个步骤
make step4  # W4A8 量化 GEMM
```

### 运行单元测试

```bash
# 编译并运行所有单元测试
make unit-test

# 运行特定测试
make test-gemm-q4
```

### 自定义 GPU 架构

```bash
# RTX 30 系列 (Ampere)
make CUDA_ARCH=sm_86

# RTX 40 系列 (Ada Lovelace)
make CUDA_ARCH=sm_89

# A100/H100 (数据中心)
make CUDA_ARCH=sm_80  # A100
make CUDA_ARCH=sm_90  # H100
```

## 项目结构

```
quant-gemm-from-scratch/
├── compat/                 # llama.cpp 兼容层
│   └── ggml_types.h       # 量化类型定义
├── kernels/               # CUDA kernel 实现
│   ├── activation/        # 激活函数 (SiLU, GELU)
│   ├── gemm/             # 矩阵乘法
│   ├── normalization/    # 归一化 (RMSNorm)
│   └── ...
├── tests/                 # 测试
│   ├── framework/        # 测试框架
│   ├── unit/             # 单元测试
│   └── step*.cu          # 步骤测试
├── tutorials/             # 教程文档
└── Makefile
```

## 学习路径

### 初学者路线

1. [CUDA 基础](tutorials/01-cuda-basics/README.md)
2. [量化基础](tutorials/02-quantization-basics/README.md)
3. 运行 `make step1` - FP32 GEMM
4. 运行 `make step2` - 量化入门
5. 运行 `make step3` - W4A16 GEMM
6. 运行 `make step4` - W4A8 GEMM (DP4A)

### 进阶路线

1. 阅读 `kernels/gemm/gemm_cuda_dp4a.cuh` 源码
2. 理解补偿公式的推导
3. 编写自己的 kernel 并添加单元测试
4. 对比 llama.cpp 的实现

## 添加新算子

1. 在 `kernels/` 下创建新目录
2. 实现 CPU 参考和 GPU kernel
3. 在 `tests/unit/` 下创建测试
4. 更新 Makefile

示例:
```cpp
// kernels/myop/myop.cuh
__global__ void myop_kernel(...) { ... }

// tests/unit/test_myop.cu
class MyOpTest : public testing::TestCase { ... }
```

## 与 llama.cpp 集成

本项目的类型定义与 llama.cpp 完全兼容:

```cpp
// 本项目
#include "compat/ggml_types.h"
block_q4_0 weight;

// llama.cpp
#include "ggml-common.h"
block_q4_0 weight;  // 相同!
```

要将 kernel 集成到 llama.cpp:

1. 复制 kernel 代码到 `llama.cpp/ggml/src/ggml-cuda/`
2. 在 `mmq.cuh` 中调用你的 kernel
3. 运行 llama.cpp 测试验证

## 常见问题

**Q: 编译时找不到 nvcc**
```bash
export PATH=/usr/local/cuda/bin:$PATH
```

**Q: NMSE 过高**
- 检查数据布局 (row-major vs column-major)
- 检查补偿公式是否正确
- 增大测试阈值 (量化本身有误差)

**Q: kernel 崩溃**
- 检查内存越界
- 检查对齐
- 使用 `cuda-memcheck ./program`

## 获取帮助

```bash
make help
```
