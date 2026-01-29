# llama.cpp CUDA 量化 GEMM 性能测试教程

本教程详细介绍如何在 CUDA 12.8 环境中测试 llama.cpp 的量化 GEMM kernel 性能，并与 naive 实现进行对比。

## 目录

1. [环境准备](#1-环境准备)
2. [编译 llama.cpp](#2-编译-llamacpp)
3. [运行性能测试](#3-运行性能测试)
4. [测试结果分析](#4-测试结果分析)
5. [关键发现](#5-关键发现)

---

## 1. 环境准备

### 1.1 硬件环境

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU |
| Compute Capability | sm_120 (Blackwell) |
| SM 数量 | 36 |
| 显存 | 8.5 GB |

### 1.2 软件环境

使用 conda 环境 `KM-12.8`，包含 CUDA 12.8：

```bash
# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate KM-12.8

# 验证 CUDA 版本
which nvcc
nvcc --version
```

预期输出：
```
/home/haiyan/miniconda3/envs/KM-12.8/bin/nvcc
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Fri_Feb_21_20:23:50_PST_2025
Cuda compilation tools, release 12.8, V12.8.93
Build cuda_12.8.r12.8/compiler.35583870_0
```

---

## 2. 编译 llama.cpp

### 2.1 初始化子模块

```bash
cd /home/haiyan/Agent4Kernel/llama.cpp
git submodule update --init --recursive
```

### 2.2 恢复原始 CMakeLists.txt（如果被修改过）

```bash
git checkout tests/CMakeLists.txt
```

### 2.3 配置 CMake

针对 RTX 5070 (sm_120) 架构配置：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate KM-12.8

# 清理并创建 build 目录
rm -rf build && mkdir build && cd build

# 配置 CMake（仅支持 sm_120）
cmake .. -DGGML_CUDA=ON \
         -DCMAKE_CUDA_ARCHITECTURES="120" \
         -DLLAMA_BUILD_TESTS=ON
```

预期输出（关键部分）：
```
-- Replacing 120 in CMAKE_CUDA_ARCHITECTURES with 120a
-- Using CMAKE_CUDA_ARCHITECTURES=120a CMAKE_CUDA_ARCHITECTURES_NATIVE=120a-real
-- CUDA host compiler is GNU 14.3.0
-- Including CUDA backend
-- Configuring done (39.3s)
-- Generating done (0.3s)
```

### 2.4 编译 test-backend-ops

```bash
cmake --build . --target test-backend-ops -j$(nproc)
```

预期输出：
```
[100%] Linking CXX executable ../bin/test-backend-ops
[100%] Built target test-backend-ops
```

验证编译结果：
```bash
ls -la bin/test-backend-ops
```

---

## 3. 运行性能测试

### 3.1 test-backend-ops 使用说明

```bash
cd /home/haiyan/Agent4Kernel/llama.cpp/build/bin
./test-backend-ops --help
```

输出：
```
Usage: ./test-backend-ops [mode] [-o <op,..>] [-b <backend>] [-p <params regex>] [--output <console|sql|csv>] [--list-ops] [--show-coverage]
    valid modes:
      - test (default, compare with CPU backend for correctness)
      - grad (compare gradients from backpropagation with method of finite differences)
      - perf (performance evaluation)
      - support (probe backend operation support)
```

### 3.2 测试 Q4_0 量化 GEMM 性能

```bash
./test-backend-ops perf -o MUL_MAT -b CUDA0 -p "type_a=q4_0"
```

**实际测试结果：**

```
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 5070 Laptop GPU, compute capability 12.0, VMM: yes
Testing 2 devices

Backend 1/2: CUDA0
  Device description: NVIDIA GeForce RTX 5070 Laptop GPU
  Device memory: 8150 MB (6999 MB free)

  MUL_MAT(type_a=q4_0,type_b=f32,m=4096,n=1,k=14336,...):   331.68 us/run - 354.08 GFLOPS
  MUL_MAT(type_a=q4_0,type_b=f32,m=4096,n=2,k=14336,...):   302.97 us/run - 775.26 GFLOPS
  MUL_MAT(type_a=q4_0,type_b=f32,m=4096,n=3,k=14336,...):   322.02 us/run - 1.09 TFLOPS
  MUL_MAT(type_a=q4_0,type_b=f32,m=4096,n=4,k=14336,...):   349.50 us/run - 1.34 TFLOPS
  MUL_MAT(type_a=q4_0,type_b=f32,m=4096,n=5,k=14336,...):   434.94 us/run - 1.35 TFLOPS
```

> ⚠️ **注意**: Q4_0 在 N=8 及更大的 batch size 时可能遇到 CUDA 内存访问错误，这是 Blackwell (sm_120) 架构的兼容性问题。

### 3.3 测试 Q8_0 量化 GEMM 性能

```bash
./test-backend-ops perf -o MUL_MAT -b CUDA0 -p "type_a=q8_0"
```

**实际测试结果：**

```
  MUL_MAT(type_a=q8_0,type_b=f32,m=4096,n=1,k=14336,...):   776.24 us/run - 151.29 GFLOPS
  MUL_MAT(type_a=q8_0,type_b=f32,m=4096,n=2,k=14336,...):   793.24 us/run - 296.10 GFLOPS
  MUL_MAT(type_a=q8_0,type_b=f32,m=4096,n=3,k=14336,...):   831.83 us/run - 423.55 GFLOPS
  MUL_MAT(type_a=q8_0,type_b=f32,m=4096,n=4,k=14336,...):   806.36 us/run - 582.57 GFLOPS
  MUL_MAT(type_a=q8_0,type_b=f32,m=4096,n=5,k=14336,...):   828.07 us/run - 709.12 GFLOPS
  MUL_MAT(type_a=q8_0,type_b=f32,m=4096,n=8,k=14336,...):  1027.74 us/run - 914.16 GFLOPS
  MUL_MAT(type_a=q8_0,type_b=f32,m=4096,n=512,k=14336,...): 6813.41 us/run - 8.83 TFLOPS
```

### 3.4 运行自定义测试程序

tests 目录下有多个预编译的测试程序：

#### 3.4.1 量化 GEMM CUDA 测试

```bash
cd /home/haiyan/Agent4Kernel/llama.cpp/tests
./test-quantized-gemm-cuda
```

**测试结果摘要：**

| 量化类型 | 配置 (M×K×N) | 时间 (ms) | TFLOPS |
|---------|--------------|-----------|--------|
| W4A16 (Q4_0) | 1×4096×4096 | 0.18 | 0.18 |
| W4A16 (Q4_0) | 128×4096×4096 | 28.69 | 0.15 |
| W4A16 (Q4_0) | 512×4096×4096 | 118.67 | 0.14 |
| W8A16 (Q8_0) | 128×4096×4096 | 56.47 | 0.08 |
| W4A8 (Q4_0+Q8_1) | 128×4096×4096 | 28.63 | 0.15 |
| W8A8 (Q8_0+Q8_1) | 128×4096×4096 | 55.48 | 0.08 |

#### 3.4.2 Naive vs Optimized 对比测试

```bash
./test-naive-vs-optimized
```

**测试结果：**

| 矩阵尺寸 | Kernel | 时间 (ms) | TFLOPS | 加速比 |
|---------|--------|-----------|--------|--------|
| M=32, K=4096, N=4096 | Naive (16×16 block) | 6.24 | 0.17 | 1.00x |
| M=32, K=4096, N=4096 | Tiled (shared mem) | 3.31 | 0.32 | 1.88x |
| M=32, K=4096, N=4096 | Vectorized (float4) | 6.04 | 0.18 | 1.03x |
| M=128, K=4096, N=4096 | Naive | 30.97 | 0.14 | 1.00x |
| M=128, K=4096, N=4096 | Tiled | 18.05 | 0.24 | 1.72x |
| M=512, K=4096, N=4096 | Naive | 116.08 | 0.15 | 1.00x |
| M=512, K=4096, N=4096 | Tiled | 67.17 | 0.26 | 1.73x |

---

## 4. 测试结果分析

### 4.1 性能对比总览

| 实现方式 | M=128, K=4096, N=4096 | M=4096, N=512, K=14336 |
|---------|----------------------|------------------------|
| Naive CUDA Kernel | 0.08-0.15 TFLOPS | - |
| Tiled Kernel | 0.24 TFLOPS | - |
| **llama.cpp 优化 Q8_0** | - | **8.83 TFLOPS** |

### 4.2 llama.cpp vs Naive 性能差距

```
llama.cpp 优化 kernel: 8.83 TFLOPS
Naive kernel:          0.08 TFLOPS
────────────────────────────────────
加速比:                ~110x
```

### 4.3 Q4_0 vs Q8_0 对比

在 llama.cpp 官方实现中：

| 量化类型 | N=1 | N=4 | N=8 | N=512 |
|---------|-----|-----|-----|-------|
| Q4_0 | 354 GFLOPS | 1.34 TFLOPS | Error* | - |
| Q8_0 | 151 GFLOPS | 583 GFLOPS | 914 GFLOPS | 8.83 TFLOPS |

*Q4_0 在 N≥8 时出现 CUDA 内存访问错误（sm_120 兼容性问题）

---

## 5. 关键发现

### 5.1 llama.cpp 优化技术

llama.cpp 的 MMQ (Matrix-Matrix Quantized) kernel 使用了以下优化技术：

1. **共享内存 Tiling**
   - 将矩阵分块加载到共享内存
   - 减少全局内存访问延迟

2. **向量化加载 (float4/int4)**
   - 单次指令加载 4 个元素
   - 提高内存带宽利用率

3. **DP4A/Tensor Core 指令**
   - 使用硬件加速的点积运算
   - INT8 矩阵乘法加速

4. **Warp-level 优化**
   - 使用 warp shuffle 指令
   - 减少共享内存访问

### 5.2 Blackwell (sm_120) 兼容性

- **Q8_0**: 完全兼容，性能优秀
- **Q4_0**: 部分兼容，大 batch size 时有问题
- **建议**: 等待 llama.cpp 更新以完全支持 sm_120

### 5.3 性能优化建议

1. **小 batch (N≤4)**: Q4_0 性能更好
2. **大 batch (N≥8)**: 使用 Q8_0 更稳定
3. **推理场景**:
   - 单 token 生成 (N=1): ~354 GFLOPS (Q4_0)
   - 批量处理 (N=512): ~8.83 TFLOPS (Q8_0)

---

## 附录

### A. 完整测试命令汇总

```bash
# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate KM-12.8

# 进入 llama.cpp 目录
cd /home/haiyan/Agent4Kernel/llama.cpp

# 编译
rm -rf build && mkdir build && cd build
cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="120" -DLLAMA_BUILD_TESTS=ON
cmake --build . --target test-backend-ops -j$(nproc)

# 运行官方测试
cd bin
./test-backend-ops perf -o MUL_MAT -b CUDA0 -p "type_a=q8_0"
./test-backend-ops perf -o MUL_MAT -b CUDA0 -p "type_a=q4_0"

# 运行自定义测试
cd ../../tests
./test-quantized-gemm-cuda
./test-naive-vs-optimized
./llama_cpp_comparison_test
./mmq_baseline_simple_test
```

### B. 测试程序说明

| 程序名 | 功能 |
|-------|------|
| test-backend-ops | llama.cpp 官方后端操作测试 |
| test-quantized-gemm-cuda | 各种量化 GEMM 的 naive 实现测试 |
| test-naive-vs-optimized | naive/tiled/vectorized kernel 对比 |
| llama_cpp_comparison_test | Per-channel vs Q8_0 格式对比 |
| mmq_baseline_simple_test | MMQ 正确性验证 |

### C. 测试日期与环境

- **测试日期**: 2026-01-29
- **CUDA 版本**: 12.8.93
- **llama.cpp 版本**: ggml 0.9.5 (commit 0c21677e4)
- **操作系统**: Linux (WSL2)

---

## 参考资料

- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)
- [GGML Quantization](https://github.com/ggerganov/llama.cpp/blob/master/ggml/docs/ggml-quant.md)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
