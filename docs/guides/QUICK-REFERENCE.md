# llama.cpp CUDA 性能测试快速参考

本文档提供快速命令参考，用于在 CUDA 12.8 环境中测试 llama.cpp 的量化 GEMM 性能。

---

## 快速开始

### 1. 激活环境

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate KM-12.8
```

### 2. 编译 llama.cpp

```bash
cd /home/haiyan/Agent4Kernel/llama.cpp
rm -rf build && mkdir build && cd build
cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="120" -DLLAMA_BUILD_TESTS=ON
cmake --build . --target test-backend-ops -j$(nproc)
```

### 3. 运行测试

```bash
# 官方 llama.cpp 测试
cd bin
./test-backend-ops perf -o MUL_MAT -b CUDA0 -p "type_a=q8_0"

# 自定义测试
cd ../../tests
./test-quantized-gemm-cuda
./test-naive-vs-optimized
```

---

## 常用测试命令

### test-backend-ops (官方测试)

```bash
cd /home/haiyan/Agent4Kernel/llama.cpp/build/bin

# Q8_0 性能测试
./test-backend-ops perf -o MUL_MAT -b CUDA0 -p "type_a=q8_0"

# Q4_0 性能测试 (注意: N≥8 可能出错)
./test-backend-ops perf -o MUL_MAT -b CUDA0 -p "type_a=q4_0"

# 测试特定矩阵尺寸
./test-backend-ops perf -o MUL_MAT -b CUDA0 -p "type_a=q8_0.*m=4096.*n=512.*k=14336"

# 正确性测试
./test-backend-ops test -o MUL_MAT -b CUDA0 -p "type_a=q8_0"

# 列出所有操作
./test-backend-ops --list-ops
```

### 自定义测试程序

```bash
cd /home/haiyan/Agent4Kernel/llama.cpp/tests

# 量化 GEMM 性能测试 (W4A16, W8A16, W4A8, W8A8)
./test-quantized-gemm-cuda

# Naive vs Optimized 对比
./test-naive-vs-optimized

# llama.cpp 格式对比
./llama_cpp_comparison_test

# MMQ 正确性验证
./mmq_baseline_simple_test

# W4 量化对比
./llama_w4_comparison_test

# 量化类型对比
./quantization_comparison_test
```

---

## 性能基准参考

### RTX 5070 Laptop GPU (sm_120)

| 测试 | 配置 | 性能 |
|------|------|------|
| **llama.cpp Q8_0** | M=4096, N=512, K=14336 | **8.83 TFLOPS** |
| llama.cpp Q8_0 | M=4096, N=8, K=14336 | 0.91 TFLOPS |
| llama.cpp Q4_0 | M=4096, N=4, K=14336 | 1.34 TFLOPS |
| Naive W4A16 | M=128, K=4096, N=4096 | 0.15 TFLOPS |
| Naive W8A16 | M=128, K=4096, N=4096 | 0.08 TFLOPS |
| Tiled W4A16 | M=128, K=4096, N=4096 | 0.24 TFLOPS |

---

## 故障排除

### Q4_0 内存访问错误

**症状:**
```
CUDA error: an illegal memory access was encountered
```

**原因:** Blackwell (sm_120) 架构兼容性问题

**解决方案:**
1. 使用 Q8_0 代替 Q4_0
2. 限制 batch size (N<8)
3. 等待 llama.cpp 更新

### CUTLASS 警告

**症状:**
```
CMake Warning: cutlass not found, please set CUTLASS_DIR
```

**影响:** 不影响基本功能，仅影响高级优化

**解决方案:** 可忽略，或安装 NVIDIA CUTLASS

### 编译错误

**症状:** CMake 配置失败

**解决方案:**
```bash
# 恢复原始 CMakeLists.txt
git checkout tests/CMakeLists.txt

# 清理并重新编译
rm -rf build && mkdir build && cd build
cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="120" -DLLAMA_BUILD_TESTS=ON
```

---

## 测试输出解读

### test-backend-ops 输出

```
MUL_MAT(type_a=q8_0,type_b=f32,m=4096,n=512,k=14336,...): 148 runs - 6813.41 us/run - 60.13 GFLOP/run - 8.83 TFLOPS
```

- **148 runs**: 运行次数
- **6813.41 us/run**: 每次运行时间（微秒）
- **60.13 GFLOP/run**: 每次运行的浮点操作数
- **8.83 TFLOPS**: 计算吞吐量

### test-quantized-gemm-cuda 输出

```
║ M128_K4096_N4096    │ W4A16 (Q4_0)     │  128 x  4096 x  4096 │  28.6871 │      0.15 ║
```

- **M128_K4096_N4096**: 矩阵尺寸 (M×K×N)
- **W4A16 (Q4_0)**: 量化类型
- **28.6871**: 时间（毫秒）
- **0.15**: TFLOPS

---

## 环境信息

### 查看 GPU 信息

```bash
nvidia-smi

# 或使用 CUDA 工具
cd /home/haiyan/Agent4Kernel/llama.cpp/build/bin
./test-backend-ops perf -o MUL_MAT -b CUDA0 -p "type_a=q8_0" 2>&1 | head -5
```

### 查看 CUDA 版本

```bash
nvcc --version
```

### 查看编译配置

```bash
cd /home/haiyan/Agent4Kernel/llama.cpp/build
cat CMakeCache.txt | grep -E "CUDA|GGML"
```

---

## 性能优化建议

### 1. 选择合适的量化类型

- **小 batch (N≤4)**: 使用 Q4_0（更快）
- **大 batch (N≥8)**: 使用 Q8_0（更稳定）
- **推理场景**: Q8_0 在大 batch 下性能最佳

### 2. 矩阵尺寸优化

- **M**: batch size，越大越好（但受显存限制）
- **K**: 输入维度，通常固定
- **N**: 输出维度，影响并行度

### 3. 编译优化

```bash
# 针对特定架构优化
cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="120"

# 启用所有优化
cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
```

---

## 相关文档

- [完整教程](CUDA-GEMM-BENCHMARK-TUTORIAL.md) - 详细的测试教程
- [测试日志](CUDA-12.8-TEST-LOG.md) - 完整的测试输出记录
- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)

---

## 快速测试脚本

创建 `quick-test.sh`:

```bash
#!/bin/bash
set -e

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate KM-12.8

# 进入目录
cd /home/haiyan/Agent4Kernel/llama.cpp/build/bin

echo "=== Q8_0 性能测试 ==="
./test-backend-ops perf -o MUL_MAT -b CUDA0 -p "type_a=q8_0" 2>&1 | grep "TFLOPS"

echo ""
echo "=== Naive Kernel 测试 ==="
cd ../../tests
./test-quantized-gemm-cuda 2>&1 | grep "M128_K4096_N4096"

echo ""
echo "测试完成!"
```

使用方法:
```bash
chmod +x quick-test.sh
./quick-test.sh
```

---

**最后更新**: 2026-01-29
**测试环境**: CUDA 12.8, RTX 5070 Laptop GPU (sm_120)
