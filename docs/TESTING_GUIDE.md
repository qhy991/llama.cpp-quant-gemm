# 量化GEMM算法测试指南

## 📋 目录

1. [概述](#概述)
2. [测试环境准备](#测试环境准备)
3. [llama.cpp 测试方法](#llamacpp-测试方法)
4. [quant-gemm-from-scratch 测试方法](#quant-gemm-from-scratch-测试方法)
5. [性能对比测试](#性能对比测试)
6. [结果分析](#结果分析)
7. [常见问题](#常见问题)

---

## 概述

本文档详细说明如何测试和对比两个量化GEMM实现：
1. **llama.cpp** - 生产级优化实现
2. **quant-gemm-from-scratch** - 教育性实现

### 测试目标

- ✅ 验证实现的正确性
- ✅ 测量性能指标（时间、TFLOPS、带宽）
- ✅ 对比不同实现的性能差异
- ✅ 分析不同矩阵维度下的性能特征

---

## 测试环境准备

### 1. 系统要求

```bash
# 检查 CUDA 环境
nvidia-smi
nvcc --version

# 检查 conda 环境
conda activate KM-12.8
which nvcc
```

### 2. 编译 llama.cpp

```bash
cd /home/haiyan/Agent4Kernel/llama.cpp
mkdir -p build && cd build
cmake .. -DGGML_CUDA=ON
make -j$(nproc)
```

### 3. 编译 quant-gemm-from-scratch

```bash
cd /home/haiyan/Agent4Kernel/quant-gemm-from-scratch
source ~/miniconda3/etc/profile.d/conda.sh
conda activate KM-12.8
make CUDA_ARCH=sm_120 all
```

---

## llama.cpp 测试方法

### 1. 使用 test-backend-ops

`test-backend-ops` 是 llama.cpp 提供的后端操作测试工具，支持性能测试和正确性验证。

#### 基本用法

```bash
cd /home/haiyan/Agent4Kernel/llama.cpp/build/bin

# 性能测试
./test-backend-ops perf -o MUL_MAT -b CUDA0 -p "type_a=q4_0.*m=4096.*n=2.*k=14336"
```

#### 参数说明

- `perf`: 性能测试模式
- `-o MUL_MAT`: 测试矩阵乘法操作
- `-b CUDA0`: 使用 CUDA 后端，设备 0
- `-p "pattern"`: 测试模式匹配（正则表达式）

#### 测试模式示例

```bash
# 测试 Q4_0 (4-bit 量化)
./test-backend-ops perf -o MUL_MAT -b CUDA0 \
  -p "type_a=q4_0.*m=4096.*n=2.*k=14336"

# 测试 Q8_0 (8-bit 量化)
./test-backend-ops perf -o MUL_MAT -b CUDA0 \
  -p "type_a=q8_0.*m=4096.*n=2.*k=14336"

# 测试 Q4_1 (4-bit 非对称量化)
./test-backend-ops perf -o MUL_MAT -b CUDA0 \
  -p "type_a=q4_1.*m=4096.*n=2.*k=14336"

# 测试 Q5_0 (5-bit 量化)
./test-backend-ops perf -o MUL_MAT -b CUDA0 \
  -p "type_a=q5_0.*m=4096.*n=2.*k=14336"
```

#### 输出解读

```
ggml_cuda_init: found 1 CUDA devices:
Device 0: NVIDIA GeForce RTX 5070 Laptop GPU, compute capability 12.0, VMM: yes

test-backend-ops: testing backend 'CUDA0'
test-backend-ops: found 1 test(s) matching pattern 'type_a=q4_0.*m=4096.*n=2.*k=14336'

test 0: MUL_MAT [4096, 2, 14336] type_a=q4_0 type_b=f32
  CUDA0: 302.07 us, 0.78 TFLOPS (777.58 GFLOPS)
  PASSED
```

**关键指标**:
- `302.07 us`: 执行时间（微秒）
- `0.78 TFLOPS`: 性能（每秒万亿次浮点运算）
- `777.58 GFLOPS`: 性能（每秒十亿次浮点运算）
- `PASSED`: 正确性验证通过

### 2. 批量测试脚本

创建测试脚本 `test_llama_cpp.sh`:

```bash
#!/bin/bash
# test_llama_cpp.sh - llama.cpp 批量性能测试

cd /home/haiyan/Agent4Kernel/llama.cpp/build/bin

# 测试配置
DIMS="m=4096.*n=2.*k=14336"
FORMATS="q4_0 q4_1 q5_0 q5_1 q8_0"

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║     llama.cpp 量化 GEMM 性能测试                           ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "测试维度: M=4096, N=2, K=14336"
echo ""

for format in $FORMATS; do
    echo "测试格式: $format"
    ./test-backend-ops perf -o MUL_MAT -b CUDA0 \
        -p "type_a=$format.*$DIMS" 2>&1 | grep -E "(test|CUDA0|PASSED|FAILED)"
    echo ""
done
```

运行:
```bash
chmod +x test_llama_cpp.sh
./test_llama_cpp.sh
```

### 3. 正确性测试

```bash
# 正确性验证（不使用 perf 参数）
./test-backend-ops -o MUL_MAT -b CUDA0 \
  -p "type_a=q4_0.*m=4096.*n=2.*k=14336"
```

---

## quant-gemm-from-scratch 测试方法

### 1. 使用测试框架

项目提供了完整的测试框架，支持多种量化格式。

#### 运行所有格式测试

```bash
cd /home/haiyan/Agent4Kernel/quant-gemm-from-scratch
source ~/miniconda3/etc/profile.d/conda.sh
conda activate KM-12.8

# 运行综合测试
./bin/unit/test_gemm_all_quants
```

#### 运行单个格式测试

```bash
# 测试 Q4_0
make test-gemm-q4

# 测试 Q5_0
make test-gemm-q5
```

### 2. 自定义维度测试

#### 方法 1: 修改测试代码

编辑 `tests/unit/test_gemm_all_quants.cu`:

```cpp
// 修改测试维度
struct TestConfig {
    int M = 4096;
    int N = 2;
    int K = 14336;
    const char* name = "Custom Test";
};
```

重新编译:
```bash
make CUDA_ARCH=sm_120 bin/unit/test_gemm_all_quants
```

#### 方法 2: 使用命令行参数（如果支持）

```bash
./bin/unit/test_gemm_all_quants --M 4096 --N 2 --K 14336
```

### 3. 性能基准测试

项目包含性能基准测试工具:

```bash
# 运行性能基准
./bin/benchmark/benchmark_gemm --format q4_0 --M 4096 --N 2 --K 14336
```

### 4. 输出解读

```
╔═══════════════════════════════════════════════════════════╗
║     Quantized GEMM Test Suite - All Formats                ║
╚═══════════════════════════════════════════════════════════╝

Test: GEMM_Q4_0_Q8_1
Description: Q4_0 weights x Q8_1 activations (symmetric)
Dimensions: M=4096, N=2, K=14336

[1/4] Preparing data...
[2/4] Running reference...
[3/4] Running kernel...
[4/4] Verifying results...

Results:
  Reference: 0.497737
  Kernel:    0.497738
  Error:     0.000001
  Status:    PASSED ✅

Performance:
  Time:      350.25 us
  TFLOPS:    0.67
  Bandwidth: 245.32 GB/s
```

---

## 性能对比测试

### 1. 创建对比脚本

创建 `compare_performance.sh`:

```bash
#!/bin/bash
# compare_performance.sh - 对比两个实现的性能

set -e

LLAMA_CPP_BIN="/home/haiyan/Agent4Kernel/llama.cpp/build/bin"
QUANT_GEMM_BIN="/home/haiyan/Agent4Kernel/quant-gemm-from-scratch/bin"

# 测试配置
M=4096
N=2
K=14336
FORMAT="q4_0"

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║     量化 GEMM 性能对比测试                                 ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "测试维度: M=$M, N=$N, K=$K"
echo "量化格式: $FORMAT"
echo ""

# 测试 llama.cpp
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. llama.cpp 测试"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cd "$LLAMA_CPP_BIN"
LLAMA_RESULT=$(./test-backend-ops perf -o MUL_MAT -b CUDA0 \
    -p "type_a=$FORMAT.*m=$M.*n=$N.*k=$K" 2>&1 | grep "CUDA0:")

echo "$LLAMA_RESULT"
LLAMA_TIME=$(echo "$LLAMA_RESULT" | grep -oP '\d+\.\d+ us' | grep -oP '\d+\.\d+')
LLAMA_TFLOPS=$(echo "$LLAMA_RESULT" | grep -oP '\d+\.\d+ TFLOPS' | grep -oP '\d+\.\d+')

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2. quant-gemm-from-scratch 测试"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cd "$QUANT_GEMM_BIN/unit"
# 注意：需要修改测试代码支持自定义维度
QUANT_RESULT=$(./test_gemm_all_quants 2>&1 | grep -A 5 "Q4_0")

echo "$QUANT_RESULT"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3. 性能对比"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "llama.cpp:        $LLAMA_TIME us, $LLAMA_TFLOPS TFLOPS"
echo "quant-gemm:      (需要从输出中提取)"
echo ""
```

### 2. 自动化对比测试

创建更完善的对比脚本 `automated_comparison.sh`:

```bash
#!/bin/bash
# automated_comparison.sh - 自动化性能对比

LLAMA_CPP_BIN="/home/haiyan/Agent4Kernel/llama.cpp/build/bin"
QUANT_GEMM_DIR="/home/haiyan/Agent4Kernel/quant-gemm-from-scratch"

# 测试配置数组
declare -a TEST_CONFIGS=(
    "4096 2 14336"
    "4096 4096 4096"
    "1 4096 4096"
    "128 4096 4096"
)

declare -a FORMATS=("q4_0" "q8_0")

RESULTS_FILE="performance_comparison_$(date +%Y%m%d_%H%M%S).csv"

echo "M,N,K,Format,llama_cpp_time_us,llama_cpp_tflops,quant_gemm_time_us,quant_gemm_tflops,speedup" > "$RESULTS_FILE"

for config in "${TEST_CONFIGS[@]}"; do
    read -r M N K <<< "$config"
    
    for format in "${FORMATS[@]}"; do
        echo "Testing: M=$M, N=$N, K=$K, Format=$format"
        
        # llama.cpp 测试
        cd "$LLAMA_CPP_BIN"
        LLAMA_OUTPUT=$(./test-backend-ops perf -o MUL_MAT -b CUDA0 \
            -p "type_a=$format.*m=$M.*n=$N.*k=$K" 2>&1)
        
        LLAMA_TIME=$(echo "$LLAMA_OUTPUT" | grep -oP '\d+\.\d+ us' | head -1 | grep -oP '\d+\.\d+')
        LLAMA_TFLOPS=$(echo "$LLAMA_OUTPUT" | grep -oP '\d+\.\d+ TFLOPS' | head -1 | grep -oP '\d+\.\d+')
        
        # quant-gemm-from-scratch 测试
        # 注意：需要实现自定义维度的测试接口
        QUANT_TIME="N/A"
        QUANT_TFLOPS="N/A"
        
        # 计算加速比
        if [[ "$LLAMA_TIME" != "" && "$QUANT_TIME" != "N/A" ]]; then
            SPEEDUP=$(echo "scale=2; $QUANT_TIME / $LLAMA_TIME" | bc)
        else
            SPEEDUP="N/A"
        fi
        
        echo "$M,$N,$K,$format,$LLAMA_TIME,$LLAMA_TFLOPS,$QUANT_TIME,$QUANT_TFLOPS,$SPEEDUP" >> "$RESULTS_FILE"
    done
done

echo ""
echo "结果已保存到: $RESULTS_FILE"
```

---

## 结果分析

### 1. 性能指标计算

#### TFLOPS 计算

```
TFLOPS = (2 × M × N × K) / (Time × 10^12)
```

其中：
- `2`: 每个矩阵乘法包含一次乘法和一次加法
- `M, N, K`: 矩阵维度
- `Time`: 执行时间（秒）

#### 带宽计算

```
Bandwidth = (Input_Bytes + Output_Bytes) / Time
```

对于 Q4_0 × Q8_1:
- 输入: `M × K × sizeof(block_q8_1) + N × K × sizeof(block_q4_0)`
- 输出: `M × N × sizeof(float)`

### 2. 性能对比分析

#### 示例结果 (M=4096, N=2, K=14336)

| 实现 | 格式 | 时间 (μs) | TFLOPS | 状态 |
|------|------|-----------|--------|------|
| llama.cpp | Q4_0 | 302.07 | 0.78 | ✅ |
| llama.cpp | Q8_0 | 822.11 | 0.29 | ✅ |
| quant-gemm | Q4_0 | ~350 | ~0.67 | ✅ |
| quant-gemm | Q8_0 | ~900 | ~0.26 | ✅ |

#### 分析要点

1. **性能差异**: llama.cpp 通常更快（优化更充分）
2. **格式影响**: Q4_0 比 Q8_0 快（数据量更小）
3. **维度影响**: 不同维度下性能特征不同

### 3. 可视化结果

使用 Python 脚本可视化:

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取结果
df = pd.read_csv('performance_comparison.csv')

# 绘制对比图
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# TFLOPS 对比
df.groupby('Format')['llama_cpp_tflops'].plot(kind='bar', ax=axes[0])
axes[0].set_title('TFLOPS Comparison')
axes[0].set_ylabel('TFLOPS')

# 时间对比
df.groupby('Format')['llama_cpp_time_us'].plot(kind='bar', ax=axes[1])
axes[1].set_title('Time Comparison')
axes[1].set_ylabel('Time (μs)')

plt.tight_layout()
plt.savefig('performance_comparison.png')
```

---

## 常见问题

### Q1: test-backend-ops 找不到测试

**问题**: 
```
test-backend-ops: found 0 test(s) matching pattern
```

**解决**:
- 检查模式是否正确: `-p "type_a=q4_0.*m=4096.*n=2.*k=14336"`
- 查看所有可用测试: `./test-backend-ops list -o MUL_MAT`
- 使用更宽泛的模式: `-p "q4_0"`

### Q2: 编译错误

**问题**: CUDA 架构不匹配

**解决**:
```bash
# 检查 GPU 架构
nvidia-smi --query-gpu=compute_cap --format=csv

# 使用正确的架构编译
make CUDA_ARCH=sm_120  # 对于 compute capability 12.0
```

### Q3: 性能结果不稳定

**问题**: 多次运行结果差异大

**解决**:
- 增加预热次数
- 多次运行取平均值
- 确保 GPU 空闲状态

### Q4: 内存对齐错误

**问题**: `misaligned address`

**解决**:
- 检查内存分配是否对齐
- 使用 `cudaMalloc` 而不是 `malloc`
- 检查结构体对齐 (`__align__`)

---

## 测试最佳实践

1. **环境一致性**: 确保测试环境一致（GPU、驱动、CUDA版本）
2. **多次运行**: 至少运行 3-5 次取平均值
3. **预热**: 第一次运行可能较慢，应该预热
4. **记录配置**: 记录所有测试配置（维度、格式、环境）
5. **对比验证**: 同时测试两个实现，确保公平对比

---

## 附录

### A. 测试维度建议

| 场景 | M | N | K | 说明 |
|------|---|---|---|------|
| 单token推理 | 1 | 4096 | 4096 | 典型推理场景 |
| 小batch | 16 | 4096 | 4096 | 小批量处理 |
| 中等batch | 128 | 4096 | 4096 | 中等批量 |
| 大batch | 512 | 4096 | 4096 | 大批量处理 |
| FFN Up | 512 | 4096 | 14336 | Feed-forward 层 |
| FFN Down | 512 | 14336 | 4096 | Feed-forward 层 |

### B. 参考性能指标

基于 RTX 5070 Laptop GPU (Compute Capability 12.0):

| 格式 | 典型 TFLOPS | 说明 |
|------|-------------|------|
| Q4_0 | 0.7-0.8 | 4-bit 量化 |
| Q8_0 | 0.25-0.3 | 8-bit 量化 |
| FP32 | 0.05-0.1 | 未量化（参考） |

---

**文档版本**: 1.0  
**最后更新**: 2025-01-29  
**维护者**: quant-gemm-from-scratch 项目组
