# 量化GEMM算法详细测试文档

## 📋 概述

本文档详细说明如何测试和对比两个量化GEMM算法的实现：
1. **llama.cpp** - 生产级优化实现（MMQ优化）
2. **quant-gemm-from-scratch** - 教育性实现（DP4A优化）

### 测试场景：M=4096, N=2, K=14336

这是一个典型的**FFN Up层**场景：
- **M=4096**: 序列长度（batch size × sequence length）
- **N=2**: 输出特征数（非常小的输出维度，用于测试极端情况）
- **K=14336**: 隐藏层维度（典型的LLM FFN层大小）

---

## 🔧 环境准备

### 1. 检查环境

```bash
# 检查 CUDA
nvidia-smi
nvcc --version

# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate KM-12.8

# 验证环境
which nvcc
echo $CUDA_HOME
```

### 2. 编译 llama.cpp

```bash
cd /home/haiyan/Agent4Kernel/llama.cpp
mkdir -p build && cd build

# 配置（如果还没配置）
cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release

# 编译测试工具
make test-backend-ops -j$(nproc)

# 验证编译
ls -lh bin/test-backend-ops
```

### 3. 编译 quant-gemm-from-scratch

```bash
cd /home/haiyan/Agent4Kernel/quant-gemm-from-scratch

# 使用 Makefile 编译
make CUDA_ARCH=sm_120 all

# 或使用 CMake
mkdir -p build && cd build
cmake .. -DCUDA_ARCHITECTURES=120
cmake --build . -j$(nproc)
```

---

## 📊 llama.cpp 测试方法

### 1. 基本性能测试

#### Q4_0 测试

```bash
cd /home/haiyan/Agent4Kernel/llama.cpp/build/bin

./test-backend-ops perf -o MUL_MAT -b CUDA0 \
  -p "type_a=q4_0.*m=4096.*n=2.*k=14336"
```

**预期输出**:
```
ggml_cuda_init: found 1 CUDA devices:
Device 0: NVIDIA GeForce RTX 5070 Laptop GPU, compute capability 12.0, VMM: yes

test-backend-ops: testing backend 'CUDA0'
test-backend-ops: found 1 test(s) matching pattern 'type_a=q4_0.*m=4096.*n=2.*k=14336'

test 0: MUL_MAT [4096, 2, 14336] type_a=q4_0 type_b=f32
  CUDA0: 302.07 us, 0.78 TFLOPS (777.58 GFLOPS)
  PASSED
```

**关键指标提取**:
- **时间**: `302.07 us` (微秒)
- **性能**: `0.78 TFLOPS` (每秒万亿次浮点运算)
- **状态**: `PASSED` (正确性验证通过)

#### Q8_0 测试

```bash
./test-backend-ops perf -o MUL_MAT -b CUDA0 \
  -p "type_a=q8_0.*m=4096.*n=2.*k=14336"
```

**预期输出**:
```
test 0: MUL_MAT [4096, 2, 14336] type_a=q8_0 type_b=f32
  CUDA0: 822.11 us, 0.29 TFLOPS (285.71 GFLOPS)
  PASSED
```

### 2. 批量测试脚本

创建 `test_llama_cpp_batch.sh`:

```bash
#!/bin/bash
# test_llama_cpp_batch.sh - llama.cpp 批量测试脚本

set -e

LLAMA_BIN="/home/haiyan/Agent4Kernel/llama.cpp/build/bin"
cd "$LLAMA_BIN"

# 测试配置
M=4096
N=2
K=14336

# 测试格式
FORMATS=("q4_0" "q4_1" "q5_0" "q5_1" "q8_0")

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║     llama.cpp 量化 GEMM 性能测试                           ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "测试维度: M=$M, N=$N, K=$K"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

# 结果文件
RESULTS_FILE="llama_cpp_results_$(date +%Y%m%d_%H%M%S).txt"
echo "Format,Time_us,TFLOPS,GFLOPS,Status" > "$RESULTS_FILE"

for format in "${FORMATS[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "测试格式: $format"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    OUTPUT=$(./test-backend-ops perf -o MUL_MAT -b CUDA0 \
        -p "type_a=$format.*m=$M.*n=$N.*k=$K" 2>&1)
    
    # 提取结果
    TIME=$(echo "$OUTPUT" | grep -oP '\d+\.\d+ us' | head -1 | grep -oP '\d+\.\d+')
    TFLOPS=$(echo "$OUTPUT" | grep -oP '\d+\.\d+ TFLOPS' | head -1 | grep -oP '\d+\.\d+')
    GFLOPS=$(echo "$OUTPUT" | grep -oP '\d+\.\d+ GFLOPS' | head -1 | grep -oP '\d+\.\d+')
    STATUS=$(echo "$OUTPUT" | grep -E "PASSED|FAILED" | head -1)
    
    echo "  时间:   ${TIME} us"
    echo "  性能:   ${TFLOPS} TFLOPS (${GFLOPS} GFLOPS)"
    echo "  状态:   ${STATUS}"
    echo ""
    
    # 保存结果
    echo "$format,$TIME,$TFLOPS,$GFLOPS,$STATUS" >> "$RESULTS_FILE"
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "结果已保存到: $RESULTS_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
```

运行:
```bash
chmod +x test_llama_cpp_batch.sh
./test_llama_cpp_batch.sh
```

### 3. 正确性验证

```bash
# 不使用 perf 参数，进行正确性验证
./test-backend-ops -o MUL_MAT -b CUDA0 \
  -p "type_a=q4_0.*m=4096.*n=2.*k=14336"
```

---

## 📊 quant-gemm-from-scratch 测试方法

### 1. 使用测试框架

#### 运行所有格式测试

```bash
cd /home/haiyan/Agent4Kernel/quant-gemm-from-scratch
source ~/miniconda3/etc/profile.d/conda.sh
conda activate KM-12.8

# 运行综合测试
./bin/unit/test_gemm_all_quants
```

#### 输出解读

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

### 2. 自定义维度测试

#### 方法 1: 修改测试代码

编辑 `tests/unit/test_gemm_all_quants.cu`，找到测试配置部分：

```cpp
// 原始配置
struct TestConfig {
    int M = 4;
    int N = 512;
    int K = 1024;
    const char* name = "Default Test";
};

// 修改为
struct TestConfig {
    int M = 4096;
    int N = 2;
    int K = 14336;
    const char* name = "FFN Up Layer Test";
};
```

重新编译:
```bash
make CUDA_ARCH=sm_120 bin/unit/test_gemm_all_quants
```

#### 方法 2: 创建自定义测试程序

创建 `tests/unit/test_custom_dim.cu`:

```cpp
#include "../framework/test_framework.cuh"
#include "../../kernels/gemm/gemm_quant_formats.cuh"

int main() {
    // 自定义测试维度
    int M = 4096;
    int N = 2;
    int K = 14336;
    
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║     自定义维度测试: M=%d, N=%d, K=%d                      ║\n", M, N, K);
    printf("╚═══════════════════════════════════════════════════════════╝\n");
    
    // 运行测试...
    // (使用测试框架的API)
    
    return 0;
}
```

编译:
```bash
make CUDA_ARCH=sm_120 bin/unit/test_custom_dim
```

### 3. 性能基准测试

如果项目包含基准测试工具:

```bash
# 运行性能基准
./bin/benchmark/benchmark_gemm \
    --format q4_0 \
    --M 4096 --N 2 --K 14336 \
    --warmup 10 --repeat 100
```

---

## 🔄 性能对比测试

### 1. 手动对比

#### Step 1: 运行 llama.cpp 测试

```bash
cd /home/haiyan/Agent4Kernel/llama.cpp/build/bin

# Q4_0 测试
./test-backend-ops perf -o MUL_MAT -b CUDA0 \
  -p "type_a=q4_0.*m=4096.*n=2.*k=14336" > llama_q4_0_result.txt 2>&1

# 提取关键信息
grep "CUDA0:" llama_q4_0_result.txt
```

#### Step 2: 运行 quant-gemm-from-scratch 测试

```bash
cd /home/haiyan/Agent4Kernel/quant-gemm-from-scratch

# 运行测试（需要先修改维度）
./bin/unit/test_gemm_all_quants > quant_q4_0_result.txt 2>&1

# 提取关键信息
grep -A 5 "Performance:" quant_q4_0_result.txt
```

#### Step 3: 对比结果

创建对比表格:

| 实现 | 格式 | 时间 (μs) | TFLOPS | 状态 |
|------|------|-----------|--------|------|
| llama.cpp | Q4_0 | 302.07 | 0.78 | ✅ PASSED |
| quant-gemm | Q4_0 | ~350 | ~0.67 | ✅ PASSED |
| llama.cpp | Q8_0 | 822.11 | 0.29 | ✅ PASSED |
| quant-gemm | Q8_0 | ~900 | ~0.26 | ✅ PASSED |

### 2. 自动化对比脚本

创建 `compare_algorithms.sh`:

```bash
#!/bin/bash
# compare_algorithms.sh - 自动化算法对比

set -e

# 配置
M=4096
N=2
K=14336
FORMAT="q4_0"

LLAMA_BIN="/home/haiyan/Agent4Kernel/llama.cpp/build/bin"
QUANT_GEMM_DIR="/home/haiyan/Agent4Kernel/quant-gemm-from-scratch"

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║     量化 GEMM 算法性能对比                                  ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "测试维度: M=$M, N=$N, K=$K"
echo "量化格式: $FORMAT"
echo ""

# 结果文件
RESULTS_CSV="comparison_results_$(date +%Y%m%d_%H%M%S).csv"
echo "Implementation,Format,Time_us,TFLOPS,Status" > "$RESULTS_CSV"

# 1. llama.cpp 测试
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. llama.cpp 测试"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cd "$LLAMA_BIN"

LLAMA_OUTPUT=$(./test-backend-ops perf -o MUL_MAT -b CUDA0 \
    -p "type_a=$FORMAT.*m=$M.*n=$N.*k=$K" 2>&1)

LLAMA_TIME=$(echo "$LLAMA_OUTPUT" | grep -oP '\d+\.\d+ us' | head -1 | grep -oP '\d+\.\d+')
LLAMA_TFLOPS=$(echo "$LLAMA_OUTPUT" | grep -oP '\d+\.\d+ TFLOPS' | head -1 | grep -oP '\d+\.\d+')
LLAMA_STATUS=$(echo "$LLAMA_OUTPUT" | grep -E "PASSED|FAILED" | head -1 | tr -d ' ')

echo "  时间:   ${LLAMA_TIME} us"
echo "  性能:   ${LLAMA_TFLOPS} TFLOPS"
echo "  状态:   ${LLAMA_STATUS}"
echo ""

echo "llama.cpp,$FORMAT,$LLAMA_TIME,$LLAMA_TFLOPS,$LLAMA_STATUS" >> "$RESULTS_CSV"

# 2. quant-gemm-from-scratch 测试
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2. quant-gemm-from-scratch 测试"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cd "$QUANT_GEMM_DIR"

# 注意：需要先修改测试维度或使用自定义测试
QUANT_OUTPUT=$(./bin/unit/test_gemm_all_quants 2>&1 | grep -A 10 "Q4_0")

QUANT_TIME=$(echo "$QUANT_OUTPUT" | grep -oP 'Time:\s+\d+\.\d+' | grep -oP '\d+\.\d+' || echo "N/A")
QUANT_TFLOPS=$(echo "$QUANT_OUTPUT" | grep -oP 'TFLOPS:\s+\d+\.\d+' | grep -oP '\d+\.\d+' || echo "N/A")
QUANT_STATUS=$(echo "$QUANT_OUTPUT" | grep -E "PASSED|FAILED" | head -1 | tr -d ' ' || echo "N/A")

echo "  时间:   ${QUANT_TIME} us"
echo "  性能:   ${QUANT_TFLOPS} TFLOPS"
echo "  状态:   ${QUANT_STATUS}"
echo ""

echo "quant-gemm,$FORMAT,$QUANT_TIME,$QUANT_TFLOPS,$QUANT_STATUS" >> "$RESULTS_CSV"

# 3. 对比分析
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3. 性能对比"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ "$LLAMA_TIME" != "" && "$QUANT_TIME" != "N/A" ]]; then
    SPEEDUP=$(echo "scale=2; $QUANT_TIME / $LLAMA_TIME" | bc)
    EFFICIENCY=$(echo "scale=2; $LLAMA_TFLOPS / $QUANT_TFLOPS" | bc)
    
    echo "  llama.cpp:        ${LLAMA_TIME} us, ${LLAMA_TFLOPS} TFLOPS"
    echo "  quant-gemm:       ${QUANT_TIME} us, ${QUANT_TFLOPS} TFLOPS"
    echo "  加速比:           ${SPEEDUP}x (llama.cpp 更快)"
    echo "  效率比:           ${EFFICIENCY}x (llama.cpp 更高效)"
else
    echo "  无法计算对比（数据不完整）"
fi

echo ""
echo "结果已保存到: $RESULTS_CSV"
```

运行:
```bash
chmod +x compare_algorithms.sh
./compare_algorithms.sh
```

---

## 📈 结果分析

### 1. 性能指标计算

#### TFLOPS 计算公式

```
TFLOPS = (2 × M × N × K) / (Time × 10^12)
```

对于 M=4096, N=2, K=14336:
```
理论 FLOPS = 2 × 4096 × 2 × 14336 = 234,881,024 次运算

如果时间 = 302.07 us = 0.00030207 秒:
TFLOPS = 234,881,024 / (0.00030207 × 10^12) = 0.78 TFLOPS
```

#### 带宽计算

对于 Q4_0 × Q8_1:
```
输入数据:
  - Activation (Q8_1): M × K × sizeof(block_q8_1) / QK8_1
    = 4096 × 14336 × 36 / 32 = 66,060,288 bytes
  - Weight (Q4_0): N × K × sizeof(block_q4_0) / QK4_0
    = 2 × 14336 × 18 / 32 = 16,128 bytes

输出数据:
  - Output (FP32): M × N × sizeof(float)
    = 4096 × 2 × 4 = 32,768 bytes

总数据量 = 66,060,288 + 16,128 + 32,768 = 66,109,184 bytes ≈ 63.05 MB

带宽 = 66,109,184 / (302.07 × 10^-6) = 219.06 GB/s
```

### 2. 性能对比分析

#### 示例结果 (M=4096, N=2, K=14336)

| 实现 | 格式 | 时间 (μs) | TFLOPS | 带宽 (GB/s) | 状态 |
|------|------|-----------|--------|-------------|------|
| llama.cpp | Q4_0 | 302.07 | 0.78 | ~219 | ✅ |
| quant-gemm | Q4_0 | ~350 | ~0.67 | ~189 | ✅ |
| llama.cpp | Q8_0 | 822.11 | 0.29 | ~80 | ✅ |
| quant-gemm | Q8_0 | ~900 | ~0.26 | ~73 | ✅ |

#### 分析要点

1. **llama.cpp 性能优势**:
   - Q4_0: 约 16% 更快 (302 vs 350 μs)
   - Q8_0: 约 9% 更快 (822 vs 900 μs)
   - 原因: MMQ优化、更好的内存访问模式

2. **格式影响**:
   - Q4_0 比 Q8_0 快约 2.7x (数据量更小)
   - 4-bit 量化显著减少内存带宽需求

3. **维度特征**:
   - N=2 是极端情况（输出维度很小）
   - 这种维度下，内存带宽可能是瓶颈
   - 计算强度较低，难以充分利用GPU

### 3. 可视化结果

使用 Python 脚本:

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取结果
df = pd.read_csv('comparison_results.csv')

# 创建对比图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 时间对比
formats = df['Format'].unique()
llama_times = [df[(df['Implementation']=='llama.cpp') & (df['Format']==f)]['Time_us'].values[0] 
               for f in formats]
quant_times = [df[(df['Implementation']=='quant-gemm') & (df['Format']==f)]['Time_us'].values[0] 
               for f in formats]

x = np.arange(len(formats))
width = 0.35
axes[0, 0].bar(x - width/2, llama_times, width, label='llama.cpp', color='#2E86AB')
axes[0, 0].bar(x + width/2, quant_times, width, label='quant-gemm', color='#A23B72')
axes[0, 0].set_xlabel('Format')
axes[0, 0].set_ylabel('Time (μs)')
axes[0, 0].set_title('Execution Time Comparison')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(formats)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. TFLOPS 对比
llama_tflops = [df[(df['Implementation']=='llama.cpp') & (df['Format']==f)]['TFLOPS'].values[0] 
                for f in formats]
quant_tflops = [df[(df['Implementation']=='quant-gemm') & (df['Format']==f)]['TFLOPS'].values[0] 
                for f in formats]

axes[0, 1].bar(x - width/2, llama_tflops, width, label='llama.cpp', color='#2E86AB')
axes[0, 1].bar(x + width/2, quant_tflops, width, label='quant-gemm', color='#A23B72')
axes[0, 1].set_xlabel('Format')
axes[0, 1].set_ylabel('TFLOPS')
axes[0, 1].set_title('Performance Comparison')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(formats)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. 加速比
speedups = [llama_times[i] / quant_times[i] for i in range(len(formats))]
axes[1, 0].bar(formats, speedups, color='#F18F01')
axes[1, 0].axhline(y=1.0, color='r', linestyle='--', label='Baseline')
axes[1, 0].set_xlabel('Format')
axes[1, 0].set_ylabel('Speedup (llama.cpp / quant-gemm)')
axes[1, 0].set_title('Speedup Ratio')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. 效率对比
efficiencies = [llama_tflops[i] / quant_tflops[i] for i in range(len(formats))]
axes[1, 1].bar(formats, efficiencies, color='#C73E1D')
axes[1, 1].axhline(y=1.0, color='r', linestyle='--', label='Baseline')
axes[1, 1].set_xlabel('Format')
axes[1, 1].set_ylabel('Efficiency Ratio')
axes[1, 1].set_title('Efficiency Comparison')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
print("图表已保存到: performance_comparison.png")
```

---

## 🐛 常见问题

### Q1: test-backend-ops 找不到测试

**错误**:
```
test-backend-ops: found 0 test(s) matching pattern
```

**解决**:
```bash
# 1. 查看所有可用测试
./test-backend-ops list -o MUL_MAT

# 2. 使用更宽泛的模式
./test-backend-ops perf -o MUL_MAT -b CUDA0 -p "q4_0"

# 3. 检查模式语法
# 正确: -p "type_a=q4_0.*m=4096.*n=2.*k=14336"
# 错误: -p "q4_0 m=4096 n=2 k=14336"
```

### Q2: 编译错误 - CUDA 架构不匹配

**错误**:
```
nvcc fatal: Unsupported gpu architecture 'sm_75'
```

**解决**:
```bash
# 1. 检查 GPU 架构
nvidia-smi --query-gpu=compute_cap --format=csv,noheader

# 2. 使用正确的架构
# RTX 5070 = compute capability 12.0 = sm_120
make CUDA_ARCH=sm_120
```

### Q3: 性能结果不稳定

**现象**: 多次运行结果差异大

**解决**:
```bash
# 1. 确保 GPU 空闲
nvidia-smi

# 2. 设置 GPU 性能模式
sudo nvidia-smi -pm 1
sudo nvidia-smi -pl 250  # 设置功耗限制（可选）

# 3. 多次运行取平均
for i in {1..5}; do
    ./test-backend-ops perf -o MUL_MAT -b CUDA0 \
        -p "type_a=q4_0.*m=4096.*n=2.*k=14336" | grep "CUDA0:"
done | awk '{sum+=$2; count++} END {print "Average:", sum/count}'
```

### Q4: 内存对齐错误

**错误**:
```
CUDA error: misaligned address
```

**解决**:
- 检查内存分配是否对齐
- 使用 `cudaMalloc` 而不是 `malloc`
- 检查结构体对齐 (`__align__`)

---

## 📝 测试报告模板

### 测试报告示例

```markdown
# 量化GEMM性能测试报告

## 测试环境
- GPU: NVIDIA GeForce RTX 5070 Laptop GPU
- Compute Capability: 12.0
- CUDA Version: 12.8
- 测试时间: 2025-01-29

## 测试配置
- 维度: M=4096, N=2, K=14336
- 格式: Q4_0, Q8_0
- 重复次数: 5次取平均

## 测试结果

### llama.cpp
| 格式 | 时间 (μs) | TFLOPS | 状态 |
|------|-----------|--------|------|
| Q4_0 | 302.07 | 0.78 | ✅ PASSED |
| Q8_0 | 822.11 | 0.29 | ✅ PASSED |

### quant-gemm-from-scratch
| 格式 | 时间 (μs) | TFLOPS | 状态 |
|------|-----------|--------|------|
| Q4_0 | 350.25 | 0.67 | ✅ PASSED |
| Q8_0 | 900.00 | 0.26 | ✅ PASSED |

## 性能分析
- llama.cpp Q4_0 比 quant-gemm 快约 16%
- Q4_0 比 Q8_0 快约 2.7x
- 两种实现都通过了正确性验证
```

---

## 🎯 最佳实践

1. **环境一致性**: 确保测试环境一致
2. **多次运行**: 至少运行 3-5 次取平均值
3. **预热**: 第一次运行可能较慢，应该预热
4. **记录配置**: 记录所有测试配置
5. **对比验证**: 同时测试两个实现，确保公平对比
6. **文档记录**: 详细记录测试过程和结果

---

**文档版本**: 1.0  
**最后更新**: 2025-01-29  
**测试场景**: M=4096, N=2, K=14336 (FFN Up Layer)
