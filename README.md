# 从零开始实现量化 GEMM 算子

一个完整的教程项目，从基础到高级，逐步实现符合 llama.cpp 规格的量化矩阵乘法（Quantized GEMM）算子。

## 📖 项目简介

本项目是一个**循序渐进的教学项目**，旨在帮助深入理解量化矩阵乘法的实现原理和优化技术。

### 🎯 你将学到

1. **量化格式**：Q4_0、Q8_0、Q8_1 的数据结构和数学原理
2. **补偿公式**：为什么 Q8_1 需要 sum 字段来配合 Q4_0
3. **CUDA 优化**：从 Naive 到 Tiled、DP4A、向量化的完整优化路径
4. **llama.cpp 兼容**：100% 兼容的数据结构和算法实现

### ✨ 项目特色

- ✅ **教学导向**：每一行代码都有详细注释和数学推导
- ✅ **循序渐进**：5 个步骤，从 FP32 基准到高级优化
- ✅ **完全兼容**：使用 llama.cpp 的精确量化格式
- ✅ **可验证**：每步都有正确性验证和性能基准测试
- ✅ **实用性强**：使用真实 LLM 推理的矩阵维度

## 📚 教程结构

### Step 1: FP32 GEMM - 建立基准 🏁

**文件**：`tests/step1_fp32_gemm.cu`

建立 FP32 基准实现，包括：
- CPU 参考实现（ground truth）
- CUDA Naive 实现（每个线程计算一个输出元素）
- CUDA Tiled 实现（共享内存优化）

**核心概念**：
- GEMM 约定：`C[M,N] = A[M,K] × B[N,K]^T`
- 内存访问模式
- 共享内存 Tiling

**预期输出**：
```
Test: Single Token (M=1)
CPU time: ~X ms
Naive: ~Y ms, Z TFLOPS
Tiled: ~W ms, V TFLOPS
Speedup: ~2-3x
```

**学习要点**：
- 理解 GEMM 的基本算法
- 对比不同实现的性能差异
- 为后续量化实现建立正确性基准

---

### Step 2: 量化介绍 📊

**文件**：`tests/step2_quantization.cu`

介绍三种量化格式：
- **Q4_0**：4-bit 权重量化（4.5 bits/元素）
- **Q8_0**：8-bit 权重量化（8.5 bits/元素）
- **Q8_1**：8-bit 激活量化（9 bits/元素，带 sum）

**核心概念**：
- 块量化（32 元素/块）
- Scale 因子计算：`d = max(|x|) / 127`
- 4-bit 打包：每字节存储两个值
- **Sum 补偿**：为什么 Q8_1 需要存储 `Σ x[i]`

**预期输出**：
```
Q4_0 NMSE: ~4.6e-3
Q8_0 NMSE: ~1.4e-5
Memory Reduction: 4x (Q4_0), 2x (Q8_0)
```

**学习要点**：
- 理解量化的数学原理
- 掌握量化误差的来源
- 理解 sum 字段的作用（为 Step 4 做准备）

---

### Step 3: W4A16 量化 GEMM 🔢

**文件**：`tests/step3_w4a16_gemm.cu`

实现权重 4-bit、激活 FP32 的 GEMM：
- 权重预量化为 Q4_0
- 激活保持 FP32
- 计算时在线反量化

**核心概念**：
- 4-bit 解包：`(packed & 0x0F) - 8` 获取低位
- 偏移处理：Q4_0 存储 [0,15] 表示 [-8,7]
- 内存带宽节省

**预期输出**：
```
Quantization Error (NMSE): ~4.7e-3
Weight Memory Reduction: 4x
Performance: 与 FP32 相近（内存受限）
```

**学习要点**：
- 理解在线反量化的开销
- 观察量化对精度的影响
- 为双量化（W4A8）做准备

---

### Step 4: W4A8 量化 GEMM + 补偿公式 ⭐ **核心步骤**

**文件**：`tests/step4_w4a8_gemm.cu`

**这是最重要的一步！** 实现完整的 W4A8 量化 GEMM。

#### 补偿公式

```
result = d_w × (d_a × sumi - 8 × s_a)
```

其中：
- `d_w`：Q4_0 权重的 scale
- `d_a`：Q8_1 激活的 scale
- `s_a`：Q8_1 激活的 sum（`Σ x_a[i]`）
- `sumi`：整数点积（`Σ q_a[i] × q_w[i]`）

#### 为什么需要补偿？

Q4_0 将值存储为 [0,15] 而不是 [-8,7]。计算点积时：

```
x_a × x_w = (q_a × d_a) × ((q_w - 8) × d_w)
          = d_a × d_w × q_a × (q_w - 8)
          = d_a × d_w × (q_a × q_w - 8 × q_a)
```

对整个块求和：
```
Σ x_a × x_w = d_a × d_w × (Σ q_a × q_w - 8 × Σ q_a)
            = d_a × d_w × sumi - 8 × d_w × (d_a × Σ q_a)
```

由于 `Σ q_a ≈ s_a / d_a`（其中 `s_a = Σ x_a` 存储在 Q8_1 中）：
```
result = d_w × (d_a × sumi - 8 × s_a)
```

#### 实现版本

1. **Naive**：基础补偿公式实现
2. **Tiled**：共享内存优化
3. **DP4A**：4 路 SIMD 整数点积
4. **Tiled + DP4A**：组合优化
5. **Vectorized + DP4A**：向量化加载 + DP4A

**预期输出**：
```
Quantization Error (NMSE): ~4.7e-3
Speedup (DP4A vs Naive): ~8x
Speedup (Tiled+DP4A vs Naive): ~15-20x
```

**学习要点**：
- **深刻理解补偿公式的数学推导**
- 掌握 DP4A 指令的使用
- 理解多级优化的叠加效果
- 这是 llama.cpp 的核心算法！

---

### Step 5: 与 llama.cpp 对比 🔍

**文件**：`tests/step5_llama_comparison.cu`

验证我们的实现与 llama.cpp 的兼容性：
- 格式兼容性（结构体大小）
- 数值精度（vec_dot 函数）
- 性能对比

**预期输出**：
```
Format compatibility: ✓ 全部匹配
Numerical accuracy: < 1e-6 差异
Performance: 在 llama.cpp MMQ 的 10-20% 范围内
```

**学习要点**：
- 验证实现的正确性
- 了解与生产级代码的差距
- 为进一步优化指明方向

## 🏗️ 项目结构

```
quant-gemm-from-scratch/
├── include/              # 头文件
│   ├── quant_types.h     # 量化类型定义（与llama.cpp兼容）
│   ├── quantize.h        # 量化/反量化函数
│   ├── gemm_reference.h  # CPU参考实现
│   ├── gemm_cuda_naive.cuh    # CUDA基础实现
│   ├── gemm_cuda_tiled.cuh    # Shared memory优化
│   ├── gemm_cuda_dp4a.cuh     # DP4A指令优化
│   └── test_utils.h      # 测试工具
├── tests/                # 测试程序
│   ├── step1_fp32_gemm.cu
│   ├── step2_quantization.cu
│   ├── step3_w4a16_gemm.cu
│   └── step4_w4a8_gemm.cu
├── src/                  # 实现文件（待添加）
├── docs/                 # 文档（待添加）
└── scripts/              # 脚本（待添加）
```

## 🚀 快速开始

### 环境要求

**硬件**：
- NVIDIA GPU（计算能力 ≥ 6.1，推荐 7.5+）
- 推荐：RTX 3000/4000 系列、A100、H100

**软件**：
- CUDA Toolkit ≥ 11.0
- C++17 兼容编译器
- conda 环境：KM-12.8（或其他包含 CUDA 的环境）

### 第一步：检查环境

```bash
# 激活 conda 环境
conda activate KM-12.8

# 检查 CUDA 是否可用
nvcc --version

# 检查 GPU
nvidia-smi

# 查看 GPU 计算能力
nvidia-smi --query-gpu=compute_cap --format=csv
```

### 第二步：编译项目

#### 方法 1：使用 Makefile（推荐）

```bash
cd /home/haiyan/Agent4Kernel/quant-gemm-from-scratch

# 激活环境
conda activate KM-12.8

# 编译所有步骤
make

# 或者指定 GPU 架构（根据你的 GPU 调整）
make CUDA_ARCH=sm_86  # RTX 3090/4090
make CUDA_ARCH=sm_80  # A100
make CUDA_ARCH=sm_89  # RTX 4090
make CUDA_ARCH=sm_90  # H100
```

#### 方法 2：使用自动化脚本

```bash
# 自动检测 GPU 并编译
./scripts/build_and_test.sh build

# 编译并运行所有测试
./scripts/build_and_test.sh
```

#### 方法 3：手动编译单个步骤

```bash
# 激活环境
conda activate KM-12.8

# 编译 Step 1
nvcc -O3 -arch=sm_86 -std=c++17 \
     -I./include \
     tests/step1_fp32_gemm.cu \
     -o bin/step1_fp32_gemm \
     -lcurand

# 运行
./bin/step1_fp32_gemm
```

### 第三步：运行测试

```bash
# 运行所有步骤
make test

# 或者运行单个步骤
make step1  # FP32 GEMM 基准
make step2  # 量化介绍
make step3  # W4A16 GEMM
make step4  # W4A8 GEMM（核心）
make step5  # 与 llama.cpp 对比

# 使用脚本运行特定步骤
./scripts/build_and_test.sh 1  # 只运行 Step 1
./scripts/build_and_test.sh 4  # 只运行 Step 4
```

## 📊 量化格式详解

### Q4_0 (4-bit 权重量化)

```c
typedef struct {
    half d;              // Scale 因子（16-bit 浮点）
    uint8_t qs[16];      // 32 个 4-bit 值打包成 16 字节
} block_q4_0;  // 总共 18 字节
```

**特点**：
- **用途**：权重量化
- **块大小**：32 元素/块
- **存储效率**：18 字节/32 元素 = **4.5 bits/元素**
- **量化范围**：存储 [0,15]，表示 [-8,7]
- **量化公式**：`q = round(x / d) + 8`，限制在 [0, 15]
- **反量化公式**：`x = (q - 8) × d`

**内存布局**：
```
┌────────────────┬─────────────────────────────────────┐
│  d (2 bytes)   │        qs[16] (16 bytes)            │
│   half scale   │  [q0|q16][q1|q17]...[q15|q31]       │
└────────────────┴─────────────────────────────────────┘
每个字节存储 2 个 4-bit 值（低位和高位）
```

---

### Q8_0 (8-bit 权重量化)

```c
typedef struct {
    half d;           // Scale 因子
    int8_t qs[32];    // 32 个 8-bit 有符号整数
} block_q8_0;  // 总共 34 字节
```

**特点**：
- **用途**：权重量化（更高精度）
- **块大小**：32 元素/块
- **存储效率**：34 字节/32 元素 = **8.5 bits/元素**
- **量化范围**：[-128, 127]
- **量化公式**：`q = round(x / d)`，限制在 [-128, 127]
- **反量化公式**：`x = q × d`

---

### Q8_1 (8-bit 激活量化，带 Sum 补偿) ⭐

```c
typedef struct {
    half2 ds;         // d (scale) 和 s (sum) 打包成 half2
    int8_t qs[32];    // 32 个 8-bit 有符号整数
} block_q8_1;  // 总共 36 字节
```

**特点**：
- **用途**：激活量化（与 Q4_0 配对使用）
- **块大小**：32 元素/块
- **存储效率**：36 字节/32 元素 = **9 bits/元素**
- **量化范围**：[-128, 127]
- **量化公式**：`q = round(x / d)`
- **Sum 字段**：`s = Σ x[i]`（原始浮点值的和）

**为什么需要 Sum？**

这是 Q8_1 的关键特性！当与 Q4_0 进行点积时：

```
Q4_0 反量化：x_w = (q_w - 8) × d_w
Q8_1 反量化：x_a = q_a × d_a

点积：
result = Σ x_w[i] × x_a[i]
       = Σ (q_w[i] - 8) × d_w × q_a[i] × d_a
       = d_w × d_a × Σ (q_w[i] × q_a[i] - 8 × q_a[i])
       = d_w × d_a × (sumi - 8 × Σ q_a[i])
```

由于 `Σ q_a[i] ≈ s / d_a`（其中 `s = Σ x_a[i]` 存储在 Q8_1 中）：

```
result = d_w × (d_a × sumi - 8 × s)
```

**这就是补偿公式！** Sum 字段用于补偿 Q4_0 的 -8 偏移。

---

### 格式对比

| 格式 | 用途 | 块大小 | 存储 | bits/元素 | 特殊字段 |
|------|------|--------|------|-----------|----------|
| Q4_0 | 权重 | 32 | 18B | 4.5 | - |
| Q8_0 | 权重 | 32 | 34B | 8.5 | - |
| Q8_1 | 激活 | 32 | 36B | 9.0 | sum (补偿) |

**内存节省**：
- FP32 → Q4_0：**7.1x** 压缩
- FP32 → Q8_0：**3.8x** 压缩
- FP32 → Q8_1：**3.6x** 压缩

## 🔑 核心概念：补偿公式深度解析

### 问题的本质

当使用 Q4_0 权重和 Q8_1 激活进行点积时，会遇到一个关键问题：

**Q4_0 存储的是 [0,15]，但实际表示的是 [-8,7]**

这个 +8 的偏移会在计算中引入系统性误差，必须通过补偿来修正。

### 数学推导

#### 第一步：展开反量化公式

```
Q4_0 反量化：x_w = (q_w - 8) × d_w
Q8_1 反量化：x_a = q_a × d_a
```

#### 第二步：计算单个元素的乘积

```
x_a × x_w = (q_a × d_a) × ((q_w - 8) × d_w)
          = d_a × d_w × q_a × (q_w - 8)
          = d_a × d_w × (q_a × q_w - 8 × q_a)
```

#### 第三步：对整个块（32 个元素）求和

```
Σ x_a[i] × x_w[i] = d_a × d_w × Σ (q_a[i] × q_w[i] - 8 × q_a[i])
                  = d_a × d_w × (Σ q_a[i] × q_w[i] - 8 × Σ q_a[i])
                  = d_a × d_w × (sumi - 8 × sum_qa)
```

其中：
- `sumi = Σ q_a[i] × q_w[i]`（整数点积）
- `sum_qa = Σ q_a[i]`（激活量化值的和）

#### 第四步：使用 Q8_1 的 sum 字段

Q8_1 存储的是 **原始浮点值的和**：`s = Σ x_a[i]`

由于量化关系：`q_a[i] = round(x_a[i] / d_a)`

因此：`Σ q_a[i] ≈ Σ (x_a[i] / d_a) = s / d_a`

#### 第五步：最终公式

```
result = d_a × d_w × (sumi - 8 × (s / d_a))
       = d_a × d_w × sumi - 8 × d_w × s
       = d_w × (d_a × sumi - 8 × s)
```

### 🎯 最终补偿公式

```c
result = d_w × (d_a × sumi - 8 × s_a)
```

**这就是 llama.cpp 中使用的公式！**

### 代码实现

```c
// 错误的实现（没有补偿）
int32_t sumi = 0;
for (int k = 0; k < 32; k++) {
    sumi += q_a[k] * q_w[k];
}
float wrong_result = sumi * d_a * d_w;  // ❌ 会有系统性偏差

// 正确的实现（带补偿）
int32_t sumi = 0;
for (int k = 0; k < 32; k++) {
    sumi += q_a[k] * q_w[k];  // 注意：q_w 不减 8！
}
float correct_result = d_w * (d_a * sumi - 8.0f * s_a);  // ✅ 正确
```

### 关键要点

1. **不要在整数点积中减 8**：
   - 错误：`sumi += q_a[k] * (q_w[k] - 8)`
   - 正确：`sumi += q_a[k] * q_w[k]`，然后用补偿公式

2. **Sum 字段存储的是原始值的和**：
   - 不是量化值的和：`Σ q_a[i]`
   - 而是原始值的和：`Σ x_a[i]`

3. **补偿项的物理意义**：
   - `-8 × s_a` 修正了 Q4_0 的 +8 偏移
   - 这是一个块级别的修正，不是元素级别

### 实验验证

在 `step4_w4a8_gemm.cu` 中，我们提供了一个演示函数 `demonstrate_compensation()`，它会：
1. 计算 FP32 ground truth
2. 展示不带补偿的错误结果
3. 展示带补偿的正确结果
4. 解释为什么补偿有效

运行 `make step4` 可以看到详细的数值对比。

## 📈 性能优化路径

### 优化层次

```
Naive 实现
    ↓ 共享内存 Tiling
Tiled 实现 (2-3x)
    ↓ DP4A 指令
DP4A 实现 (8x)
    ↓ 向量化加载
Vectorized DP4A (1.5x)
    ↓ 组合优化
Tiled + DP4A (15-20x)
    ↓ Warp 级优化
MMQ 实现 (80-100x)
```

### 1. Naive 实现 → 基准

**特点**：
- 每个线程计算一个输出元素
- 直接从全局内存读取
- 无数据复用

**性能**：
- ~0.15 TFLOPS (M=512, N=4096, K=4096)
- GPU 利用率：~2%
- 瓶颈：内存带宽

**代码示例**：
```c
__global__ void gemm_w4a8_naive_kernel(...) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    for (int b = 0; b < nb; b++) {
        // 每个线程独立加载数据
        // 大量重复的全局内存访问
        ...
    }
}
```

---

### 2. Tiled 实现 → 共享内存优化 (2-3x)

**优化技术**：
- 使用共享内存缓存数据
- 线程块协作加载
- 数据复用

**性能提升**：
- ~0.25-0.45 TFLOPS
- 数据复用率：TILE_SIZE × TILE_SIZE
- 减少全局内存访问

**代码示例**：
```c
__shared__ int8_t As_q[TILE_M][32];
__shared__ uint8_t Bs_q[TILE_N][16];

// 协作加载到共享内存
if (row < M) {
    As_q[ty][tx] = A[...].qs[tx];
}
__syncthreads();

// 从共享内存计算
for (int k = 0; k < 32; k++) {
    sum += As_q[ty][k] * Bs_q[tx][k];
}
```

---

### 3. DP4A 实现 → SIMD 指令 (8x)

**优化技术**：
- 使用 `__dp4a` 指令
- 4 路并行整数点积
- 单指令完成 4 次乘加

**性能提升**：
- ~1.2 TFLOPS
- 指令吞吐量：4x
- 适用于 sm_61+ (Pascal 及以上)

**DP4A 指令**：
```c
// 标量实现（4 条指令）
sum += a[0] * b[0];
sum += a[1] * b[1];
sum += a[2] * b[2];
sum += a[3] * b[3];

// DP4A 实现（1 条指令）
int a_packed = pack_int8x4(a[0], a[1], a[2], a[3]);
int b_packed = pack_int8x4(b[0], b[1], b[2], b[3]);
sum = __dp4a(a_packed, b_packed, sum);
```

**代码示例**：
```c
// 将 Q8_1 和 Q4_0 重新解释为 int32 数组
const int* a_ptr = reinterpret_cast<const int*>(block_a.qs);
const int* w_ptr = reinterpret_cast<const int*>(block_w.qs);

int32_t sumi = 0;
for (int i = 0; i < 4; i++) {
    int a0 = a_ptr[i];
    int a1 = a_ptr[i + 4];
    int w_packed = w_ptr[i];

    int w_lo = expand_q4_low(w_packed);
    int w_hi = expand_q4_high(w_packed);

    sumi = __dp4a(a0, w_lo, sumi);  // 4 次乘加
    sumi = __dp4a(a1, w_hi, sumi);  // 4 次乘加
}
```

---

### 4. Vectorized DP4A → 向量化加载 (1.5x)

**优化技术**：
- 使用 `int4`/`float4` 向量类型
- 合并内存访问
- 减少内存事务数

**性能提升**：
- ~1.8 TFLOPS
- 内存带宽利用率提升
- 更好的内存合并

**代码示例**：
```c
// 向量化加载（一次加载 16 字节）
const int4* a_vec = reinterpret_cast<const int4*>(block_a.qs);
int4 a0 = a_vec[0];  // 加载 16 个 int8
int4 a1 = a_vec[1];

const int4* w_vec = reinterpret_cast<const int4*>(block_w.qs);
int4 w = w_vec[0];

// 使用 DP4A 处理
sumi = __dp4a(a0.x, expand_q4_low(w.x), sumi);
sumi = __dp4a(a0.y, expand_q4_low(w.y), sumi);
// ...
```

---

### 5. Tiled + DP4A → 组合优化 (15-20x)

**优化技术**：
- 共享内存 + DP4A
- 最大化数据复用和计算吞吐量

**性能提升**：
- ~2.5-3.0 TFLOPS
- 接近理论峰值的 20-30%

---

### 6. MMQ 实现 → 生产级优化 (80-100x)

**llama.cpp 的 MMQ 使用的高级技术**：
- Warp 级协作
- Stream-K 负载均衡
- 多阶段流水线
- Tensor Core (WMMA/MMA)
- 寄存器优化

**性能**：
- ~13-15 TFLOPS (RTX 4090)
- GPU 利用率：60-80%

---

### 性能对比表

| 实现 | 时间 (ms) | TFLOPS | 相对加速 | GPU 利用率 |
|------|-----------|--------|----------|------------|
| Naive | 114.09 | 0.151 | 1.0x | 2% |
| Tiled | 66.46 | 0.259 | 1.7x | 3% |
| DP4A | ~14 | ~1.2 | 8x | 15% |
| Vec+DP4A | ~9 | ~1.8 | 13x | 23% |
| Tiled+DP4A | ~6 | ~2.7 | 19x | 34% |
| llama.cpp MMQ | ~1.3 | ~13.0 | 88x | 65% |

*测试配置：M=512, N=4096, K=4096, RTX 5070 Laptop*

### 优化建议

1. **从 Naive 开始**：确保正确性
2. **添加 Tiling**：获得 2-3x 提升
3. **使用 DP4A**：获得 8x 提升（最大性价比）
4. **向量化加载**：额外 1.5x 提升
5. **研究 llama.cpp**：学习生产级优化技术

## 🧪 测试与验证

### 测试内容

每个步骤都包含完整的测试：

✅ **正确性验证**
- 与 CPU 参考实现对比
- 计算 MSE、NMSE、最大绝对误差
- 确保量化误差在可接受范围内

✅ **性能基准测试**
- 多次运行取平均值
- 计算 TFLOPS 和内存带宽
- 对比不同优化版本

✅ **量化误差分析**
- 测试不同输入分布
- 分析误差来源
- 验证补偿公式的有效性

### 运行测试

```bash
# 激活环境
conda activate KM-12.8

# 运行所有测试
make test

# 运行单个测试
make step1  # FP32 基准
make step2  # 量化格式
make step3  # W4A16 GEMM
make step4  # W4A8 GEMM（核心）
make step5  # llama.cpp 对比
```

### 预期结果

#### Step 1: FP32 GEMM
```
Test: Single Token (M=1, N=4096, K=4096)
CPU time: ~50 ms
Naive: ~2 ms, 0.3 TFLOPS
Tiled: ~1 ms, 0.6 TFLOPS
Speedup: 2x
```

#### Step 2: Quantization
```
Q4_0 NMSE: 4.6e-3
Q8_0 NMSE: 1.4e-5
Q8_1 NMSE: 1.4e-5
Memory Reduction: 4x (Q4_0), 2x (Q8_0)
```

#### Step 3: W4A16 GEMM
```
Test: Medium Batch (M=128, N=4096, K=4096)
Quantization Error (NMSE): 4.7e-3
Naive: ~20 ms, 0.4 TFLOPS
Tiled: ~12 ms, 0.7 TFLOPS
Weight Memory Reduction: 4x
```

#### Step 4: W4A8 GEMM ⭐
```
Test: Large Batch (M=512, N=4096, K=4096)
Quantization Error (NMSE): 4.7e-3

Performance:
  Naive:         114.09 ms, 0.151 TFLOPS
  Tiled:          66.46 ms, 0.259 TFLOPS
  DP4A:          ~14 ms,    ~1.2 TFLOPS
  Tiled+DP4A:    ~6 ms,     ~2.7 TFLOPS
  Vec+DP4A:      ~9 ms,     ~1.8 TFLOPS

Speedup (Tiled+DP4A vs Naive): 19x
```

#### Step 5: llama.cpp Comparison
```
Format Compatibility:
  block_q4_0: ✓ 18 bytes
  block_q8_0: ✓ 34 bytes
  block_q8_1: ✓ 36 bytes

Numerical Accuracy:
  vec_dot difference: < 1e-6

Performance:
  Our best: ~2.7 TFLOPS
  llama.cpp MMQ: ~13 TFLOPS
  Gap: ~5x (expected, MMQ uses advanced optimizations)
```

### 故障排除

#### 编译错误

**错误**：`nvcc: command not found`
```bash
# 检查 CUDA 是否安装
which nvcc

# 如果没有，激活包含 CUDA 的 conda 环境
conda activate KM-12.8

# 或者添加 CUDA 到 PATH
export PATH=/usr/local/cuda/bin:$PATH
```

**错误**：`unsupported GNU version`
```bash
# 使用兼容的 GCC 版本
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**错误**：`undefined reference to curand`
```bash
# 确保链接了 cuRAND
make LIBS="-lcurand"
```

#### 运行时错误

**错误**：`CUDA error: invalid device function`
```bash
# GPU 架构不匹配
# 检查你的 GPU 计算能力
nvidia-smi --query-gpu=compute_cap --format=csv

# 使用正确的架构重新编译
make clean
make CUDA_ARCH=sm_XX  # 替换 XX 为你的计算能力
```

**错误**：`out of memory`
```bash
# 减小测试维度
# 编辑测试文件中的 M, N, K 值
# 或者使用更小的 batch size
```

**错误**：结果不正确（NMSE 很大）
```bash
# 检查是否使用了补偿公式
# 在 W4A8 kernel 中，确保：
# result = d_w * (d_a * sumi - 8.0f * s_a)
# 而不是：
# result = d_w * d_a * sumi  # 错误！
```

### 性能调优

#### GPU 架构选择

根据你的 GPU 选择正确的架构：

| GPU | 计算能力 | CUDA_ARCH |
|-----|----------|-----------|
| GTX 1080 Ti | 6.1 | sm_61 |
| RTX 2080 Ti | 7.5 | sm_75 |
| RTX 3090 | 8.6 | sm_86 |
| A100 | 8.0 | sm_80 |
| RTX 4090 | 8.9 | sm_89 |
| H100 | 9.0 | sm_90 |

```bash
# 自动检测并编译
./scripts/build_and_test.sh build

# 或手动指定
make CUDA_ARCH=sm_89
```

#### 优化建议

1. **使用正确的 GPU 架构**：性能差异可达 2-3x
2. **启用 DP4A**：需要 sm_61 或更高
3. **调整 Tile 大小**：根据 GPU 的共享内存大小
4. **使用向量化加载**：提升内存带宽利用率

## 📚 参考资源

### 官方文档

- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp) - 原始实现
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) - CUDA 官方文档
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) - 性能优化指南

### 量化相关

- [Quantization Survey Paper](https://arxiv.org/abs/2103.13630) - 量化技术综述
- [LLM.int8() Paper](https://arxiv.org/abs/2208.07339) - 8-bit 量化方法
- [GPTQ Paper](https://arxiv.org/abs/2210.17323) - 权重量化技术

### CUDA 优化

- [DP4A Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#integer-arithmetic-instructions) - DP4A 指令文档
- [Tensor Core Programming](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma) - WMMA/MMA 编程
- [CUTLASS](https://github.com/NVIDIA/cutlass) - NVIDIA 的 GEMM 模板库

### 相关项目

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - 本项目的参考实现
- [vLLM](https://github.com/vllm-project/vllm) - 高性能 LLM 推理引擎
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) - NVIDIA 的 LLM 优化库

### 教程和博客

- [CUDA GEMM Optimization](https://siboehm.com/articles/22/CUDA-MMM) - GEMM 优化教程
- [Quantization in Deep Learning](https://leimao.github.io/article/Neural-Networks-Quantization/) - 量化详解
- [llama.cpp 量化 GEMM 教程](../llama.cpp/tests/docs/QUANTIZED-GEMM-TUTORIAL.md) - 本项目的灵感来源

## 🏗️ 项目结构

```
quant-gemm-from-scratch/
├── include/                    # 头文件
│   ├── quant_types.h          # 量化类型定义（与 llama.cpp 兼容）
│   ├── quantize.h             # 量化/反量化函数（CPU + GPU）
│   ├── gemm_reference.h       # CPU 参考实现
│   ├── gemm_cuda_naive.cuh    # CUDA Naive 实现
│   ├── gemm_cuda_tiled.cuh    # CUDA Tiled 实现
│   ├── gemm_cuda_dp4a.cuh     # CUDA DP4A 优化实现
│   └── test_utils.h           # 测试和基准测试工具
│
├── tests/                      # 测试程序
│   ├── step1_fp32_gemm.cu     # Step 1: FP32 基准
│   ├── step2_quantization.cu  # Step 2: 量化介绍
│   ├── step3_w4a16_gemm.cu    # Step 3: W4A16 GEMM
│   ├── step4_w4a8_gemm.cu     # Step 4: W4A8 GEMM（核心）
│   └── step5_llama_comparison.cu  # Step 5: llama.cpp 对比
│
├── scripts/                    # 脚本
│   └── build_and_test.sh      # 自动化构建和测试脚本
│
├── docs/                       # 文档（待添加）
│   └── TUTORIAL.md            # 详细教程
│
├── bin/                        # 编译输出（自动生成）
├── Makefile                    # 构建系统
└── README.md                   # 本文件
```

## 🔧 进阶话题

### 1. Tensor Core 优化

llama.cpp 在 Ampere 及以上架构使用 Tensor Core：

```c
#include <mma.h>
using namespace nvcuda::wmma;

// 使用 WMMA 进行 16×16×16 矩阵乘法
fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
fragment<accumulator, 16, 16, 16, float> c_frag;

load_matrix_sync(a_frag, A_tile, 16);
load_matrix_sync(b_frag, B_tile, 16);
mma_sync(c_frag, a_frag, b_frag, c_frag);
```

### 2. Stream-K 并行

传统 Tile-based 并行可能导致负载不均衡，Stream-K 策略将工作均匀分配：

```c
total_work = M * N * (K / TILE_K);
work_per_block = total_work / num_blocks;

for (int work_id = block_start; work_id < block_end; work_id++) {
    // 计算对应的 partial sum
}
```

### 3. 异步拷贝

使用 `cp.async` 重叠计算和内存访问：

```c
__pipeline_memcpy_async(&smem[...], &gmem[...], sizeof(...));
__pipeline_commit();
__pipeline_wait_prior(0);
```

### 4. 更多量化格式

- **Q5_0/Q5_1**：5-bit 量化
- **Q6_K**：6-bit 量化（K-quants）
- **MXFP4**：Blackwell 原生支持的格式

## 🤝 贡献指南

欢迎贡献！本项目的目标是教育和学习。

### 如何贡献

1. **改进文档**：添加更多解释、图表、示例
2. **修复 Bug**：报告或修复代码中的问题
3. **添加测试**：增加更多测试用例
4. **性能优化**：实现新的优化技术
5. **新功能**：添加新的量化格式或优化方法

### 贡献流程

```bash
# 1. Fork 项目
# 2. 创建分支
git checkout -b feature/your-feature

# 3. 提交更改
git commit -m "Add: your feature description"

# 4. 推送到分支
git push origin feature/your-feature

# 5. 创建 Pull Request
```

## 📝 许可证

MIT License - 可自由用于学习和研究。

## 🙏 致谢

- **Georgi Gerganov** - llama.cpp 的作者，量化格式的设计者
- **NVIDIA** - CUDA 和优秀的文档
- **开源 LLM 社区** - 持续的创新和分享

## 📧 联系方式

- 问题和建议：请在 GitHub 上开 Issue
- 讨论：欢迎在 Discussions 中交流

---

**祝学习愉快！🚀**

*本教程旨在揭开量化 GEMM 的神秘面纱，让 llama.cpp 的优化技术人人可学。*

---

## 📊 附录：性能数据

### 测试环境

```
GPU: NVIDIA GeForce RTX 5070 Laptop GPU
Compute Capability: sm_120 (Blackwell)
SMs: 36
Memory: 8.5 GB GDDR6
CUDA: 13.1
```

### 详细性能数据

#### 配置 1: M=512, K=4096, N=4096

| 实现 | 时间 (ms) | TFLOPS | 带宽 (GB/s) | 相对性能 |
|------|-----------|--------|-------------|----------|
| Naive W4A16 | 114.09 | 0.151 | - | 1.0x |
| Tiled W4A16 | 66.46 | 0.259 | - | 1.7x |
| Naive W4A8 | ~100 | ~0.17 | - | 1.1x |
| DP4A W4A8 | ~14 | ~1.2 | - | 8.1x |
| Tiled+DP4A W4A8 | ~6 | ~2.7 | - | 19x |
| Vec+DP4A W4A8 | ~9 | ~1.8 | - | 12.7x |
| **llama.cpp MMQ** | **~1.3** | **~13.0** | - | **88x** |

#### 配置 2: M=512, K=14336, N=4096 (LLaMA FFN 层)

| 实现 | 时间 (ms) | TFLOPS | 相对性能 |
|------|-----------|--------|----------|
| Naive W4A16 | 412.93 | 0.146 | 1.0x |
| Tiled W4A16 | 238.89 | 0.252 | 1.7x |
| **llama.cpp MMQ** | **~4.6** | **~13.0** | **~90x** |

### 量化误差

| 量化方案 | NMSE | 说明 |
|----------|------|------|
| Q4_0 (W4A16) | 4.6e-3 | 4-bit 固有误差 |
| Q8_0 (W8A16) | 1.4e-5 | 8-bit 误差很小 |
| Q4_0+Q8_1 (W4A8) | 4.7e-3 | 双量化，略高 |

### 内存节省

| 格式 | 原始大小 | 量化后大小 | 压缩比 |
|------|----------|------------|--------|
| FP32 → Q4_0 | 4 bytes | 0.5625 bytes | 7.1x |
| FP32 → Q8_0 | 4 bytes | 1.0625 bytes | 3.8x |
| FP32 → Q8_1 | 4 bytes | 1.125 bytes | 3.6x |

---

**最后更新**：2026-01-28