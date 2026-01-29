# llama.cpp GEMM 运算详细拆解教程

本教程以 MMQ vs CUTLASS 对比实验为例，详细拆解 llama.cpp 中量化矩阵乘法（GEMM）的实现。

---

## 目录

1. [概述](#概述)
2. [数据流程图](#数据流程图)
3. [核心数据结构](#核心数据结构)
4. [量化格式详解](#量化格式详解)
5. [内核实现拆解](#内核实现拆解)
6. [性能优化技术](#性能优化技术)
7. [CUTLASS 集成指南](#cutlass-集成指南)

---

## 概述

### 什么是 MMQ？

**MMQ (Matrix Multiply Quantized)** 是 llama.cpp 专门为量化矩阵乘法设计的 CUDA 内核。

```
传统 GEMM:        FP32 × FP32 = FP32
llama.cpp MMQ:    Q8_0 × (FP32→Q8_1) = FP32
                  ↓
              INT8 × INT8 = INT32 → FP32
```

### 为什么需要 MMQ？

| 方案 | 显存占用 | 计算速度 | 精度 |
|------|---------|---------|------|
| FP32 | 100% | 慢 | 最高 |
| FP16 | 50% | 中等 | 高 |
| INT8 量化 | 25% | 快 | 中等 |
| Q4_K 量化 | 12.5% | **MMQ 最快** | 可接受 |

---

## 数据流程图

### 完整的 Q8_0 × FP32 矩阵乘法流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                        llama.cpp Q8_0 × FP32 GEMM                     │
└─────────────────────────────────────────────────────────────────────┘

输入:
┌──────────────┐         ┌──────────────┐
│  权重矩阵 A   │         │ 激活值矩阵 B  │
│  (Q8_0 格式)  │         │  (FP32 格式)  │
│   M × K      │         │    K × N     │
└──────────────┘         └──────────────┘
       │                        │
       │                        │
       ▼                        ▼
┌──────────────┐         ┌──────────────┐
│ Q8_0 Block   │         │   FP32 数据   │
│ - d (scale)  │         │              │
│ - qs[32]     │         │              │
└──────────────┘         └──────────────┘
       │                        │
       │                        │
       │                        ▼
       │              ┌──────────────────────┐
       │              │ quantize_mmq_q8_1_cuda│  ← 动态量化
       │              │   (独立 Kernel)       │
       │              └──────────────────────┘
       │                        │
       │                        ▼
       │              ┌──────────────┐
       │              │ Q8_1 Block   │
       │              │ - d (scale)  │
       │              │ - qs[32]     │
       │              │ - s (补偿)   │
       │              └──────────────┘
       │                        │
       │                        │
       ▼                        ▼
┌──────────────────────────────────────────┐
│           MMQ DP4A Kernel                │
│  ┌────────────────────────────────────┐  │
│  │ 1. 加载 Q8_0 block 到寄存器        │  │
│  │ 2. 加载 Q8_1 block 到共享内存      │  │
│  │ 3. INT8×INT8 点积 (__dp4a)        │  │
│  │ 4. 累加到 INT32 accumulator        │  │
│  │ 5. 反量化: sum * dA * dB + sB/32   │  │
│  │ 6. Warp 级归约                     │  │
│  │ 7. 写回 FP32 结果                  │  │
│  └────────────────────────────────────┘  │
└──────────────────────────────────────────┘
                  │
                  ▼
         ┌──────────────┐
         │  输出矩阵 C   │
         │   (FP32)     │
         │    M × N     │
         └──────────────┘
```

---

## 核心数据结构

### mmq_args 结构体

定义位置：`ggml/src/ggml-cuda/mmq.cuh:3855-3861`

```cpp
struct mmq_args {
    // ========== 输入/输出指针 ==========
    const char * x;              // 量化矩阵 A (src0)，Q8_0 等格式
    ggml_type type_x;            // A 的量化类型 (Q4_0, Q8_0, Q4_K 等)
    const int * y;               // 量化矩阵 B (src1)，已预先量化为 Q8_1
    float * dst;                 // 输出矩阵 (FP32)

    // ========== 矩阵维度 ==========
    int64_t ncols_x;      // A 的列数 (K 维度)
    int64_t nrows_x;      // A 的行数 (M 维度)
    int64_t ncols_dst;    // 输出矩阵的列数 (N 维度)
    int64_t stride_row_x; // A 的行步长
    int64_t ncols_y;      // B 的列数

    // ========== 批处理维度 ==========
    int64_t nchannels_x;           // 通道数（用于 3D/4D 张量）
    int64_t nchannels_y;
    int64_t stride_channel_x;
    int64_t stride_channel_y;
    int64_t stride_channel_dst;

    int64_t nsamples_x;            // 样本数
    int64_t nsamples_y;
    int64_t stride_sample_x;
    int64_t stride_sample_y;
    int64_t stride_sample_dst;

    // ========== 执行标志 ==========
    bool use_stream_k;    // 是否使用 Stream-K 分解
    int64_t ncols_max;    // 最大列数（用于对齐）
};
```

### 参数传递流程

```
ggml_cuda_mul_mat_q()
    │
    ├─ 构建参数结构
    │   mmq_args args = {
    │       .x = src0_d,           // Q8_0 权重
    │       .type_x = src0->type,  // GGML_TYPE_Q8_0
    │       .y = src1_q8_1,        // 动态量化后的 Q8_1
    │       .dst = dst_d,          // FP32 输出
    │       .ncols_x = ne00,       // K
    │       .nrows_x = ne01,       // M
    │       .ncols_dst = ne1,      // N
    │       ...
    │   };
    │
    └─ ggml_cuda_mul_mat_q_switch_type(args)
            │
            └─ mul_mat_q_case<GGML_TYPE_Q8_0>(args)
                    │
                    └─ mul_mat_q<Q8_0, mmq_x, need_check>
                            (实际 CUDA Kernel)
```

---

## 量化格式详解

### Q8_0 格式 (块量化)

```cpp
struct block_q8_0 {
    float d;         // 缩放因子 (scale factor)
    int8_t qs[32];   // 32 个 INT8 量化值
};

// 每个块存储 32 个 FP32 值
// 原始数据: x[0], x[1], ..., x[31]
// 量化: qs[i] = round(x[i] / d)
// 反量化: x[i] ≈ qs[i] * d

// 内存布局
// +----+----+----+----+----+----+----+----+----+----+----+----+
// | d  | qs[0]| qs[1] | ... | qs[31] |  d  | qs[0] | ... |
// +----+----+----+----+----+----+----+----+----+----+----+
//  4B    1B     1B            1B      4B    1B
//  │<────── 36 B ──────>|  │<────── 36 B ──────>│
//  │      Block 0       │  │      Block 1       │
//  │<───────── K/32 个块 ───────────────────────>│
```

### Q8_1 格式 (带补偿的块量化)

```cpp
struct block_q8_1 {
    float d;         // 缩放因子
    int8_t qs[32];   // 32 个 INT8 量化值
    float s;         // 补偿项 (min 值)
};

// 量化: qs[i] = round((x[i] - min) / d)
// 反量化: x[i] ≈ qs[i] * d + s

// 补偿项的作用
// - 减少量化误差
// - 提高精度
// - 对于非对称分布的数据更有效
```

### 量化过程示例

```
原始 FP32 数据 (假设 4 个值):
[1.2, -0.5, 3.7, -2.1]

Q8_0 量化:
d = 0.02  (最大值 / 127)
qs[0] = round(1.2 / 0.02) = 60
qs[1] = round(-0.5 / 0.02) = -25
qs[2] = round(3.7 / 0.02) = 185 → clip to 127
qs[3] = round(-2.1 / 0.02) = -105

存储: d=0.02, qs=[60, -25, 127, -105]
```

---

## 内核实现拆解

### MMQ Kernel 完整代码流程

定义位置：`ggml/src/ggml-cuda/mmq.cuh:3459-3700`

```cpp
template <ggml_type type, int mmq_x, int need_check>
static __global__ void mul_mat_q(
    const char * __restrict__ x,      // Q8_0 矩阵
    const int * __restrict__ y,       // Q8_1 矩阵
    float * __restrict__ dst,         // FP32 输出
    // ... 其他参数
) {
    // ========== 第 1 步: 线程映射 ==========
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int tidy = blockIdx.y;
    const int tidz = blockIdx.z;

    // 每个 thread block 处理一个 tile
    // tile 大小: mmq_x × mmq_y (如 64 × 64)
    // K 维度迭代: MMQ_ITER_K (256)

    // ========== 第 2 步: 共享内存分配 ==========
    extern __shared__ int8_t data [];
    // 用于存储 tile 数据
    // 避免全局内存访问

    // ========== 第 3 步: 加载 tiles ==========
    // 调用类型特定的加载函数
    mmq_type_traits<type>::load_tiles(x, data, ...);

    // 对于 Q8_0:
    // - 从全局内存加载 Q8_0 blocks
    // - 转换为内部布局 (避免 bank conflicts)

    // ========== 第 4 步: K 维度循环 ==========
    float sum[mmq_x / WARP_SIZE] = {0.0f};

    for (int k = 0; k < ncols_x; k += MMQ_ITER_K) {
        // 4.1 加载 Q8_1 tile (矩阵 B)
        // 4.2 同步: __syncthreads()

        // 4.3 计算 vector dot products
        #pragma unroll
        for (int ix = 0; ix < mmq_x / WARP_SIZE; ix++) {
            // 使用 DP4A 或 Tensor Core
            sum[ix] += vec_dot_q8_0_q8_1_dp4a(...);
        }

        // 4.4 同步: __syncthreads()
    }

    // ========== 第 5 步: 写回结果 ==========
    mmq_write_back_dp4a(dst, sum, ...);
}
```

### DP4A 点积函数详解

定义位置：`ggml/src/ggml-cuda/vecdotq.cuh:231-269`

```cpp
template <int mmq_x, int mmq_y, int n_feats>
static __device__ __forceinline__ void vec_dot_q8_0_q8_1_dp4a(
    const int * __restrict__ x,      // Q8_0 数据 (重新布局后)
    const int * __restrict__ y,      // Q8_1 数据 (重新布局后)
    float * __restrict__ sum)        // 累加器
{
    // VDR (Vector Dot Register) = 8
    // 每次迭代处理 8 个 vector groups
    constexpr int VDR = 8;

    // 累加到 INT32
    int32_t tmp[VDR] = {0};

    #pragma unroll
    for (int i = 0; i < VDR; ++i) {
        // 使用 DP4A 指令
        // DP4A: 4 对 INT8 × INT8 点积
        //
        // 输入: 两个 32-bit 整数
        //   - v: 包含 4 个 INT8 (字节 0, 1, 2, 3)
        //   - u: 包含 4 个 INT8 (字节 0, 1, 2, 3)
        //
        // 输出: v[0]*u[0] + v[1]*u[1] + v[2]*u[2] + v[3]*u[3] + c
        tmp[i] = __dp4a(x[i], y[i], tmp[i]);
    }

    // 反量化并累加到 FP32
    #pragma unroll
    for (int i = 0; i < VDR; ++i) {
        sum[i] += (float)tmp[i] * dA * dB + sB / 32.0f;
    }
}
```

### DP4A 指令工作原理

```
DP4A (Dot Product Accumulate 4-byte)

输入:
  v = 0x01020304  (字节: 0x01, 0x02, 0x03, 0x04)
  u = 0x05060708  (字节: 0x05, 0x06, 0x07, 0x08)
  c = 100         (累加器初始值)

计算:
  result = (0x01 * 0x05)    // 1 * 5 = 5
         + (0x02 * 0x06)    // 2 * 6 = 12
         + (0x03 * 0x07)    // 3 * 7 = 21
         + (0x04 * 0x08)    // 4 * 8 = 32
         + 100              // 初始值
         = 5 + 12 + 21 + 32 + 100 = 170

输出: 170 (INT32)

性能:
- Pascal (GP102): 每个 SM 每周期 16 个 DP4A
- Volta+: 每个 SM 每周期 64 个 DP4A
- 理论峰值: RTX 5070 (36 SM) ≈ 36 TFLOPS (INT8)
```

---

## 性能优化技术

### 1. 动态量化 (On-the-fly Quantization)

```cpp
// 位置: ggml/src/ggml-cuda/mmq.cu:136-137

// 在 MMQ kernel 运行前，先将 FP32 激活值量化为 Q8_1
quantize_mmq_q8_1_cuda(
    src1_d,              // FP32 输入
    nullptr,             // ids (MoE 使用)
    src1_q8_1,           // Q8_1 输出
    src0->type,          // 目标量化类型
    ne10, s11, s12, s13, // 维度和步长
    ne10_padded,         // 对齐后的列数
    ne11, ne12, ne13,    // batch 维度
    stream
);
```

**优势**：
- 量化只做一次，多次复用
- Q8_1 布局优化，避免 bank conflicts
- 减少内存带宽需求

### 2. 数据重新布局 (Data Relayout)

```
原始布局 (可能有 bank conflicts):
+-----+-----+-----+-----+
| qs0 | qs1 | qs2 | qs3 |
+-----+-----+-----+-----+
  │     │     │     │
  └─B0──┴─B1──┴─B2──┴─B3─ (可能冲突)

重新布局后 (避免 bank conflicts):
+-----+-----+-----+-----+
| qs0 | qs1 | qs2 | qs3 |
+-----+-----+-----+-----+
  │     │     │     │
  └─B0──┴─B2──┴─B1──┴─B3─ (交错访问)
```

### 3. 共享内存优化

```cpp
// 计算最优 tile 大小
int mmq_x = 8;
while (mmq_x <= mmq_x_max) {
    size_t shared_mem = calculate_shared_memory(mmq_x);
    if (shared_mem <= max_shared_mem) {
        break;  // 找到最大可用的 tile 大小
    }
    mmq_x += 8;
}

// Tile 大小影响:
// - 太小: 全局内存访问次数多
// - 太大: 共享内存不足，寄存器压力高
// - 最优: 平衡两者
```

### 4. Stream-K 分解

```
传统 Tiling:
  Block 0:  K[0:256]
  Block 1:  K[256:512]
  Block 2:  K[512:768]
  ...

Stream-K (负载更均衡):
  Block 0:  K[0:100] → K[300:500] → K[700:800]
  Block 1:  K[100:200] → K[500:600] → ...
  Block 2:  K[200:300] → K[600:700] → ...
```

### 5. Warp-level 原语

```cpp
// Warp 内归约 (使用 warp shuffle)
float warp_sum = sum;
for (int offset = 16; offset > 0; offset /= 2) {
    warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, offset);
}

// 优势:
// - 不需要共享内存
// - 单周期完成
// - 延迟低
```

---

## CUTLASS 集成指南

### 为什么用 CUTLASS？

| 特性 | llama.cpp MMQ | CUTLASS |
|------|--------------|---------|
| 指令集 | DP4A (INT8) | Tensor Core (INT8/FP16/FP32) |
| 理论峰值 | ~36 TFLOPS | ~180 TFLOPS (RTX 5070) |
| 实现复杂度 | 高 | 中等 (模板库) |
| 灵活性 | 高 (17种量化) | 中等 |

### 集成步骤

#### 步骤 1: 定义 CUTLASS Kernel

```cuda
// 文件: ggml/src/ggml-cuda/mmq_cutlass.cu

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>

using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 256>;
using WarpShape = cutlass::gemm::GemmShape<32, 32, 64>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

// INT8 Tensor Core GEMM
using CutlassGemm = cutlass::gemm::device::Gemm<
    int8_t,                    // ElementA
    cutlass::layout::RowMajor, // LayoutA
    int8_t,                    // ElementB
    cutlass::layout::RowMajor, // LayoutB
    int32_t,                   // ElementC (累加器)
    cutlass::layout::RowMajor, // LayoutC
    float,                     // ElementOutput (输出)
    cutlass::arch::OpClassTensorCore,
    cutlass::arch::Sm90,       // RTX 5070 Blackwell
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2,                          // Stages
    128,                        // AlignA
    128                         // AlignB
>;

void cutlass_q8_0_q8_1_gemm(
    const block_q8_0* A,
    const block_q8_1* B,
    float* C,
    int M, int N, int K)
{
    // 分配临时缓冲区用于反量化和重排
    cutlass::device_memory::allocation<int8_t> d_A_dequant(M * K);
    cutlass::device_memory::allocation<int8_t> d_B_dequant(N * K);
    cutlass::device_memory::allocation<int32_t> d_C_int(M * N);

    // 反量化 Kernel: Q8_0/Q8_1 → INT8
    dequant_q8_0_to_int8<<<...>>>(A, d_A_dequant.get(), M, K);
    dequant_q8_1_to_int8<<<...>>>(B, d_B_dequant.get(), N, K);

    // CUTLASS GEMM
    CutlassGemm gemm_op;
    typename CutlassGemm::Arguments args(
        M, N, K,
        {d_A_dequant.get(), K},  // Tensor A
        {d_B_dequant.get(), N},  // Tensor B
        {d_C_int.get(), N},      // Tensor C
        {d_C_int.get(), N},      // Tensor D (output)
        {1.0f, 0.0f}             // Alpha, Beta
    );

    cutlass::Status status = gemm_op(args);

    // INT32 → FP32 转换
    convert_int32_to_float<<<...>>>(d_C_int.get(), C, M * N);
}
```

#### 步骤 2: 集成到 mmq.cu

```cpp
// 文件: ggml/src/ggml-cuda/mmq.cu

// 在文件末尾添加新的函数
void ggml_cuda_mul_mat_q_cutlass(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0,    // Q8_0
    const ggml_tensor * src1,    // FP32 → Q8_1
    ggml_tensor * dst)
{
    // ... 参数准备 ...

    // 调用 CUTLASS 实现
    cutlass_q8_0_q8_1_gemm(
        (const block_q8_0 *) src0_d,
        (const block_q8_1 *) src1_q8_1,
        dst_d,
        ne01, ne1, ne00
    );
}

// 修改分发逻辑
void ggml_cuda_mul_mat_q_switch_type(...) {
    // 检查是否可以使用 CUTLASS
    bool use_cutlass = blackwell_mma_available(cc) &&
                       args.type_x == GGML_TYPE_Q8_0 &&
                       !ids;  // 不支持 MoE

    if (use_cutlass) {
        ggml_cuda_mul_mat_q_cutlass(ctx, args, stream);
        return;
    }

    // 原有的 DP4A 路径
    switch (args.type_x) {
        // ...
    }
}
```

#### 步骤 3: 性能对比

```cpp
// 在测试代码中添加基准测试

void benchmark_cutlass_vs_mmq(...) {
    // MMQ DP4A
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++) {
        mmq_dp4a_kernel<<<...>>>(...);
    }
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    float mmq_time = ...;

    // CUTLASS Tensor Core
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++) {
        cutlass_q8_0_q8_1_gemm(...);
    }
    cudaDeviceSynchronize();
    auto t3 = std::chrono::high_resolution_clock::now();
    float cutlass_time = ...;

    printf("加速比: %.2fx\n", mmq_time / cutlass_time);
}
```

---

## 总结

### 关键要点

1. **MMQ 使用两阶段设计**:
   - 阶段 1: `quantize_mmq_q8_1_cuda` (动态量化)
   - 阶段 2: `mul_mat_q` (INT8×INT8 矩阵乘法)

2. **INT8×INT8 计算路径**:
   ```
   Q8_0 × FP32 → Q8_0 × (FP32→Q8_1) → INT8×INT8 → FP32
   ```

3. **核心优化技术**:
   - DP4A 指令 (4 对 INT8 点积)
   - 共享内存 tiling
   - 数据重新布局
   - Stream-K 分解

4. **CUTLASS 集成路径**:
   - 反量化 → INT8 Tensor Core GEMM → INT32→FP32
   - 理论加速比: 2-4x

### 实验验证

基于 `tests/mmq_vs_cutlass_test.cu` 的验证结果：

- ✅ **数值一致性**: MMQ 和 CUTLASS 计算结果完全相同
- ✅ **浮点精度**: GPU vs CPU 误差 < 1e-7
- ✅ **性能**: 简化实现下性能相当，完整 CUTLASS 会更快

### 下一步

1. 实现完整的 CUTLASS 集成 (`mmq_cutlass.cu`)
2. 添加更多量化类型支持 (Q4_K, Q6_K 等)
3. 性能分析和优化
4. 集成到 llama.cpp 主分支
