# Naive 实现 vs llama.cpp 详细性能对比

**测试日期**: 2026-01-29
**测试环境**: CUDA 12.8, RTX 5070 Laptop GPU (sm_120)
**测试目的**: 对比 quant-gemm-from-scratch 的 naive 实现与 llama.cpp 在 Llama 模型典型尺寸下的性能

---

## 执行摘要

本次测试在 M=4096, K=14336, N=1,2,3,4,5,8 的矩阵尺寸下，对比了 naive 实现和 llama.cpp 的性能。

**关键发现**:
- **W4A16**: llama.cpp 比 naive 实现快 **16-18 倍**
- **W4A8**: llama.cpp 比 naive 实现快 **13-17 倍**
- **W8A8**: llama.cpp 比 naive 实现快 **7-14 倍**

---

## 测试环境

### 硬件配置

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU |
| 架构 | Blackwell (sm_120) |
| SM 数量 | 36 |
| 显存 | 8.5 GB |
| 内存带宽 | 384 GB/s |

### 软件配置

| 项目 | 版本 |
|------|------|
| CUDA | 12.8.93 |
| llama.cpp | ggml 0.9.5 |
| 编译器 | nvcc 12.8 + GCC 14.3.0 |
| 操作系统 | Linux (WSL2) |

---

## 性能测试结果

### 1. W4A16 (Q4_0) 性能对比

| N | Naive 时间 (μs) | llama.cpp 时间 (μs) | Naive GFLOPS | llama.cpp GFLOPS | 加速比 |
|---|----------------|---------------------|--------------|------------------|--------|
| 1 | 5375.46 | 331.68 | 21.85 | 354.08 | **16.2x** |
| 2 | 5419.74 | 302.97 | 43.34 | 775.26 | **17.9x** |
| 3 | 5767.97 | 322.02 | 61.08 | 1094.00 | **17.9x** |
| 4 | 6108.50 | 349.50 | 76.90 | 1344.00 | **17.5x** |
| 5 | 7007.54 | 434.94 | 83.80 | 1350.00 | **16.1x** |
| 8 | 8535.96 | - | 110.07 | - | - |

**观察**:
- Naive 实现性能: 21.85 - 110.07 GFLOPS
- llama.cpp 性能: 354.08 - 1350.00 GFLOPS
- 平均加速比: **17.1x**

### 2. W4A8 (Q8_1×Q4_0) 性能对比

| N | Naive 时间 (μs) | llama.cpp 时间 (μs) | Naive GFLOPS | llama.cpp GFLOPS | 加速比 |
|---|----------------|---------------------|--------------|------------------|--------|
| 1 | 4392.71 | 331.68 | 26.74 | 354.08 | **13.2x** |
| 2 | 5012.61 | 302.97 | 46.86 | 775.26 | **16.5x** |
| 3 | 5144.53 | 322.02 | 68.48 | 1094.00 | **16.0x** |
| 4 | 5813.71 | 349.50 | 80.80 | 1344.00 | **16.6x** |
| 5 | 7086.04 | 434.94 | 82.87 | 1350.00 | **16.3x** |
| 8 | 8600.43 | - | 109.24 | - | - |

**观察**:
- Naive 实现性能: 26.74 - 109.24 GFLOPS
- llama.cpp 性能: 354.08 - 1350.00 GFLOPS
- 平均加速比: **15.7x**

### 3. W8A8 (Q8_1×Q8_0) 性能对比

| N | Naive 时间 (μs) | llama.cpp 时间 (μs) | Naive GFLOPS | llama.cpp GFLOPS | 加速比 |
|---|----------------|---------------------|--------------|------------------|--------|
| 1 | 5567.74 | 776.24 | 21.09 | 151.29 | **7.2x** |
| 2 | 5883.15 | 793.24 | 39.92 | 296.10 | **7.4x** |
| 3 | 6895.98 | 831.83 | 51.09 | 423.55 | **8.3x** |
| 4 | 8077.27 | 806.36 | 58.16 | 582.57 | **10.0x** |
| 5 | 10330.50 | 828.07 | 56.84 | 709.12 | **12.5x** |
| 8 | 14518.04 | 1027.74 | 64.71 | 914.16 | **14.1x** |

**观察**:
- Naive 实现性能: 21.09 - 64.71 GFLOPS
- llama.cpp 性能: 151.29 - 914.16 GFLOPS
- 平均加速比: **9.9x**

---

## 性能分析

### 1. 性能差距原因

#### Naive 实现的瓶颈

1. **内存访问模式**
   - 非 coalesced memory access
   - 大量的全局内存访问
   - 没有使用 shared memory 缓存

2. **计算效率**
   - 没有使用 DP4A 指令 (INT8 点积加速)
   - 没有向量化加载
   - 没有 warp-level 优化

3. **线程配置**
   - 简单的 2D grid 配置
   - 没有针对不同矩阵尺寸优化

#### llama.cpp 的优化

1. **MMQ (Matrix Multiplication Quantized) Kernel**
   - 使用 DP4A 指令加速 INT8 点积
   - 优化的内存访问模式
   - Warp-level 优化

2. **自适应 Kernel 选择**
   - 根据矩阵尺寸选择最优 kernel
   - 小 batch 和大 batch 使用不同策略

3. **内存优化**
   - Coalesced memory access
   - Shared memory 缓存
   - 向量化加载

### 2. Batch Size 的影响

#### Naive 实现

| N | W4A16 GFLOPS | W4A8 GFLOPS | W8A8 GFLOPS |
|---|--------------|-------------|-------------|
| 1 | 21.85 | 26.74 | 21.09 |
| 2 | 43.34 | 46.86 | 39.92 |
| 4 | 76.90 | 80.80 | 58.16 |
| 8 | 110.07 | 109.24 | 64.71 |

- 性能提升: 1x → 2x → 3.5x → 5x (N=1 到 N=8)
- 提升缓慢，说明内存访问是主要瓶颈

#### llama.cpp

| N | W4A16 GFLOPS | W8A8 GFLOPS |
|---|--------------|-------------|
| 1 | 354.08 | 151.29 |
| 2 | 775.26 | 296.10 |
| 4 | 1344.00 | 582.57 |
| 8 | - | 914.16 |

- 性能提升: 1x → 2.2x → 3.8x → 6x (N=1 到 N=8)
- 提升显著，说明优化在大 batch 时更有效

### 3. 量化类型的影响

#### 性能对比 (N=2)

| 量化类型 | Naive GFLOPS | llama.cpp GFLOPS | 加速比 |
|---------|--------------|------------------|--------|
| W4A16 | 43.34 | 775.26 | 17.9x |
| W4A8 | 46.86 | 775.26 | 16.5x |
| W8A8 | 39.92 | 296.10 | 7.4x |

**观察**:
- W4A16 和 W4A8 的 naive 性能相近
- W8A8 的 naive 性能略低
- llama.cpp 在 W4A16/W4A8 上的优化更显著

---

## 优化建议

### 1. 内存访问优化

**问题**: Naive 实现使用非 coalesced memory access

**解决方案**:
```cuda
// Bad: 非 coalesced access
for (int i = 0; i < nb; i++) {
    const block_q4_0& block = B[j * nb + i];  // 不同线程访问不连续内存
    ...
}

// Good: Coalesced access
__shared__ block_q4_0 s_B[TILE_SIZE];
// 使用 shared memory 缓存，确保连续访问
```

### 2. 使用 DP4A 指令

**问题**: Naive 实现使用标量乘法

**解决方案**:
```cuda
// Bad: 标量乘法
int32_t sumi = 0;
for (int k = 0; k < QK4_0; k++) {
    sumi += (int32_t)q_a[k] * q_w[k];
}

// Good: DP4A 指令 (4x INT8 点积)
int32_t sumi = __dp4a(a_vec, w_vec, 0);
```

预期加速: **4-8x**

### 3. Warp-level 优化

**问题**: Naive 实现没有利用 warp 内的并行性

**解决方案**:
```cuda
// 使用 warp shuffle 减少 shared memory 使用
float sum = ...;
for (int offset = 16; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
}
```

预期加速: **1.5-2x**

### 4. 向量化加载

**问题**: Naive 实现使用标量加载

**解决方案**:
```cuda
// Bad: 标量加载
float a = A[i];

// Good: 向量化加载
float4 a = *reinterpret_cast<const float4*>(&A[i]);
```

预期加速: **1.2-1.5x**

---

## 总结

### 性能对比总结

| 实现 | W4A16 (GFLOPS) | W4A8 (GFLOPS) | W8A8 (GFLOPS) |
|------|----------------|---------------|---------------|
| Naive | 21.85 - 110.07 | 26.74 - 109.24 | 21.09 - 64.71 |
| llama.cpp | 354.08 - 1350.00 | 354.08 - 1350.00 | 151.29 - 914.16 |
| **加速比** | **16-18x** | **13-17x** | **7-14x** |

### 关键结论

1. **Naive 实现的价值**
   - 清晰展示量化 GEMM 的基本原理
   - 为优化提供基准
   - 帮助理解 llama.cpp 优化的价值

2. **性能差距的原因**
   - 内存访问模式 (最大瓶颈)
   - 计算效率 (DP4A 指令)
   - 线程配置和优化

3. **优化潜力**
   - 通过实现上述优化，预期可以达到 llama.cpp 性能的 50-70%
   - 完全匹配 llama.cpp 需要更多细节优化

4. **教学意义**
   - 从 naive 到优化的演进过程非常有价值
   - 帮助理解 GPU 编程的核心概念
   - 为深入学习 llama.cpp 源码打下基础

---

## 附录: 测试命令

### Naive 实现测试
```bash
cd /home/haiyan/Agent4Kernel/quant-gemm-from-scratch
nvcc -O3 -arch=sm_120 -I./include tests/test_llama_shapes.cu -o bin/test_llama_shapes
./bin/test_llama_shapes
```

### llama.cpp 测试
```bash
cd /home/haiyan/Agent4Kernel/llama.cpp/build/bin
./test-backend-ops perf -o MUL_MAT -b CUDA0 -p "type_a=q4_0.*m=4096.*n=2.*k=14336"
./test-backend-ops perf -o MUL_MAT -b CUDA0 -p "type_a=q8_0.*m=4096.*n=2.*k=14336"
```
