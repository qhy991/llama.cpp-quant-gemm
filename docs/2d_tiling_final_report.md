# 2D Tiling 优化最终报告

## 🎉 性能突破总结

### 最佳性能记录

在 GPU 正常温度下测试（NVIDIA GeForce RTX 5070 Laptop GPU）：

| Kernel 版本 | GFLOPS | Speedup | 状态 |
|------------|--------|---------|------|
| Naive (基线) | 120.8 | 1.00x | ✓ |
| Warp Multirow | 494.2 | 4.09x | ✓ |
| Shared Memory | 519.0 | 4.30x | ✓ |
| **2D Tile (N=4)** | **610.4** | **5.05x** | ✓ |
| **2D Tile (8192×2×14336)** | **630.4** | **5.20x** | ✓ |

### 目标达成情况

- **当前最佳：** 630.4 GFLOPS
- **目标 (llama.cpp)：** 775 GFLOPS
- **达成率：** **81.3%**
- **剩余差距：** 144.6 GFLOPS (18.7%)

## 关键优化技术

### 1. 2D Tiling 架构

**核心思想：** 每个 thread block 处理 TILE_M × TILE_N 的输出 tile

```
之前 (1D): 每个 block 处理 (64 × 1) 输出
           ↓
           每列独立加载权重矩阵 → 巨大的内存浪费

现在 (2D): 每个 block 处理 (64 × 4) 输出
           ↓
           权重加载一次，复用 4 次 → 显著减少带宽
```

**配置参数：**
- TILE_M = 64 (每个 block 处理 64 行)
- TILE_N = 4 (每个 block 处理 4 列)
- TILE_K = 128 (K 维度分块大小)
- ROWS = 4 (每个 warp 处理 4 行)

**内存层次：**
```
Global Memory (权重 + 激活)
    ↓
Shared Memory (TILE_N × TILE_K 的激活值)
    ↓
Registers (预加载的权重和激活)
    ↓
Computation (dp4a)
```

### 2. 性能提升分解

| 优化阶段 | GFLOPS | 提升 | 累计加速比 |
|---------|--------|------|-----------|
| Naive | 120.8 | - | 1.00x |
| + Warp 协作 | 494.2 | +309% | 4.09x |
| + Shared Memory | 519.0 | +5% | 4.30x |
| + **2D Tiling** | **630.4** | **+21%** | **5.20x** |

### 3. 不同矩阵尺寸的性能

| 矩阵尺寸 (M×N×K) | 2D Tile GFLOPS | Speedup |
|-----------------|----------------|---------|
| 4096×1×4096 | 622.3 | 5.16x |
| 4096×1×14336 | 594.5 | 4.92x |
| 4096×2×14336 | 610.4 | 5.05x |
| 4096×4×14336 | 623.4 | 5.17x |
| **8192×2×14336** | **630.4** | **5.20x** |

**结论：** 性能在不同尺寸下非常稳定 (594-630 GFLOPS)

## 正确性验证

✅ **所有测试通过**
- 错误率：0/8192 (100% 正确)
- 最大数值误差：0.012695
- 相对误差：< 0.1%

数值误差来源于浮点运算顺序差异，完全在可接受范围内。

## 性能瓶颈分析

### 当前限制因素

1. **内存带宽** (主要瓶颈)
   - 权重矩阵仍需从全局内存加载
   - 虽然 2D tiling 减少了重复加载，但仍有优化空间

2. **计算吞吐量**
   - dp4a 指令吞吐量
   - half 精度转换开销

3. **占用率**
   - 每个 block 512 threads (16 warps)
   - 可能未充分利用所有 SM

### 为什么还差 18.7%？

llama.cpp 可能使用的额外优化：
1. **Persistent kernels** - 保持 thread blocks 常驻
2. **Async copy** - 使用 `cp.async` 异步拷贝
3. **Double buffering** - 计算与内存传输重叠
4. **更激进的 tiling** - 更大的 tile 尺寸
5. **汇编级优化** - 手写 PTX 代码

## 下一步优化建议

### 优先级 1: Async Copy + Double Buffering (预期 +15-20%)

```cuda
// 使用 cp.async 异步加载到 shared memory
__pipeline_memcpy_async(&smem[buffer], &gmem[offset], size);
__pipeline_commit();

// 在加载下一个 tile 的同时计算当前 tile
compute_tile(smem[current_buffer]);
__pipeline_wait_prior(0);
current_buffer ^= 1;  // 切换缓冲区
```

**预期性能：** 630 × 1.15 = **725 GFLOPS**

### 优先级 2: 使用 Nsight Compute 分析

```bash
ncu --set full -o profile ./tests/benchmark_best 4096 2 14336
```

分析指标：
- Memory bandwidth utilization
- Compute utilization
- Occupancy
- L2 cache hit rate
- Warp stall reasons

### 优先级 3: 调整 Tile 参数

尝试不同配置：
- TILE_M = 64, TILE_N = 8, TILE_K = 256
- TILE_M = 128, TILE_N = 4, TILE_K = 128
- ROWS = 8 (每个 warp 处理更多行)

### 优先级 4: 向量化加载

修复之前的 bug，使用 int4/float4 向量化加载：
```cuda
int4 u_vec = *reinterpret_cast<const int4*>(&bq8->qs[0]);
```

**预期提升：** +10-15%

## 热节流问题

⚠️ **重要提示：** 在连续测试中观察到严重的 GPU 热节流

| 状态 | 2D Tile 性能 | 下降幅度 |
|------|-------------|---------|
| 正常温度 | 610-630 GFLOPS | - |
| 热节流 | 70-80 GFLOPS | **-88%** |

**建议：**
1. 在测试间增加冷却时间
2. 改善笔记本散热
3. 降低 GPU 功耗限制
4. 使用台式机 GPU 进行最终测试

## 代码结构

```
kernels/gemm/gemm_warp_optimized.cuh
├── gemm_q4_0_q8_1_warp_multirow_kernel<4>     # 4 行/warp
├── gemm_q4_0_q8_1_smem_kernel                 # Shared memory (TILE_K=128)
├── gemm_q4_0_q8_1_smem_large_kernel           # Shared memory (TILE_K=256)
└── gemm_q4_0_q8_1_tile2d_kernel<64,4,4>       # 2D Tiling (最佳)
    ├── TILE_M = 64
    ├── TILE_N = 4
    ├── TILE_K = 128
    └── ROWS = 4

tests/
├── benchmark_warp_optimized.cu                # 完整 benchmark
└── benchmark_best.cu                          # 简化 benchmark (避免热节流)
```

## 与 llama.cpp 对比

| 指标 | 我们的实现 | llama.cpp | 差距 |
|------|-----------|-----------|------|
| 性能 | 630 GFLOPS | 775 GFLOPS | -18.7% |
| 优化技术 | Warp + Shared Mem + 2D Tile | + Async + Persistent | 缺少高级特性 |
| 代码复杂度 | 中等 | 高 | 我们更易读 |
| 可移植性 | 好 | 好 | 相当 |

## 结论

### 成就

✅ **达到 630.4 GFLOPS** (llama.cpp 的 81.3%)
✅ **5.20x 加速比** (相对 naive 实现)
✅ **100% 正确性** (所有测试通过)
✅ **稳定性能** (不同矩阵尺寸下一致)

### 关键洞察

1. **2D Tiling 是关键** - 单这一项优化就带来了 21% 的提升
2. **内存复用至关重要** - 减少全局内存访问是性能的核心
3. **Warp 协作高效** - warp shuffle 比 atomic 快得多
4. **Shared memory 必不可少** - 但 tile 大小需要仔细调优

### 下一步

要达到 775 GFLOPS，需要：
1. 实现 async copy + double buffering (**最重要**)
2. 使用 Nsight Compute 找到真实瓶颈
3. 调优 tile 参数
4. 考虑 persistent kernel 设计

**预计：** 通过 async copy 可以达到 **720-750 GFLOPS**，非常接近目标。

---

**生成时间：** 2026-01-30
**GPU：** NVIDIA GeForce RTX 5070 Laptop GPU
**CUDA 架构：** sm_120
