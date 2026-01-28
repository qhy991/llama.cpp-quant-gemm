# 自定义 Kernel 测试流程总结

## 测试结果

✅ **所有测试通过！**

我们成功实现并测试了自定义的 W4A8 GEMM kernel，使用 DP4A 指令进行 Q4_0 × Q8_1 矩阵乘法。

## 测试配置

### 小规模测试 (M=256, N=256, K=512)
- **数据量**: 131K 激活值 + 131K 权重值
- **量化误差**:
  - 激活 (Q8_1): NMSE = 1.42e-05
  - 权重 (Q4_0): NMSE = 0.00462
- **计算精度**: NMSE = 5.67e-05 ✅
- **性能**: 2193.67 GFLOPS
- **执行时间**: 0.031 ms

### 大规模测试 (M=1024, N=1024, K=2048)
- **数据量**: 2.1M 激活值 + 2.1M 权重值
- **量化误差**:
  - 激活 (Q8_1): NMSE = 1.42e-05
  - 权重 (Q4_0): NMSE = 0.00465
- **计算精度**: NMSE = 5.38e-05 ✅
- **性能**: 311.78 GFLOPS
- **执行时间**: 13.78 ms

## 关键问题与解决方案

### 问题 1: 编译时间过长 (32+ 分钟)

**原因**: llama.cpp 默认为 13 个 GPU 架构编译每个 CUDA 文件

**解决方案**:
```bash
cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="120a"
```
只编译 RTX 5070 (sm_120a) 架构，编译时间降至 ~3 分钟

### 问题 2: 符号重复定义

**原因**: Kernel 函数在多个编译单元中被包含

**解决方案**: 将 kernel 函数声明为 `static __global__`

### 问题 3: 计算结果错误 (NMSE > 1.0)

**原因**: 4-bit 权重的 nibble 展开顺序不正确

**详细分析**:
- Q4_0 格式：每个字节存储 2 个 4-bit 值 `(high << 4) | low`
- 4 个字节包含 8 个 4-bit 值：`[w0,w1, w2,w3, w4,w5, w6,w7]`
- 原始实现：分别提取所有低位和高位 nibble
  - `expand_q4_low`: `[w0, w2, w4, w6]`
  - `expand_q4_high`: `[w1, w3, w5, w7]`
  - 导致与激活值的对齐错误

**解决方案**: 实现 `expand_q4_interleaved` 函数
```cuda
// 正确的展开方式：
out0 = [w0, w1, w2, w3]  // 与 a[0-3] 对齐
out1 = [w4, w5, w6, w7]  // 与 a[4-7] 对齐
```

### 问题 4: 补偿公式

**正确的公式**:
```
result = d_w * (d_a * sumi - 8 * s_a)
```

其中：
- `d_w`: Q4_0 权重的 scale
- `d_a`: Q8_1 激活的 scale
- `s_a`: Q8_1 激活的 sum（原始浮点值的和）
- `sumi`: 量化值的点积 `sum(q_w * q_a)`
- `8`: Q4_0 的偏移量（将 [-8,7] 映射到 [0,15]）

## 实现细节

### Kernel 结构

```cuda
gemm_w4a8_dp4a_kernel(
    const block_q8_1* A,  // M × K_blocks (激活)
    const block_q4_0* B,  // N × K_blocks (权重)
    float* C,             // M × N (输出)
    int M, int N, int K
)
```

### 核心循环

```cuda
for (int b = 0; b < nb; b++) {
    // 加载 scale 和 sum
    float d_a = __half2float(__low2half(block_a.ds));
    float s_a = __half2float(__high2half(block_a.ds));
    float d_w = __half2float(block_w.d);

    int32_t sumi = 0;

    // 处理 32 个元素（4 次迭代，每次 8 个元素）
    for (int i = 0; i < 4; i++) {
        int a0 = load_int_b4(block_a.qs, i * 2);
        int a1 = load_int_b4(block_a.qs, i * 2 + 1);
        int w_packed = load_int_b2(block_w.qs, i);

        int w0, w1;
        expand_q4_interleaved(w_packed, w0, w1);

        sumi = dp4a(a0, w0, sumi);
        sumi = dp4a(a1, w1, sumi);
    }

    // 应用补偿公式
    sum += d_w * (d_a * sumi - 8.0f * s_a);
}
```

## 性能分析

### 理论性能

RTX 5070 Laptop GPU:
- 计算能力: 12.0 (Blackwell)
- INT8 Tensor Core 性能: ~200 TOPS (理论值)

### 实际性能

- 小矩阵 (256×256×512): 2193 GFLOPS
- 大矩阵 (1024×1024×2048): 311 GFLOPS

### 性能差距原因

1. **未使用 Tensor Core**: 当前实现使用 DP4A 指令，而非 Tensor Core
2. **内存带宽限制**: 量化数据需要频繁访问内存
3. **Tile 大小**: 16×16 的 block 可能不是最优的
4. **缺少共享内存优化**: 未使用 shared memory 缓存

### 优化方向

1. **使用 Tensor Core**: 利用 WMMA 或 MMA 指令
2. **Tile 优化**: 使用更大的 tile (如 64×64)
3. **共享内存**: 缓存激活和权重块
4. **向量化加载**: 使用 float4/int4 向量加载
5. **流水线**: 使用异步拷贝和双缓冲

## 文件清单

### 核心实现
- `include/quant_types.h` - 量化类型定义
- `include/gemm_cuda_dp4a.cuh` - DP4A kernel 实现

### 测试文件
- `tests/test_custom_kernel.cu` - 完整测试套件
- `tests/test_simple.cu` - 简单调试测试
- `run_test.sh` - 测试脚本

### 文档
- `COMPILATION_ANALYSIS.md` - 编译问题分析
- `TEST_FLOW_SUMMARY.md` - 本文档

## 集成到 llama.cpp

自定义 kernel 已成功集成到 llama.cpp 中：

- **集成位置**: `llama.cpp/ggml/src/ggml-cuda/mmq.cuh`
- **触发条件**:
  - 量化类型为 `GGML_TYPE_Q4_0`
  - 单通道、单样本场景
  - 无专家路由
- **回退机制**: 其他情况自动使用 llama.cpp 原始实现

详见 `llama.cpp/CUSTOM_KERNEL_INTEGRATION.md`

## 下一步

1. **性能优化**: 实现上述优化方向
2. **更多量化格式**: 支持 Q4_1, Q5_0, Q5_1 等
3. **Benchmark**: 与 llama.cpp 原始实现对比
4. **实际模型测试**: 使用真实的 .gguf 模型文件测试

## 总结

我们成功实现了一个功能正确的 W4A8 GEMM kernel，并解决了所有编译和计算问题。虽然性能还有优化空间，但这是一个很好的起点，为后续优化奠定了基础。

关键成就：
- ✅ 正确实现 Q4_0 × Q8_1 矩阵乘法
- ✅ 计算精度 NMSE < 0.0001
- ✅ 成功集成到 llama.cpp
- ✅ 完整的测试流程和文档

---

生成时间: 2026-01-28
GPU: NVIDIA GeForce RTX 5070 Laptop GPU (sm_120a)
