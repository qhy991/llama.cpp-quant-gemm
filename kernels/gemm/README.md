# GEMM 内核实现

本目录包含各种 GEMM (General Matrix Multiply) 内核的实现，从基础版本到优化版本。

## 文件列表

| 文件 | 描述 | 性能等级 |
|------|------|----------|
| `gemm_cuda_naive.cuh` | 朴素实现 | 基准 |
| `gemm_cuda_tiled.cuh` | Tiled 共享内存优化 | 2-5x |
| `gemm_cuda_dp4a.cuh` | DP4A 量化 GEMM | 5-10x |

## 量化格式

### Q4_0 (权重)
```
struct block_q4_0 {
    half d;              // 缩放因子 (2 bytes)
    uint8_t qs[16];      // 32个4-bit量化值 (16 bytes)
};  // 总计 18 bytes
```

**量化公式:**
- `d = max(|x|) / 7.0`
- `q = round(x / d) + 8`, 范围 [0, 15]

**反量化:**
- `x = (q - 8) * d`

### Q8_1 (激活)
```
struct block_q8_1 {
    half2 ds;            // d=缩放, s=原值求和 (4 bytes)
    int8_t qs[32];       // 32个8-bit量化值 (32 bytes)
};  // 总计 36 bytes
```

**量化公式:**
- `d = max(|x|) / 127.0`
- `q = round(x / d)`, 范围 [-127, 127]
- `s = Σx[i]` (原始值求和)

## 补偿公式

Q4_0 × Q8_1 点积使用补偿公式来处理 Q4_0 的偏移 8:

```
result = Σ (q_w - 8) * d_w * q_a * d_a
       = d_w * d_a * Σ (q_w - 8) * q_a
       = d_w * d_a * (Σ q_w * q_a - 8 * Σ q_a)
       = d_w * d_a * sumi - 8 * d_w * d_a * Σ q_a
```

由于 `Σ q_a ≈ (Σ x_a) / d_a = s / d_a`:

```
result = d_w * (d_a * sumi - 8 * s)
```

这就是为什么 Q8_1 需要存储原值求和 `s`。

## DP4A 指令

DP4A (Dot Product of 4 Accumulated) 是 NVIDIA Pascal 及更新架构的特性:

```c
int __dp4a(int a, int b, int c)
    = c + a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w
```

其中 a 和 b 被视为 4 个打包的 int8 值。

**优势:** 单指令完成 4 次 int8 乘法和累加，吞吐量提升 4 倍。

## 使用示例

### 基础使用
```cpp
#include "gemm_cuda_dp4a.cuh"

// 准备量化数据
block_q4_0* d_weight;  // GPU 上的 Q4_0 权重
block_q8_1* d_act;     // GPU 上的 Q8_1 激活
float* d_output;       // 输出

// 启动 kernel
dim3 block(16, 16);
dim3 grid((M + 15) / 16, (N + 15) / 16);
gemm_q4_0_q8_1_kernel<<<grid, block>>>(
    d_weight, d_act, d_output, M, N, K
);
```

### 与 llama.cpp 集成

本项目的类型定义与 llama.cpp 完全兼容:

```cpp
// llama.cpp 使用
#include "ggml-common.h"
block_q4_0* weights;
block_q8_1* activations;

// 本项目使用
#include "compat/ggml_types.h"
block_q4_0* weights;  // 相同的类型
block_q8_1* activations;
```

两者的内存布局完全一致，可以直接替换。

## 性能优化技巧

1. **对齐访问**: 确保量化块对齐到 4 字节边界
2. **合并内存访问**: 连续线程访问连续内存
3. **共享内存**: 使用 tile 技术减少全局内存访问
4. **寄存器复用**: 最大化每次加载的计算量

## 测试

运行单元测试:
```bash
cd tests/unit
nvcc -o test_gemm_q4 test_gemm_q4.cu -I../../ --gpu-architecture=sm_86
./test_gemm_q4
```

命令行选项:
- `-M <value>`: 设置 M 维度
- `-N <value>`: 设置 N 维度
- `-K <value>`: 设置 K 维度
- `-seed <value>`: 设置随机种子
- `-q`: 安静模式

## 开发路线

- [x] Naive GEMM 实现
- [x] Tiled GEMM 优化
- [x] DP4A 量化 GEMM (Q4_0 x Q8_1)
- [ ] Tensor Core 支持 (WMMA)
- [ ] K-quant 支持 (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K)
- [ ] Flash Attention 集成
