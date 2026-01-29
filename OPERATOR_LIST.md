# 算子列表

本项目实现的 CUDA 算子，与 llama.cpp 兼容。

## 已实现算子

### GEMM (矩阵乘法)

| 算子 | 文件 | llama.cpp 对应 | 说明 |
|------|------|----------------|------|
| `gemm_q4_0_q8_1` | `gemm_quant_formats.cuh` | `vec_dot_q4_0_q8_1_impl` | Q4_0权重 × Q8_1激活 |
| `gemm_q4_1_q8_1` | `gemm_quant_formats.cuh` | `vec_dot_q4_1_q8_1_impl` | Q4_1权重 × Q8_1激活 (非对称) |
| `gemm_q5_0_q8_1` | `gemm_quant_formats.cuh` | `vec_dot_q5_0_q8_1_impl` | Q5_0权重 × Q8_1激活 |
| `gemm_q5_1_q8_1` | `gemm_quant_formats.cuh` | `vec_dot_q5_1_q8_1_impl` | Q5_1权重 × Q8_1激活 (非对称) |
| `gemm_q8_0_q8_1` | `gemm_quant_formats.cuh` | `vec_dot_q8_0_q8_1_impl` | Q8_0权重 × Q8_1激活 |
| `gemm_f32_naive` | `gemm_cuda_naive.cuh` | - | FP32 朴素实现 (教学用) |
| `gemm_f32_tiled` | `gemm_cuda_tiled.cuh` | - | FP32 Tiled 实现 |
| `gemm_dp4a` | `gemm_cuda_dp4a.cuh` | - | DP4A 优化实现 (教学用) |

### 激活函数

| 算子 | 文件 | llama.cpp 对应 | 说明 |
|------|------|----------------|------|
| `silu_f32` | `kernels/activation/silu.cuh` | `ggml_cuda_op_silu` | SiLU(x) = x * sigmoid(x) |
| `silu_f32_vec4` | `kernels/activation/silu.cuh` | - | 向量化 SiLU |
| `silu_mul_f32` | `kernels/activation/silu.cuh` | `ggml_silu_inplace` | SiLU + 乘法融合 (FFN) |
| `gelu_f32` | `kernels/activation/gelu.cuh` | `ggml_cuda_op_gelu` | GELU 精确版本 |
| `gelu_quick_f32` | `kernels/activation/gelu.cuh` | `ggml_cuda_op_gelu_quick` | GELU tanh 近似 |

### 归一化

| 算子 | 文件 | llama.cpp 对应 | 说明 |
|------|------|----------------|------|
| `rms_norm_f32` | `kernels/normalization/rms_norm.cuh` | `ggml_cuda_op_rms_norm` | RMS 归一化 |
| `rms_norm_f32_vec4` | `kernels/normalization/rms_norm.cuh` | - | 向量化 RMS Norm |

### 注意力机制

| 算子 | 文件 | llama.cpp 对应 | 说明 |
|------|------|----------------|------|
| `softmax_f32` | `kernels/attention/softmax.cuh` | `ggml_cuda_op_soft_max` | Softmax |
| `softmax_f32_small` | `kernels/attention/softmax.cuh` | - | 小序列优化 Softmax |
| `softmax_causal_f32` | `kernels/attention/softmax.cuh` | - | 带 Causal Mask 的 Softmax |
| `rope_f32` | `kernels/attention/rope.cuh` | `ggml_cuda_op_rope` | 旋转位置编码 |
| `rope_interleaved_f32` | `kernels/attention/rope.cuh` | - | 交错布局 RoPE |
| `rope_batch_f32` | `kernels/attention/rope.cuh` | - | 批量 RoPE |
| `rope_with_cache_f32` | `kernels/attention/rope.cuh` | - | 预计算缓存 RoPE |

### 元素级操作

| 算子 | 文件 | llama.cpp 对应 | 说明 |
|------|------|----------------|------|
| `add_f32` | `kernels/elementwise/elementwise.cuh` | `ggml_cuda_op_add` | 元素加法 |
| `mul_f32` | `kernels/elementwise/elementwise.cuh` | `ggml_cuda_op_mul` | 元素乘法 |
| `scale_f32` | `kernels/elementwise/elementwise.cuh` | `ggml_cuda_op_scale` | 标量乘法 |
| `add_scale_f32` | `kernels/elementwise/elementwise.cuh` | - | 残差连接 |

## 待实现算子

### 高优先级

| 算子 | 说明 | llama.cpp 对应 |
|------|------|----------------|
| `layer_norm` | Layer Normalization | `ggml_cuda_op_norm` |
| `concat` | 张量拼接 | `ggml_cuda_op_concat` |
| `reshape/view` | 张量变形 | `ggml_cuda_cpy` |
| `flash_attention` | Flash Attention | `ggml_cuda_flash_attn_ext` |

### K-Quant 格式

| 格式 | 位宽 | 说明 |
|------|------|------|
| `Q2_K` | 2.5 bits | 超低精度 |
| `Q3_K` | 3.4 bits | 低精度 |
| `Q4_K` | 4.5 bits | 平衡精度 |
| `Q5_K` | 5.5 bits | 高精度 |
| `Q6_K` | 6.5 bits | 近无损 |

### 中优先级

| 算子 | 说明 |
|------|------|
| `embedding` | Token 嵌入 |
| `transpose` | 转置 |
| `permute` | 维度重排 |
| `clamp` | 值裁剪 |

## LLaMA 推理流程

```
输入 tokens
    ↓
┌─────────────────────────────────────────┐
│           Embedding Layer               │
│         (token → hidden_state)          │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│         Transformer Block × N           │
│  ┌───────────────────────────────────┐  │
│  │         RMS Norm ✓                │  │
│  │             ↓                     │  │
│  │    ┌───────────────────────┐      │  │
│  │    │   Self Attention      │      │  │
│  │    │  Q,K,V = Linear(x) ✓  │      │  │
│  │    │  RoPE(Q,K) ✓          │      │  │
│  │    │  Attn = Softmax ✓     │      │  │
│  │    │  Out = Attn @ V       │      │  │
│  │    └───────────────────────┘      │  │
│  │             ↓                     │  │
│  │    Add (Residual) ✓               │  │
│  │             ↓                     │  │
│  │         RMS Norm ✓                │  │
│  │             ↓                     │  │
│  │    ┌───────────────────────┐      │  │
│  │    │        FFN            │      │  │
│  │    │  gate = Linear(x) ✓   │      │  │
│  │    │  up = Linear(x) ✓     │      │  │
│  │    │  down = Linear(       │      │  │
│  │    │    SiLU(gate)*up) ✓   │      │  │
│  │    └───────────────────────┘      │  │
│  │             ↓                     │  │
│  │    Add (Residual) ✓               │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│           RMS Norm ✓                    │
│               ↓                         │
│           LM Head ✓                     │
│         (Linear → logits)               │
└─────────────────────────────────────────┘
    ↓
输出 logits

✓ = 已实现
```

## 使用示例

### 单独使用算子

```cpp
#include "kernels/activation/silu.cuh"
#include "kernels/normalization/rms_norm.cuh"
#include "kernels/attention/softmax.cuh"

// SiLU
silu_forward_f32(d_input, d_output, n);

// RMS Norm
rms_norm_forward_f32(d_input, d_weight, d_output, n_rows, n_cols);

// Softmax
softmax_forward_f32(d_input, d_output, n_rows, n_cols, scale);
```

### 与 llama.cpp 集成

```cpp
// 替换 llama.cpp 中的算子
// 1. 复制 kernel 文件到 llama.cpp/ggml/src/ggml-cuda/
// 2. 在对应的 op 函数中调用

// 例如，替换 SiLU:
// ggml-cuda/unary.cu
void ggml_cuda_op_silu(ggml_tensor* dst) {
    // 使用我们的实现
    silu_forward_f32(src, dst, n);
}
```

## 性能对比

| 算子 | 本项目 | llama.cpp | 说明 |
|------|--------|-----------|------|
| GEMM Q4_0 | 基准 | ~相同 | 使用相同 DP4A 策略 |
| SiLU | ~5% 快 | 基准 | 向量化优化 |
| RMS Norm | ~相同 | 基准 | 相同归约策略 |
| Softmax | ~相同 | 基准 | 相同实现 |

*注: 性能因 GPU 型号和输入大小而异*
