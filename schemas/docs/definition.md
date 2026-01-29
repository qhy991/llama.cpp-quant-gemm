# Definition Schema

## 概述

本文档描述了 quant-gemm-from-scratch 项目的 **Definition** JSON schema。

`Definition` 提供了计算工作负载的形式化、机器可读的规范。它是指导人类和 AI 代理进行 kernel 开发的唯一真实来源。

## JSON Schema 结构

### 顶层字段

| 字段 | 类型 | 必需 | 描述 |
|------|------|------|------|
| `name` | string | 是 | 唯一标识符，命名规范: `{op_type}_{format}_{constants}` |
| `description` | string | 否 | 人类可读的描述 |
| `op_type` | string | 是 | 操作类型 (见下表) |
| `tags` | array[string] | 否 | 标签用于分组和过滤 |
| `axes` | Dict[string, Axis] | 是 | 维度定义 |
| `constraints` | array[string] | 否 | 维度约束表达式 |
| `inputs` | Dict[string, TensorSpec] | 是 | 输入张量定义 |
| `outputs` | Dict[string, TensorSpec] | 是 | 输出张量定义 |
| `reference` | string | 是 | PyTorch 参考实现 |
| `quantization_spec` | object | 否 | 量化格式详细规范 |
| `compensation_formula` | object | 否 | 补偿公式 (用于量化 GEMM) |

### op_type 类型

| op_type | 描述 |
|---------|------|
| `gemm_fp32` | FP32 矩阵乘法 |
| `gemm_q4_0` | Q4_0 量化 GEMM (W4A16) |
| `gemm_q4_0_q8_1` | Q4_0×Q8_1 量化 GEMM (W4A8) |
| `quantize` | 量化操作 |
| `dequantize` | 反量化操作 |
| `silu` | SiLU 激活函数 |
| `gelu` | GELU 激活函数 |
| `rmsnorm` | RMS 归一化 |
| `softmax` | Softmax |
| `rope` | 旋转位置编码 |

### tags 命名空间

| 命名空间 | 描述 | 示例 |
|----------|------|------|
| `status:` | 验证状态 | `status:verified`, `status:draft` |
| `stage:` | 教程阶段 | `stage:step1`, `stage:step4` |
| `format:` | 量化格式 | `format:q4_0`, `format:q8_1` |
| `quantization:` | 量化类型 | `quantization:w4a16`, `quantization:w4a8` |
| `model:` | 兼容模型 | `model:llama-compatible` |
| `compensation:` | 补偿需求 | `compensation:required` |

### axes 维度定义

#### const (常量维度)

```json
"hidden_size": {
  "type": "const",
  "value": 4096,
  "description": "Hidden dimension"
}
```

#### var (变量维度)

```json
"batch_size": {
  "type": "var",
  "description": "Batch dimension"
}
```

### inputs/outputs 张量规范

```json
"A": {
  "shape": ["M", "K"],
  "dtype": "float32",
  "description": "Input matrix"
}
```

支持的 dtype:
- 浮点: `float32`, `float16`, `bfloat16`
- 整数: `int8`, `uint8`, `int32`, `int64`
- 量化: `q4_0`, `q8_0`, `q8_1`
- 其他: `bool`

标量使用 `"shape": null`。

### quantization_spec 量化规范

```json
"quantization_spec": {
  "q4_0": {
    "block_size": 32,
    "bytes_per_block": 18,
    "bits_per_element": 4.5,
    "structure": {
      "d": {"dtype": "float16", "bytes": 2},
      "qs": {"dtype": "uint8", "count": 16, "bytes": 16}
    },
    "quantize": "d = max(|x|) / 7.0; q = round(x / d) + 8",
    "dequantize": "x = (q - 8) * d"
  }
}
```

### compensation_formula 补偿公式

```json
"compensation_formula": {
  "formula": "result = d_w × (d_a × sumi - 8 × s_a)",
  "variables": {
    "d_w": "Q4_0 weight scale",
    "d_a": "Q8_1 activation scale",
    "sumi": "Integer dot product",
    "s_a": "Sum of original activations"
  }
}
```

## 完整示例

### 示例 1: FP32 GEMM

```json
{
  "name": "gemm_fp32_baseline",
  "description": "FP32 baseline GEMM: C = A @ B.T",
  "op_type": "gemm_fp32",
  "tags": ["status:verified", "stage:step1"],
  "axes": {
    "M": {"type": "var"},
    "N": {"type": "var"},
    "K": {"type": "var"}
  },
  "inputs": {
    "A": {"shape": ["M", "K"], "dtype": "float32"},
    "B": {"shape": ["N", "K"], "dtype": "float32"}
  },
  "outputs": {
    "C": {"shape": ["M", "N"], "dtype": "float32"}
  },
  "reference": "import torch\n\ndef run(A, B):\n    return torch.matmul(A, B.T)"
}
```

### 示例 2: Q4_0 × Q8_1 量化 GEMM

```json
{
  "name": "gemm_q4_0_q8_1_w4a8",
  "description": "W4A8 quantized GEMM with compensation",
  "op_type": "gemm_q4_0_q8_1",
  "tags": ["status:verified", "quantization:w4a8", "compensation:required"],
  "axes": {
    "M": {"type": "var"},
    "N": {"type": "var"},
    "K": {"type": "var"},
    "QK": {"type": "const", "value": 32}
  },
  "constraints": ["K % QK == 0"],
  "inputs": {
    "A_q8_1": {"shape": ["M", "K/QK"], "dtype": "q8_1"},
    "B_q4_0": {"shape": ["N", "K/QK"], "dtype": "q4_0"}
  },
  "outputs": {
    "C": {"shape": ["M", "N"], "dtype": "float32"}
  },
  "compensation_formula": {
    "formula": "result = d_w × (d_a × sumi - 8 × s_a)"
  }
}
```

### 示例 3: RMS Normalization

```json
{
  "name": "rmsnorm_h4096",
  "description": "RMS Layer Normalization",
  "op_type": "rmsnorm",
  "tags": ["status:verified", "model:llama-compatible"],
  "axes": {
    "batch_size": {"type": "var"},
    "hidden_size": {"type": "const", "value": 4096}
  },
  "inputs": {
    "hidden_states": {"shape": ["batch_size", "hidden_size"], "dtype": "float32"},
    "weight": {"shape": ["hidden_size"], "dtype": "float32"},
    "eps": {"shape": null, "dtype": "float32"}
  },
  "outputs": {
    "output": {"shape": ["batch_size", "hidden_size"], "dtype": "float32"}
  }
}
```

## 与 flashinfer-bench 的兼容性

本 schema 设计与 flashinfer-bench 的 Definition schema 兼容，主要扩展：

1. **quantization_spec**: 详细描述量化格式的结构和算法
2. **compensation_formula**: 记录量化 GEMM 的补偿公式推导
3. **自定义 dtype**: 支持 `q4_0`, `q8_0`, `q8_1` 等量化类型

这使得定义可以在两个项目间共享和复用。
