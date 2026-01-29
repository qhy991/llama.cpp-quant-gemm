# Quant-GEMM Schema System

本项目采用与 flashinfer-bench 兼容的三层 Schema 系统来描述量化 GEMM 及相关算子。

## Schema 架构

```
schemas/
├── definitions/          # 算子定义 (Definition)
│   ├── gemm/            # GEMM 相关定义
│   ├── quantization/    # 量化格式定义
│   ├── activation/      # 激活函数定义
│   ├── normalization/   # 归一化定义
│   └── attention/       # 注意力机制定义
├── solutions/           # 实现方案 (Solution)
└── README.md
```

## 核心概念

### 1. Definition (算子定义)

描述计算工作负载的形式化规范，包含：
- `name`: 唯一标识符
- `op_type`: 操作类型
- `axes`: 维度定义 (const/var)
- `inputs`/`outputs`: 张量规范
- `reference`: PyTorch 参考实现

### 2. Solution (实现方案)

针对 Definition 的具体实现：
- `name`: 方案名称
- `definition`: 引用的 Definition
- `spec`: 实现规范 (语言、硬件、入口点)
- `sources`: 源代码文件

### 3. Trace (执行记录)

基准测试的不可变记录：
- `workload`: 具体输入配置
- `evaluation`: 正确性和性能指标

## 支持的操作类型 (op_type)

| op_type | 描述 | 示例 |
|---------|------|------|
| `gemm_fp32` | FP32 矩阵乘法 | `gemm_fp32_baseline` |
| `gemm_q4_0` | Q4_0 量化 GEMM | `gemm_q4_0_w4a16` |
| `gemm_q4_0_q8_1` | Q4_0×Q8_1 量化 GEMM | `gemm_q4_0_q8_1_w4a8` |
| `quantize` | 量化操作 | `quantize_q4_0`, `quantize_q8_1` |
| `dequantize` | 反量化操作 | `dequantize_q4_0` |
| `silu` | SiLU 激活函数 | `silu_h4096` |
| `gelu` | GELU 激活函数 | `gelu_h4096` |
| `rmsnorm` | RMS 归一化 | `rmsnorm_h4096` |
| `softmax` | Softmax | `softmax_d128` |
| `rope` | 旋转位置编码 | `rope_d128` |

## 量化格式

### Q4_0 (4-bit 权重)
- 结构: `{half d, uint8_t qs[16]}` (18 bytes / 32 elements)
- 量化: `q = round(x / d) + 8`, `d = max(|x|) / 7`
- 反量化: `x = (q - 8) * d`

### Q8_0 (8-bit 对称)
- 结构: `{half d, int8_t qs[32]}` (34 bytes / 32 elements)
- 量化: `q = round(x / d)`, `d = max(|x|) / 127`

### Q8_1 (8-bit 带补偿)
- 结构: `{half2 ds, int8_t qs[32]}` (36 bytes / 32 elements)
- 特殊: 存储原始值之和 `s = Σx[i]` 用于补偿

## 补偿公式

Q4_0 × Q8_1 点积的核心公式：

```
result = d_w × (d_a × sumi - 8 × s_a)

其中:
  d_w  = Q4_0 scale
  d_a  = Q8_1 scale
  sumi = Σ q_w[i] × q_a[i]  (整数点积)
  s_a  = Σ x_a[i]           (原始激活值之和)
```

## 数据类型

支持的 dtype:
- `float32`, `float16`, `bfloat16`
- `int8`, `uint8`, `int32`
- `q4_0`, `q8_0`, `q8_1` (自定义量化类型)

## 使用示例

```python
import json

# 加载 Definition
with open("schemas/definitions/gemm/gemm_q4_0_q8_1_bs32.json") as f:
    definition = json.load(f)

print(f"Name: {definition['name']}")
print(f"Op Type: {definition['op_type']}")
print(f"Inputs: {list(definition['inputs'].keys())}")
```
