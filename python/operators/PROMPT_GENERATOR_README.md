# Kernel Implementation Prompt Generator

自动从 JSON spec 生成详细的 kernel 实现指南。

## 功能

根据 `spec.json` 自动生成完整的 kernel 实现指南，包括：
- Kernel 函数签名
- 输入/输出格式详解
- 量化格式说明（Q4_0, Q4_1, Q8_0, Q8_1）
- 数学公式
- Pybind11 集成代码
- 测试验证流程
- 实现检查清单

## 使用方法

### 基本用法

```bash
cd python
python3 operators/kernel_prompt_generator.py <spec_json_path>
```

### 示例

```bash
# 为 w4a16_q4_0_fp32 生成指南
python3 operators/kernel_prompt_generator.py operators/quant_gemm/variants/w4a16_q4_0_fp32/spec.json

# 为 w4a8_q4_0_q8_1 生成指南
python3 operators/kernel_prompt_generator.py operators/quant_gemm/variants/w4a8_q4_0_q8_1/spec.json

# 使用自定义 spec.json
python3 operators/kernel_prompt_generator.py my_custom_spec.json
```

### 输出

生成的指南保存在：`<variant_path>/IMPLEMENTATION_PROMPT.md`

## 工作原理

```
spec.json ──► Prompt Generator ──► IMPLEMENTATION_PROMPT.md
                │
                ├── 解析 kernel 配置
                ├── 查找量化格式定义
                ├── 生成函数签名
                ├── 生成数学公式
                ├── 生成测试说明
                └── 生成检查清单
```

## 支持的量化格式

| 格式 | 块大小 | 字节数/块 | 描述 |
|------|--------|-----------|------|
| `block_q4_0` | 32 | 18 | 4-bit + scale |
| `block_q4_1` | 32 | 20 | 4-bit + scale + min |
| `block_q8_0` | 32 | 34 | 8-bit + scale |
| `block_q8_1` | 32 | 36 | 8-bit + scale + min |
| `float32` | - | 4 | 标准浮点 |

## 添加新的量化格式

编辑 `kernel_prompt_generator.py` 中的 `QUANTIZATION_FORMATS` 字典：

```python
QUANTIZATION_FORMATS = {
    "your_format": {
        "name": "FORMAT_NAME",
        "block_size": 32,
        "bytes_per_block": XX,
        "scale_type": "float16",
        "data_type": "uint8",
        "description": "Description here",
        "layout": "Memory layout here",
        "dequant_formula": "value = ...",
        "dequant_code": """
// C++ code here
"""
    }
}
```

## 与 LLM 集成

可以直接将生成的 prompt 输入到 LLM：

```bash
# 直接输出到终端供复制
cat operators/quant_gemm/variants/w4a16_q4_0_fp32/IMPLEMENTATION_PROMPT.md

# 或使用 llm 工具（如果安装）
cat operators/quant_gemm/variants/w4a16_q4_0_fp32/IMPLEMENTATION_PROMPT.md | llm
```

## 生成的指南结构

```
# Kernel 实现指南: <name>

## 概述
## Kernel 函数签名
## 输入格式
  - Weight/Activation 详解
  - 内存布局
  - 反量化代码
## 输出格式
## 数学公式
## Pybind11 集成
## 测试验证
## 实现检查清单
## 参考资源
## 快速开始
```

## 示例输出

参见已生成的文件：
- `operators/quant_gemm/variants/w4a16_q4_0_fp32/IMPLEMENTATION_PROMPT.md`
- `operators/quant_gemm/variants/w4a8_q4_0_q8_1/IMPLEMENTATION_PROMPT.md`
