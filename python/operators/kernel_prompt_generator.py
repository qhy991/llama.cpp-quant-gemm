#!/usr/bin/env python3
"""
Kernel Implementation Prompt Generator

根据 JSON schema 自动生成 kernel 实现指南。
用于指导 LLM 或开发者实现符合测试框架要求的 kernel。

Usage:
    python kernel_prompt_generator.py <spec_json_path>

Example:
    python kernel_prompt_generator.py operators/quant_gemm/variants/w4a16_q4_0_fp32/spec.json
    python kernel_prompt_generator.py my_spec.json
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any


# 量化类型定义
QUANTIZATION_FORMATS = {
    "block_q4_0": {
        "name": "Q4_0",
        "block_size": 32,
        "bytes_per_block": 18,
        "scale_type": "float16",
        "data_type": "uint8",
        "description": "4-bit quantization with scale. Each block of 32 values stored in 18 bytes.",
        "layout": """
Block layout (18 bytes per block, 32 values):
  Bytes 0-1:   scale (fp16)
  Bytes 2-17:  packed 4-bit values (16 bytes = 128 bits = 32 values)

Memory layout: [scale][q0][q1]...[q31] where each qi is 4 bits
""",
        "dequant_formula": "value = (q - 8) * scale",
        "dequant_code": """
// Q4_0 dequantization
__half scale = *(__half*)&block[0];  // First 2 bytes
uint8_t packed = block[i/2 + 2];     // Data starts at byte 2
uint8_t q = (i % 2 == 0) ? (packed & 0x0F) : (packed >> 4);
float value = (float(q) - 8.0f) * __half2float(scale);
"""
    },
    "block_q4_1": {
        "name": "Q4_1",
        "block_size": 32,
        "bytes_per_block": 20,
        "scale_type": "float16",
        "min_type": "float16",
        "data_type": "uint8",
        "description": "4-bit quantization with scale and min. Each block of 32 values stored in 20 bytes.",
        "layout": """
Block layout (20 bytes per block, 32 values):
  Bytes 0-1:   scale (fp16)
  Bytes 2-3:   min (fp16)
  Bytes 4-19:  packed 4-bit values (16 bytes = 128 bits = 32 values)
""",
        "dequant_formula": "value = q * scale + min",
        "dequant_code": """
// Q4_1 dequantization
__half scale = *(__half*)&block[0];
__half min = *(__half*)&block[2];
uint8_t packed = block[i/2 + 4];
uint8_t q = (i % 2 == 0) ? (packed & 0x0F) : (packed >> 4);
float value = float(q) * __half2float(scale) + __half2float(min);
"""
    },
    "block_q8_0": {
        "name": "Q8_0",
        "block_size": 32,
        "bytes_per_block": 34,
        "scale_type": "float16",
        "data_type": "int8",
        "description": "8-bit quantization with scale. Each block of 32 values stored in 34 bytes.",
        "layout": """
Block layout (34 bytes per block, 32 values):
  Bytes 0-1:   scale (fp16)
  Bytes 2-33:  int8 values (32 bytes)
""",
        "dequant_formula": "value = q * scale",
        "dequant_code": """
// Q8_0 dequantization
__half scale = *(__half*)&block[0];
int8_t q = block[i + 2];
float value = float(q) * __half2float(scale);
"""
    },
    "block_q8_1": {
        "name": "Q8_1",
        "block_size": 32,
        "bytes_per_block": 36,
        "scale_type": "float16",
        "min_type": "float16",
        "data_type": "int8",
        "description": "8-bit quantization with scale and min. Each block of 32 values stored in 36 bytes.",
        "layout": """
Block layout (36 bytes per block, 32 values):
  Bytes 0-1:   scale (fp16)
  Bytes 2-3:   min (fp16)
  Bytes 4-35:  int8 values (32 bytes)
""",
        "dequant_formula": "value = q * scale + min",
        "dequant_code": """
// Q8_1 dequantization
__half scale = *(__half*)&block[0];
__half min = *(__half*)&block[2];
int8_t q = block[i + 4];
float value = float(q) * __half2float(scale) + __half2float(min);
"""
    },
    "float32": {
        "name": "FP32",
        "description": "Standard 32-bit floating point",
        "layout": "Standard FP32 (4 bytes per value)",
        "dequant_formula": "value = x",  # No quantization
        "dequant_code": "// No dequantization needed for FP32\nfloat value = x;"
    }
}


def format_shape(shape, params: Dict[str, Any]) -> str:
    """将 shape 表达式转换为可读格式"""
    context = params.copy()
    # 评估 K/32 这样的表达式
    result = []
    for dim in shape:
        dim_str = str(dim)
        if "/" in dim_str:
            parts = dim_str.split("/")
            try:
                val = int(context[parts[0]]) // int(parts[1])
                result.append(f"{val} (computed as {dim})")
            except:
                result.append(dim_str)
        else:
            val = context.get(dim_str, dim)
            result.append(str(val))
    return f"[{', '.join(result)}]"


def generate_input_description(input_name: str, input_spec: Dict[str, Any],
                               quant_info: Dict, params: Dict[str, Any]) -> str:
    """生成单个输入的描述"""
    dtype = input_spec["dtype"]
    shape = input_spec["shape"]

    lines = [
        f"### {input_name.title()}",
        "",
        f"**Data Type:** `{dtype}`",
        ""
    ]

    # 添加量化格式信息
    if dtype in QUANTIZATION_FORMATS:
        qinfo = QUANTIZATION_FORMATS[dtype]
        lines.extend([
            f"**Format:** {qinfo['name']}",
            "",
            qinfo["description"],
            "",
            "**Memory Layout:**",
            "```",
            qinfo["layout"],
            "```",
            "",
            "**Dequantization Formula:**",
            f"```cpp",
            f"{qinfo['dequant_formula']}",
            "```",
            "",
            "**Dequantization Code:**",
            "```cpp",
            qinfo["dequant_code"],
            "```"
        ])

    lines.extend([
        "",
        f"**Shape:** `{format_shape(shape, params)}`",
        "",
        input_spec.get("description", ""),
        ""
    ])

    return "\n".join(lines)


def generate_kernel_signature(spec: Dict[str, Any]) -> str:
    """生成 kernel 函数签名"""
    kernel = spec["kernel"]
    entry_point = kernel["entry_point"]
    inputs = spec["inputs"]
    outputs = spec["outputs"]

    # 收集参数
    params_list = []

    # 输入参数
    for name, input_spec in inputs.items():
        dtype = input_spec["dtype"]
        shape = input_spec["shape"]
        shape_str = ", ".join(str(s) for s in shape)

        # 确定指针类型
        if dtype == "float32":
            params_list.append(f"const float* {name}")
        elif dtype in QUANTIZATION_FORMATS:
            params_list.append(f"const uint8_t* {name}")
        else:
            params_list.append(f"const void* {name}")

    # 输出参数
    for name, output_spec in outputs.items():
        dtype = output_spec["dtype"]
        if dtype == "float32":
            params_list.append(f"float* {name}")
        else:
            params_list.append(f"void* {name}")

    # 维度参数
    params_list.extend(["int M", "int N", "int K"])

    signature = f"__global__ void {entry_point}(\n    "
    signature += ",\n    ".join(params_list)
    signature += "\n)"

    return signature


def generate_math_formula(spec: Dict[str, Any]) -> str:
    """生成数学公式"""
    lines = [
        "## 数学公式",
        "",
        "### 高层公式",
        "```"
    ]

    formula = spec.get("formula", {})
    outputs = spec["outputs"]
    inputs = spec["inputs"]

    output_shape = outputs["output"]["shape"]
    input_names = list(inputs.keys())

    if "gemm" in formula:
        lines.append(formula["gemm"])
    elif "dot_product" in formula:
        lines.append(formula["dot_product"])
    else:
        # 默认 GEMM 公式
        lines.append(f"C[m,n] = sum_k({input_names[0]}[n,k] * {input_names[1]}[m,k])")

    lines.append("```")
    lines.append("")

    # 添加解释
    if "explanation" in formula:
        lines.extend([
            "**解释:**",
            "",
            formula["explanation"],
            ""
        ])

    # 如果有反量化公式
    if "dequantize" in formula:
        lines.extend([
            "### 反量化",
            "```cpp",
            formula["dequantize"],
            "```"
        ])

    return "\n".join(lines)


def generate_pybind_section(spec: Dict[str, Any]) -> str:
    """生成 pybind11 集成说明"""
    kernel = spec["kernel"]
    entry_point = kernel["entry_point"]
    name = spec["name"]

    # 生成参数类型
    params = []
    for input_name, input_spec in spec["inputs"].items():
        dtype = input_spec["dtype"]
        if dtype == "float32":
            params.append("py::array_t<float>")
        else:
            params.append("py::array_t<uint8_t>")

    for output_name, output_spec in spec["outputs"].items():
        dtype = output_spec["dtype"]
        if dtype == "float32":
            params.append("py::array_t<float>")
        else:
            params.append("py::array_t<uint8_t>")

    params_str = ", ".join(params)

    return f"""## Pybind11 集成

在 `bindings.cpp` 中添加以下声明:

```cpp
// Include header (if separate)
// #include "kernels/{kernel['file']}"

// Binding declaration
m.def("{entry_point}",
    []({params_str}) {{
        // TODO: Implement buffer_info extraction and kernel launch
        // See w4a8_q4_0_q8_1 variant for reference
        py::gil_scoped_release release;
        // Launch kernel here
    }},
    py::arg({", ".join([f'"{n}"' for n in spec["inputs"].keys()])}),
    "Kernel implementation for {name}"
);
```

**重要提示:**
- 函数名必须与 `spec.json` 中的 `kernel.entry_point` 一致
- 参数顺序必须与 spec 中定义的顺序一致
- 必须释放 GIL (`py::gil_scoped_release`)
"""


def generate_test_section(spec: Dict[str, Any]) -> str:
    """生成测试验证说明"""
    accuracy = spec.get("accuracy", {})
    test_configs = spec.get("test_configs", [])
    reference = spec.get("reference", "")

    metric = accuracy.get("metric", "nmse")
    threshold = accuracy.get("threshold", 0.05)

    lines = [
        "## 测试框架验证",
        "",
        "### 验证流程",
        "",
        "```",
        "1. 测试框架生成随机输入数据",
        "2. 调用 reference.py 生成正确输出",
        "3. 调用你的 kernel 生成实际输出",
        "4. 比较两者并计算 " + metric.upper(),
        "5. 验证 " + metric.upper() + f" 是否 ≤ {threshold}",
        "```",
        "",
        "### 精度要求",
        "",
        f"- **指标:** {metric.upper()}",
        f"- **阈值:** {threshold}",
        "",
        f"**NMSE 计算公式:**",
        "```python",
        "nmse = np.mean((ref - actual) ** 2) / np.mean(ref ** 2)",
        "```",
        "",
        "### 测试配置",
        ""
    ]

    for config in test_configs:
        lines.append(f"- `{config['name']}`: M={config['M']}, N={config['N']}, K={config['K']}")

    lines.extend([
        "",
        "### 参考实现",
        "",
        f"位置: `{reference}`",
        "",
        "### 验收标准",
        "",
        "1. **正确性**: 所有测试配置的 NMSE ≤ " + str(threshold),
        "2. **性能**: 需要达到最低性能目标",
        "3. **稳定性**: 多次运行结果一致",
        ""
    ])

    return "\n".join(lines)


def generate_common_pitfalls_section(spec: Dict[str, Any]) -> str:
    """生成常见错误警告部分"""
    inputs = spec["inputs"]

    # 检测使用的量化格式
    used_formats = set()
    has_fp32_activation = False
    has_quant_activation = False

    for input_name, input_spec in inputs.items():
        dtype = input_spec["dtype"]
        if dtype in QUANTIZATION_FORMATS and dtype != "float32":
            used_formats.add(dtype)
            if "activation" in input_name.lower():
                has_quant_activation = True
        elif dtype == "float32" and "activation" in input_name.lower():
            has_fp32_activation = True

    lines = [
        "## ⚠️ 常见错误和陷阱",
        "",
        "**请仔细阅读本节以避免常见的实现错误！**",
        "",
    ]

    # Q4_0 特定警告
    if "block_q4_0" in used_formats:
        lines.extend([
            "### 🚨 CRITICAL: Q4_0 Packing Format",
            "",
            "**Q4_0 使用 SPLIT-BY-16 打包，不是连续对！**",
            "",
            "✅ **正确的理解:**",
            "```",
            "byte[0]  = weight[0]  (low nibble) | weight[16] (high nibble)",
            "byte[1]  = weight[1]  (low nibble) | weight[17] (high nibble)",
            "...",
            "byte[15] = weight[15] (low nibble) | weight[31] (high nibble)",
            "```",
            "",
            "❌ **错误的理解 (常见错误):**",
            "```",
            "byte[0] = weight[0] (low) | weight[1] (high)  // WRONG!",
            "byte[1] = weight[2] (low) | weight[3] (high)  // WRONG!",
            "```",
            "",
            "✅ **正确的解包代码:**",
            "```cpp",
            "for (int i = 0; i < 16; i++) {",
            "    uint8_t packed = data_ptr[i];",
            "    ",
            "    // Low nibble -> weight[i]",
            "    uint8_t q0 = packed & 0x0F;",
            "    float w0 = (float(q0) - 8.0f) * scale;",
            "    sum += activation[k_start + i] * w0;",
            "    ",
            "    // High nibble -> weight[i + 16]",
            "    uint8_t q1 = packed >> 4;",
            "    float w1 = (float(q1) - 8.0f) * scale;",
            "    sum += activation[k_start + i + 16] * w1;",
            "}",
            "```",
            "",
            "**验证方法:**",
            "1. 先测试 quantize -> dequantize 往返",
            "2. 使用简单固定值测试 (weight=0.5, activation=2.0)",
            "3. 确保 NMSE < 0.05",
            "",
        ])

    # 维度约定警告
    lines.extend([
        "### 🚨 Dimension Conventions",
        "",
    ])

    if has_quant_activation:
        lines.extend([
            "**本 kernel 使用量化 activation (w4a8 约定):**",
            "",
            "```cpp",
            "// Kernel 计算: C[N, M] = W[N, K] @ A[M, K]^T",
            "// 调用约定: kernel(weight, activation, N, M, K)",
            "// 输出需要转置: output.T 得到 [M, N]",
            "```",
            "",
        ])
    elif has_fp32_activation:
        lines.extend([
            "**本 kernel 使用 FP32 activation (w4a16 约定):**",
            "",
            "```cpp",
            "// Kernel 计算: C[M, N] = A[M, K] @ W[N, K]^T",
            "// 调用约定: kernel(weight, activation, M, N, K)",
            "// 输出直接是 [M, N]，无需转置",
            "```",
            "",
        ])

    lines.extend([
        "### 🚨 Memory and Performance Pitfalls",
        "",
        "1. **Integer Overflow**",
        "   ```cpp",
        "   // ❌ WRONG: 可能溢出",
        "   int offset = n * num_blocks * 18;",
        "   ",
        "   // ✅ CORRECT: 使用 long long",
        "   long long offset = (long long)(n * num_blocks) * 18;",
        "   ```",
        "",
        "2. **Memory Alignment (float4)**",
        "   ```cpp",
        "   // float4 需要 16-byte 对齐",
        "   // 如果地址未对齐，会退化为 4 次单独读取",
        "   // 确保 K 是 4 的倍数且起始地址对齐",
        "   ```",
        "",
        "3. **Quantization Offset**",
        "   ```cpp",
        "   // Q4_0: 值偏移 8 (范围 [0,15] -> [-8,7])",
        "   float w = (float(q) - 8.0f) * scale;  // 必须减 8!",
        "   ```",
        "",
        "4. **Block Size Assumptions**",
        "   ```cpp",
        "   // 不要硬编码 32，使用常量",
        "   int num_blocks = K / QK4_0;  // QK4_0 = 32",
        "   ```",
        "",
    ])

    lines.extend([
        "### ✅ Testing Best Practices",
        "",
        "**测试顺序 (从简单到复杂):**",
        "",
        "1. **Quantization Roundtrip**",
        "   ```python",
        "   x -> quantize -> dequantize -> x'",
        "   max_error = (x - x').abs().max()",
        "   assert max_error < 1.0  # Q4_0 有显著误差",
        "   ```",
        "",
        "2. **Fixed Values**",
        "   ```python",
        "   weight = torch.full((N, K), 0.5)",
        "   activation = torch.full((M, K), 2.0)",
        "   expected_output = K * 0.5 * 2.0 = K",
        "   ```",
        "",
        "3. **Different Data Patterns**",
        "   - All zeros",
        "   - All ones",
        "   - Positive only (torch.rand)",
        "   - Mixed signs (torch.randn) ← 最容易暴露 bug",
        "",
        "4. **NMSE Thresholds**",
        "   - Q4_0: NMSE < 0.05 (5%)",
        "   - Q8_1: NMSE < 0.01 (1%)",
        "   - FP16: NMSE < 0.001 (0.1%)",
        "",
    ])

    lines.extend([
        "### 📚 Reference Implementations",
        "",
        "**在实现前，请参考:**",
        "",
        "1. **Dequantization Reference**",
        "   - 查看 `dequantize_q4_0_kernel` in gemm_ops.cu",
        "   - 确保你的解包逻辑与之一致",
        "",
        "2. **llama.cpp Q4_0 Format**",
        "   - https://github.com/ggerganov/llama.cpp/blob/master/ggml.c",
        "   - 搜索 `dequantize_row_q4_0`",
        "",
        "3. **Working Kernels**",
        "   - w4a8_q4_0_q8_1: 参考量化 activation 的实现",
        "   - w4a16_q4_0_fp32: 参考 FP32 activation 的实现",
        "",
    ])

    return "\n".join(lines)


def generate_implementation_checklist(spec: Dict[str, Any]) -> str:
    """生成实现检查清单"""
    name = spec["name"]
    entry_point = spec["kernel"]["entry_point"]

    inputs_desc = []
    for input_name, input_spec in spec["inputs"].items():
        dtype = input_spec["dtype"]
        shape = input_spec["shape"]
        inputs_desc.append(f"  - {input_name}: {dtype}, shape={'x'.join(str(s) for s in shape)}")

    outputs_desc = []
    for output_name, output_spec in spec["outputs"].items():
        dtype = output_spec["dtype"]
        shape = output_spec["shape"]
        outputs_desc.append(f"  - {output_name}: {dtype}, shape={'x'.join(str(s) for s in shape)}")

    accuracy = spec.get("accuracy", {})
    threshold = accuracy.get("threshold", 0.05)

    return f"""## 实现检查清单

### 开始实现前

- [ ] 阅读 `KERNEL_IMPLEMENTATION_GUIDE.md` 了解 GEMM 基础
- [ ] 理解本指南中的所有输入格式和数学公式
- [ ] 阅读 `{spec['kernel']['file']}` 中的参考实现（如果存在）

### 实现步骤

1. [ ] **创建文件**: `operators/quant_gemm/variants/{name}/{spec['kernel']['file']}`
2. [ ] **实现 kernel 函数**:
   ```cpp
   {generate_kernel_signature(spec)}
   {{
       // TODO: 实现 kernel 逻辑
   }}
   ```
3. [ ] **添加 pybind11 声明**: 在 `bindings.cpp` 中注册函数
4. [ ] **编译验证**: `python setup.py build_ext --inplace`
5. [ ] **运行测试**: `python test_operator.py {name} operators/quant_gemm/variants/{name}`

### 输入参数

{chr(10).join(inputs_desc)}

### 输出参数

{chr(10).join(outputs_desc)}

### 验收标准

- [ ] 所有测试配置通过
- [ ] NMSE ≤ {threshold}
- [ ] 无内存泄漏
- [ ] 代码符合项目规范
"""


def generate_prompt(spec: Dict[str, Any], variant_path: Path) -> str:
    """生成完整的 kernel 实现提示"""
    name = spec["name"]
    description = spec.get("description", "")
    family = spec.get("family", "")
    version = spec.get("version", "1.0.0")

    # 获取默认参数
    default_params = {
        "M": 1,
        "N": spec["params"]["N"].get("default", 4096),
        "K": spec["params"]["K"].get("default", 4096)
    }

    lines = [
        f"# Kernel 实现指南: {name}",
        "",
        f"**Family:** {family}",
        f"**Version:** {version}",
        "",
        description,
        "",
        "---",
        "",
        "## 概述",
        "",
        f"本指南描述如何实现 `{name}` kernel。该 kernel 是量化矩阵乘法(GEMM)的一个变体。",
        "",
        "## 目录",
        "",
        "1. [Kernel 函数签名](#kernel-函数签名)",
        "2. [输入格式](#输入格式)",
        "3. [输出格式](#输出格式)",
        "4. [数学公式](#数学公式)",
        "5. [Pybind11 集成](#pybind11-集成)",
        "6. [测试验证](#测试验证)",
        "7. [实现检查清单](#实现检查清单)",
        "",
        "---",
        "",
        "## Kernel 函数签名",
        "",
        "### 必须使用的函数名和签名",
        "",
        "```cpp",
        generate_kernel_signature(spec),
        "```",
        "",
        "**参数说明:**",
        ""
    ]

    # 添加参数说明
    for param_name, param_spec in spec["params"].items():
        default = param_spec.get("default", "N/A")
        constraint = param_spec.get("constraint", "无")
        lines.extend([
            f"- **{param_name}**: {param_spec['description']}",
            f"  - 默认值: {default}",
            f"  - 约束: {constraint}",
            ""
        ])

    lines.extend([
        "",
        "## 输入格式",
        ""
    ])

    # 添加输入描述
    for input_name, input_spec in spec["inputs"].items():
        dtype = input_spec["dtype"]
        quant_info = QUANTIZATION_FORMATS.get(dtype, {})
        lines.append(generate_input_description(input_name, input_spec, quant_info, default_params))
        lines.append("")

    lines.extend([
        "### 内存布局约定",
        "",
        "- 所有输入都是行优先 (row-major) 存储",
        "- K 维度必须是 32 的倍数 (量化 block size)",
        "- 对于量化数据，最后一个维度包含完整的 block 字节数",
        "",
        "---",
        "",
        "## 输出格式",
        ""
    ])

    # 添加输出描述
    for output_name, output_spec in spec["outputs"].items():
        dtype = output_spec["dtype"]
        shape = output_spec["shape"]
        lines.extend([
            f"### {output_name.title()}",
            "",
            f"**Data Type:** `{dtype}`",
            f"**Shape:** `{'x'.join(shape)}`",
            "",
            output_spec.get("description", ""),
            ""
        ])

    lines.extend([
        "**注意:** 输出是行优先存储的 [M, N] 矩阵。",
        "",
        generate_math_formula(spec),
        "",
        "---",
        "",
        generate_common_pitfalls_section(spec),
        "",
        "---",
        "",
        generate_pybind_section(spec),
        "",
        "---",
        "",
        generate_test_section(spec),
        "",
        "---",
        "",
        generate_implementation_checklist(spec),
        "",
        "---",
        "",
        "## 参考资源",
        "",
        f"- **Variant 目录:** `{variant_path}`",
        f"- **Spec 文件:** `{variant_path}/spec.json`",
        f"- **Kernel 文件:** `{variant_path}/{spec['kernel']['file']}`",
        f"- **Reference 实现:** `{variant_path}/{spec.get('reference', 'reference.py')}`",
        "",
        "---",
        "",
        "## 快速开始",
        "",
        f"```bash",
        f"# 1. 创建 kernel 文件",
        f"touch operators/quant_gemm/variants/{name}/{spec['kernel']['file']}",
        f"",
        f"# 2. 实现 kernel (参考上面的函数签名)",
        f"",
        f"# 3. 在 bindings.cpp 中添加声明",
        f"",
        f"# 4. 编译",
        f"python setup.py build_ext --inplace",
        f"",
        f"# 5. 测试",
        f"python test_operator.py {name} operators/quant_gemm/variants/{name}",
        f"```"
    ])

    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python kernel_prompt_generator.py <spec_json_path>")
        print("")
        print("Examples:")
        print("  python kernel_prompt_generator.py operators/quant_gemm/variants/w4a16_q4_0_fp32/spec.json")
        print("  python kernel_prompt_generator.py my_spec.json")
        sys.exit(1)

    spec_file = Path(sys.argv[1])

    if not spec_file.exists():
        print(f"Error: spec file not found: {spec_file}")
        sys.exit(1)

    variant_path = spec_file.parent

    with open(spec_file, 'r') as f:
        spec = json.load(f)

    prompt = generate_prompt(spec, variant_path)

    # 输出到文件
    output_file = variant_path / "IMPLEMENTATION_PROMPT.md"
    with open(output_file, 'w') as f:
        f.write(prompt)

    print(f"Generated implementation prompt: {output_file}")
    print("")
    print(f"To view: cat {output_file}")
    print(f"Or use with LLM: cat {output_file} | llm ...")


if __name__ == "__main__":
    main()
