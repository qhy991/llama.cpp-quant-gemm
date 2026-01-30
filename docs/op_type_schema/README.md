# llama.cpp Op Type Schema

This directory contains standardized operation type definitions for llama.cpp compatible CUDA kernels. Each schema follows a consistent format to enable:

1. **AI-assisted kernel development**: LLMs can generate implementations from these specs
2. **Automated testing**: Test cases can be generated from input/output definitions
3. **Performance benchmarking**: Workload traces can be standardized

## Schema Format

Each operation schema follows this structure:

```markdown
# op_name

[Description paragraph explaining the operation's purpose and use case]

Variants:
- variant_1: brief description
- variant_2: brief description

## variant_name

Axes (N dimensions):
- `dim_name`: variable | constant

Inputs (N tensors + M scalars):
- `tensor_name`: description [shape]
- `scalar_name`: description (scalar)

Outputs (N tensors):
- `tensor_name`: description [shape]

Constraints:
- constraint expressions

[Additional sections as needed: Block Structure, Formula, etc.]
```

## Available Operations

| Operation | File | Variants | Description |
|-----------|------|----------|-------------|
| quant_gemm | [quant_gemm.md](quant_gemm.md) | W4A16, W4A8, W4_1A8, W5A8, W8A8 | Quantized matrix multiplication |
| quant_vec_dot | [quant_vec_dot.md](quant_vec_dot.md) | q4_0_q8_1, q4_1_q8_1, q5_0_q8_1, q8_0_q8_1, q4_0_f32 | Block-wise vector dot product |
| quantize | [quantize.md](quantize.md) | q4_0, q4_1, q5_0, q5_1, q8_0, q8_1 | FP32 to quantized format conversion |
| dequantize | [dequantize.md](dequantize.md) | q4_0, q4_1, q5_0, q8_0, q8_1 | Quantized to FP32 format conversion |
| activation | [activation.md](activation.md) | silu, gelu, gelu_quick, relu, silu_mul | Activation functions |
| rmsnorm | [rmsnorm.md](rmsnorm.md) | standard, fused_add | RMS normalization |
| rope | [rope.md](rope.md) | standard, neox, glm | Rotary position embedding |
| softmax | [softmax.md](softmax.md) | standard, scaled, masked, online | Softmax attention scores |

## Task Template

See [TASK_TEMPLATE.md](TASK_TEMPLATE.md) for the standardized format to describe kernel implementation tasks for LLMs.

## Quantization Type Reference

| Type | Bits/Value | Block Size | Bytes/Block | Symmetry | Primary Use |
|------|------------|------------|-------------|----------|-------------|
| Q4_0 | 4.5 | 32 | 18 | Symmetric (offset +8) | Weight |
| Q4_1 | 5.0 | 32 | 20 | Asymmetric (min value) | Weight |
| Q5_0 | 5.5 | 32 | 22 | Symmetric (offset +16) | Weight |
| Q5_1 | 6.0 | 32 | 24 | Asymmetric | Weight |
| Q8_0 | 8.5 | 32 | 34 | Symmetric | Weight |
| Q8_1 | 9.0 | 32 | 36 | Symmetric + sum | Activation |

## Dimension Conventions

llama.cpp uses the following conventions:

```
GEMM: C[M, N] = A[M, K] × B[N, K]^T

Where:
- M = batch_size × seq_len (variable)
- N = output_features (constant per layer)
- K = input_features (constant per layer)

Memory Layout:
- All matrices are row-major
- Quantized weights: [N, K/block_size] blocks
- Quantized activations: [M, K/block_size] blocks
```

## Usage for LLM Code Generation

To generate a kernel implementation, provide:

1. The operation schema
2. Target optimization level (naive, tiled, dp4a, tensorcore)
3. Target GPU architecture (sm_61, sm_70, sm_80, sm_89)

Example prompt:
```
Generate a CUDA kernel for quant_gemm/W4A8 variant with:
- Optimization: dp4a
- Target: sm_80
- Follow the dot product formula exactly
- Ensure block_q4_0 and block_q8_1 are compatible with llama.cpp
```
