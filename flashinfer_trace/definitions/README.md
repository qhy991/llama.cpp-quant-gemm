# llama.cpp Operation Definitions

This directory contains JSON schema definitions for llama.cpp compatible CUDA kernels, following the flashinfer-bench format.

## Directory Structure

```
definitions/
├── quant_gemm/           # Quantized GEMM operations
│   ├── w4a8_q4_0_q8_1_n4096_k4096.json
│   ├── w4a16_q4_0_fp32_n4096_k4096.json
│   └── w4_1a8_q4_1_q8_1_n4096_k4096.json
├── quant_vec_dot/        # Block-wise dot products
│   └── vec_dot_q4_0_q8_1.json
├── quantize/             # Quantization operations
│   └── quantize_q8_1_k4096.json
├── rmsnorm/              # RMS normalization
├── activation/           # Activation functions
└── README.md
```

## JSON Schema Format

Each definition follows this structure:

```json
{
  "name": "operation_name",
  "op_type": "quant_gemm | quantize | quant_vec_dot | ...",
  "variant": "W4A8 | q8_1 | ...",
  "description": "Human-readable description",
  "tags": ["status:verified", "framework:llama.cpp", "quantization:q4_0"],

  "axes": {
    "M": {"type": "var", "description": "..."},
    "N": {"type": "const", "value": 4096, "description": "..."}
  },

  "inputs": {
    "tensor_name": {
      "shape": ["M", "K"],
      "dtype": "float32 | block_q4_0 | block_q8_1",
      "description": "..."
    }
  },

  "outputs": {
    "tensor_name": {
      "shape": ["M", "N"],
      "dtype": "float32",
      "description": "..."
    }
  },

  "constraints": ["K % 32 == 0", "sizeof(block_q4_0) == 18"],

  "types": {
    "block_q4_0": {
      "size": 18,
      "fields": [...]
    }
  },

  "formula": {
    "dot_product": "result = d_w * (d_a * sumi - 8.0f * s_a)",
    "explanation": "..."
  },

  "reference": "Python reference implementation as string"
}
```

## Key Differences from flashinfer-bench

| Feature | flashinfer-bench | llama.cpp (this) |
|---------|-----------------|------------------|
| Data types | float16, float8_e4m3fn | block_q4_0, block_q8_1, etc. |
| Block structure | N/A | Explicit type definitions |
| Formula | Simple GEMM | Compensation formulas |
| Quantization | Block-scale FP8 | Per-block symmetric/asymmetric |

## Supported Quantization Types

| Type | Size | Description | Primary Use |
|------|------|-------------|-------------|
| block_q4_0 | 18 bytes | 4-bit symmetric, offset +8 | Weight |
| block_q4_1 | 20 bytes | 4-bit asymmetric, min-max | Weight |
| block_q5_0 | 22 bytes | 5-bit symmetric | Weight |
| block_q5_1 | 24 bytes | 5-bit asymmetric | Weight |
| block_q8_0 | 34 bytes | 8-bit symmetric | Weight |
| block_q8_1 | 36 bytes | 8-bit with sum | Activation |

## Naming Convention

```
{op}_{variant}_{weight_type}_{act_type}_n{N}_k{K}.json

Examples:
- w4a8_q4_0_q8_1_n4096_k4096.json   # W4A8 with N=4096, K=4096
- w4a16_q4_0_fp32_n4096_k4096.json  # W4A16 with FP32 activation
- quantize_q8_1_k4096.json           # Q8_1 quantization
```

## Usage for LLM Code Generation

1. **Select a definition** that matches your target operation
2. **Parse the JSON** to extract:
   - Input/output tensor shapes and types
   - Type definitions (block structures)
   - Formula for computation
   - Reference implementation
3. **Generate CUDA kernel** following the spec

Example prompt:
```
Using the definition in w4a8_q4_0_q8_1_n4096_k4096.json:
1. Generate a naive CUDA kernel
2. Use the exact formula: d_w * (d_a * sumi - 8.0f * s_a)
3. Ensure block_q4_0 is 18 bytes, block_q8_1 is 36 bytes
4. Add proper bounds checking for M, N, K
```

## Validation

Each definition includes:
- `static_assert` size constraints in `constraints` field
- Python reference implementation in `reference` field
- Expected relative error threshold (typically < 1e-3)
