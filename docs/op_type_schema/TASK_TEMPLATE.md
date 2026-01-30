# LLM Task Template for llama.cpp Kernel Implementation

This document defines a standardized task description format for instructing LLMs to implement llama.cpp compatible CUDA kernels.

## Task Description Format

```yaml
# ============================================================
# KERNEL IMPLEMENTATION TASK
# ============================================================

task:
  name: "{{TASK_NAME}}"
  operation: "{{OP_TYPE}}"           # quant_gemm | quant_vec_dot | quantize | dequantize | activation | rmsnorm | rope | softmax
  variant: "{{VARIANT}}"             # e.g., W4A8, q4_0_q8_1, silu

# ------------------------------------------------------------
# TARGET CONFIGURATION
# ------------------------------------------------------------

target:
  optimization: "{{OPT_LEVEL}}"      # naive | tiled | dp4a | tensorcore
  architecture: "{{SM_VERSION}}"     # sm_61 | sm_70 | sm_80 | sm_89 | sm_100
  compatibility: "llama.cpp"         # Must be compatible with llama.cpp types

# ------------------------------------------------------------
# OPERATION SPECIFICATION (from op_type_schema)
# ------------------------------------------------------------

spec:
  # Copy relevant section from op_type_schema/*.md
  axes:
    - name: "M"
      type: "variable"
    - name: "N"
      type: "constant"
    - name: "K"
      type: "constant"

  inputs:
    - name: "activation"
      dtype: "{{ACT_TYPE}}"          # block_q8_1 | float | half
      shape: "[M, K/32]"
    - name: "weight"
      dtype: "{{WEIGHT_TYPE}}"       # block_q4_0 | block_q4_1 | ...
      shape: "[N, K/32]"

  outputs:
    - name: "output"
      dtype: "float"
      shape: "[M, N]"

  constraints:
    - "K % 32 == 0"

  formula: |
    # Core computation formula
    result = d_w * (d_a * sumi - 8 * s_a)

# ------------------------------------------------------------
# TYPE DEFINITIONS (must match llama.cpp exactly)
# ------------------------------------------------------------

types:
  block_q4_0:
    size: 18
    layout: |
      half d;           // 2 bytes - scale factor
      uint8_t qs[16];   // 16 bytes - 32 x 4-bit packed

  block_q8_1:
    size: 36
    layout: |
      half2 ds;         // 4 bytes - d (scale) + s (sum)
      int8_t qs[32];    // 32 bytes - 32 x 8-bit signed

# ------------------------------------------------------------
# EXPECTED OUTPUT FILES
# ------------------------------------------------------------

outputs:
  files:
    - path: "kernels/gemm/gemm_{{VARIANT}}_{{OPT_LEVEL}}.cuh"
      content: "CUDA kernel implementation"

    - path: "include/gemm_{{VARIANT}}_api.h"
      content: "C++ API wrapper"

    - path: "tests/test_gemm_{{VARIANT}}.cu"
      content: "Correctness tests"

# ------------------------------------------------------------
# ACCEPTANCE CRITERIA
# ------------------------------------------------------------

acceptance:
  correctness:
    - "Output matches CPU reference within relative error < 1e-3"
    - "Handles edge cases: M=1, large M, K at minimum (32)"

  compatibility:
    - "block_q4_0 size == 18 bytes (static_assert)"
    - "block_q8_1 size == 36 bytes (static_assert)"
    - "Uses exact formula from spec"

  code_quality:
    - "No compiler warnings"
    - "Proper CUDA error checking"
    - "Clear comments explaining key steps"
```

---

## Example Task Instances

### Example 1: W4A8 Naive GEMM

```yaml
task:
  name: "W4A8 Naive GEMM Implementation"
  operation: "quant_gemm"
  variant: "W4A8"

target:
  optimization: "naive"
  architecture: "sm_61"
  compatibility: "llama.cpp"

spec:
  axes:
    - name: "M"
      type: "variable"
      description: "batch_size * seq_len"
    - name: "N"
      type: "constant"
      description: "output features"
    - name: "K"
      type: "constant"
      description: "input features"

  inputs:
    - name: "activation"
      dtype: "block_q8_1"
      shape: "[M, K/32]"
      description: "Q8_1 quantized activations"
    - name: "weight"
      dtype: "block_q4_0"
      shape: "[N, K/32]"
      description: "Q4_0 quantized weights"

  outputs:
    - name: "output"
      dtype: "float"
      shape: "[M, N]"

  constraints:
    - "K % 32 == 0"
    - "K >= 32"

  formula: |
    For each output element C[m, n]:
      sum = 0
      for b in range(K / 32):
        weight_block = weight[n, b]
        activation_block = activation[m, b]

        d_w = weight_block.d
        d_a = __low2half(activation_block.ds)
        s_a = __high2half(activation_block.ds)

        sumi = 0
        for i in range(16):
          q_w_low  = weight_block.qs[i] & 0x0F
          q_w_high = weight_block.qs[i] >> 4
          q_a_low  = activation_block.qs[i]
          q_a_high = activation_block.qs[i + 16]
          sumi += q_w_low * q_a_low + q_w_high * q_a_high

        sum += d_w * (d_a * sumi - 8.0f * s_a)

      C[m, n] = sum

types:
  block_q4_0:
    size: 18
    layout: |
      typedef struct {
          half d;              // scale factor
          uint8_t qs[16];      // 32 x 4-bit values packed
      } block_q4_0;
      static_assert(sizeof(block_q4_0) == 18);

  block_q8_1:
    size: 36
    layout: |
      typedef struct {
          half2 ds;            // d (scale) and s (sum)
          int8_t qs[32];       // 32 x 8-bit signed
      } block_q8_1;
      static_assert(sizeof(block_q8_1) == 36);

outputs:
  files:
    - path: "kernels/gemm/gemm_w4a8_naive.cuh"
    - path: "include/gemm_w4a8_api.h"
    - path: "tests/test_gemm_w4a8.cu"

acceptance:
  correctness:
    - "Relative error < 1e-3 vs CPU reference"
    - "Test with M ∈ {1, 64, 256, 1024}"
    - "Test with N ∈ {128, 512, 4096}"
    - "Test with K ∈ {32, 256, 4096}"
```

### Example 2: W4A8 DP4A Optimized GEMM

```yaml
task:
  name: "W4A8 DP4A Optimized GEMM"
  operation: "quant_gemm"
  variant: "W4A8"

target:
  optimization: "dp4a"
  architecture: "sm_61"
  compatibility: "llama.cpp"

spec:
  # Same as naive...

  optimization_hints:
    - "Use __dp4a() intrinsic for 4-way int8 dot product"
    - "Pack 4 consecutive q4 values into int32 for DP4A input"
    - "Pack 4 consecutive q8 values into int32 for DP4A input"
    - "Each DP4A computes 4 multiply-adds in single instruction"

  dp4a_usage: |
    // DP4A: c += dot(a, b) where a, b are int32 holding 4 x int8
    int dp4a(int a, int b, int c) {
        return __dp4a(a, b, c);
    }

    // Example packing for q4:
    int pack_q4(uint8_t* qs, int offset) {
        // Pack 4 consecutive 4-bit values into int32
        int result = 0;
        result |= ((qs[offset/2] >> (4*(offset%2))) & 0x0F) << 0;
        result |= ((qs[(offset+1)/2] >> (4*((offset+1)%2))) & 0x0F) << 8;
        result |= ((qs[(offset+2)/2] >> (4*((offset+2)%2))) & 0x0F) << 16;
        result |= ((qs[(offset+3)/2] >> (4*((offset+3)%2))) & 0x0F) << 24;
        return result;
    }

acceptance:
  correctness:
    - "Same as naive"
  performance:
    - "At least 2x faster than naive on target architecture"
```

### Example 3: Q8_1 Quantization Kernel

```yaml
task:
  name: "Q8_1 Activation Quantization"
  operation: "quantize"
  variant: "q8_1"

target:
  optimization: "naive"
  architecture: "sm_61"

spec:
  axes:
    - name: "M"
      type: "variable"
      description: "number of rows"
    - name: "K"
      type: "variable"
      description: "number of columns (must be multiple of 32)"

  inputs:
    - name: "input"
      dtype: "float"
      shape: "[M, K]"

  outputs:
    - name: "output"
      dtype: "block_q8_1"
      shape: "[M, K/32]"

  constraints:
    - "K % 32 == 0"

  formula: |
    For each block of 32 elements:
      // Find maximum absolute value
      amax = max(|input[i]|) for i in [0, 31]

      // Compute scale
      d = amax / 127.0f

      // Compute sum of original values (CRITICAL!)
      s = sum(input[i]) for i in [0, 31]

      // Quantize
      for i in [0, 31]:
        qs[i] = round(input[i] / d)
        qs[i] = clamp(qs[i], -128, 127)

      // Pack output
      block.ds = make_half2((half)d, (half)s)
      block.qs = qs

types:
  block_q8_1:
    size: 36
    layout: |
      typedef struct {
          half2 ds;        // ds.x = d (scale), ds.y = s (sum)
          int8_t qs[32];   // quantized values
      } block_q8_1;

outputs:
  files:
    - path: "kernels/quantization/quantize_q8_1.cuh"
    - path: "tests/test_quantize_q8_1.cu"

acceptance:
  correctness:
    - "Dequantized values within 1% of original"
    - "Sum (s) matches sum of original values within FP16 precision"
```

---

## How to Use This Template

1. **Select operation type** from `op_type_schema/` directory
2. **Fill in target configuration** (optimization level, architecture)
3. **Copy relevant spec** from the schema document
4. **Define acceptance criteria** specific to your needs
5. **Submit to LLM** with the filled template

The LLM should produce:
- Correct CUDA kernel implementation
- Matching API header
- Test file with specified test cases
- All type definitions compatible with llama.cpp
