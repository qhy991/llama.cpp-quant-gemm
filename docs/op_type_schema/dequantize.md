# dequantize

Dequantization operations that convert quantized block formats back to floating-point. Used for verification, debugging, and operations that require FP precision.

Variants:
- q4_0: Q4_0 → FP32
- q4_1: Q4_1 → FP32
- q5_0: Q5_0 → FP32
- q5_1: Q5_1 → FP32
- q8_0: Q8_0 → FP32
- q8_1: Q8_1 → FP32

## q4_0

Dequantize Q4_0 (4-bit symmetric with offset) to FP32.

Axes (1 dimension):
- `K`: variable (output elements, must be multiple of 32)

Inputs (1 tensor):
- `input`: Q4_0 quantized blocks [K/32] as block_q4_0

Outputs (1 tensor):
- `output`: FP32 tensor [K]

Algorithm:
```
for each block:
    d = (float)block.d

    for i in [0, 15]:
        q_low  = (block.qs[i] & 0x0F) - 8   // subtract offset
        q_high = (block.qs[i] >> 4) - 8

        output[i]      = q_low * d
        output[i + 16] = q_high * d
```

Constraints:
- `K % 32 == 0`
- Input size: `K / 32 * 18` bytes

## q4_1

Dequantize Q4_1 (4-bit asymmetric) to FP32.

Axes (1 dimension):
- `K`: variable

Inputs (1 tensor):
- `input`: Q4_1 quantized blocks [K/32] as block_q4_1

Outputs (1 tensor):
- `output`: FP32 tensor [K]

Algorithm:
```
for each block:
    d = (float)block.d
    m = (float)block.m    // minimum value offset

    for i in [0, 15]:
        q_low  = (block.qs[i] & 0x0F)
        q_high = (block.qs[i] >> 4)

        output[i]      = q_low * d + m
        output[i + 16] = q_high * d + m
```

## q5_0

Dequantize Q5_0 (5-bit symmetric) to FP32.

Axes (1 dimension):
- `K`: variable

Inputs (1 tensor):
- `input`: Q5_0 quantized blocks [K/32] as block_q5_0

Outputs (1 tensor):
- `output`: FP32 tensor [K]

Algorithm:
```
for each block:
    d = (float)block.d

    for i in [0, 31]:
        // Extract low 4 bits
        q_low4 = (block.qs[i/2] >> (4 * (i % 2))) & 0x0F

        // Extract 5th bit from qh
        q_high1 = (block.qh[i/8] >> (i % 8)) & 1

        // Combine to 5-bit value and apply offset
        q5 = (q_low4 | (q_high1 << 4)) - 16

        output[i] = q5 * d
```

## q8_0

Dequantize Q8_0 (8-bit symmetric) to FP32.

Axes (1 dimension):
- `K`: variable

Inputs (1 tensor):
- `input`: Q8_0 quantized blocks [K/32] as block_q8_0

Outputs (1 tensor):
- `output`: FP32 tensor [K]

Algorithm:
```
for each block:
    d = (float)block.d

    for i in [0, 31]:
        output[i] = block.qs[i] * d   // qs is already signed
```

## q8_1

Dequantize Q8_1 (8-bit with sum) to FP32.

Axes (1 dimension):
- `K`: variable

Inputs (1 tensor):
- `input`: Q8_1 quantized blocks [K/32] as block_q8_1

Outputs (1 tensor):
- `output`: FP32 tensor [K]

Algorithm:
```
for each block:
    d = (float)__low2half(block.ds)   // scale factor
    // Note: block.ds.y (sum) is not used for dequantization

    for i in [0, 31]:
        output[i] = block.qs[i] * d
```

## Verification Usage

Dequantization is primarily used for:

1. **Accuracy verification**:
```python
# Verify quantization roundtrip
original = torch.randn(1024)
quantized = quantize_q4_0(original)
recovered = dequantize_q4_0(quantized)
error = torch.abs(original - recovered).max()
# Expected: error < original.abs().max() / 7 (Q4_0 has 15 levels)
```

2. **Reference GEMM implementation**:
```python
# W4A16 reference: dequantize weights, then FP32 GEMM
weights_fp32 = dequantize_q4_0(weights_q4)
output = activation @ weights_fp32.T
```

3. **Debugging quantized kernels**:
```python
# Compare quantized GEMM with reference
output_quant = gemm_w4a8(activation_q8, weight_q4)
output_ref = dequantize_q8_1(activation_q8) @ dequantize_q4_0(weight_q4).T
assert torch.allclose(output_quant, output_ref, rtol=1e-3)
```
