# quant_gemm

Quantized General Matrix Multiplication for llama.cpp inference. Computes C = A × B^T where weights (B) are stored in low-bit quantized formats to reduce memory bandwidth, and activations (A) can be either floating-point or quantized. This is the core operation for efficient LLM inference with reduced memory footprint.

Variants:
- W4A16: 4-bit weights (Q4_0/Q4_1) with FP32/FP16 activations
- W4A8: 4-bit weights (Q4_0/Q4_1) with 8-bit activations (Q8_1)
- W5A8: 5-bit weights (Q5_0/Q5_1) with 8-bit activations (Q8_1)
- W8A8: 8-bit weights (Q8_0) with 8-bit activations (Q8_1)

## W4A16

4-bit quantized weights with floating-point activations. Simpler implementation without activation quantization overhead.

Axes (3 dimensions):
- `M`: variable (batch size × sequence length)
- `N`, `K`: constant (model dimensions)

Inputs (2 tensors):
- `activation`: FP32 activation matrix [M, K]
- `weight`: Q4_0 quantized weight [N, K/32] as block_q4_0

Outputs (1 tensor):
- `output`: FP32 result matrix [M, N]

Constraints:
- `K % 32 == 0` (K must be multiple of block size)

Block Structure (block_q4_0):
```
+----------------+--------------------------------+
| d (2 bytes)    |        qs[16] (16 bytes)       |
| half scale     | 32 x 4-bit values packed       |
+----------------+--------------------------------+
Total: 18 bytes per 32 elements
```

Dequantization Formula:
```
x[i] = (qs[i] - 8) * d
where qs[i] ∈ [0, 15], offset by 8 for signed range [-8, 7]
```

## W4A8

4-bit weights with 8-bit quantized activations. Uses integer dot product with compensation for maximum throughput.

Axes (3 dimensions):
- `M`: variable
- `N`, `K`: constant

Inputs (2 tensors):
- `activation`: Q8_1 quantized activation [M, K/32] as block_q8_1
- `weight`: Q4_0 quantized weight [N, K/32] as block_q4_0

Outputs (1 tensor):
- `output`: FP32 result matrix [M, N]

Constraints:
- `K % 32 == 0`

Block Structures:

block_q4_0 (weight):
```
+----------------+--------------------------------+
| d (2 bytes)    |        qs[16] (16 bytes)       |
| half scale     | 32 x 4-bit values packed       |
+----------------+--------------------------------+
Total: 18 bytes per 32 elements
```

block_q8_1 (activation):
```
+------------------+--------------------------------+
| ds (4 bytes)     |        qs[32] (32 bytes)       |
| half2 (d + s)    | 32 x 8-bit signed values       |
+------------------+--------------------------------+
Total: 36 bytes per 32 elements

ds.x = d (scale factor)
ds.y = s (sum of original values before quantization)
```

Dot Product Formula:
```
For each block pair (weight_block, activation_block):
  d_w = weight_block.d
  d_a = activation_block.ds.x
  s_a = activation_block.ds.y

  sumi = Σ (q_w[i] * q_a[i])  // integer dot product

  result = d_w * (d_a * sumi - 8 * s_a)  // compensation for Q4_0 offset
```

Why compensation is needed:
```
Q4_0 stores: q_stored = q_real + 8
Dot product: Σ (q_stored - 8) * d_w * q_a * d_a
           = d_w * d_a * Σ q_stored * q_a - 8 * d_w * d_a * Σ q_a
           = d_w * (d_a * sumi - 8 * s_a)
where s_a ≈ d_a * Σ q_a (precomputed in block_q8_1)
```

## W4_1A8

4-bit asymmetric weights with 8-bit activations. Uses minimum value offset instead of fixed offset.

Axes (3 dimensions):
- `M`: variable
- `N`, `K`: constant

Inputs (2 tensors):
- `activation`: Q8_1 quantized activation [M, K/32] as block_q8_1
- `weight`: Q4_1 quantized weight [N, K/32] as block_q4_1

Outputs (1 tensor):
- `output`: FP32 result matrix [M, N]

Constraints:
- `K % 32 == 0`

Block Structure (block_q4_1):
```
+----------------+----------------+------------------------+
| d (2 bytes)    | m (2 bytes)    |    qs[16] (16 bytes)   |
| half scale     | half minimum   | 32 x 4-bit values      |
+----------------+----------------+------------------------+
Total: 20 bytes per 32 elements
```

Dot Product Formula:
```
For each block pair:
  d_w = weight_block.d
  m_w = weight_block.m  // minimum value
  d_a = activation_block.ds.x
  s_a = activation_block.ds.y

  sumi = Σ (q_w[i] * q_a[i])

  result = d_w * d_a * sumi + m_w * s_a
```

## W5A8

5-bit weights with 8-bit activations. Higher precision weights with extra high-bit storage.

Axes (3 dimensions):
- `M`: variable
- `N`, `K`: constant

Inputs (2 tensors):
- `activation`: Q8_1 quantized activation [M, K/32] as block_q8_1
- `weight`: Q5_0 quantized weight [N, K/32] as block_q5_0

Outputs (1 tensor):
- `output`: FP32 result matrix [M, N]

Constraints:
- `K % 32 == 0`

Block Structure (block_q5_0):
```
+----------------+----------------+------------------------+
| d (2 bytes)    | qh[4] (4 bytes)|    qs[16] (16 bytes)   |
| half scale     | 5th bits       | low 4-bit values       |
+----------------+----------------+------------------------+
Total: 22 bytes per 32 elements
```

Dot Product Formula:
```
For each block pair:
  d_w = weight_block.d
  d_a = activation_block.ds.x
  s_a = activation_block.ds.y

  // Reconstruct 5-bit values: q5[i] = qs[i] | (qh_bit[i] << 4)
  sumi = Σ (q5_w[i] * q_a[i])

  result = d_w * (d_a * sumi - 16 * s_a)  // offset is 16 for 5-bit
```

## W8A8

8-bit weights with 8-bit activations. Highest precision quantized variant.

Axes (3 dimensions):
- `M`: variable
- `N`, `K`: constant

Inputs (2 tensors):
- `activation`: Q8_1 quantized activation [M, K/32] as block_q8_1
- `weight`: Q8_0 quantized weight [N, K/32] as block_q8_0

Outputs (1 tensor):
- `output`: FP32 result matrix [M, N]

Constraints:
- `K % 32 == 0`

Block Structure (block_q8_0):
```
+----------------+--------------------------------+
| d (2 bytes)    |        qs[32] (32 bytes)       |
| half scale     | 32 x 8-bit signed values       |
+----------------+--------------------------------+
Total: 34 bytes per 32 elements
```

Dot Product Formula:
```
For each block pair:
  d_w = weight_block.d
  d_a = activation_block.ds.x

  sumi = Σ (q_w[i] * q_a[i])  // both symmetric, no offset

  result = d_w * d_a * sumi  // no compensation needed
```
