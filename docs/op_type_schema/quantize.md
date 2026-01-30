# quantize

Quantization operations that convert floating-point tensors to low-bit quantized block formats. These are used to quantize activations at runtime (Q8_1) or weights offline (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0).

Variants:
- q4_0: FP32 → Q4_0 (4-bit symmetric, for weights)
- q4_1: FP32 → Q4_1 (4-bit asymmetric, for weights)
- q5_0: FP32 → Q5_0 (5-bit symmetric, for weights)
- q5_1: FP32 → Q5_1 (5-bit asymmetric, for weights)
- q8_0: FP32 → Q8_0 (8-bit symmetric, for weights)
- q8_1: FP32 → Q8_1 (8-bit with sum, for activations)

## q4_0

4-bit symmetric quantization with fixed offset.

Axes (1 dimension):
- `K`: variable (must be multiple of 32)

Inputs (1 tensor):
- `input`: FP32 source tensor [K]

Outputs (1 tensor):
- `output`: Q4_0 quantized blocks [K/32] as block_q4_0

Algorithm:
```
for each block of 32 elements:
    // Find scale
    amax = max(|input[i]|) for i in block
    d = amax / 7.0f

    // Quantize with offset +8
    for i in [0, 31]:
        q[i] = round(input[i] / d) + 8
        q[i] = clamp(q[i], 0, 15)

    // Pack into qs[16]
    for i in [0, 15]:
        qs[i] = q[i] | (q[i + 16] << 4)

    // Store
    block.d = (half)d
    block.qs = qs
```

Constraints:
- `K % 32 == 0`
- Output size: `K / 32 * 18` bytes

## q4_1

4-bit asymmetric quantization with minimum value.

Axes (1 dimension):
- `K`: variable (must be multiple of 32)

Inputs (1 tensor):
- `input`: FP32 source tensor [K]

Outputs (1 tensor):
- `output`: Q4_1 quantized blocks [K/32] as block_q4_1

Algorithm:
```
for each block of 32 elements:
    // Find range
    vmin = min(input[i]) for i in block
    vmax = max(input[i]) for i in block

    d = (vmax - vmin) / 15.0f
    m = vmin

    // Quantize
    for i in [0, 31]:
        q[i] = round((input[i] - m) / d)
        q[i] = clamp(q[i], 0, 15)

    // Pack and store
    block.d = (half)d
    block.m = (half)m
    block.qs = pack(q)
```

Constraints:
- `K % 32 == 0`
- Output size: `K / 32 * 20` bytes

## q8_1

8-bit symmetric quantization with precomputed sum. This is the key format for runtime activation quantization.

Axes (1 dimension):
- `K`: variable (must be multiple of 32)

Inputs (1 tensor):
- `input`: FP32 source tensor [K]

Outputs (1 tensor):
- `output`: Q8_1 quantized blocks [K/32] as block_q8_1

Algorithm:
```
for each block of 32 elements:
    // Find scale
    amax = max(|input[i]|) for i in block
    d = amax / 127.0f

    // Compute sum of original values (CRITICAL for dot product compensation)
    s = sum(input[i]) for i in block

    // Quantize
    for i in [0, 31]:
        qs[i] = round(input[i] / d)
        qs[i] = clamp(qs[i], -128, 127)

    // Store with packed d and s
    block.ds = make_half2((half)d, (half)s)
    block.qs = qs
```

Constraints:
- `K % 32 == 0`
- Output size: `K / 32 * 36` bytes

Why sum is stored:
```
In Q4_0 × Q8_1 dot product:
  result = d_w * (d_a * sumi - 8 * s_a)

The s_a term compensates for Q4_0's +8 offset.
Storing s = Σ original_values allows exact compensation.
```

## q8_0

8-bit symmetric quantization for weights (no sum needed).

Axes (1 dimension):
- `K`: variable (must be multiple of 32)

Inputs (1 tensor):
- `input`: FP32 source tensor [K]

Outputs (1 tensor):
- `output`: Q8_0 quantized blocks [K/32] as block_q8_0

Algorithm:
```
for each block of 32 elements:
    // Find scale
    amax = max(|input[i]|) for i in block
    d = amax / 127.0f

    // Quantize
    for i in [0, 31]:
        qs[i] = round(input[i] / d)
        qs[i] = clamp(qs[i], -128, 127)

    // Store
    block.d = (half)d
    block.qs = qs
```

Constraints:
- `K % 32 == 0`
- Output size: `K / 32 * 34` bytes

## CUDA Implementation Notes

Row-wise quantization for matrices:
```c
// Quantize entire activation matrix
// Input:  [M, K] FP32
// Output: [M, K/32] block_q8_1

__global__ void quantize_q8_1_kernel(
    const float* input,      // [M, K]
    block_q8_1* output,      // [M, K/32]
    int M, int K
) {
    int row = blockIdx.y;
    int block_idx = blockIdx.x;

    if (row >= M || block_idx >= K/32) return;

    // Each thread block processes one quantization block
    const float* src = input + row * K + block_idx * 32;
    block_q8_1* dst = output + row * (K/32) + block_idx;

    // Collaborative reduction for amax and sum
    // ... (shared memory reduction)
}
```

Performance considerations:
- Use warp-level primitives for reduction (amax, sum)
- Ensure coalesced memory access patterns
- For activations: fuse with previous operation when possible
