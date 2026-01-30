# quant_vec_dot

Block-wise vector dot product for quantized formats. This is the fundamental building block for quantized GEMM operations. Each variant computes the dot product between one quantized weight block and one quantized/float activation block.

Variants:
- q4_0_q8_1: Q4_0 weight × Q8_1 activation (most common in llama.cpp)
- q4_1_q8_1: Q4_1 weight × Q8_1 activation
- q5_0_q8_1: Q5_0 weight × Q8_1 activation
- q5_1_q8_1: Q5_1 weight × Q8_1 activation
- q8_0_q8_1: Q8_0 weight × Q8_1 activation
- q4_0_f32: Q4_0 weight × FP32 activation

## q4_0_q8_1

Axes (1 dimension):
- `n`: constant (always 32, elements per block)

Inputs (2 blocks):
- `weight`: block_q4_0 containing 32 quantized 4-bit values
- `activation`: block_q8_1 containing 32 quantized 8-bit values with sum

Outputs (1 scalar):
- `result`: FP32 dot product result

Block Structures:

block_q4_0:
```c
typedef struct {
    half d;              // scale factor (2 bytes)
    uint8_t qs[16];      // 32 x 4-bit values packed (16 bytes)
} block_q4_0;            // Total: 18 bytes

// Unpacking: qs[i] contains q[i] (low nibble) and q[i+16] (high nibble)
// q_low  = qs[i] & 0x0F
// q_high = qs[i] >> 4
```

block_q8_1:
```c
typedef struct {
    half2 ds;            // d (scale) and s (sum) packed (4 bytes)
    int8_t qs[32];       // 32 x 8-bit signed values (32 bytes)
} block_q8_1;            // Total: 36 bytes

// Accessing:
// d = __low2half(ds)   or ds.x
// s = __high2half(ds)  or ds.y
```

Formula:
```
d_w = weight.d
d_a = __low2half(activation.ds)
s_a = __high2half(activation.ds)

sumi = 0
for i in [0, 15]:
    q_w_low  = (weight.qs[i] & 0x0F)      // values 0-15
    q_w_high = (weight.qs[i] >> 4)         // values 0-15
    q_a_low  = activation.qs[i]            // values -128 to 127
    q_a_high = activation.qs[i + 16]       // values -128 to 127

    sumi += q_w_low * q_a_low + q_w_high * q_a_high

result = d_w * (d_a * sumi - 8.0f * s_a)
```

Explanation of compensation term `-8.0f * s_a`:
```
Q4_0 stores values with +8 offset: q_stored = q_real + 8
To get correct dot product: Σ q_real * activation
  = Σ (q_stored - 8) * activation
  = Σ q_stored * activation - 8 * Σ activation
  = sumi - 8 * (s_a / d_a)  // where s_a = Σ original_activation

After scaling:
  result = d_w * d_a * (sumi - 8 * s_a / d_a)
         = d_w * (d_a * sumi - 8 * s_a)
```

## q4_1_q8_1

Axes (1 dimension):
- `n`: constant (32)

Inputs (2 blocks):
- `weight`: block_q4_1
- `activation`: block_q8_1

Outputs (1 scalar):
- `result`: FP32

Block Structure (block_q4_1):
```c
typedef struct {
    half d;              // scale factor
    half m;              // minimum value
    uint8_t qs[16];      // 32 x 4-bit values
} block_q4_1;            // Total: 20 bytes
```

Formula:
```
d_w = weight.d
m_w = weight.m
d_a = __low2half(activation.ds)
s_a = __high2half(activation.ds)

sumi = Σ (q_w[i] * q_a[i])

result = d_w * d_a * sumi + m_w * s_a
```

## q5_0_q8_1

Axes (1 dimension):
- `n`: constant (32)

Inputs (2 blocks):
- `weight`: block_q5_0
- `activation`: block_q8_1

Outputs (1 scalar):
- `result`: FP32

Block Structure (block_q5_0):
```c
typedef struct {
    half d;              // scale factor
    uint8_t qh[4];       // high bits (5th bit for each of 32 values)
    uint8_t qs[16];      // low 4 bits
} block_q5_0;            // Total: 22 bytes

// Reconstruction:
// q5[i] = (qs[i/2] >> (4*(i%2))) & 0x0F | ((qh[i/8] >> (i%8)) & 1) << 4
```

Formula:
```
d_w = weight.d
d_a = __low2half(activation.ds)
s_a = __high2half(activation.ds)

sumi = Σ (q5_w[i] * q_a[i])  // q5 is reconstructed 5-bit value

result = d_w * (d_a * sumi - 16.0f * s_a)  // offset is 16 for 5-bit
```

## q8_0_q8_1

Axes (1 dimension):
- `n`: constant (32)

Inputs (2 blocks):
- `weight`: block_q8_0
- `activation`: block_q8_1

Outputs (1 scalar):
- `result`: FP32

Block Structure (block_q8_0):
```c
typedef struct {
    half d;              // scale factor
    int8_t qs[32];       // 32 x 8-bit signed values
} block_q8_0;            // Total: 34 bytes
```

Formula:
```
d_w = weight.d
d_a = __low2half(activation.ds)

sumi = Σ (q_w[i] * q_a[i])  // both signed, symmetric

result = d_w * d_a * sumi   // no compensation needed
```

## q4_0_f32

Axes (1 dimension):
- `n`: constant (32)

Inputs (1 block + 32 floats):
- `weight`: block_q4_0
- `activation`: float[32]

Outputs (1 scalar):
- `result`: FP32

Formula:
```
d_w = weight.d
sum = 0.0f

for i in [0, 15]:
    q_w_low  = (weight.qs[i] & 0x0F) - 8  // dequantize inline
    q_w_high = (weight.qs[i] >> 4) - 8

    sum += q_w_low * d_w * activation[i]
    sum += q_w_high * d_w * activation[i + 16]

result = sum
```

## DP4A Optimization

For SM61+ GPUs, integer dot products can use DP4A instruction:

```c
// DP4A: 4-way 8-bit integer dot product
int dp4a(int a, int b, int c) {
    // c += a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
    return __dp4a(a, b, c);
}

// Usage in q4_0_q8_1:
// Pack 4 consecutive q4 values into int32
// Pack 4 consecutive q8 values into int32
// Use dp4a for 4 multiplies in single instruction
```
