# softmax

Softmax operation for attention score normalization. Converts raw attention scores (logits) into probability distributions. Critical for numerical stability with large sequence lengths.

Variants:
- standard: Basic softmax along last dimension
- scaled: Softmax with scale factor (for attention)
- masked: Softmax with causal or padding mask
- online: Online/streaming softmax for memory efficiency

## standard

Standard softmax: exp(x) / sum(exp(x)) with numerical stability.

Axes (2 dimensions):
- `batch_size`: variable (batch * num_heads * seq_len for attention)
- `seq_len`: variable (sequence length being softmaxed)

Inputs (1 tensor):
- `input`: attention scores [batch_size, seq_len]

Outputs (1 tensor):
- `output`: probabilities [batch_size, seq_len]

Formula:
```
For each row x[i]:
  // Numerical stability: subtract max
  m = max(x[i])
  x_shifted = x[i] - m

  // Compute exp and sum
  exp_x = exp(x_shifted)
  sum_exp = sum(exp_x)

  // Normalize
  output[i] = exp_x / sum_exp
```

## scaled

Scaled softmax for attention: applies 1/sqrt(d_k) scale before softmax.

Axes (2 dimensions):
- `batch_size`: variable
- `seq_len`: variable

Inputs (1 tensor + 1 scalar):
- `input`: attention scores [batch_size, seq_len]
- `scale`: typically 1/sqrt(head_dim) (scalar)

Outputs (1 tensor):
- `output`: probabilities [batch_size, seq_len]

Formula:
```
output = softmax(input * scale)
```

## masked

Masked softmax for causal attention or padding.

Axes (2 dimensions):
- `batch_size`: variable
- `seq_len`: variable

Inputs (2 tensors + 1 scalar):
- `input`: attention scores [batch_size, seq_len]
- `mask`: boolean or float mask [batch_size, seq_len] or [seq_len, seq_len]
- `scale`: scale factor (scalar)

Outputs (1 tensor):
- `output`: masked probabilities [batch_size, seq_len]

Formula:
```
// Apply mask (set masked positions to -inf)
masked_input = where(mask, input * scale, -inf)

// Standard softmax
output = softmax(masked_input)
```

Causal mask pattern:
```
For query position q and key position k:
  mask[q, k] = (k <= q)  // can attend to current and past only
```

## online

Online (streaming) softmax for memory-efficient attention.

Used in FlashAttention-style implementations to compute softmax incrementally without materializing full attention matrix.

Formula:
```
// Process in blocks
for each block:
  // Update running max
  m_new = max(m_old, max(block))

  // Update running sum with correction
  sum_new = sum_old * exp(m_old - m_new) + sum(exp(block - m_new))

  // Update output with correction
  output = output * exp(m_old - m_new) + block_output
```

## CUDA Implementation

Safe softmax with warp reduction:

```c
__global__ void softmax_kernel(
    float* output,
    const float* input,
    int batch_size, int seq_len
) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    const float* row_input = input + row * seq_len;
    float* row_output = output + row * seq_len;

    // Step 1: Find max (parallel reduction)
    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        local_max = fmaxf(local_max, row_input[i]);
    }
    // Warp reduction for max
    for (int offset = 16; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
    }
    __shared__ float shared_max;
    if (threadIdx.x == 0) shared_max = local_max;
    __syncthreads();
    float row_max = shared_max;

    // Step 2: Compute exp and sum
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        float exp_val = expf(row_input[i] - row_max);
        row_output[i] = exp_val;  // Store intermediate
        local_sum += exp_val;
    }
    // Warp reduction for sum
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
    }
    __shared__ float shared_sum;
    if (threadIdx.x == 0) shared_sum = local_sum;
    __syncthreads();
    float inv_sum = 1.0f / shared_sum;

    // Step 3: Normalize
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        row_output[i] *= inv_sum;
    }
}
```

## Performance Notes

- Softmax is memory-bound for typical attention sizes
- Two passes required: max reduction, then sum reduction
- Consider fusing with attention score computation
- For long sequences, online softmax avoids O(n^2) memory
