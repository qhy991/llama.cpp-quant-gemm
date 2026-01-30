# rmsnorm

Root Mean Square Layer Normalization used in LLaMA and other modern transformers. Normalizes input by RMS value and applies learned scale parameters. Simpler and faster than LayerNorm as it skips mean subtraction.

Variants:
- standard: Basic RMSNorm with learned weights
- fused_add: Fused residual addition + RMSNorm

## standard

Standard RMSNorm: normalize by RMS and scale by learned weights.

Axes (2 dimensions):
- `batch_size`: variable (tokens in batch)
- `hidden_size`: constant (model dimension)

Inputs (2 tensors + 1 scalar):
- `input`: hidden states [batch_size, hidden_size]
- `weight`: learned scale parameters [hidden_size]
- `eps`: epsilon for numerical stability (scalar, typically 1e-6)

Outputs (1 tensor):
- `output`: normalized output [batch_size, hidden_size]

Formula:
```
For each row x[i] (i ∈ [0, batch_size)):
  // Compute RMS
  rms = sqrt(mean(x[i]^2) + eps)
      = sqrt(sum(x[i,j]^2) / hidden_size + eps)

  // Normalize and scale
  output[i,j] = (x[i,j] / rms) * weight[j]
```

CUDA Implementation Notes:
```c
// Per-row reduction for sum of squares
__shared__ float shared_sum[BLOCK_SIZE];

float local_sum = 0.0f;
for (int j = threadIdx.x; j < hidden_size; j += blockDim.x) {
    float val = input[row * hidden_size + j];
    local_sum += val * val;
}

// Warp/block reduction...
float rms = sqrtf(sum_sq / hidden_size + eps);
float inv_rms = 1.0f / rms;

// Apply normalization
for (int j = threadIdx.x; j < hidden_size; j += blockDim.x) {
    output[row * hidden_size + j] =
        input[row * hidden_size + j] * inv_rms * weight[j];
}
```

## fused_add

Fused residual addition + RMSNorm. Adds residual before normalization in single kernel.

Axes (2 dimensions):
- `batch_size`: variable
- `hidden_size`: constant

Inputs (3 tensors + 1 scalar):
- `input`: current hidden states [batch_size, hidden_size]
- `residual`: residual connection [batch_size, hidden_size]
- `weight`: learned scale parameters [hidden_size]
- `eps`: epsilon (scalar)

Outputs (2 tensors):
- `output`: normalized output [batch_size, hidden_size]
- `residual_out`: updated residual for next layer [batch_size, hidden_size]

Formula:
```
For each row i:
  // Fuse addition
  hidden[i] = input[i] + residual[i]

  // RMSNorm on fused result
  rms = sqrt(mean(hidden[i]^2) + eps)
  output[i] = (hidden[i] / rms) * weight

  // Pass through residual
  residual_out[i] = hidden[i]
```

Benefit:
- Single pass over data instead of two
- Reduced memory bandwidth
- Common pattern in transformer inference

## Performance Considerations

1. **Memory bound**: RMSNorm is typically memory-bound, not compute-bound
2. **Vectorized loads**: Use float4 for coalesced memory access
3. **Warp reduction**: Use `__shfl_xor_sync` for fast reduction
4. **Fused operations**: Combine with previous/next ops when possible

Typical hidden_size values:
- LLaMA-7B: 4096
- LLaMA-13B: 5120
- LLaMA-70B: 8192
