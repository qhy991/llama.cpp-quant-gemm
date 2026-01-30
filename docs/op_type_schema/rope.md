# rope

Rotary Position Embedding (RoPE) applies rotation-based position encoding to query and key tensors in attention. Used in LLaMA, Mistral, and many modern LLMs for relative position awareness.

Variants:
- standard: Basic RoPE with precomputed cos/sin tables
- neox: GPT-NeoX style (interleaved rotation)
- glm: GLM style rotation

## standard

Standard RoPE: rotates pairs of adjacent dimensions using position-dependent angles.

Axes (4 dimensions):
- `batch_size`: variable
- `seq_len`: variable
- `num_heads`: constant
- `head_dim`: constant

Inputs (3 tensors + 1 scalar):
- `input`: query or key tensor [batch_size, seq_len, num_heads, head_dim]
- `cos`: precomputed cosine values [seq_len, head_dim/2]
- `sin`: precomputed sine values [seq_len, head_dim/2]
- `position_offset`: starting position for the sequence (scalar, for KV cache)

Outputs (1 tensor):
- `output`: rotated tensor [batch_size, seq_len, num_heads, head_dim]

Formula:
```
For position p, head_dim dimension d (d even):
  // Get rotation angle components
  cos_θ = cos[p, d/2]
  sin_θ = sin[p, d/2]

  // Rotate adjacent pairs
  x0 = input[..., d]
  x1 = input[..., d+1]

  output[..., d]   = x0 * cos_θ - x1 * sin_θ
  output[..., d+1] = x0 * sin_θ + x1 * cos_θ
```

Frequency computation (for precomputing cos/sin):
```python
def precompute_freqs(head_dim, max_seq_len, base=10000.0):
    # Inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2) / head_dim))

    # Position indices
    t = torch.arange(max_seq_len)

    # Outer product: [seq_len, head_dim/2]
    freqs = torch.outer(t, inv_freq)

    return torch.cos(freqs), torch.sin(freqs)
```

## neox

GPT-NeoX style RoPE: rotates first half with second half (non-adjacent).

Formula:
```
// Split into halves
x_first  = input[..., :head_dim/2]
x_second = input[..., head_dim/2:]

// Rotate halves
output[..., :head_dim/2]  = x_first * cos - x_second * sin
output[..., head_dim/2:]  = x_first * sin + x_second * cos
```

## CUDA Implementation Notes

```c
__global__ void rope_kernel(
    float* output,
    const float* input,
    const float* cos_table,
    const float* sin_table,
    int batch_size, int seq_len, int num_heads, int head_dim,
    int position_offset
) {
    // Each thread handles one (batch, seq, head, dim_pair)
    int b = blockIdx.z;
    int s = blockIdx.y;
    int h = blockIdx.x;
    int d = threadIdx.x;  // handles pair (2*d, 2*d+1)

    if (d >= head_dim / 2) return;

    int pos = s + position_offset;

    float cos_val = cos_table[pos * (head_dim/2) + d];
    float sin_val = sin_table[pos * (head_dim/2) + d];

    int idx0 = ((b * seq_len + s) * num_heads + h) * head_dim + 2*d;
    int idx1 = idx0 + 1;

    float x0 = input[idx0];
    float x1 = input[idx1];

    output[idx0] = x0 * cos_val - x1 * sin_val;
    output[idx1] = x0 * sin_val + x1 * cos_val;
}
```

## Fused RoPE

In practice, RoPE is often fused with:
- QKV projection output splitting
- Attention score computation

Typical head_dim values:
- LLaMA: 128
- Mistral: 128
- GPT-NeoX: 64-128
