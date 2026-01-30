# activation

Element-wise activation functions used in transformer feed-forward networks. These non-linear operations are applied after linear projections.

Variants:
- silu: Sigmoid Linear Unit (SiLU/Swish), used in LLaMA, Mistral
- gelu: Gaussian Error Linear Unit, used in GPT, BERT
- gelu_quick: Fast approximation of GELU
- relu: Rectified Linear Unit

## silu

SiLU (Swish) activation: x * sigmoid(x). Default activation in LLaMA models.

Axes (2 dimensions):
- `batch_size`: variable
- `hidden_size`: constant

Inputs (1 tensor):
- `input`: [batch_size, hidden_size]

Outputs (1 tensor):
- `output`: [batch_size, hidden_size]

Formula:
```
output[i] = input[i] * sigmoid(input[i])
          = input[i] / (1 + exp(-input[i]))
```

CUDA Implementation:
```c
__device__ float silu(float x) {
    return x / (1.0f + expf(-x));
}
```

## gelu

Gaussian Error Linear Unit with exact formula.

Axes (2 dimensions):
- `batch_size`: variable
- `hidden_size`: constant

Inputs (1 tensor):
- `input`: [batch_size, hidden_size]

Outputs (1 tensor):
- `output`: [batch_size, hidden_size]

Formula:
```
output[i] = 0.5 * x * (1 + erf(x / sqrt(2)))
```

CUDA Implementation:
```c
__device__ float gelu(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
}
```

## gelu_quick

Fast GELU approximation using tanh.

Formula:
```
output[i] = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

CUDA Implementation:
```c
__device__ float gelu_quick(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}
```

## relu

Rectified Linear Unit.

Formula:
```
output[i] = max(0, input[i])
```

## Fused SiLU-Mul (silu_mul)

Fused operation for LLaMA FFN: SiLU(gate) * up

Axes (2 dimensions):
- `batch_size`: variable
- `hidden_size`: constant

Inputs (2 tensors):
- `gate`: [batch_size, hidden_size] - gate projection output
- `up`: [batch_size, hidden_size] - up projection output

Outputs (1 tensor):
- `output`: [batch_size, hidden_size]

Formula:
```
output[i] = silu(gate[i]) * up[i]
```

Usage in LLaMA FFN:
```
hidden = silu(gate_proj(x)) * up_proj(x)
output = down_proj(hidden)
```
