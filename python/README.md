# quant_gemm: Python Bindings for Quantized GEMM

This package provides Python bindings for the quantized GEMM CUDA kernels.

## Installation

### Prerequisites

- Python >= 3.8
- PyTorch >= 2.0.0 with CUDA support
- CUDA Toolkit (matching your PyTorch version)

### Install from source

```bash
cd python
pip install -e .
```

If you encounter compilation issues, you can specify the CUDA architecture:

```bash
TORCH_CUDA_ARCH_LIST="8.6" pip install -e .  # For RTX 3090
TORCH_CUDA_ARCH_LIST="8.9" pip install -e .  # For RTX 4090
```

## Quick Start

```python
import torch
import quant_gemm

# Create test data
M, N, K = 4096, 2, 14336
weight = torch.randn(M, K, device='cuda', dtype=torch.float32)
activation = torch.randn(N, K, device='cuda', dtype=torch.float32)

# Quantize
weight_q = quant_gemm.quantize_q4_0(weight)       # [M, K] -> [M, K//32, 18]
activation_q = quant_gemm.quantize_q8_1(activation)  # [N, K] -> [N, K//32, 36]

# Run quantized GEMM
output = quant_gemm.gemm_q4_0_q8_1(weight_q, activation_q, M, N, K)

# Compare with FP32 reference
output_ref = weight @ activation.T
nmse = torch.mean((output - output_ref)**2) / torch.mean(output_ref**2)
print(f"NMSE: {nmse.item():.6e}")
```

## API Reference

### Quantization Functions

#### `quantize_q4_0(x: torch.Tensor) -> torch.Tensor`
Quantize FP32 tensor to Q4_0 format (4-bit symmetric quantization).

- **Input**: FP32 tensor, shape `[..., K]` where K is divisible by 32
- **Output**: uint8 tensor, shape `[..., K//32, 18]`

#### `quantize_q8_1(x: torch.Tensor) -> torch.Tensor`
Quantize FP32 tensor to Q8_1 format (8-bit quantization with sum).

- **Input**: FP32 tensor, shape `[..., K]` where K is divisible by 32
- **Output**: uint8 tensor, shape `[..., K//32, 36]`

#### `dequantize_q4_0(x_q: torch.Tensor, K: int) -> torch.Tensor`
Dequantize Q4_0 tensor back to FP32.

- **Input**: Q4_0 quantized tensor, shape `[..., K//32, 18]`
- **Output**: FP32 tensor, shape `[..., K]`

### GEMM Functions

#### `gemm_q4_0_q8_1(weight_q, activation_q, M, N, K) -> torch.Tensor`
Quantized GEMM: `C[M,N] = W[M,K] @ A[N,K]^T`

- **weight_q**: Q4_0 quantized weights, shape `[M, K//32, 18]`, dtype uint8
- **activation_q**: Q8_1 quantized activations, shape `[N, K//32, 36]`, dtype uint8
- **M, N, K**: Matrix dimensions
- **Returns**: FP32 tensor, shape `[M, N]`

## Testing

```bash
# Run quick test
python tests/test_gemm_q4_0.py

# Run pytest
pytest tests/ -v
```

## Block Formats

### Q4_0 (18 bytes per 32 elements)
```
+----------------+--------------------------------+
| d (2 bytes)    |        qs[16] (16 bytes)       |
| half scale     | 4-bit values (2 per byte)      |
+----------------+--------------------------------+
```

### Q8_1 (36 bytes per 32 elements)
```
+------------------+--------------------------------+
| ds (4 bytes)     |        qs[32] (32 bytes)       |
| half2 (d + sum)  | 8-bit signed values            |
+------------------+--------------------------------+
```

## Performance

Typical performance on RTX 3090 (M=4096, N=2, K=14336):
- Naive implementation: ~40-50 GFLOPS
- Reference (llama.cpp): ~775 GFLOPS

Note: This is a naive implementation for educational purposes. Production use cases should use optimized implementations like llama.cpp.
