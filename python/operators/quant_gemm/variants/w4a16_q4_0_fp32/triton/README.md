# Triton Q4_0 GEMM Implementation

This directory contains a Triton implementation of the W4A16 GEMM kernel with Q4_0 quantization, equivalent to the CUDA implementation in the parent directory.

## Q4_0 Format (llama.cpp style)

- **Block size**: 32 elements
- **Storage**: 18 bytes per block
  - 2 bytes: FP16 scale (d)
  - 16 bytes: 32 x 4-bit quantized values (packed, 2 per byte)
- **Dequantization formula**: `w[i] = (q[i] - 8) * d`

## Files

- `kernel.py`: Triton kernel implementation
- `test_correctness.py`: Test script comparing against reference implementation
- `__init__.py`: Module exports

## Usage

```python
from triton.kernel import gemm_q4_0_triton

# weight: [N, K/32, 18] uint8 (Q4_0 format)
# activation: [M, K] float32
# Returns: [M, N] float32
output = gemm_q4_0_triton(weight, activation, M, N, K)
```

## Testing

```bash
python test_correctness.py
```

## Requirements

- PyTorch with CUDA support
- Triton (`pip install triton`)

## Notes

- For RTX 50 series (sm_120), you need PyTorch built with CUDA 12.8+
- The kernel computes: `C[m,n] = sum_k(A[m,k] * dequant(B[n,k]))`
