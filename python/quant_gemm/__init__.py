"""
quant_gemm: Python bindings for quantized GEMM CUDA kernels

Example usage:
    import torch
    import quant_gemm

    # Create test data
    M, N, K = 4096, 2, 14336
    weight = torch.randn(M, K, device='cuda', dtype=torch.float32)
    activation = torch.randn(N, K, device='cuda', dtype=torch.float32)

    # Quantize
    weight_q = quant_gemm.quantize_q4_0(weight)
    activation_q = quant_gemm.quantize_q8_1(activation)

    # Run GEMM
    output = quant_gemm.gemm_q4_0_q8_1(weight_q, activation_q, M, N, K)
"""

from . import _C

# Version
__version__ = "0.1.0"

# Block sizes
QK4_0 = 32  # Q4_0 block size
QK8_1 = 32  # Q8_1 block size
BLOCK_Q4_0_BYTES = 18  # sizeof(block_q4_0)
BLOCK_Q8_1_BYTES = 36  # sizeof(block_q8_1)


def quantize_q4_0(x):
    """
    Quantize FP32 tensor to Q4_0 format.

    Args:
        x: FP32 tensor, shape [..., K] where K must be divisible by 32

    Returns:
        Quantized tensor as uint8, shape [..., K//32, 18]
    """
    return _C.quantize_q4_0(x.contiguous())


def quantize_q8_1(x):
    """
    Quantize FP32 tensor to Q8_1 format.

    Args:
        x: FP32 tensor, shape [..., K] where K must be divisible by 32

    Returns:
        Quantized tensor as uint8, shape [..., K//32, 36]
    """
    return _C.quantize_q8_1(x.contiguous())


def gemm_q4_0_q8_1(weight_q, activation_q, M, N, K):
    """
    Quantized GEMM: C[M,N] = W[M,K] @ A[N,K]^T

    Args:
        weight_q: Q4_0 quantized weights, shape [M, K//32, 18], dtype uint8
        activation_q: Q8_1 quantized activations, shape [N, K//32, 36], dtype uint8
        M, N, K: Matrix dimensions

    Returns:
        Output tensor, shape [M, N], dtype float32
    """
    return _C.gemm_q4_0_q8_1(
        weight_q.contiguous(),
        activation_q.contiguous(),
        M, N, K
    )


def dequantize_q4_0(x_q, K):
    """
    Dequantize Q4_0 tensor back to FP32.

    Args:
        x_q: Q4_0 quantized tensor, shape [..., K//32, 18], dtype uint8
        K: Original K dimension

    Returns:
        FP32 tensor, shape [..., K]
    """
    return _C.dequantize_q4_0(x_q.contiguous(), K)


# Export all public functions
__all__ = [
    'quantize_q4_0',
    'quantize_q8_1',
    'gemm_q4_0_q8_1',
    'dequantize_q4_0',
    'QK4_0',
    'QK8_1',
    'BLOCK_Q4_0_BYTES',
    'BLOCK_Q8_1_BYTES',
]
