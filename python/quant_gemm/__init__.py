"""
quant_gemm: Python bindings for quantized GEMM CUDA kernels

This package provides:
1. Low-level CUDA kernel bindings (quantize, gemm, dequantize)
2. Registry system for quantization types and kernels
3. JSON-driven test framework for generic GEMM testing

Example usage (low-level API):
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

Example usage (registry API):
    from quant_gemm.registry import GemmKernelRegistry, quantize

    # Quantize using registry
    weight_q = quantize(weight, "block_q4_0")
    activation_q = quantize(activation, "block_q8_1")

    # Get kernel from registry
    kernel = GemmKernelRegistry.get_kernel("block_q4_0", "block_q8_1")
    output = kernel(weight_q, activation_q, M, N, K)

Example usage (test framework):
    from quant_gemm.test_framework import GemmTestSpec, discover_gemm_specs
    from quant_gemm.test_runner import GemmTestRunner

    # Load spec from JSON
    spec = GemmTestSpec.from_json("definitions/quant_gemm/w4a8_q4_0_q8_1.json")

    # Run test
    runner = GemmTestRunner(spec)
    result = runner.run_test(M=128)
    print(f"NMSE: {result.nmse:.2e}, Passed: {result.passed}")
"""

from . import _C

# Version
__version__ = "0.2.0"

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


def gemm_q4_0_fp32(weight_q, activation, M, N, K):
    """
    Quantized GEMM with Q4_0 weights and FP32 activations.

    Args:
        weight_q: Q4_0 quantized weights, shape [N, K//32, 18], dtype uint8
        activation: FP32 activations, shape [M, K], dtype float32
        M, N, K: Matrix dimensions

    Returns:
        Output tensor, shape [M, N], dtype float32
    """
    return _C.gemm_q4_0_fp32(
        weight_q.contiguous(),
        activation.contiguous(),
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


# Import submodules (lazy import to avoid circular dependencies)
def _import_registry():
    from . import registry
    return registry

def _import_test_framework():
    from . import test_framework
    return test_framework

def _import_test_runner():
    from . import test_runner
    return test_runner


# Convenience re-exports from registry
def quantize(tensor, dtype: str):
    """
    Quantize a tensor to the specified type using the registry.

    Args:
        tensor: Input FP32 tensor
        dtype: Target quantization type (e.g., "block_q4_0", "block_q8_1")

    Returns:
        Quantized tensor
    """
    from .registry import quantize as _quantize
    return _quantize(tensor, dtype)


def dequantize(tensor, dtype: str, K: int):
    """
    Dequantize a tensor back to FP32 using the registry.

    Args:
        tensor: Quantized tensor
        dtype: Quantization type
        K: Original K dimension

    Returns:
        FP32 tensor
    """
    from .registry import dequantize as _dequantize
    return _dequantize(tensor, dtype, K)


def get_kernel(weight_type: str, activation_type: str):
    """
    Get GEMM kernel for the specified type pair.

    Args:
        weight_type: Weight quantization type (e.g., "block_q4_0")
        activation_type: Activation quantization type (e.g., "block_q8_1")

    Returns:
        Kernel function or None if not available
    """
    from .registry import get_kernel as _get_kernel
    return _get_kernel(weight_type, activation_type)


# Export all public functions and modules
__all__ = [
    # Low-level API
    'quantize_q4_0',
    'quantize_q8_1',
    'gemm_q4_0_q8_1',
    'gemm_q4_0_fp32',
    'dequantize_q4_0',
    # Constants
    'QK4_0',
    'QK8_1',
    'BLOCK_Q4_0_BYTES',
    'BLOCK_Q8_1_BYTES',
    # Registry API
    'quantize',
    'dequantize',
    'get_kernel',
    # Submodules (import via quant_gemm.registry, etc.)
    'registry',
    'test_framework',
    'test_runner',
]
