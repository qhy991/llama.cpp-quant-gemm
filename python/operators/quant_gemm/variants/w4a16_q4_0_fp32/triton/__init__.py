"""
Triton implementation of W4A16 GEMM with Q4_0 quantization.

This module provides a Triton kernel equivalent to the CUDA kernel
in the parent directory, implementing llama.cpp style Q4_0 quantized
matrix multiplication.
"""

from .kernel import gemm_q4_0_triton, run

__all__ = ['gemm_q4_0_triton', 'run']
