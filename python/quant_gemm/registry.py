"""
Registry for quantization types and GEMM kernels.

This module provides a centralized registry for:
1. Quantization type definitions (block sizes, byte sizes, etc.)
2. Quantization functions (FP32 -> quantized)
3. Dequantization functions (quantized -> FP32)
4. GEMM kernel implementations (for different weight/activation type pairs)

Usage:
    from quant_gemm.registry import QuantTypeRegistry, GemmKernelRegistry

    # Get type info
    info = QuantTypeRegistry.get("block_q4_0")

    # Get kernel for specific type pair
    kernel = GemmKernelRegistry.get_kernel("block_q4_0", "block_q8_1")
"""

from typing import Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
import torch


@dataclass
class QuantTypeInfo:
    """Information about a quantization type."""
    name: str
    size_bytes: int          # Size of one quantization block in bytes
    block_size: int          # Number of elements per block (usually 32)
    bits: int                # Bits per element (4, 5, 8, etc.)
    symmetric: bool          # Whether it's symmetric quantization
    has_sum: bool = False    # Whether block stores sum (like Q8_1)
    has_min: bool = False    # Whether block stores min offset (like Q4_1)

    @property
    def bits_per_element(self) -> float:
        """Effective bits per element including metadata."""
        return (self.size_bytes * 8) / self.block_size


class QuantTypeRegistry:
    """
    Registry for quantization type definitions.

    Stores information about each quantization format including:
    - Block size (elements per block)
    - Byte size (bytes per block)
    - Bit width
    - Quantization scheme (symmetric/asymmetric)
    """

    _types: Dict[str, QuantTypeInfo] = {}
    _initialized: bool = False

    @classmethod
    def _init_builtin_types(cls):
        """Initialize built-in quantization types."""
        if cls._initialized:
            return

        builtin_types = [
            # Symmetric 4-bit (llama.cpp Q4_0)
            QuantTypeInfo(
                name="block_q4_0",
                size_bytes=18,      # 2 (half d) + 16 (uint8[16])
                block_size=32,
                bits=4,
                symmetric=True,
            ),
            # Asymmetric 4-bit with min (llama.cpp Q4_1)
            QuantTypeInfo(
                name="block_q4_1",
                size_bytes=20,      # 2 (half d) + 2 (half m) + 16 (uint8[16])
                block_size=32,
                bits=4,
                symmetric=False,
                has_min=True,
            ),
            # Symmetric 5-bit (llama.cpp Q5_0)
            QuantTypeInfo(
                name="block_q5_0",
                size_bytes=22,      # 2 (half d) + 4 (high bits) + 16 (low 4 bits)
                block_size=32,
                bits=5,
                symmetric=True,
            ),
            # Asymmetric 5-bit with min (llama.cpp Q5_1)
            QuantTypeInfo(
                name="block_q5_1",
                size_bytes=24,      # 2 (half d) + 2 (half m) + 4 (high bits) + 16 (low 4 bits)
                block_size=32,
                bits=5,
                symmetric=False,
                has_min=True,
            ),
            # Symmetric 8-bit (llama.cpp Q8_0)
            QuantTypeInfo(
                name="block_q8_0",
                size_bytes=34,      # 2 (half d) + 32 (int8[32])
                block_size=32,
                bits=8,
                symmetric=True,
            ),
            # 8-bit with sum (llama.cpp Q8_1, for activation)
            QuantTypeInfo(
                name="block_q8_1",
                size_bytes=36,      # 4 (half2 ds) + 32 (int8[32])
                block_size=32,
                bits=8,
                symmetric=True,     # Symmetric but stores sum for compensation
                has_sum=True,
            ),
            # Standard float types
            QuantTypeInfo(
                name="float32",
                size_bytes=4,
                block_size=1,
                bits=32,
                symmetric=True,     # N/A for float
            ),
            QuantTypeInfo(
                name="float16",
                size_bytes=2,
                block_size=1,
                bits=16,
                symmetric=True,
            ),
        ]

        for type_info in builtin_types:
            cls._types[type_info.name] = type_info

        cls._initialized = True

    @classmethod
    def get(cls, name: str) -> Optional[QuantTypeInfo]:
        """Get quantization type info by name."""
        cls._init_builtin_types()
        return cls._types.get(name)

    @classmethod
    def register(cls, type_info: QuantTypeInfo):
        """Register a new quantization type."""
        cls._init_builtin_types()
        cls._types[type_info.name] = type_info

    @classmethod
    def list_types(cls) -> list:
        """List all registered type names."""
        cls._init_builtin_types()
        return list(cls._types.keys())

    @classmethod
    def get_block_bytes(cls, name: str) -> int:
        """Get the byte size of a quantization block."""
        type_info = cls.get(name)
        if type_info:
            return type_info.size_bytes
        raise ValueError(f"Unknown quantization type: {name}")

    @classmethod
    def get_block_size(cls, name: str) -> int:
        """Get the element count per quantization block."""
        type_info = cls.get(name)
        if type_info:
            return type_info.block_size
        raise ValueError(f"Unknown quantization type: {name}")


# Type alias for kernel function
KernelFunc = Callable[[torch.Tensor, torch.Tensor, int, int, int], torch.Tensor]
QuantizerFunc = Callable[[torch.Tensor], torch.Tensor]
DequantizerFunc = Callable[[torch.Tensor, int], torch.Tensor]


class GemmKernelRegistry:
    """
    Registry for GEMM kernel implementations.

    Maps (weight_type, activation_type) pairs to kernel functions.
    Also manages quantization and dequantization functions.

    Usage:
        # Register a kernel
        GemmKernelRegistry.register_kernel("block_q4_0", "block_q8_1", my_kernel_func)

        # Get kernel
        kernel = GemmKernelRegistry.get_kernel("block_q4_0", "block_q8_1")
        output = kernel(weight_q, activation_q, M, N, K)
    """

    # (weight_type, activation_type) -> kernel function
    _kernels: Dict[Tuple[str, str], KernelFunc] = {}

    # dtype -> quantizer function
    _quantizers: Dict[str, QuantizerFunc] = {}

    # dtype -> dequantizer function
    _dequantizers: Dict[str, DequantizerFunc] = {}

    # Track initialization state
    _initialized: bool = False

    @classmethod
    def _init_from_c_module(cls):
        """Initialize registry from _C module if available."""
        if cls._initialized:
            return

        try:
            from . import _C

            # Register quantizers
            if hasattr(_C, 'quantize_q4_0'):
                cls._quantizers['block_q4_0'] = _C.quantize_q4_0
            if hasattr(_C, 'quantize_q8_1'):
                cls._quantizers['block_q8_1'] = _C.quantize_q8_1
            if hasattr(_C, 'quantize_q4_1'):
                cls._quantizers['block_q4_1'] = _C.quantize_q4_1
            if hasattr(_C, 'quantize_q5_0'):
                cls._quantizers['block_q5_0'] = _C.quantize_q5_0
            if hasattr(_C, 'quantize_q5_1'):
                cls._quantizers['block_q5_1'] = _C.quantize_q5_1
            if hasattr(_C, 'quantize_q8_0'):
                cls._quantizers['block_q8_0'] = _C.quantize_q8_0

            # Register dequantizers
            if hasattr(_C, 'dequantize_q4_0'):
                cls._dequantizers['block_q4_0'] = _C.dequantize_q4_0
            if hasattr(_C, 'dequantize_q4_1'):
                cls._dequantizers['block_q4_1'] = _C.dequantize_q4_1
            if hasattr(_C, 'dequantize_q8_1'):
                cls._dequantizers['block_q8_1'] = _C.dequantize_q8_1

            # Register GEMM kernels
            if hasattr(_C, 'gemm_q4_0_q8_1'):
                cls._kernels[('block_q4_0', 'block_q8_1')] = _C.gemm_q4_0_q8_1
            if hasattr(_C, 'gemm_q4_1_q8_1'):
                cls._kernels[('block_q4_1', 'block_q8_1')] = _C.gemm_q4_1_q8_1
            if hasattr(_C, 'gemm_q5_0_q8_1'):
                cls._kernels[('block_q5_0', 'block_q8_1')] = _C.gemm_q5_0_q8_1
            if hasattr(_C, 'gemm_q5_1_q8_1'):
                cls._kernels[('block_q5_1', 'block_q8_1')] = _C.gemm_q5_1_q8_1
            if hasattr(_C, 'gemm_q8_0_q8_1'):
                cls._kernels[('block_q8_0', 'block_q8_1')] = _C.gemm_q8_0_q8_1
            # W4A16 variants
            if hasattr(_C, 'gemm_q4_0_fp32'):
                cls._kernels[('block_q4_0', 'float32')] = _C.gemm_q4_0_fp32

        except ImportError:
            pass  # _C module not available, will use manual registration

        cls._initialized = True

    @classmethod
    def register_kernel(cls, weight_type: str, activation_type: str, func: KernelFunc):
        """
        Register a GEMM kernel for a specific (weight_type, activation_type) pair.

        Args:
            weight_type: Quantization type for weights (e.g., "block_q4_0")
            activation_type: Quantization type for activations (e.g., "block_q8_1")
            func: Kernel function with signature (weight_q, activation_q, M, N, K) -> output
        """
        cls._init_from_c_module()
        cls._kernels[(weight_type, activation_type)] = func

    @classmethod
    def register_quantizer(cls, dtype: str, func: QuantizerFunc):
        """
        Register a quantization function.

        Args:
            dtype: Quantization type name (e.g., "block_q4_0")
            func: Function with signature (tensor) -> quantized_tensor
        """
        cls._init_from_c_module()
        cls._quantizers[dtype] = func

    @classmethod
    def register_dequantizer(cls, dtype: str, func: DequantizerFunc):
        """
        Register a dequantization function.

        Args:
            dtype: Quantization type name
            func: Function with signature (quantized_tensor, K) -> tensor
        """
        cls._init_from_c_module()
        cls._dequantizers[dtype] = func

    @classmethod
    def get_kernel(cls, weight_type: str, activation_type: str) -> Optional[KernelFunc]:
        """Get GEMM kernel for specified type pair."""
        cls._init_from_c_module()
        return cls._kernels.get((weight_type, activation_type))

    @classmethod
    def get_quantizer(cls, dtype: str) -> Optional[QuantizerFunc]:
        """Get quantization function for specified type."""
        cls._init_from_c_module()
        return cls._quantizers.get(dtype)

    @classmethod
    def get_dequantizer(cls, dtype: str) -> Optional[DequantizerFunc]:
        """Get dequantization function for specified type."""
        cls._init_from_c_module()
        return cls._dequantizers.get(dtype)

    @classmethod
    def list_kernels(cls) -> list:
        """List all registered (weight_type, activation_type) pairs."""
        cls._init_from_c_module()
        return list(cls._kernels.keys())

    @classmethod
    def list_quantizers(cls) -> list:
        """List all registered quantizer types."""
        cls._init_from_c_module()
        return list(cls._quantizers.keys())

    @classmethod
    def has_kernel(cls, weight_type: str, activation_type: str) -> bool:
        """Check if a kernel is registered for the type pair."""
        cls._init_from_c_module()
        return (weight_type, activation_type) in cls._kernels

    @classmethod
    def has_quantizer(cls, dtype: str) -> bool:
        """Check if a quantizer is registered for the type."""
        cls._init_from_c_module()
        return dtype in cls._quantizers


# Convenience functions
def get_quant_type(name: str) -> Optional[QuantTypeInfo]:
    """Get quantization type info by name."""
    return QuantTypeRegistry.get(name)


def get_kernel(weight_type: str, activation_type: str) -> Optional[KernelFunc]:
    """Get GEMM kernel for specified type pair."""
    return GemmKernelRegistry.get_kernel(weight_type, activation_type)


def quantize(tensor: torch.Tensor, dtype: str) -> torch.Tensor:
    """
    Quantize a tensor to the specified type.

    Args:
        tensor: Input FP32 tensor
        dtype: Target quantization type (e.g., "block_q4_0")

    Returns:
        Quantized tensor as uint8

    Raises:
        ValueError: If quantizer not available for dtype
    """
    quantizer = GemmKernelRegistry.get_quantizer(dtype)
    if quantizer is None:
        raise ValueError(f"No quantizer registered for type: {dtype}")
    return quantizer(tensor.contiguous())


def dequantize(tensor: torch.Tensor, dtype: str, K: int) -> torch.Tensor:
    """
    Dequantize a tensor back to FP32.

    Args:
        tensor: Quantized tensor
        dtype: Quantization type
        K: Original K dimension

    Returns:
        FP32 tensor

    Raises:
        ValueError: If dequantizer not available for dtype
    """
    dequantizer = GemmKernelRegistry.get_dequantizer(dtype)
    if dequantizer is None:
        raise ValueError(f"No dequantizer registered for type: {dtype}")
    return dequantizer(tensor.contiguous(), K)
