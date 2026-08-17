"""
Quantization type definitions.

This module defines all supported quantization types and their properties.
These definitions are used by the operator framework to:
1. Validate tensor shapes
2. Calculate memory requirements
3. Select appropriate quantizers/dequantizers
"""

from dataclasses import dataclass
from typing import Dict, Optional, List
from enum import Enum


@dataclass
class QuantType:
    """Definition of a quantization type."""

    name: str                    # e.g., "block_q4_0"
    size_bytes: int              # Bytes per quantization block
    block_size: int              # Elements per block (usually 32)
    bits: int                    # Bits per element
    symmetric: bool              # Symmetric vs asymmetric quantization
    has_scale: bool = True       # Has scale factor
    has_zero_point: bool = False # Has zero point / min value
    has_sum: bool = False        # Has sum for compensation (Q8_1)

    # Field layout for documentation
    fields: Optional[List[Dict]] = None

    @property
    def bits_per_element(self) -> float:
        """Effective bits per element including metadata."""
        return (self.size_bytes * 8) / self.block_size

    def get_quantized_shape(self, original_shape: tuple, axis: int = -1) -> tuple:
        """
        Calculate the shape of quantized tensor.

        Args:
            original_shape: Shape of FP32 tensor
            axis: Axis along which to quantize (default: last)

        Returns:
            Shape of quantized tensor: (..., K//block_size, size_bytes)
        """
        shape = list(original_shape)
        K = shape[axis]

        if K % self.block_size != 0:
            raise ValueError(
                f"Dimension {K} not divisible by block_size {self.block_size}"
            )

        num_blocks = K // self.block_size
        shape[axis] = num_blocks
        shape.append(self.size_bytes)

        return tuple(shape)

    def get_original_size(self, quantized_shape: tuple, axis: int = -2) -> int:
        """
        Calculate the original K dimension from quantized shape.

        Args:
            quantized_shape: Shape of quantized tensor
            axis: Axis of num_blocks (default: second to last)

        Returns:
            Original K dimension
        """
        num_blocks = quantized_shape[axis]
        return num_blocks * self.block_size


# Standard quantization types from llama.cpp / GGML
QUANT_TYPES: Dict[str, QuantType] = {
    # 4-bit symmetric (Q4_0)
    "block_q4_0": QuantType(
        name="block_q4_0",
        size_bytes=18,      # 2 (half d) + 16 (uint8[16])
        block_size=32,
        bits=4,
        symmetric=True,
        has_scale=True,
        fields=[
            {"name": "d", "dtype": "half", "size": 2, "offset": 0},
            {"name": "qs", "dtype": "uint8[16]", "size": 16, "offset": 2},
        ],
    ),

    # 4-bit asymmetric with min (Q4_1)
    "block_q4_1": QuantType(
        name="block_q4_1",
        size_bytes=20,      # 2 (half d) + 2 (half m) + 16 (uint8[16])
        block_size=32,
        bits=4,
        symmetric=False,
        has_scale=True,
        has_zero_point=True,
        fields=[
            {"name": "d", "dtype": "half", "size": 2, "offset": 0},
            {"name": "m", "dtype": "half", "size": 2, "offset": 2},
            {"name": "qs", "dtype": "uint8[16]", "size": 16, "offset": 4},
        ],
    ),

    # 5-bit symmetric (Q5_0)
    "block_q5_0": QuantType(
        name="block_q5_0",
        size_bytes=22,      # 2 (half d) + 4 (high bits) + 16 (low 4 bits)
        block_size=32,
        bits=5,
        symmetric=True,
        has_scale=True,
        fields=[
            {"name": "d", "dtype": "half", "size": 2, "offset": 0},
            {"name": "qh", "dtype": "uint8[4]", "size": 4, "offset": 2},
            {"name": "qs", "dtype": "uint8[16]", "size": 16, "offset": 6},
        ],
    ),

    # 5-bit asymmetric with min (Q5_1)
    "block_q5_1": QuantType(
        name="block_q5_1",
        size_bytes=24,      # 2 (half d) + 2 (half m) + 4 (high bits) + 16 (low 4 bits)
        block_size=32,
        bits=5,
        symmetric=False,
        has_scale=True,
        has_zero_point=True,
        fields=[
            {"name": "d", "dtype": "half", "size": 2, "offset": 0},
            {"name": "m", "dtype": "half", "size": 2, "offset": 2},
            {"name": "qh", "dtype": "uint8[4]", "size": 4, "offset": 4},
            {"name": "qs", "dtype": "uint8[16]", "size": 16, "offset": 8},
        ],
    ),

    # 8-bit symmetric (Q8_0)
    "block_q8_0": QuantType(
        name="block_q8_0",
        size_bytes=34,      # 2 (half d) + 32 (int8[32])
        block_size=32,
        bits=8,
        symmetric=True,
        has_scale=True,
        fields=[
            {"name": "d", "dtype": "half", "size": 2, "offset": 0},
            {"name": "qs", "dtype": "int8[32]", "size": 32, "offset": 2},
        ],
    ),

    # 8-bit with sum for compensation (Q8_1) - used for activations
    "block_q8_1": QuantType(
        name="block_q8_1",
        size_bytes=36,      # 4 (half2 ds) + 32 (int8[32])
        block_size=32,
        bits=8,
        symmetric=True,
        has_scale=True,
        has_sum=True,
        fields=[
            {"name": "ds", "dtype": "half2", "size": 4, "offset": 0,
             "description": "d (scale) in low half, s (sum) in high half"},
            {"name": "qs", "dtype": "int8[32]", "size": 32, "offset": 4},
        ],
    ),

    # Standard floating point types
    "float32": QuantType(
        name="float32",
        size_bytes=4,
        block_size=1,
        bits=32,
        symmetric=True,
    ),

    "float16": QuantType(
        name="float16",
        size_bytes=2,
        block_size=1,
        bits=16,
        symmetric=True,
    ),

    "bfloat16": QuantType(
        name="bfloat16",
        size_bytes=2,
        block_size=1,
        bits=16,
        symmetric=True,
    ),
}


def get_quant_type(name: str) -> Optional[QuantType]:
    """Get quantization type by name."""
    return QUANT_TYPES.get(name)


def get_block_size(dtype: str) -> int:
    """Get block size for a quantization type."""
    qtype = get_quant_type(dtype)
    if qtype:
        return qtype.block_size
    raise ValueError(f"Unknown dtype: {dtype}")


def get_block_bytes(dtype: str) -> int:
    """Get bytes per block for a quantization type."""
    qtype = get_quant_type(dtype)
    if qtype:
        return qtype.size_bytes
    raise ValueError(f"Unknown dtype: {dtype}")


def is_quantized_type(dtype: str) -> bool:
    """Check if dtype is a quantized type (not float32/float16)."""
    return dtype.startswith("block_")


def register_quant_type(qtype: QuantType):
    """Register a custom quantization type."""
    QUANT_TYPES[qtype.name] = qtype


def list_quant_types() -> List[str]:
    """List all registered quantization type names."""
    return list(QUANT_TYPES.keys())
