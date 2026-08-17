"""
Test framework for loading GEMM test specifications from JSON.

This module provides:
1. GemmTestSpec - Dataclass representing a GEMM test specification
2. JSON loading utilities
3. Test case discovery from definitions directory

The JSON schema supports:
- Multiple quantization types (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, FP32, FP16)
- Different GEMM variants (W4A8, W4A16, W4_1A8, etc.)
- Reference implementations in Python
- Type definitions and formulas

Usage:
    from quant_gemm.test_framework import GemmTestSpec, discover_test_specs

    # Load a single spec
    spec = GemmTestSpec.from_json("path/to/w4a8_q4_0_q8_1.json")

    # Discover all specs in definitions directory
    specs = discover_test_specs()
"""

import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from pathlib import Path


@dataclass
class TypeFieldInfo:
    """Information about a field in a quantization block type."""
    name: str
    dtype: str
    size: int
    offset: Optional[int] = None
    description: str = ""


@dataclass
class QuantBlockTypeInfo:
    """Detailed information about a quantization block type from JSON."""
    name: str
    size: int
    fields: List[TypeFieldInfo]
    quantization: Optional[Dict[str, Any]] = None
    unpacking: Optional[Dict[str, str]] = None

    @classmethod
    def from_dict(cls, name: str, data: dict) -> "QuantBlockTypeInfo":
        """Parse from JSON dict."""
        fields = []
        for f in data.get("fields", data.get("layout", [])):
            fields.append(TypeFieldInfo(
                name=f["name"],
                dtype=f["dtype"],
                size=f["size"],
                offset=f.get("offset"),
                description=f.get("description", ""),
            ))
        return cls(
            name=name,
            size=data["size"],
            fields=fields,
            quantization=data.get("quantization"),
            unpacking=data.get("unpacking"),
        )


@dataclass
class AxisInfo:
    """Information about an axis (dimension) in the GEMM."""
    name: str
    type: str  # "var" or "const"
    value: Optional[int] = None
    description: str = ""

    @property
    def is_variable(self) -> bool:
        return self.type == "var"

    @classmethod
    def from_dict(cls, name: str, data: dict) -> "AxisInfo":
        return cls(
            name=name,
            type=data["type"],
            value=data.get("value"),
            description=data.get("description", ""),
        )


@dataclass
class TensorInfo:
    """Information about an input or output tensor."""
    name: str
    shape: List[str]  # e.g., ["M", "K/32"]
    dtype: str
    size_bytes: Optional[int] = None
    description: str = ""

    @classmethod
    def from_dict(cls, name: str, data: dict) -> "TensorInfo":
        return cls(
            name=name,
            shape=data.get("shape") or [],
            dtype=data["dtype"],
            size_bytes=data.get("size_bytes"),
            description=data.get("description", ""),
        )


@dataclass
class GemmTestSpec:
    """
    Complete specification for a GEMM test case loaded from JSON.

    This dataclass captures all information needed to:
    1. Generate test data with correct shapes
    2. Apply quantization
    3. Run the kernel
    4. Compare with reference implementation
    """

    # Basic identification
    name: str
    op_type: str
    variant: str
    description: str
    tags: List[str]

    # Dimensions
    axes: Dict[str, AxisInfo]

    # Input/output specifications
    inputs: Dict[str, TensorInfo]
    outputs: Dict[str, TensorInfo]

    # Type definitions
    types: Dict[str, QuantBlockTypeInfo]

    # Constraints (e.g., "K % 32 == 0")
    constraints: List[str]

    # Formula information
    formula: Dict[str, str]

    # Reference implementation (Python code string)
    reference_code: str

    # Source file path
    source_path: Optional[str] = None

    # Convenience properties
    @property
    def M(self) -> Optional[int]:
        """Get M dimension value (None if variable)."""
        axis = self.axes.get("M")
        return axis.value if axis else None

    @property
    def N(self) -> int:
        """Get N dimension value."""
        axis = self.axes.get("N")
        if axis and axis.value is not None:
            return axis.value
        raise ValueError("N dimension not defined or is variable")

    @property
    def K(self) -> int:
        """Get K dimension value."""
        axis = self.axes.get("K")
        if axis and axis.value is not None:
            return axis.value
        raise ValueError("K dimension not defined or is variable")

    @property
    def block_size(self) -> int:
        """Get quantization block size (default 32)."""
        axis = self.axes.get("block_size")
        return axis.value if axis and axis.value else 32

    @property
    def weight_dtype(self) -> str:
        """Get weight quantization type."""
        weight_info = self.inputs.get("weight")
        if weight_info:
            return weight_info.dtype
        raise ValueError("Weight input not defined")

    @property
    def activation_dtype(self) -> str:
        """Get activation quantization type."""
        activation_info = self.inputs.get("activation")
        if activation_info:
            return activation_info.dtype
        raise ValueError("Activation input not defined")

    @property
    def output_dtype(self) -> str:
        """Get output data type."""
        output_info = self.outputs.get("output")
        if output_info:
            return output_info.dtype
        return "float32"  # Default

    @property
    def is_w4a8(self) -> bool:
        """Check if this is a W4A8 (4-bit weight, 8-bit activation) variant."""
        return "q4" in self.weight_dtype.lower() and "q8" in self.activation_dtype.lower()

    @property
    def is_w4a16(self) -> bool:
        """Check if this is a W4A16 (4-bit weight, FP16/FP32 activation) variant."""
        return "q4" in self.weight_dtype.lower() and "float" in self.activation_dtype.lower()

    def get_weight_block_bytes(self) -> int:
        """Get the byte size of weight quantization blocks."""
        type_info = self.types.get(self.weight_dtype)
        if type_info:
            return type_info.size
        # Fallback to registry
        from .registry import QuantTypeRegistry
        reg_info = QuantTypeRegistry.get(self.weight_dtype)
        if reg_info:
            return reg_info.size_bytes
        raise ValueError(f"Unknown weight type: {self.weight_dtype}")

    def get_activation_block_bytes(self) -> int:
        """Get the byte size of activation quantization blocks."""
        if self.activation_dtype in ("float32", "float16"):
            return 4 if self.activation_dtype == "float32" else 2
        type_info = self.types.get(self.activation_dtype)
        if type_info:
            return type_info.size
        from .registry import QuantTypeRegistry
        reg_info = QuantTypeRegistry.get(self.activation_dtype)
        if reg_info:
            return reg_info.size_bytes
        raise ValueError(f"Unknown activation type: {self.activation_dtype}")

    def get_weight_shape(self, M: Optional[int] = None) -> tuple:
        """
        Get the expected weight tensor shape.

        For quantized weights, returns (N, num_blocks, block_bytes).
        """
        N = self.N
        K = self.K
        block_size = self.block_size

        if self.weight_dtype in ("float32", "float16"):
            return (N, K)
        else:
            num_blocks = K // block_size
            block_bytes = self.get_weight_block_bytes()
            return (N, num_blocks, block_bytes)

    def get_activation_shape(self, M: int) -> tuple:
        """
        Get the expected activation tensor shape.

        Args:
            M: Batch dimension

        Returns:
            Shape tuple. For quantized activations: (M, num_blocks, block_bytes).
            For FP32/FP16: (M, K).
        """
        K = self.K
        block_size = self.block_size

        if self.activation_dtype in ("float32", "float16"):
            return (M, K)
        else:
            num_blocks = K // block_size
            block_bytes = self.get_activation_block_bytes()
            return (M, num_blocks, block_bytes)

    def get_output_shape(self, M: int) -> tuple:
        """Get the expected output tensor shape."""
        return (M, self.N)

    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> "GemmTestSpec":
        """
        Load a GemmTestSpec from a JSON file.

        Args:
            json_path: Path to the JSON definition file

        Returns:
            GemmTestSpec instance
        """
        json_path = Path(json_path)

        with open(json_path, 'r') as f:
            data = json.load(f)

        # Parse axes
        axes = {}
        for name, axis_data in data.get("axes", {}).items():
            axes[name] = AxisInfo.from_dict(name, axis_data)

        # Parse inputs
        inputs = {}
        for name, input_data in data.get("inputs", {}).items():
            inputs[name] = TensorInfo.from_dict(name, input_data)

        # Parse outputs
        outputs = {}
        for name, output_data in data.get("outputs", {}).items():
            outputs[name] = TensorInfo.from_dict(name, output_data)

        # Parse types
        types = {}
        for name, type_data in data.get("types", {}).items():
            types[name] = QuantBlockTypeInfo.from_dict(name, type_data)

        return cls(
            name=data["name"],
            op_type=data.get("op_type", "quant_gemm"),
            variant=data.get("variant", ""),
            description=data.get("description", ""),
            tags=data.get("tags", []),
            axes=axes,
            inputs=inputs,
            outputs=outputs,
            types=types,
            constraints=data.get("constraints", []),
            formula=data.get("formula", {}),
            reference_code=data.get("reference", ""),
            source_path=str(json_path),
        )

    @classmethod
    def from_dict(cls, data: dict, source_path: Optional[str] = None) -> "GemmTestSpec":
        """
        Create a GemmTestSpec from a dictionary.

        Useful for programmatic test spec creation.
        """
        # Parse axes
        axes = {}
        for name, axis_data in data.get("axes", {}).items():
            axes[name] = AxisInfo.from_dict(name, axis_data)

        # Parse inputs
        inputs = {}
        for name, input_data in data.get("inputs", {}).items():
            inputs[name] = TensorInfo.from_dict(name, input_data)

        # Parse outputs
        outputs = {}
        for name, output_data in data.get("outputs", {}).items():
            outputs[name] = TensorInfo.from_dict(name, output_data)

        # Parse types
        types = {}
        for name, type_data in data.get("types", {}).items():
            types[name] = QuantBlockTypeInfo.from_dict(name, type_data)

        return cls(
            name=data["name"],
            op_type=data.get("op_type", "quant_gemm"),
            variant=data.get("variant", ""),
            description=data.get("description", ""),
            tags=data.get("tags", []),
            axes=axes,
            inputs=inputs,
            outputs=outputs,
            types=types,
            constraints=data.get("constraints", []),
            formula=data.get("formula", {}),
            reference_code=data.get("reference", ""),
            source_path=source_path,
        )

    def to_dict(self) -> dict:
        """Convert back to dictionary (for serialization)."""
        return {
            "name": self.name,
            "op_type": self.op_type,
            "variant": self.variant,
            "description": self.description,
            "tags": self.tags,
            "axes": {
                name: {"type": axis.type, "value": axis.value, "description": axis.description}
                for name, axis in self.axes.items()
            },
            "inputs": {
                name: {"shape": inp.shape, "dtype": inp.dtype, "description": inp.description}
                for name, inp in self.inputs.items()
            },
            "outputs": {
                name: {"shape": out.shape, "dtype": out.dtype, "description": out.description}
                for name, out in self.outputs.items()
            },
            "constraints": self.constraints,
            "formula": self.formula,
            "reference": self.reference_code,
        }

    def __repr__(self) -> str:
        return (
            f"GemmTestSpec(name={self.name!r}, variant={self.variant!r}, "
            f"weight={self.weight_dtype}, activation={self.activation_dtype})"
        )


def get_definitions_dir() -> Path:
    """Get the path to the definitions directory."""
    # Try relative to this file
    module_dir = Path(__file__).parent
    candidates = [
        module_dir.parent.parent / "flashinfer_trace" / "definitions",
        module_dir.parent.parent.parent / "flashinfer_trace" / "definitions",
        Path("/home/haiyan/Agent4Kernel/quant-gemm-from-scratch/flashinfer_trace/definitions"),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError("Could not find definitions directory")


def discover_test_specs(
    definitions_dir: Optional[Union[str, Path]] = None,
    op_type: Optional[str] = None,
    variant: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> List[GemmTestSpec]:
    """
    Discover all test specifications from JSON files.

    Args:
        definitions_dir: Path to definitions directory. If None, auto-detect.
        op_type: Filter by operation type (e.g., "quant_gemm")
        variant: Filter by variant (e.g., "W4A8")
        tags: Filter by tags (all specified tags must be present)

    Returns:
        List of GemmTestSpec instances
    """
    if definitions_dir is None:
        definitions_dir = get_definitions_dir()
    else:
        definitions_dir = Path(definitions_dir)

    specs = []

    # Search for JSON files recursively
    for json_file in definitions_dir.rglob("*.json"):
        try:
            spec = GemmTestSpec.from_json(json_file)

            # Apply filters
            if op_type and spec.op_type != op_type:
                continue
            if variant and spec.variant != variant:
                continue
            if tags:
                spec_tags = set(spec.tags)
                if not all(tag in spec_tags for tag in tags):
                    continue

            specs.append(spec)
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")

    return specs


def discover_gemm_specs(
    definitions_dir: Optional[Union[str, Path]] = None,
    variant: Optional[str] = None,
) -> List[GemmTestSpec]:
    """
    Discover GEMM test specifications (convenience function).

    Args:
        definitions_dir: Path to definitions directory
        variant: Filter by variant (e.g., "W4A8", "W4A16")

    Returns:
        List of GemmTestSpec instances for GEMM operations
    """
    return discover_test_specs(
        definitions_dir=definitions_dir,
        op_type="quant_gemm",
        variant=variant,
    )


def list_available_variants(definitions_dir: Optional[Union[str, Path]] = None) -> List[str]:
    """List all available GEMM variants in the definitions directory."""
    specs = discover_gemm_specs(definitions_dir)
    variants = set(spec.variant for spec in specs if spec.variant)
    return sorted(variants)


def get_spec_by_name(name: str, definitions_dir: Optional[Union[str, Path]] = None) -> Optional[GemmTestSpec]:
    """Get a specific test spec by name."""
    specs = discover_test_specs(definitions_dir)
    for spec in specs:
        if spec.name == name:
            return spec
    return None
