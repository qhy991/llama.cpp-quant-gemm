"""
Operator Registry - Discovery and registration of operators.

This module provides:
1. Automatic discovery of operators from directory structure
2. Registration of operator implementations
3. Lookup of operators by name or type
"""

from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Type
import json
import importlib.util
import sys

from .base import (
    OperatorSpec,
    OperatorFamily,
    BaseOperator,
    TensorSpec,
)
from .common.types import get_quant_type


class OperatorRegistry:
    """
    Central registry for all operators.

    Discovers operators from a directory structure and provides
    lookup by name, family, or type.

    Directory Structure Expected:
        operators_dir/
        ├── <family_name>/
        │   ├── manifest.json
        │   ├── csrc/
        │   │   └── bindings.cpp
        │   └── variants/
        │       ├── <variant_name>/
        │       │   ├── spec.json
        │       │   ├── kernel.cu
        │       │   └── reference.py
        │       └── ...
        └── ...
    """

    def __init__(self, operators_dir: Optional[Path] = None):
        """
        Initialize the registry.

        Args:
            operators_dir: Root directory containing operator families.
                          If None, uses the default 'operators/' directory.
        """
        if operators_dir is None:
            operators_dir = Path(__file__).parent
        self.operators_dir = Path(operators_dir)

        # Storage
        self._families: Dict[str, OperatorFamily] = {}
        self._specs: Dict[str, OperatorSpec] = {}
        self._operators: Dict[str, BaseOperator] = {}
        self._kernels: Dict[str, Callable] = {}
        self._quantizers: Dict[str, Callable] = {}
        self._dequantizers: Dict[str, Callable] = {}

        # Discovery state
        self._discovered = False

    def discover(self, reload: bool = False):
        """
        Discover all operators from the directory structure.

        Args:
            reload: If True, rediscover even if already discovered
        """
        if self._discovered and not reload:
            return

        self._families.clear()
        self._specs.clear()

        # Scan for operator families
        for item in self.operators_dir.iterdir():
            if not item.is_dir():
                continue
            if item.name.startswith(("_", ".", "common")):
                continue

            manifest_path = item / "manifest.json"
            if manifest_path.exists():
                self._load_family(item, manifest_path)

        self._discovered = True

    def _load_family(self, family_dir: Path, manifest_path: Path):
        """Load an operator family from its manifest."""
        try:
            family = OperatorFamily.from_manifest(manifest_path)
            self._families[family.name] = family

            # Load variants
            variants_dir = family_dir / "variants"
            if variants_dir.exists():
                for variant_dir in variants_dir.iterdir():
                    if not variant_dir.is_dir():
                        continue

                    spec_path = variant_dir / "spec.json"
                    if spec_path.exists():
                        self._load_variant(family, variant_dir, spec_path)

        except Exception as e:
            print(f"Warning: Failed to load family from {manifest_path}: {e}")

    def _load_variant(
        self,
        family: OperatorFamily,
        variant_dir: Path,
        spec_path: Path
    ):
        """Load an operator variant from its spec."""
        try:
            spec = OperatorSpec.from_json(spec_path)
            spec.family = family.name  # Ensure family is set

            family.add_variant(spec)
            self._specs[spec.name] = spec

            # Try to load reference implementation
            self._load_reference(spec, variant_dir)

        except Exception as e:
            print(f"Warning: Failed to load variant from {spec_path}: {e}")

    def _load_reference(self, spec: OperatorSpec, variant_dir: Path):
        """Load the Python reference implementation for a variant."""
        if not spec.reference:
            return

        try:
            # Parse "reference.py:run" format
            if ":" in spec.reference:
                file_name, func_name = spec.reference.split(":")
            else:
                file_name = spec.reference
                func_name = "run"

            ref_path = variant_dir / file_name
            if not ref_path.exists():
                return

            # Dynamically load the module
            module_name = f"_ref_{spec.name}"
            loader_spec = importlib.util.spec_from_file_location(module_name, ref_path)
            module = importlib.util.module_from_spec(loader_spec)
            sys.modules[module_name] = module
            loader_spec.loader.exec_module(module)

            # Get the function
            if hasattr(module, func_name):
                spec._reference_fn = getattr(module, func_name)

        except Exception as e:
            print(f"Warning: Failed to load reference for {spec.name}: {e}")

    # =========================================================================
    # Registration Methods
    # =========================================================================

    def register_kernel(self, name: str, kernel: Callable):
        """
        Register a compiled kernel function.

        Args:
            name: Kernel name (usually matches variant name)
            kernel: The kernel function
        """
        self._kernels[name] = kernel

    def register_quantizer(self, dtype: str, quantizer: Callable):
        """
        Register a quantizer function.

        Args:
            dtype: Data type name (e.g., "block_q4_0")
            quantizer: Function to quantize FP32 to this dtype
        """
        self._quantizers[dtype] = quantizer

    def register_dequantizer(self, dtype: str, dequantizer: Callable):
        """
        Register a dequantizer function.

        Args:
            dtype: Data type name
            dequantizer: Function to dequantize back to FP32
        """
        self._dequantizers[dtype] = dequantizer

    def register_operator(self, operator: BaseOperator):
        """
        Register an operator instance.

        Args:
            operator: The operator instance
        """
        self._operators[operator.name] = operator

    def register_from_module(self, module, prefix: str = ""):
        """
        Auto-register kernels and quantizers from a compiled module.

        Looks for functions matching patterns:
        - gemm_<name> or <prefix>_<name>: kernels
        - quantize_<dtype>: quantizers
        - dequantize_<dtype>: dequantizers

        Args:
            module: Compiled pybind module
            prefix: Prefix for kernel functions
        """
        for attr_name in dir(module):
            if attr_name.startswith("_"):
                continue

            func = getattr(module, attr_name)
            if not callable(func):
                continue

            # Register quantizers
            if attr_name.startswith("quantize_"):
                dtype = "block_" + attr_name[9:]  # quantize_q4_0 -> block_q4_0
                self.register_quantizer(dtype, func)

            # Register dequantizers
            elif attr_name.startswith("dequantize_"):
                dtype = "block_" + attr_name[11:]
                self.register_dequantizer(dtype, func)

            # Register kernels
            elif attr_name.startswith("gemm_") or (prefix and attr_name.startswith(prefix)):
                kernel_name = attr_name
                self.register_kernel(kernel_name, func)

    # =========================================================================
    # Lookup Methods
    # =========================================================================

    def get_family(self, name: str) -> Optional[OperatorFamily]:
        """Get an operator family by name."""
        self.discover()
        return self._families.get(name)

    def get_spec(self, name: str) -> Optional[OperatorSpec]:
        """Get an operator spec by name."""
        self.discover()
        return self._specs.get(name)

    def get_kernel(self, name: str) -> Optional[Callable]:
        """Get a registered kernel by name."""
        return self._kernels.get(name)

    def get_quantizer(self, dtype: str) -> Optional[Callable]:
        """Get a registered quantizer by dtype."""
        return self._quantizers.get(dtype)

    def get_dequantizer(self, dtype: str) -> Optional[Callable]:
        """Get a registered dequantizer by dtype."""
        return self._dequantizers.get(dtype)

    def get_operator(self, name: str) -> Optional[BaseOperator]:
        """Get a registered operator by name."""
        return self._operators.get(name)

    def list_families(self) -> List[str]:
        """List all discovered family names."""
        self.discover()
        return list(self._families.keys())

    def list_specs(self) -> List[str]:
        """List all discovered spec names."""
        self.discover()
        return list(self._specs.keys())

    def list_variants(self, family: str) -> List[str]:
        """List all variants in a family."""
        self.discover()
        fam = self._families.get(family)
        if fam:
            return fam.list_variants()
        return []

    def list_kernels(self) -> List[str]:
        """List all registered kernel names."""
        return list(self._kernels.keys())

    def list_quantizers(self) -> List[str]:
        """List all registered quantizer dtypes."""
        return list(self._quantizers.keys())

    def has_kernel(self, name: str) -> bool:
        """Check if a kernel is registered."""
        return name in self._kernels

    def has_quantizer(self, dtype: str) -> bool:
        """Check if a quantizer is registered for dtype."""
        return dtype in self._quantizers

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_specs_by_family(self, family: str) -> List[OperatorSpec]:
        """Get all specs belonging to a family."""
        self.discover()
        fam = self._families.get(family)
        if fam:
            return list(fam.variants.values())
        return []

    def get_specs_by_type(self, op_type: str) -> List[OperatorSpec]:
        """Get all specs matching an operator type (gemm, attention, etc.)."""
        self.discover()
        result = []
        for family in self._families.values():
            if family.op_type == op_type:
                result.extend(family.variants.values())
        return result

    def get_available_specs(self) -> List[OperatorSpec]:
        """
        Get specs that have all required components registered.

        Returns specs where:
        - Kernel is registered
        - All required quantizers are registered
        """
        self.discover()
        available = []

        for spec in self._specs.values():
            if self._is_spec_available(spec):
                available.append(spec)

        return available

    def _is_spec_available(self, spec: OperatorSpec) -> bool:
        """Check if a spec has all required components."""
        # Check for kernel
        kernel_name = self._get_kernel_name_for_spec(spec)
        if kernel_name not in self._kernels:
            return False

        # Check for quantizers
        for input_spec in spec.inputs.values():
            if input_spec.is_quantized:
                if input_spec.dtype not in self._quantizers:
                    return False

        return True

    def _get_kernel_name_for_spec(self, spec: OperatorSpec) -> str:
        """
        Determine the kernel function name for a spec.

        Convention: gemm_<weight_dtype>_<activation_dtype>
        e.g., gemm_q4_0_q8_1
        """
        # dtype normalization for kernel naming
        dtype_normalization = {
            "float32": "fp32",
            "float16": "fp16",
            "bfloat16": "bf16",
        }

        # Extract dtypes from inputs
        weight_dtype = ""
        activation_dtype = ""

        for name, input_spec in spec.inputs.items():
            dtype_short = input_spec.dtype.replace("block_", "")
            # Normalize dtype
            dtype_short = dtype_normalization.get(dtype_short, dtype_short)
            if "weight" in name.lower():
                weight_dtype = dtype_short
            elif "activation" in name.lower():
                activation_dtype = dtype_short

        if weight_dtype and activation_dtype:
            return f"gemm_{weight_dtype}_{activation_dtype}"

        # Fallback to spec name
        return spec.name

    def print_status(self):
        """Print registration status."""
        self.discover()

        print("=" * 60)
        print("Operator Registry Status")
        print("=" * 60)

        print(f"\nFamilies: {len(self._families)}")
        for name, family in self._families.items():
            print(f"  - {name}: {len(family.variants)} variants")

        print(f"\nSpecs: {len(self._specs)}")
        for name, spec in self._specs.items():
            status = "available" if self._is_spec_available(spec) else "missing components"
            print(f"  - {name}: {status}")

        print(f"\nKernels: {len(self._kernels)}")
        for name in self._kernels:
            print(f"  - {name}")

        print(f"\nQuantizers: {len(self._quantizers)}")
        for dtype in self._quantizers:
            print(f"  - {dtype}")

        print(f"\nDequantizers: {len(self._dequantizers)}")
        for dtype in self._dequantizers:
            print(f"  - {dtype}")


# Global registry instance
_global_registry: Optional[OperatorRegistry] = None


def get_registry() -> OperatorRegistry:
    """Get the global registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = OperatorRegistry()
    return _global_registry


def register_kernel(name: str, kernel: Callable):
    """Register a kernel to the global registry."""
    get_registry().register_kernel(name, kernel)


def register_quantizer(dtype: str, quantizer: Callable):
    """Register a quantizer to the global registry."""
    get_registry().register_quantizer(dtype, quantizer)


def register_dequantizer(dtype: str, dequantizer: Callable):
    """Register a dequantizer to the global registry."""
    get_registry().register_dequantizer(dtype, dequantizer)
