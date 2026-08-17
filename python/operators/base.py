"""
Base classes for the operator framework.

This module defines the core abstractions:
1. TensorSpec - Specification for input/output tensors
2. ParamSpec - Specification for operator parameters
3. OperatorSpec - Complete specification for an operator variant
4. BaseOperator - Abstract base class for operator implementations
5. OperatorFamily - Collection of related operator variants
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
from pathlib import Path
import json
import torch

from .common.types import get_quant_type, is_quantized_type, QuantType


@dataclass
class TensorSpec:
    """
    Specification for an input or output tensor.

    Attributes:
        name: Tensor name (e.g., "weight", "activation", "output")
        dtype: Data type (e.g., "block_q4_0", "float32")
        shape: Shape specification using symbolic dimensions (e.g., ["M", "K/32", 18])
        description: Human-readable description
        quantizer: Name of quantizer function (if applicable)
        optional: Whether this tensor is optional
    """
    name: str
    dtype: str
    shape: List[Union[str, int]]
    description: str = ""
    quantizer: Optional[str] = None
    optional: bool = False

    @property
    def is_quantized(self) -> bool:
        """Check if this tensor uses a quantized dtype."""
        return is_quantized_type(self.dtype)

    @property
    def quant_type(self) -> Optional[QuantType]:
        """Get the QuantType if this is a quantized tensor."""
        if self.is_quantized:
            return get_quant_type(self.dtype)
        return None

    def resolve_shape(self, params: Dict[str, int]) -> Tuple[int, ...]:
        """
        Resolve symbolic shape to concrete dimensions.

        Args:
            params: Dictionary mapping dimension names to values
                   e.g., {"M": 128, "N": 4096, "K": 4096}

        Returns:
            Tuple of concrete dimensions
        """
        resolved = []
        for dim in self.shape:
            if isinstance(dim, int):
                resolved.append(dim)
            elif isinstance(dim, str):
                # Handle expressions like "K/32"
                if "/" in dim:
                    var, divisor = dim.split("/")
                    resolved.append(params[var.strip()] // int(divisor))
                else:
                    resolved.append(params[dim])
            else:
                raise ValueError(f"Invalid shape dimension: {dim}")
        return tuple(resolved)

    @classmethod
    def from_dict(cls, name: str, data: Dict) -> "TensorSpec":
        """Create TensorSpec from dictionary."""
        return cls(
            name=name,
            dtype=data["dtype"],
            shape=data.get("shape", []),
            description=data.get("description", ""),
            quantizer=data.get("quantizer"),
            optional=data.get("optional", False),
        )


@dataclass
class ParamSpec:
    """
    Specification for an operator parameter.

    Attributes:
        name: Parameter name (e.g., "M", "N", "K")
        dtype: Parameter type ("int", "float", "bool")
        default: Default value (None if required)
        constraint: Constraint expression (e.g., "K % 32 == 0")
        description: Human-readable description
    """
    name: str
    dtype: str = "int"
    default: Optional[Any] = None
    constraint: Optional[str] = None
    description: str = ""

    @property
    def is_required(self) -> bool:
        """Check if this parameter is required (has no default)."""
        return self.default is None

    def validate(self, value: Any) -> bool:
        """Validate a value against the constraint."""
        if self.constraint is None:
            return True

        # Create a safe eval environment
        env = {self.name: value}
        try:
            return eval(self.constraint, {"__builtins__": {}}, env)
        except Exception:
            return False

    @classmethod
    def from_dict(cls, name: str, data: Dict) -> "ParamSpec":
        """Create ParamSpec from dictionary."""
        return cls(
            name=name,
            dtype=data.get("type", "int"),
            default=data.get("default"),
            constraint=data.get("constraint"),
            description=data.get("description", ""),
        )


@dataclass
class TestConfig:
    """Configuration for a single test case."""
    name: str
    params: Dict[str, int]
    description: str = ""
    threshold: Optional[float] = None  # Override default accuracy threshold

    @classmethod
    def from_dict(cls, data: Dict) -> "TestConfig":
        """Create TestConfig from dictionary."""
        name = data.pop("name", "unnamed")
        description = data.pop("description", "")
        threshold = data.pop("threshold", None)
        return cls(
            name=name,
            params=data,  # Remaining keys are params
            description=description,
            threshold=threshold,
        )


@dataclass
class AccuracySpec:
    """Specification for accuracy requirements."""
    metric: str = "nmse"  # "nmse", "mse", "max_error", "cosine_similarity"
    threshold: float = 0.1
    per_element_threshold: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict) -> "AccuracySpec":
        """Create AccuracySpec from dictionary."""
        return cls(
            metric=data.get("metric", "nmse"),
            threshold=data.get("threshold", 0.1),
            per_element_threshold=data.get("per_element_threshold"),
        )


@dataclass
class KernelSpec:
    """Specification for the CUDA kernel."""
    file: str
    entry_point: str
    launch_config: Optional[Dict] = None

    @classmethod
    def from_dict(cls, data: Dict) -> "KernelSpec":
        """Create KernelSpec from dictionary."""
        return cls(
            file=data.get("file", "kernel.cu"),
            entry_point=data.get("entry_point", ""),
            launch_config=data.get("launch_config"),
        )


@dataclass
class OperatorSpec:
    """
    Complete specification for an operator variant.

    This is loaded from spec.json and contains all information needed to:
    1. Prepare input tensors (shapes, dtypes, quantization)
    2. Execute the kernel
    3. Validate outputs against reference
    4. Run test cases
    """
    # Identity
    name: str
    family: str  # Parent operator family
    version: str = "1.0.0"
    description: str = ""

    # Kernel info
    kernel: Optional[KernelSpec] = None

    # IO specifications
    inputs: Dict[str, TensorSpec] = field(default_factory=dict)
    outputs: Dict[str, TensorSpec] = field(default_factory=dict)
    params: Dict[str, ParamSpec] = field(default_factory=dict)

    # Reference implementation
    reference: Optional[str] = None  # "reference.py:run"

    # Test configurations
    test_configs: List[TestConfig] = field(default_factory=list)

    # Accuracy requirements
    accuracy: AccuracySpec = field(default_factory=AccuracySpec)

    # Source path
    source_dir: Optional[Path] = None

    # Formula / computation description
    formula: Dict[str, str] = field(default_factory=dict)

    def get_input_names(self) -> List[str]:
        """Get list of input tensor names."""
        return list(self.inputs.keys())

    def get_output_names(self) -> List[str]:
        """Get list of output tensor names."""
        return list(self.outputs.keys())

    def get_param_names(self) -> List[str]:
        """Get list of parameter names."""
        return list(self.params.keys())

    def get_required_params(self) -> List[str]:
        """Get list of required parameter names."""
        return [name for name, spec in self.params.items() if spec.is_required]

    def validate_params(self, **kwargs) -> Dict[str, Any]:
        """
        Validate and fill in default parameters.

        Returns complete params dict with defaults filled in.
        Raises ValueError if required params missing or constraints violated.
        """
        result = {}

        for name, spec in self.params.items():
            if name in kwargs:
                value = kwargs[name]
                if not spec.validate(value):
                    raise ValueError(
                        f"Parameter {name}={value} violates constraint: {spec.constraint}"
                    )
                result[name] = value
            elif spec.default is not None:
                result[name] = spec.default
            else:
                raise ValueError(f"Missing required parameter: {name}")

        return result

    @classmethod
    def from_json(cls, json_path: Path) -> "OperatorSpec":
        """Load OperatorSpec from a spec.json file."""
        with open(json_path) as f:
            data = json.load(f)

        # Parse inputs
        inputs = {}
        for name, input_data in data.get("inputs", {}).items():
            inputs[name] = TensorSpec.from_dict(name, input_data)

        # Parse outputs
        outputs = {}
        for name, output_data in data.get("outputs", {}).items():
            outputs[name] = TensorSpec.from_dict(name, output_data)

        # Parse params
        params = {}
        for name, param_data in data.get("params", {}).items():
            params[name] = ParamSpec.from_dict(name, param_data)

        # Parse test configs
        test_configs = []
        for config_data in data.get("test_configs", []):
            test_configs.append(TestConfig.from_dict(config_data.copy()))

        # Parse kernel spec
        kernel = None
        if "kernel" in data:
            kernel = KernelSpec.from_dict(data["kernel"])

        # Parse accuracy spec
        accuracy = AccuracySpec()
        if "accuracy" in data:
            accuracy = AccuracySpec.from_dict(data["accuracy"])

        return cls(
            name=data["name"],
            family=data.get("family", data.get("parent", "")),
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            kernel=kernel,
            inputs=inputs,
            outputs=outputs,
            params=params,
            reference=data.get("reference"),
            test_configs=test_configs,
            accuracy=accuracy,
            source_dir=json_path.parent,
            formula=data.get("formula", {}),
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "family": self.family,
            "version": self.version,
            "description": self.description,
            "inputs": {
                name: {
                    "dtype": spec.dtype,
                    "shape": spec.shape,
                    "description": spec.description,
                    "quantizer": spec.quantizer,
                }
                for name, spec in self.inputs.items()
            },
            "outputs": {
                name: {
                    "dtype": spec.dtype,
                    "shape": spec.shape,
                    "description": spec.description,
                }
                for name, spec in self.outputs.items()
            },
            "params": {
                name: {
                    "type": spec.dtype,
                    "default": spec.default,
                    "constraint": spec.constraint,
                    "description": spec.description,
                }
                for name, spec in self.params.items()
            },
            "test_configs": [
                {"name": tc.name, **tc.params}
                for tc in self.test_configs
            ],
            "accuracy": {
                "metric": self.accuracy.metric,
                "threshold": self.accuracy.threshold,
            },
            "formula": self.formula,
        }


class BaseOperator(ABC):
    """
    Abstract base class for operator implementations.

    Subclasses must implement:
    - prepare_inputs(): Quantize/prepare input tensors
    - run(): Execute the operator
    - run_reference(): Run reference implementation

    The framework will automatically:
    - Load the spec from JSON
    - Discover and register quantizers
    - Run test cases
    - Compute accuracy metrics
    """

    def __init__(self, spec: OperatorSpec):
        self.spec = spec
        self._kernel = None
        self._quantizers: Dict[str, Callable] = {}
        self._dequantizers: Dict[str, Callable] = {}
        self._reference_fn: Optional[Callable] = None

    @property
    def name(self) -> str:
        """Operator name."""
        return self.spec.name

    @property
    def family(self) -> str:
        """Operator family name."""
        return self.spec.family

    def set_kernel(self, kernel: Callable):
        """Set the compiled kernel function."""
        self._kernel = kernel

    def set_quantizer(self, dtype: str, quantizer: Callable):
        """Register a quantizer for a dtype."""
        self._quantizers[dtype] = quantizer

    def set_dequantizer(self, dtype: str, dequantizer: Callable):
        """Register a dequantizer for a dtype."""
        self._dequantizers[dtype] = dequantizer

    def set_reference(self, reference_fn: Callable):
        """Set the reference implementation function."""
        self._reference_fn = reference_fn

    def get_quantizer(self, dtype: str) -> Optional[Callable]:
        """Get quantizer for a dtype."""
        return self._quantizers.get(dtype)

    def get_dequantizer(self, dtype: str) -> Optional[Callable]:
        """Get dequantizer for a dtype."""
        return self._dequantizers.get(dtype)

    def quantize(self, tensor: torch.Tensor, dtype: str) -> torch.Tensor:
        """Quantize a tensor to the specified dtype."""
        if dtype in ("float32", "float16", "bfloat16"):
            return tensor

        quantizer = self.get_quantizer(dtype)
        if quantizer is None:
            raise RuntimeError(f"No quantizer registered for {dtype}")

        return quantizer(tensor.contiguous())

    def dequantize(self, tensor: torch.Tensor, dtype: str, K: int) -> torch.Tensor:
        """Dequantize a tensor back to float32."""
        if dtype in ("float32", "float16", "bfloat16"):
            return tensor

        dequantizer = self.get_dequantizer(dtype)
        if dequantizer is None:
            raise RuntimeError(f"No dequantizer registered for {dtype}")

        return dequantizer(tensor.contiguous(), K)

    @abstractmethod
    def prepare_inputs(
        self,
        raw_inputs: Dict[str, torch.Tensor],
        params: Dict[str, int],
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare input tensors (quantization, reshaping, etc.).

        Args:
            raw_inputs: Dictionary of raw FP32 input tensors
            params: Dictionary of parameter values

        Returns:
            Dictionary of prepared (possibly quantized) tensors
        """
        pass

    @abstractmethod
    def run(
        self,
        inputs: Dict[str, torch.Tensor],
        params: Dict[str, int],
    ) -> Dict[str, torch.Tensor]:
        """
        Execute the operator.

        Args:
            inputs: Dictionary of prepared input tensors
            params: Dictionary of parameter values

        Returns:
            Dictionary of output tensors
        """
        pass

    @abstractmethod
    def run_reference(
        self,
        raw_inputs: Dict[str, torch.Tensor],
        params: Dict[str, int],
    ) -> Dict[str, torch.Tensor]:
        """
        Run the reference implementation.

        Args:
            raw_inputs: Dictionary of raw FP32 input tensors
            params: Dictionary of parameter values

        Returns:
            Dictionary of reference output tensors
        """
        pass

    def generate_random_inputs(
        self,
        params: Dict[str, int],
        device: str = "cuda",
        seed: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate random input tensors for testing.

        Args:
            params: Dictionary of parameter values
            device: Target device
            seed: Random seed

        Returns:
            Dictionary of random FP32 tensors matching input specs
        """
        if seed is not None:
            torch.manual_seed(seed)

        inputs = {}
        for name, spec in self.spec.inputs.items():
            shape = spec.resolve_shape(params)
            # For quantized types, we generate FP32 and quantize later
            tensor = torch.randn(shape, device=device, dtype=torch.float32)
            inputs[name] = tensor

        return inputs

    def is_ready(self) -> bool:
        """Check if operator is ready to run (all components registered)."""
        if self._kernel is None:
            return False

        # Check quantizers for quantized inputs
        for spec in self.spec.inputs.values():
            if spec.is_quantized and spec.dtype not in self._quantizers:
                return False

        return True

    def get_missing_components(self) -> List[str]:
        """Get list of missing components."""
        missing = []

        if self._kernel is None:
            missing.append("kernel")

        for spec in self.spec.inputs.values():
            if spec.is_quantized and spec.dtype not in self._quantizers:
                missing.append(f"quantizer:{spec.dtype}")

        return missing


@dataclass
class OperatorFamily:
    """
    Collection of related operator variants.

    An operator family groups variants that share common structure
    but differ in quantization types or other parameters.

    Example: quant_gemm family contains:
    - w4a8_q4_0_q8_1
    - w4a16_q4_0_fp32
    - w4_1a8_q4_1_q8_1
    """
    name: str
    version: str = "1.0.0"
    description: str = ""
    op_type: str = ""  # "gemm", "attention", "normalization", etc.

    # Common dependencies
    common_dependencies: List[str] = field(default_factory=list)

    # Interface definition
    interface: Dict[str, List[str]] = field(default_factory=dict)

    # Variants
    variants: Dict[str, OperatorSpec] = field(default_factory=dict)

    # Source directory
    source_dir: Optional[Path] = None

    def add_variant(self, spec: OperatorSpec):
        """Add a variant to this family."""
        self.variants[spec.name] = spec

    def get_variant(self, name: str) -> Optional[OperatorSpec]:
        """Get a variant by name."""
        return self.variants.get(name)

    def list_variants(self) -> List[str]:
        """List all variant names."""
        return list(self.variants.keys())

    @classmethod
    def from_manifest(cls, manifest_path: Path) -> "OperatorFamily":
        """Load OperatorFamily from a manifest.json file."""
        with open(manifest_path) as f:
            data = json.load(f)

        return cls(
            name=data["name"],
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            op_type=data.get("op_type", ""),
            common_dependencies=data.get("common_dependencies", []),
            interface=data.get("interface", {}),
            source_dir=manifest_path.parent,
        )
