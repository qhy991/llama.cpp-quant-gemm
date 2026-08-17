"""
Universal Test Framework for Operators.

This module provides:
1. TestResult - Structured test results
2. TestFramework - Runs tests for any operator based on its spec
3. Accuracy metrics computation
4. Benchmark utilities
"""

import torch
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
from pathlib import Path

from .base import OperatorSpec, TestConfig, BaseOperator
from .registry import OperatorRegistry, get_registry
from .common.types import get_quant_type, is_quantized_type


@dataclass
class ErrorMetrics:
    """Error metrics for comparing outputs."""
    mse: float = 0.0           # Mean Squared Error
    nmse: float = 0.0          # Normalized MSE (MSE / signal power)
    max_error: float = 0.0     # Maximum absolute error
    avg_error: float = 0.0     # Average absolute error
    relative_error: float = 0.0  # Mean relative error
    cosine_sim: float = 1.0    # Cosine similarity

    @classmethod
    def compute(cls, actual: torch.Tensor, expected: torch.Tensor) -> "ErrorMetrics":
        """Compute all error metrics."""
        actual_flat = actual.flatten().float()
        expected_flat = expected.flatten().float()

        diff = actual_flat - expected_flat

        mse = torch.mean(diff ** 2).item()
        signal_power = torch.mean(expected_flat ** 2).item()
        nmse = mse / signal_power if signal_power > 1e-10 else 0.0

        abs_diff = diff.abs()
        max_error = abs_diff.max().item()
        avg_error = abs_diff.mean().item()

        # Relative error
        rel_diff = abs_diff / (expected_flat.abs() + 1e-8)
        relative_error = rel_diff.mean().item()

        # Cosine similarity
        dot = torch.dot(actual_flat, expected_flat)
        norm_a = torch.norm(actual_flat)
        norm_b = torch.norm(expected_flat)
        cosine_sim = (dot / (norm_a * norm_b + 1e-8)).item()

        return cls(
            mse=mse,
            nmse=nmse,
            max_error=max_error,
            avg_error=avg_error,
            relative_error=relative_error,
            cosine_sim=cosine_sim,
        )

    def check(self, metric: str, threshold: float) -> bool:
        """Check if a metric is within threshold."""
        value = getattr(self, metric, None)
        if value is None:
            return False

        if metric == "cosine_sim":
            return value >= threshold  # Higher is better
        else:
            return value <= threshold  # Lower is better

    def get_metric(self, metric: str) -> float:
        """Get a specific metric value."""
        return getattr(self, metric, 0.0)


@dataclass
class TestResult:
    """Result of a single test run."""
    # Test identification
    name: str
    spec_name: str
    config_name: str

    # Status
    passed: bool
    skipped: bool = False
    error_message: str = ""

    # Metrics
    metrics: ErrorMetrics = field(default_factory=ErrorMetrics)

    # Parameters
    params: Dict[str, int] = field(default_factory=dict)

    # Performance (optional)
    time_ms: Optional[float] = None
    gflops: Optional[float] = None

    def __str__(self) -> str:
        if self.skipped:
            return f"[SKIP] {self.name}: {self.error_message}"
        status = "PASS" if self.passed else "FAIL"
        perf = f", {self.time_ms:.3f}ms" if self.time_ms else ""
        return f"[{status}] {self.name}: NMSE={self.metrics.nmse:.2e}{perf}"


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    name: str
    params: Dict[str, int]
    time_ms: float
    gflops: float
    iterations: int


class OperatorRunner:
    """
    Runner for a single operator variant.

    Handles:
    - Input generation
    - Quantization
    - Kernel execution
    - Reference computation
    - Metrics calculation
    """

    def __init__(
        self,
        spec: OperatorSpec,
        registry: OperatorRegistry,
    ):
        self.spec = spec
        self.registry = registry

    def _get_kernel(self) -> Callable:
        """Get the kernel function for this spec."""
        # Use registry's method to get kernel name
        kernel_name = self.registry._get_kernel_name_for_spec(self.spec)

        kernel = self.registry.get_kernel(kernel_name)
        if kernel is not None:
            return kernel

        # Fallback: try spec name directly
        kernel = self.registry.get_kernel(self.spec.name)
        if kernel is not None:
            return kernel

        raise RuntimeError(f"No kernel found for {self.spec.name} (tried: {kernel_name}, {self.spec.name})")

    def generate_inputs(
        self,
        params: Dict[str, int],
        seed: Optional[int] = None,
        device: str = "cuda",
    ) -> Dict[str, torch.Tensor]:
        """
        Generate random FP32 inputs.

        For quantized types, generates the original FP32 shape (not quantized shape).
        The shape in spec.json for quantized types represents the quantized tensor,
        but we need to generate FP32 data for quantization.
        """
        if seed is not None:
            torch.manual_seed(seed)

        inputs = {}
        for name, tensor_spec in self.spec.inputs.items():
            if tensor_spec.is_quantized:
                # For quantized types, generate original FP32 shape
                # Extract dimensions from params based on input name
                if "weight" in name.lower():
                    # Weight is [N, K] in FP32
                    N = params.get("N", 4096)
                    K = params.get("K", 4096)
                    shape = (N, K)
                elif "activation" in name.lower():
                    # Activation is [M, K] in FP32
                    M = params.get("M", 1)
                    K = params.get("K", 4096)
                    shape = (M, K)
                else:
                    # Fallback: try to infer from quantized shape
                    qtype = tensor_spec.quant_type
                    if qtype:
                        # Remove the bytes dimension and multiply back
                        q_shape = tensor_spec.resolve_shape(params)
                        # Typically: [dim1, num_blocks, bytes] -> [dim1, num_blocks * block_size]
                        shape = (q_shape[0], q_shape[1] * qtype.block_size)
                    else:
                        shape = tensor_spec.resolve_shape(params)
            else:
                # For FP32/FP16, use shape directly
                shape = tensor_spec.resolve_shape(params)

            tensor = torch.randn(shape, device=device, dtype=torch.float32)
            inputs[name] = tensor

        return inputs

    def quantize_inputs(
        self,
        raw_inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Quantize inputs according to their specs."""
        quantized = {}

        for name, tensor in raw_inputs.items():
            tensor_spec = self.spec.inputs.get(name)
            if tensor_spec is None:
                quantized[name] = tensor
                continue

            if tensor_spec.is_quantized:
                quantizer = self.registry.get_quantizer(tensor_spec.dtype)
                if quantizer is None:
                    raise RuntimeError(f"No quantizer for {tensor_spec.dtype}")
                quantized[name] = quantizer(tensor.contiguous())
            else:
                quantized[name] = tensor

        return quantized

    def run_kernel(
        self,
        quantized_inputs: Dict[str, torch.Tensor],
        params: Dict[str, int],
    ) -> torch.Tensor:
        """Run the kernel."""
        kernel = self._get_kernel()

        # For GEMM operators, handle the dimension convention
        # Quantized kernels (w4a8): C[M_k, N_k] = W[M_k, K] @ A[N_k, K]^T
        # FP32 activation kernels (w4a16): C[M, N] = A[M, K] @ W[N, K].T
        weight_q = quantized_inputs.get("weight")
        activation = quantized_inputs.get("activation")

        if weight_q is not None and activation is not None:
            M = params.get("M", 1)
            N = params.get("N", weight_q.shape[0])
            K = params.get("K", 4096)

            # Check if activation is quantized
            act_spec = self.spec.inputs.get("activation")
            is_quantized_activation = act_spec.is_quantized if act_spec else False

            if is_quantized_activation:
                # w4a8: swap M/N and transpose output
                output = kernel(weight_q, activation, N, M, K)
                return output.T
            else:
                # w4a16: direct call, output is already [M, N]
                return kernel(weight_q, activation, M, N, K)

        # Generic fallback
        raise RuntimeError("Unknown input configuration")

    def run_reference(
        self,
        raw_inputs: Dict[str, torch.Tensor],
        params: Dict[str, int],
    ) -> torch.Tensor:
        """Run reference implementation."""
        # Default: FP32 matmul
        weight = raw_inputs.get("weight")
        activation = raw_inputs.get("activation")

        if weight is not None and activation is not None:
            # output[M, N] = activation[M, K] @ weight[N, K].T
            return torch.matmul(activation, weight.T)

        raise RuntimeError("No reference implementation available")

    def run_test(
        self,
        config: TestConfig,
        seed: Optional[int] = 42,
    ) -> TestResult:
        """Run a single test case."""
        params = self.spec.validate_params(**config.params)

        try:
            # 1. Generate inputs
            raw_inputs = self.generate_inputs(params, seed=seed)

            # 2. Quantize
            quantized_inputs = self.quantize_inputs(raw_inputs)

            # 3. Run kernel
            output = self.run_kernel(quantized_inputs, params)

            # 4. Run reference
            output_ref = self.run_reference(raw_inputs, params)

            # 5. Check for NaN/Inf
            if torch.isnan(output).any():
                return TestResult(
                    name=f"{self.spec.name}_{config.name}",
                    spec_name=self.spec.name,
                    config_name=config.name,
                    passed=False,
                    error_message="Output contains NaN",
                    params=params,
                )

            if torch.isinf(output).any():
                return TestResult(
                    name=f"{self.spec.name}_{config.name}",
                    spec_name=self.spec.name,
                    config_name=config.name,
                    passed=False,
                    error_message="Output contains Inf",
                    params=params,
                )

            # 6. Compute metrics
            metrics = ErrorMetrics.compute(output, output_ref)

            # 7. Check threshold
            threshold = config.threshold or self.spec.accuracy.threshold
            passed = metrics.check(self.spec.accuracy.metric, threshold)

            return TestResult(
                name=f"{self.spec.name}_{config.name}",
                spec_name=self.spec.name,
                config_name=config.name,
                passed=passed,
                metrics=metrics,
                params=params,
                error_message="" if passed else f"Metric {self.spec.accuracy.metric} exceeds threshold",
            )

        except Exception as e:
            return TestResult(
                name=f"{self.spec.name}_{config.name}",
                spec_name=self.spec.name,
                config_name=config.name,
                passed=False,
                error_message=str(e),
                params=params,
            )

    def run_benchmark(
        self,
        config: TestConfig,
        warmup: int = 10,
        iterations: int = 100,
        seed: Optional[int] = 42,
    ) -> BenchmarkResult:
        """Run a benchmark."""
        params = self.spec.validate_params(**config.params)
        raw_inputs = self.generate_inputs(params, seed=seed)
        quantized_inputs = self.quantize_inputs(raw_inputs)

        kernel = self._get_kernel()
        weight_q = quantized_inputs.get("weight")
        activation = quantized_inputs.get("activation")

        M = params.get("M", 1)
        N = params.get("N", weight_q.shape[0])
        K = params.get("K", 4096)

        # Check if activation is quantized
        act_spec = self.spec.inputs.get("activation")
        is_quantized_activation = act_spec.is_quantized if act_spec else False

        # Warmup
        if is_quantized_activation:
            for _ in range(warmup):
                _ = kernel(weight_q, activation, N, M, K)
        else:
            for _ in range(warmup):
                _ = kernel(weight_q, activation, M, N, K)
        torch.cuda.synchronize()

        # Benchmark
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(iterations):
            if is_quantized_activation:
                _ = kernel(weight_q, activation, N, M, K)
            else:
                _ = kernel(weight_q, activation, M, N, K)
        end.record()
        torch.cuda.synchronize()

        total_ms = start.elapsed_time(end)
        avg_ms = total_ms / iterations

        # Compute GFLOPS
        flops = 2 * M * N * K
        gflops = flops / (avg_ms / 1000) / 1e9

        return BenchmarkResult(
            name=f"{self.spec.name}_{config.name}",
            params=params,
            time_ms=avg_ms,
            gflops=gflops,
            iterations=iterations,
        )

    def is_available(self) -> bool:
        """Check if this operator can be run."""
        # Check kernel
        try:
            self._get_kernel()
        except RuntimeError:
            return False

        # Check quantizers
        for tensor_spec in self.spec.inputs.values():
            if tensor_spec.is_quantized:
                if self.registry.get_quantizer(tensor_spec.dtype) is None:
                    return False

        return True

    def get_missing_components(self) -> List[str]:
        """Get list of missing components."""
        missing = []

        try:
            self._get_kernel()
        except RuntimeError:
            missing.append(f"kernel:{self.spec.name}")

        for tensor_spec in self.spec.inputs.values():
            if tensor_spec.is_quantized:
                if self.registry.get_quantizer(tensor_spec.dtype) is None:
                    missing.append(f"quantizer:{tensor_spec.dtype}")

        return missing


class TestFramework:
    """
    Universal test framework for running operator tests.

    Supports:
    - Running tests from JSON specs
    - Automatic test discovery
    - Benchmarking
    - Result aggregation
    """

    def __init__(self, registry: Optional[OperatorRegistry] = None):
        """
        Initialize the test framework.

        Args:
            registry: Operator registry. If None, uses global registry.
        """
        self.registry = registry or get_registry()

    def run_spec(
        self,
        spec: OperatorSpec,
        configs: Optional[List[TestConfig]] = None,
        skip_unavailable: bool = True,
    ) -> List[TestResult]:
        """
        Run all tests for a single spec.

        Args:
            spec: Operator specification
            configs: Test configurations. If None, uses spec's test_configs.
            skip_unavailable: Skip if components are missing

        Returns:
            List of test results
        """
        runner = OperatorRunner(spec, self.registry)

        if not runner.is_available():
            if skip_unavailable:
                missing = runner.get_missing_components()
                return [TestResult(
                    name=spec.name,
                    spec_name=spec.name,
                    config_name="all",
                    passed=True,
                    skipped=True,
                    error_message=f"Missing: {', '.join(missing)}",
                )]
            else:
                raise RuntimeError(
                    f"Missing components for {spec.name}: "
                    f"{runner.get_missing_components()}"
                )

        configs = configs or spec.test_configs
        if not configs:
            # Create a default config
            configs = [TestConfig(
                name="default",
                params={"M": 128, "N": spec.params.get("N", {}).default or 4096,
                        "K": spec.params.get("K", {}).default or 4096},
            )]

        results = []
        for config in configs:
            result = runner.run_test(config)
            results.append(result)

        return results

    def run_family(
        self,
        family_name: str,
        skip_unavailable: bool = True,
    ) -> List[TestResult]:
        """Run tests for all variants in a family."""
        self.registry.discover()

        family = self.registry.get_family(family_name)
        if family is None:
            raise ValueError(f"Unknown family: {family_name}")

        results = []
        for spec in family.variants.values():
            results.extend(self.run_spec(spec, skip_unavailable=skip_unavailable))

        return results

    def run_all(
        self,
        skip_unavailable: bool = True,
    ) -> List[TestResult]:
        """Run tests for all discovered specs."""
        self.registry.discover()

        results = []
        for spec in self.registry._specs.values():
            results.extend(self.run_spec(spec, skip_unavailable=skip_unavailable))

        return results

    def run_available(self) -> List[TestResult]:
        """Run tests only for available specs."""
        self.registry.discover()

        results = []
        for spec in self.registry.get_available_specs():
            results.extend(self.run_spec(spec, skip_unavailable=False))

        return results

    def benchmark_spec(
        self,
        spec: OperatorSpec,
        configs: Optional[List[TestConfig]] = None,
        warmup: int = 10,
        iterations: int = 100,
    ) -> List[BenchmarkResult]:
        """Run benchmarks for a spec."""
        runner = OperatorRunner(spec, self.registry)

        if not runner.is_available():
            return []

        configs = configs or spec.test_configs
        if not configs:
            configs = [TestConfig(
                name="default",
                params={"M": 128, "N": 4096, "K": 4096},
            )]

        results = []
        for config in configs:
            try:
                result = runner.run_benchmark(
                    config, warmup=warmup, iterations=iterations
                )
                results.append(result)
            except Exception as e:
                print(f"Benchmark failed for {spec.name}/{config.name}: {e}")

        return results

    def print_results(self, results: List[TestResult]):
        """Print test results summary."""
        passed = sum(1 for r in results if r.passed and not r.skipped)
        failed = sum(1 for r in results if not r.passed and not r.skipped)
        skipped = sum(1 for r in results if r.skipped)

        print("\n" + "=" * 60)
        print(f"Test Results: {passed} passed, {failed} failed, {skipped} skipped")
        print("=" * 60)

        for result in results:
            print(result)

        if failed > 0:
            print("\nFailed tests:")
            for r in results:
                if not r.passed and not r.skipped:
                    print(f"  - {r.name}: {r.error_message}")

    def print_benchmarks(self, results: List[BenchmarkResult]):
        """Print benchmark results."""
        print("\n" + "=" * 60)
        print("Benchmark Results")
        print("=" * 60)
        print(f"{'Name':<40} {'M':>6} {'Time(ms)':>10} {'GFLOPS':>10}")
        print("-" * 60)

        for result in results:
            M = result.params.get("M", 0)
            print(f"{result.name:<40} {M:>6} {result.time_ms:>10.3f} {result.gflops:>10.2f}")
