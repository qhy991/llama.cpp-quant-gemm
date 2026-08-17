"""
Test runner for executing GEMM tests from JSON specifications.

This module provides:
1. ReferenceRunner - Executes Python reference implementations from JSON
2. GemmTestRunner - Orchestrates test execution and comparison
3. TestResult - Structured test results with metrics
4. Benchmarking utilities

Usage:
    from quant_gemm.test_framework import GemmTestSpec
    from quant_gemm.test_runner import GemmTestRunner

    spec = GemmTestSpec.from_json("path/to/spec.json")
    runner = GemmTestRunner(spec)

    # Run correctness test
    result = runner.run_test(M=128)
    print(f"NMSE: {result.nmse:.2e}, Passed: {result.passed}")

    # Run benchmark
    bench = runner.run_benchmark(M=4096, warmup=10, iterations=100)
    print(f"Time: {bench.time_ms:.3f} ms, GFLOPS: {bench.gflops:.2f}")
"""

import torch
import struct
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, Callable, List
from contextlib import contextmanager

from .test_framework import GemmTestSpec
from .registry import GemmKernelRegistry, QuantTypeRegistry


@dataclass
class ErrorMetrics:
    """Error metrics for comparing outputs."""
    mse: float           # Mean Squared Error
    nmse: float          # Normalized Mean Squared Error (MSE / signal power)
    max_error: float     # Maximum absolute error
    avg_error: float     # Average absolute error
    relative_error: float  # Mean relative error

    @classmethod
    def compute(cls, actual: torch.Tensor, expected: torch.Tensor) -> "ErrorMetrics":
        """Compute all error metrics between actual and expected tensors."""
        diff = actual - expected

        mse = torch.mean(diff ** 2).item()
        signal_power = torch.mean(expected ** 2).item()
        nmse = mse / signal_power if signal_power > 1e-10 else 0.0

        abs_diff = diff.abs()
        max_error = abs_diff.max().item()
        avg_error = abs_diff.mean().item()

        # Relative error (avoid division by zero)
        rel_diff = abs_diff / (expected.abs() + 1e-8)
        relative_error = rel_diff.mean().item()

        return cls(
            mse=mse,
            nmse=nmse,
            max_error=max_error,
            avg_error=avg_error,
            relative_error=relative_error,
        )

    def check(self, threshold: float) -> bool:
        """Check if NMSE is below threshold."""
        return self.nmse < threshold

    def __str__(self) -> str:
        return (
            f"ErrorMetrics(nmse={self.nmse:.2e}, mse={self.mse:.2e}, "
            f"max={self.max_error:.4f}, avg={self.avg_error:.4f})"
        )


@dataclass
class TestResult:
    """Result of a single test run."""
    name: str
    passed: bool
    metrics: ErrorMetrics
    time_ms: Optional[float] = None
    gflops: Optional[float] = None
    message: str = ""
    spec_variant: str = ""
    M: int = 0
    N: int = 0
    K: int = 0

    @property
    def nmse(self) -> float:
        return self.metrics.nmse

    @property
    def max_error(self) -> float:
        return self.metrics.max_error

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        perf = f", {self.time_ms:.3f}ms" if self.time_ms else ""
        return f"[{status}] {self.name}: NMSE={self.nmse:.2e}{perf}"


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    name: str
    M: int
    N: int
    K: int
    time_ms: float
    gflops: float
    iterations: int
    throughput_gb_s: Optional[float] = None

    def __str__(self) -> str:
        return (
            f"{self.name} [{self.M}x{self.N}x{self.K}]: "
            f"{self.time_ms:.3f}ms, {self.gflops:.2f} GFLOPS"
        )


class ReferenceRunner:
    """
    Executes Python reference implementations from JSON specs.

    The reference code is compiled once and cached for repeated execution.
    Supports both:
    1. Full reference implementations (with `run(activation, weight)` function)
    2. Fallback to FP32 matmul when no reference is provided
    """

    def __init__(self, spec: GemmTestSpec):
        self.spec = spec
        self._compiled_run: Optional[Callable] = None
        self._exec_globals: Dict[str, Any] = {}

    def compile(self):
        """Compile the reference implementation code."""
        if not self.spec.reference_code:
            return

        # Set up execution environment
        self._exec_globals = {
            "torch": torch,
            "struct": struct,
            "__builtins__": __builtins__,
        }

        try:
            exec(self.spec.reference_code, self._exec_globals)
            self._compiled_run = self._exec_globals.get("run")
        except Exception as e:
            print(f"Warning: Failed to compile reference code: {e}")
            self._compiled_run = None

    def run(self, activation: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Run the reference implementation.

        Args:
            activation: Activation tensor (quantized or FP32)
            weight: Weight tensor (quantized or FP32)

        Returns:
            Output tensor
        """
        if self._compiled_run is None:
            self.compile()

        if self._compiled_run is not None:
            try:
                return self._compiled_run(activation, weight)
            except Exception as e:
                print(f"Warning: Reference execution failed: {e}, falling back to FP32")

        # Fallback: use FP32 matmul (requires dequantized inputs)
        return self._fallback_fp32(activation, weight)

    def _fallback_fp32(self, activation: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Fallback FP32 matmul reference."""
        # Assume inputs are already FP32 or need dequantization
        if activation.dtype != torch.float32:
            raise ValueError("Fallback requires FP32 activation")
        if weight.dtype != torch.float32:
            raise ValueError("Fallback requires FP32 weight")

        # C[M,N] = A[M,K] @ W[N,K].T
        return torch.matmul(activation, weight.T)

    def run_with_fp32_inputs(
        self,
        activation_fp32: torch.Tensor,
        weight_fp32: torch.Tensor
    ) -> torch.Tensor:
        """
        Run reference with FP32 inputs directly.

        This is the most common use case: compare quantized kernel output
        against what the output would be with perfect FP32 arithmetic.
        """
        # C[M,N] = A[M,K] @ W[N,K].T
        return torch.matmul(activation_fp32, weight_fp32.T)


class DataGenerator:
    """Generates test data for GEMM tests."""

    @staticmethod
    def uniform(shape: tuple, device='cuda', low=-1.0, high=1.0) -> torch.Tensor:
        """Generate uniform random data."""
        return torch.empty(shape, device=device).uniform_(low, high)

    @staticmethod
    def normal(shape: tuple, device='cuda', mean=0.0, std=1.0) -> torch.Tensor:
        """Generate normal random data."""
        return torch.randn(shape, device=device) * std + mean

    @staticmethod
    def xavier(shape: tuple, device='cuda') -> torch.Tensor:
        """Generate Xavier-initialized data."""
        fan_in = shape[-1]
        fan_out = shape[-2] if len(shape) > 1 else fan_in
        std = (2.0 / (fan_in + fan_out)) ** 0.5
        return torch.randn(shape, device=device) * std

    @staticmethod
    def he(shape: tuple, device='cuda') -> torch.Tensor:
        """Generate He-initialized data."""
        fan_in = shape[-1]
        std = (2.0 / fan_in) ** 0.5
        return torch.randn(shape, device=device) * std

    @staticmethod
    def create_test_data(
        spec: GemmTestSpec,
        M: int,
        device='cuda',
        seed: Optional[int] = None,
        distribution: str = 'normal',
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create test data for a GEMM spec.

        Args:
            spec: Test specification
            M: Batch dimension
            device: Target device
            seed: Random seed (optional)
            distribution: 'normal', 'uniform', 'xavier', 'he'

        Returns:
            (weight, activation) tuple of FP32 tensors
        """
        if seed is not None:
            torch.manual_seed(seed)

        N, K = spec.N, spec.K

        gen_func = {
            'normal': DataGenerator.normal,
            'uniform': DataGenerator.uniform,
            'xavier': DataGenerator.xavier,
            'he': DataGenerator.he,
        }.get(distribution, DataGenerator.normal)

        weight = gen_func((N, K), device=device)
        activation = gen_func((M, K), device=device)

        return weight, activation


class GemmTestRunner:
    """
    Main test runner for GEMM operations.

    Orchestrates:
    1. Data generation
    2. Quantization
    3. Kernel execution
    4. Reference computation
    5. Metric calculation
    """

    # Default NMSE thresholds by variant
    DEFAULT_THRESHOLDS = {
        "W4A8": 0.1,    # 4-bit weights have higher quantization error
        "W4A16": 0.05,
        "W4_1A8": 0.1,
        "W5A8": 0.08,
        "W8A8": 0.02,
        "default": 0.1,
    }

    def __init__(
        self,
        spec: GemmTestSpec,
        threshold: Optional[float] = None,
    ):
        """
        Initialize test runner.

        Args:
            spec: GEMM test specification
            threshold: NMSE threshold for pass/fail (auto-detected if None)
        """
        self.spec = spec
        self.reference = ReferenceRunner(spec)

        # Set threshold based on variant if not specified
        if threshold is None:
            threshold = self.DEFAULT_THRESHOLDS.get(
                spec.variant,
                self.DEFAULT_THRESHOLDS["default"]
            )
        self.threshold = threshold

    def create_test_data(
        self,
        M: int,
        seed: Optional[int] = 42,
        distribution: str = 'normal',
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create test data (FP32 weight and activation)."""
        return DataGenerator.create_test_data(
            self.spec, M, seed=seed, distribution=distribution
        )

    def quantize_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """Quantize weight tensor according to spec."""
        dtype = self.spec.weight_dtype

        if dtype in ("float32", "float16"):
            return weight if dtype == "float32" else weight.half()

        quantizer = GemmKernelRegistry.get_quantizer(dtype)
        if quantizer is None:
            raise NotImplementedError(f"No quantizer for {dtype}")

        return quantizer(weight.contiguous())

    def quantize_activation(self, activation: torch.Tensor) -> torch.Tensor:
        """Quantize activation tensor according to spec."""
        dtype = self.spec.activation_dtype

        if dtype in ("float32", "float16"):
            return activation if dtype == "float32" else activation.half()

        quantizer = GemmKernelRegistry.get_quantizer(dtype)
        if quantizer is None:
            raise NotImplementedError(f"No quantizer for {dtype}")

        return quantizer(activation.contiguous())

    def get_kernel(self) -> Callable:
        """Get the CUDA kernel for this spec's type pair."""
        kernel = GemmKernelRegistry.get_kernel(
            self.spec.weight_dtype,
            self.spec.activation_dtype
        )
        if kernel is None:
            raise NotImplementedError(
                f"No kernel for ({self.spec.weight_dtype}, {self.spec.activation_dtype})"
            )
        return kernel

    def run_kernel(
        self,
        weight_q: torch.Tensor,
        activation_q: torch.Tensor,
        M: int,
    ) -> torch.Tensor:
        """
        Run the CUDA kernel.

        Note on dimension conventions:
        - JSON spec: weight[N, K], activation[M, K] -> output[M, N]
          where M=batch, N=output_features
        - Kernel: W[kernel_M, K] @ A[kernel_N, K]^T -> C[kernel_M, kernel_N]

        To get output[batch, output_features] = [M, N]:
        - kernel_M = spec.N (weight rows = output_features)
        - kernel_N = M (activation rows = batch)
        - Output from kernel is [spec.N, M], need to transpose to [M, spec.N]
        """
        kernel = self.get_kernel()
        # Kernel computes: C[kernel_M, kernel_N] = W[kernel_M, K] @ A[kernel_N, K]^T
        # With our data: weight[spec.N, K], activation[M, K]
        # So kernel_M = spec.N, kernel_N = M
        output = kernel(weight_q, activation_q, self.spec.N, M, self.spec.K)
        # Transpose to get [M, spec.N] = [batch, output_features]
        return output.T

    def compute_reference(
        self,
        weight_fp32: torch.Tensor,
        activation_fp32: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute reference output using FP32 matmul.

        JSON convention:
        - activation: [M, K] (batch x input_features)
        - weight: [N, K] (output_features x input_features)
        - output: [M, N] = activation @ weight.T
        """
        # output[M, N] = activation[M, K] @ weight[N, K].T
        return torch.matmul(activation_fp32, weight_fp32.T)

    def run_test(
        self,
        M: int,
        seed: Optional[int] = 42,
        threshold: Optional[float] = None,
    ) -> TestResult:
        """
        Run a single test case.

        Args:
            M: Batch dimension
            seed: Random seed
            threshold: NMSE threshold (uses default if None)

        Returns:
            TestResult with metrics
        """
        threshold = threshold or self.threshold

        # 1. Generate data
        weight_fp32, activation_fp32 = self.create_test_data(M, seed=seed)

        # 2. Quantize
        weight_q = self.quantize_weight(weight_fp32)
        activation_q = self.quantize_activation(activation_fp32)

        # 3. Run kernel
        try:
            output = self.run_kernel(weight_q, activation_q, M)
        except NotImplementedError as e:
            return TestResult(
                name=f"{self.spec.name}_M{M}",
                passed=False,
                metrics=ErrorMetrics(0, 0, 0, 0, 0),
                message=str(e),
                spec_variant=self.spec.variant,
                M=M, N=self.spec.N, K=self.spec.K,
            )

        # 4. Compute reference
        output_ref = self.compute_reference(weight_fp32, activation_fp32)

        # 5. Check for NaN/Inf
        if torch.isnan(output).any():
            return TestResult(
                name=f"{self.spec.name}_M{M}",
                passed=False,
                metrics=ErrorMetrics(float('inf'), float('inf'), float('inf'), float('inf'), float('inf')),
                message="Output contains NaN",
                spec_variant=self.spec.variant,
                M=M, N=self.spec.N, K=self.spec.K,
            )

        if torch.isinf(output).any():
            return TestResult(
                name=f"{self.spec.name}_M{M}",
                passed=False,
                metrics=ErrorMetrics(float('inf'), float('inf'), float('inf'), float('inf'), float('inf')),
                message="Output contains Inf",
                spec_variant=self.spec.variant,
                M=M, N=self.spec.N, K=self.spec.K,
            )

        # 6. Compute metrics
        metrics = ErrorMetrics.compute(output, output_ref)
        passed = metrics.check(threshold)

        return TestResult(
            name=f"{self.spec.name}_M{M}",
            passed=passed,
            metrics=metrics,
            message="" if passed else f"NMSE {metrics.nmse:.2e} > threshold {threshold}",
            spec_variant=self.spec.variant,
            M=M, N=self.spec.N, K=self.spec.K,
        )

    def run_shape_test(self, M: int) -> TestResult:
        """Test that output shape is correct."""
        weight_fp32, activation_fp32 = self.create_test_data(M)
        weight_q = self.quantize_weight(weight_fp32)
        activation_q = self.quantize_activation(activation_fp32)

        try:
            output = self.run_kernel(weight_q, activation_q, M)
        except NotImplementedError as e:
            return TestResult(
                name=f"{self.spec.name}_shape_M{M}",
                passed=False,
                metrics=ErrorMetrics(0, 0, 0, 0, 0),
                message=str(e),
                spec_variant=self.spec.variant,
                M=M, N=self.spec.N, K=self.spec.K,
            )

        expected_shape = (M, self.spec.N)
        passed = output.shape == expected_shape and output.dtype == torch.float32

        return TestResult(
            name=f"{self.spec.name}_shape_M{M}",
            passed=passed,
            metrics=ErrorMetrics(0, 0, 0, 0, 0),
            message="" if passed else f"Shape {output.shape} != expected {expected_shape}",
            spec_variant=self.spec.variant,
            M=M, N=self.spec.N, K=self.spec.K,
        )

    def run_benchmark(
        self,
        M: int,
        warmup: int = 10,
        iterations: int = 100,
        seed: Optional[int] = 42,
    ) -> BenchmarkResult:
        """
        Run a performance benchmark.

        Args:
            M: Batch dimension
            warmup: Number of warmup iterations
            iterations: Number of timed iterations
            seed: Random seed

        Returns:
            BenchmarkResult with timing and GFLOPS
        """
        weight_fp32, activation_fp32 = self.create_test_data(M, seed=seed)
        weight_q = self.quantize_weight(weight_fp32)
        activation_q = self.quantize_activation(activation_fp32)

        kernel = self.get_kernel()
        N, K = self.spec.N, self.spec.K

        # Note: kernel expects (weight[N,K], activation[M,K], N, M, K)
        # because kernel computes C[kernel_M, kernel_N] = W @ A^T
        # and we want output[M, N] which requires kernel_M=N, kernel_N=M

        # Warmup
        for _ in range(warmup):
            _ = kernel(weight_q, activation_q, N, M, K)
        torch.cuda.synchronize()

        # Benchmark using CUDA events for accurate timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(iterations):
            _ = kernel(weight_q, activation_q, N, M, K)
        end_event.record()
        torch.cuda.synchronize()

        total_ms = start_event.elapsed_time(end_event)
        avg_ms = total_ms / iterations

        # Compute GFLOPS (2 * M * N * K for matmul)
        flops = 2 * M * N * K
        gflops = flops / (avg_ms / 1000) / 1e9

        return BenchmarkResult(
            name=self.spec.name,
            M=M, N=N, K=K,
            time_ms=avg_ms,
            gflops=gflops,
            iterations=iterations,
        )

    def is_kernel_available(self) -> bool:
        """Check if the kernel for this spec is available."""
        return GemmKernelRegistry.has_kernel(
            self.spec.weight_dtype,
            self.spec.activation_dtype
        )

    def get_missing_components(self) -> List[str]:
        """Get list of missing components (quantizers, kernels)."""
        missing = []

        w_dtype = self.spec.weight_dtype
        a_dtype = self.spec.activation_dtype

        if w_dtype not in ("float32", "float16"):
            if not GemmKernelRegistry.has_quantizer(w_dtype):
                missing.append(f"quantizer:{w_dtype}")

        if a_dtype not in ("float32", "float16"):
            if not GemmKernelRegistry.has_quantizer(a_dtype):
                missing.append(f"quantizer:{a_dtype}")

        if not GemmKernelRegistry.has_kernel(w_dtype, a_dtype):
            missing.append(f"kernel:{w_dtype}x{a_dtype}")

        return missing


class TestSuite:
    """
    Collection of tests for multiple specs or configurations.

    Usage:
        suite = TestSuite()
        suite.add_spec(spec1)
        suite.add_spec(spec2)

        results = suite.run_all(M_values=[1, 4, 128, 4096])
    """

    def __init__(self, specs: Optional[List[GemmTestSpec]] = None):
        self.specs = specs or []
        self.runners: Dict[str, GemmTestRunner] = {}

    def add_spec(self, spec: GemmTestSpec):
        """Add a test specification."""
        self.specs.append(spec)
        self.runners[spec.name] = GemmTestRunner(spec)

    def add_specs(self, specs: List[GemmTestSpec]):
        """Add multiple test specifications."""
        for spec in specs:
            self.add_spec(spec)

    def run_all(
        self,
        M_values: List[int] = [1, 4, 128],
        skip_unavailable: bool = True,
    ) -> List[TestResult]:
        """
        Run all tests for all specs and M values.

        Args:
            M_values: List of M dimensions to test
            skip_unavailable: Skip tests where kernel is not available

        Returns:
            List of TestResult
        """
        results = []

        for spec in self.specs:
            runner = self.runners.get(spec.name) or GemmTestRunner(spec)

            if skip_unavailable and not runner.is_kernel_available():
                # Create skip result
                missing = runner.get_missing_components()
                for M in M_values:
                    results.append(TestResult(
                        name=f"{spec.name}_M{M}",
                        passed=True,  # Skipped counts as pass
                        metrics=ErrorMetrics(0, 0, 0, 0, 0),
                        message=f"SKIPPED: {', '.join(missing)}",
                        spec_variant=spec.variant,
                        M=M, N=spec.N, K=spec.K,
                    ))
                continue

            for M in M_values:
                result = runner.run_test(M)
                results.append(result)

        return results

    def run_benchmarks(
        self,
        M_values: List[int] = [1, 4, 128, 4096],
        warmup: int = 10,
        iterations: int = 100,
        skip_unavailable: bool = True,
    ) -> List[BenchmarkResult]:
        """Run benchmarks for all specs."""
        results = []

        for spec in self.specs:
            runner = self.runners.get(spec.name) or GemmTestRunner(spec)

            if skip_unavailable and not runner.is_kernel_available():
                continue

            for M in M_values:
                try:
                    result = runner.run_benchmark(M, warmup=warmup, iterations=iterations)
                    results.append(result)
                except Exception as e:
                    print(f"Warning: Benchmark failed for {spec.name} M={M}: {e}")

        return results

    def print_summary(self, results: List[TestResult]):
        """Print a summary of test results."""
        passed = sum(1 for r in results if r.passed)
        failed = sum(1 for r in results if not r.passed and "SKIPPED" not in r.message)
        skipped = sum(1 for r in results if "SKIPPED" in r.message)

        print(f"\n{'='*60}")
        print(f"Test Summary: {passed} passed, {failed} failed, {skipped} skipped")
        print(f"{'='*60}")

        if failed > 0:
            print("\nFailed tests:")
            for r in results:
                if not r.passed and "SKIPPED" not in r.message:
                    print(f"  {r}")
