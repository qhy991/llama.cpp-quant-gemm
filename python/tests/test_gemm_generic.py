"""
Generic GEMM test suite driven by JSON specifications.

This test file automatically discovers all GEMM test specs from JSON definitions
and runs correctness and shape tests for each. Tests are parameterized by:
1. JSON spec file (defining quant types, dimensions, etc.)
2. Batch size M

Usage:
    # Run all tests
    pytest tests/test_gemm_generic.py -v

    # Run only for specific variant
    pytest tests/test_gemm_generic.py -v -k "W4A8"

    # Run only small batch sizes
    pytest tests/test_gemm_generic.py -v -k "M1 or M4"

    # Run with benchmark markers
    pytest tests/test_gemm_generic.py -v -m benchmark

    # List available tests
    pytest tests/test_gemm_generic.py --collect-only
"""

import torch
import pytest
import sys
import os
from pathlib import Path
from typing import List, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_gemm.test_framework import (
    GemmTestSpec,
    discover_gemm_specs,
    get_definitions_dir,
    list_available_variants,
)
from quant_gemm.test_runner import (
    GemmTestRunner,
    TestResult,
    ErrorMetrics,
    TestSuite,
)
from quant_gemm.registry import (
    GemmKernelRegistry,
    QuantTypeRegistry,
)


# ============================================================================
# Test Discovery and Fixtures
# ============================================================================

def get_all_gemm_specs() -> List[GemmTestSpec]:
    """Discover all GEMM specs from definitions directory."""
    try:
        return discover_gemm_specs()
    except FileNotFoundError:
        # Fallback: return empty list if definitions not found
        return []


def get_spec_ids(specs: List[GemmTestSpec]) -> List[str]:
    """Generate test IDs from specs."""
    return [spec.name for spec in specs]


# Cache specs to avoid repeated file I/O
_CACHED_SPECS: Optional[List[GemmTestSpec]] = None


def load_specs() -> List[GemmTestSpec]:
    """Load and cache specs."""
    global _CACHED_SPECS
    if _CACHED_SPECS is None:
        _CACHED_SPECS = get_all_gemm_specs()
    return _CACHED_SPECS


# Batch sizes to test
BATCH_SIZES = [1, 4, 128]
LARGE_BATCH_SIZES = [1, 4, 128, 4096]


# ============================================================================
# Pytest Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def all_specs():
    """Fixture providing all discovered GEMM specs."""
    return load_specs()


@pytest.fixture(scope="module")
def available_kernels():
    """Fixture listing all available kernel type pairs."""
    return GemmKernelRegistry.list_kernels()


@pytest.fixture(scope="module")
def available_quantizers():
    """Fixture listing all available quantizer types."""
    return GemmKernelRegistry.list_quantizers()


# ============================================================================
# Helper Functions
# ============================================================================

def skip_if_no_cuda():
    """Skip test if CUDA is not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def skip_if_kernel_unavailable(spec: GemmTestSpec):
    """Skip test if kernel is not available for this spec."""
    if not GemmKernelRegistry.has_kernel(spec.weight_dtype, spec.activation_dtype):
        pytest.skip(
            f"Kernel not implemented: {spec.weight_dtype} x {spec.activation_dtype}"
        )


def get_runner(spec: GemmTestSpec) -> GemmTestRunner:
    """Get a test runner for the spec."""
    return GemmTestRunner(spec)


# ============================================================================
# Parameterized Test Generation
# ============================================================================

def pytest_generate_tests(metafunc):
    """Generate parameterized tests based on discovered specs."""

    # For tests that need gemm_spec fixture
    if "gemm_spec" in metafunc.fixturenames:
        specs = load_specs()

        if specs:
            metafunc.parametrize(
                "gemm_spec",
                specs,
                ids=get_spec_ids(specs),
            )
        else:
            # No specs found, create a dummy that will skip
            metafunc.parametrize("gemm_spec", [None], ids=["no_specs"])

    # For tests that need batch_size fixture
    if "batch_size" in metafunc.fixturenames:
        sizes = BATCH_SIZES
        # Use larger sizes for benchmark tests
        if "benchmark" in metafunc.function.__name__:
            sizes = LARGE_BATCH_SIZES
        metafunc.parametrize("batch_size", sizes, ids=[f"M{m}" for m in sizes])


# ============================================================================
# Test Classes
# ============================================================================

class TestGemmCorrectness:
    """Correctness tests for GEMM operations."""

    def test_gemm_correctness(self, gemm_spec, batch_size):
        """
        Test GEMM correctness by comparing against FP32 reference.

        For each (spec, batch_size) pair:
        1. Generate random FP32 data
        2. Quantize according to spec
        3. Run CUDA kernel
        4. Compare against FP32 matmul
        5. Check NMSE is below threshold
        """
        skip_if_no_cuda()

        if gemm_spec is None:
            pytest.skip("No test specs found in definitions directory")

        skip_if_kernel_unavailable(gemm_spec)

        runner = get_runner(gemm_spec)
        result = runner.run_test(M=batch_size)

        # Print result for visibility
        print(f"\n{result}")

        assert result.passed, result.message

    def test_gemm_shape(self, gemm_spec):
        """
        Test GEMM output shape correctness.

        Verifies:
        - Output shape is (M, N)
        - Output dtype is float32
        - Output is on CUDA
        """
        skip_if_no_cuda()

        if gemm_spec is None:
            pytest.skip("No test specs found")

        skip_if_kernel_unavailable(gemm_spec)

        runner = get_runner(gemm_spec)
        M = 8  # Fixed M for shape test

        weight_fp32, activation_fp32 = runner.create_test_data(M)
        weight_q = runner.quantize_weight(weight_fp32)
        activation_q = runner.quantize_activation(activation_fp32)

        output = runner.run_kernel(weight_q, activation_q, M)

        expected_shape = (M, gemm_spec.N)
        assert output.shape == expected_shape, \
            f"Shape mismatch: {output.shape} != {expected_shape}"
        assert output.dtype == torch.float32, \
            f"Dtype mismatch: {output.dtype} != float32"
        assert output.is_cuda, "Output should be on CUDA"

    def test_gemm_no_nan_inf(self, gemm_spec, batch_size):
        """Test that GEMM output contains no NaN or Inf values."""
        skip_if_no_cuda()

        if gemm_spec is None:
            pytest.skip("No test specs found")

        skip_if_kernel_unavailable(gemm_spec)

        runner = get_runner(gemm_spec)

        weight_fp32, activation_fp32 = runner.create_test_data(batch_size)
        weight_q = runner.quantize_weight(weight_fp32)
        activation_q = runner.quantize_activation(activation_fp32)

        output = runner.run_kernel(weight_q, activation_q, batch_size)

        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"


class TestQuantization:
    """Tests for quantization functions."""

    @pytest.fixture
    def sample_tensor(self):
        """Create a sample tensor for quantization tests."""
        skip_if_no_cuda()
        K = 1024
        return torch.randn(4, K, device='cuda')

    def test_q4_0_quantization(self, sample_tensor):
        """Test Q4_0 quantization shape and dtype."""
        quantizer = GemmKernelRegistry.get_quantizer("block_q4_0")
        if quantizer is None:
            pytest.skip("Q4_0 quantizer not available")

        K = sample_tensor.shape[-1]
        x_q = quantizer(sample_tensor)

        # Expected shape: (batch, K//32, 18)
        expected_shape = (sample_tensor.shape[0], K // 32, 18)
        assert x_q.shape == expected_shape, \
            f"Shape mismatch: {x_q.shape} != {expected_shape}"
        assert x_q.dtype == torch.uint8

    def test_q8_1_quantization(self, sample_tensor):
        """Test Q8_1 quantization shape and dtype."""
        quantizer = GemmKernelRegistry.get_quantizer("block_q8_1")
        if quantizer is None:
            pytest.skip("Q8_1 quantizer not available")

        K = sample_tensor.shape[-1]
        x_q = quantizer(sample_tensor)

        # Expected shape: (batch, K//32, 36)
        expected_shape = (sample_tensor.shape[0], K // 32, 36)
        assert x_q.shape == expected_shape, \
            f"Shape mismatch: {x_q.shape} != {expected_shape}"
        assert x_q.dtype == torch.uint8

    def test_q4_0_roundtrip(self, sample_tensor):
        """Test Q4_0 quantization roundtrip error."""
        quantizer = GemmKernelRegistry.get_quantizer("block_q4_0")
        dequantizer = GemmKernelRegistry.get_dequantizer("block_q4_0")

        if quantizer is None or dequantizer is None:
            pytest.skip("Q4_0 quantizer/dequantizer not available")

        K = sample_tensor.shape[-1]
        x_q = quantizer(sample_tensor)
        x_dq = dequantizer(x_q, K)

        # Check shape preserved
        assert x_dq.shape == sample_tensor.shape

        # Check quantization error is bounded (4-bit has significant error)
        max_error = (sample_tensor - x_dq).abs().max().item()
        assert max_error < 2.0, f"Roundtrip error too large: {max_error}"


class TestRegistry:
    """Tests for registry functionality."""

    def test_quant_type_registry(self):
        """Test that quantization types are registered."""
        types = QuantTypeRegistry.list_types()
        assert "block_q4_0" in types
        assert "block_q8_1" in types

    def test_quant_type_info(self):
        """Test quantization type info retrieval."""
        info = QuantTypeRegistry.get("block_q4_0")
        assert info is not None
        assert info.size_bytes == 18
        assert info.block_size == 32
        assert info.bits == 4

    def test_kernel_registry_initialization(self):
        """Test that kernel registry initializes from _C module."""
        skip_if_no_cuda()

        # Force re-initialization
        GemmKernelRegistry._initialized = False
        GemmKernelRegistry._init_from_c_module()

        kernels = GemmKernelRegistry.list_kernels()
        print(f"\nAvailable kernels: {kernels}")

        # At minimum, Q4_0 x Q8_1 should be available
        assert ("block_q4_0", "block_q8_1") in kernels, \
            "Expected Q4_0 x Q8_1 kernel to be registered"


class TestSpecLoading:
    """Tests for spec loading and discovery."""

    def test_discover_specs(self):
        """Test that specs can be discovered from definitions directory."""
        try:
            specs = discover_gemm_specs()
            print(f"\nDiscovered {len(specs)} specs:")
            for spec in specs:
                print(f"  - {spec.name} ({spec.variant})")

            assert len(specs) > 0, "Expected at least one spec"
        except FileNotFoundError:
            pytest.skip("Definitions directory not found")

    def test_spec_properties(self, all_specs):
        """Test that spec properties are accessible."""
        if not all_specs:
            pytest.skip("No specs available")

        for spec in all_specs:
            # These should not raise
            _ = spec.N
            _ = spec.K
            _ = spec.weight_dtype
            _ = spec.activation_dtype
            _ = spec.block_size

    def test_list_variants(self):
        """Test listing available variants."""
        try:
            variants = list_available_variants()
            print(f"\nAvailable variants: {variants}")
            # Expect at least W4A8 variant
            # (may vary depending on what's defined)
        except FileNotFoundError:
            pytest.skip("Definitions directory not found")


@pytest.mark.benchmark
class TestPerformance:
    """Performance benchmark tests."""

    def test_gemm_benchmark(self, gemm_spec, batch_size):
        """
        Benchmark GEMM performance.

        Reports:
        - Average time per iteration
        - GFLOPS achieved
        """
        skip_if_no_cuda()

        if gemm_spec is None:
            pytest.skip("No test specs found")

        skip_if_kernel_unavailable(gemm_spec)

        runner = get_runner(gemm_spec)

        # Run benchmark
        result = runner.run_benchmark(
            M=batch_size,
            warmup=10,
            iterations=50,  # Fewer iterations for test speed
        )

        print(f"\n{result}")

        # Basic sanity checks
        assert result.time_ms > 0
        assert result.gflops > 0


# ============================================================================
# Test Suite Runner (for direct execution)
# ============================================================================

def run_all_tests():
    """Run all tests programmatically."""
    print("=" * 60)
    print("GEMM Generic Test Suite")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return False

    print(f"CUDA device: {torch.cuda.get_device_name()}")

    # Discover specs
    try:
        specs = discover_gemm_specs()
        print(f"\nDiscovered {len(specs)} specs:")
        for spec in specs:
            print(f"  - {spec.name}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return False

    # Create test suite
    suite = TestSuite(specs)

    # Run tests
    print("\n" + "=" * 60)
    print("Running Correctness Tests")
    print("=" * 60)

    results = suite.run_all(M_values=[1, 4, 128])

    # Print results
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        if "SKIPPED" in result.message:
            status = "SKIP"
        print(f"[{status}] {result.name}: NMSE={result.nmse:.2e}")

    suite.print_summary(results)

    # Run benchmarks for available kernels
    print("\n" + "=" * 60)
    print("Running Benchmarks")
    print("=" * 60)

    bench_results = suite.run_benchmarks(M_values=[4, 128, 4096])
    for result in bench_results:
        print(result)

    # Return success status
    failed = sum(1 for r in results if not r.passed and "SKIPPED" not in r.message)
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
