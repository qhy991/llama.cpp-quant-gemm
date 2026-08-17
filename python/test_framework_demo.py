#!/usr/bin/env python3
"""
Test script to verify the generic GEMM test framework.

This script tests:
1. Registry module functionality
2. Test framework JSON loading
3. Test runner execution
4. Integration with existing pybind module

Run with:
    conda activate KM-12.8
    cd /home/haiyan/Agent4Kernel/quant-gemm-from-scratch/python
    python test_framework_demo.py
"""

import sys
import traceback

def print_section(title):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def test_basic_imports():
    """Test that all modules can be imported."""
    print_section("1. Testing Basic Imports")

    try:
        import torch
        print(f"[OK] torch imported (version: {torch.__version__})")
        print(f"     CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"     CUDA device: {torch.cuda.get_device_name()}")
    except ImportError as e:
        print(f"[FAIL] torch import failed: {e}")
        return False

    try:
        import quant_gemm
        print(f"[OK] quant_gemm imported (version: {quant_gemm.__version__})")
    except ImportError as e:
        print(f"[FAIL] quant_gemm import failed: {e}")
        print("     Try: pip install -e .")
        return False

    try:
        from quant_gemm import registry
        print("[OK] quant_gemm.registry imported")
    except ImportError as e:
        print(f"[FAIL] registry import failed: {e}")
        return False

    try:
        from quant_gemm import test_framework
        print("[OK] quant_gemm.test_framework imported")
    except ImportError as e:
        print(f"[FAIL] test_framework import failed: {e}")
        return False

    try:
        from quant_gemm import test_runner
        print("[OK] quant_gemm.test_runner imported")
    except ImportError as e:
        print(f"[FAIL] test_runner import failed: {e}")
        return False

    return True


def test_registry():
    """Test registry functionality."""
    print_section("2. Testing Registry Module")

    from quant_gemm.registry import QuantTypeRegistry, GemmKernelRegistry

    # Test QuantTypeRegistry
    print("\n[QuantTypeRegistry]")
    types = QuantTypeRegistry.list_types()
    print(f"  Registered types: {types}")

    info = QuantTypeRegistry.get("block_q4_0")
    if info:
        print(f"  block_q4_0 info:")
        print(f"    - size_bytes: {info.size_bytes}")
        print(f"    - block_size: {info.block_size}")
        print(f"    - bits: {info.bits}")
        print(f"    - symmetric: {info.symmetric}")
    else:
        print("  [WARN] block_q4_0 not found in registry")

    # Test GemmKernelRegistry
    print("\n[GemmKernelRegistry]")
    kernels = GemmKernelRegistry.list_kernels()
    print(f"  Registered kernels: {kernels}")

    quantizers = GemmKernelRegistry.list_quantizers()
    print(f"  Registered quantizers: {quantizers}")

    # Check specific kernel
    has_q4_q8 = GemmKernelRegistry.has_kernel("block_q4_0", "block_q8_1")
    print(f"  Has Q4_0 x Q8_1 kernel: {has_q4_q8}")

    return True


def test_json_loading():
    """Test JSON spec loading."""
    print_section("3. Testing JSON Spec Loading")

    from quant_gemm.test_framework import (
        GemmTestSpec,
        discover_gemm_specs,
        get_definitions_dir,
    )

    # Find definitions directory
    try:
        defs_dir = get_definitions_dir()
        print(f"[OK] Definitions directory: {defs_dir}")
    except FileNotFoundError as e:
        print(f"[WARN] Definitions directory not found: {e}")
        print("       Skipping JSON loading tests")
        return True  # Not a failure, just no JSON files

    # Discover specs
    specs = discover_gemm_specs()
    print(f"[OK] Discovered {len(specs)} GEMM specs")

    for spec in specs:
        print(f"\n  Spec: {spec.name}")
        print(f"    Variant: {spec.variant}")
        print(f"    Weight dtype: {spec.weight_dtype}")
        print(f"    Activation dtype: {spec.activation_dtype}")
        print(f"    N={spec.N}, K={spec.K}")
        print(f"    Has reference code: {bool(spec.reference_code)}")

    return True


def test_low_level_api():
    """Test low-level quant_gemm API (original functions)."""
    print_section("4. Testing Low-Level API (Original pybind)")

    import torch
    import quant_gemm

    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return True

    # Test quantization
    print("\n[Quantization Test]")
    K = 1024
    x = torch.randn(4, K, device='cuda', dtype=torch.float32)
    print(f"  Input shape: {x.shape}, dtype: {x.dtype}")

    # Q4_0
    x_q4 = quant_gemm.quantize_q4_0(x)
    expected_q4_shape = (4, K // 32, 18)
    print(f"  Q4_0 output: {x_q4.shape}, expected: {expected_q4_shape}")
    assert x_q4.shape == expected_q4_shape, f"Shape mismatch!"
    print("  [OK] Q4_0 quantization")

    # Q8_1
    x_q8 = quant_gemm.quantize_q8_1(x)
    expected_q8_shape = (4, K // 32, 36)
    print(f"  Q8_1 output: {x_q8.shape}, expected: {expected_q8_shape}")
    assert x_q8.shape == expected_q8_shape, f"Shape mismatch!"
    print("  [OK] Q8_1 quantization")

    # Dequantization roundtrip
    print("\n[Dequantization Roundtrip Test]")
    x_dq = quant_gemm.dequantize_q4_0(x_q4, K)
    print(f"  Dequantized shape: {x_dq.shape}")
    max_err = (x - x_dq).abs().max().item()
    print(f"  Max roundtrip error: {max_err:.4f}")
    print("  [OK] Dequantization")

    # GEMM test
    print("\n[GEMM Test]")
    M, N, K = 128, 64, 1024
    weight = torch.randn(M, K, device='cuda', dtype=torch.float32)
    activation = torch.randn(N, K, device='cuda', dtype=torch.float32)

    weight_q = quant_gemm.quantize_q4_0(weight)
    activation_q = quant_gemm.quantize_q8_1(activation)

    output = quant_gemm.gemm_q4_0_q8_1(weight_q, activation_q, M, N, K)
    print(f"  Output shape: {output.shape}, expected: ({M}, {N})")
    assert output.shape == (M, N), "GEMM output shape mismatch!"

    # Compare with FP32 reference
    output_ref = weight @ activation.T
    mse = torch.mean((output - output_ref) ** 2).item()
    ref_power = torch.mean(output_ref ** 2).item()
    nmse = mse / ref_power if ref_power > 0 else 0

    print(f"  NMSE vs FP32: {nmse:.2e}")
    print(f"  Max error: {(output - output_ref).abs().max().item():.4f}")
    print("  [OK] GEMM execution")

    return True


def test_registry_api():
    """Test registry-based API."""
    print_section("5. Testing Registry API")

    import torch
    from quant_gemm.registry import (
        GemmKernelRegistry,
        quantize,
        get_kernel,
    )

    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return True

    # Check if quantizers are available
    if not GemmKernelRegistry.has_quantizer("block_q4_0"):
        print("[SKIP] Q4_0 quantizer not registered")
        return True

    # Test using registry functions
    print("\n[Using Registry Functions]")
    K = 1024
    x = torch.randn(4, K, device='cuda', dtype=torch.float32)

    # Quantize via registry
    x_q4 = quantize(x, "block_q4_0")
    print(f"  quantize(x, 'block_q4_0'): {x_q4.shape}")

    x_q8 = quantize(x, "block_q8_1")
    print(f"  quantize(x, 'block_q8_1'): {x_q8.shape}")

    # Get kernel via registry
    kernel = get_kernel("block_q4_0", "block_q8_1")
    if kernel:
        print(f"  get_kernel('block_q4_0', 'block_q8_1'): {kernel}")

        # Use kernel
        M, N, K = 64, 32, 1024
        w = torch.randn(M, K, device='cuda')
        a = torch.randn(N, K, device='cuda')
        w_q = quantize(w, "block_q4_0")
        a_q = quantize(a, "block_q8_1")

        output = kernel(w_q, a_q, M, N, K)
        print(f"  Kernel output shape: {output.shape}")
        print("  [OK] Registry API works")
    else:
        print("  [WARN] Kernel not found in registry")

    return True


def test_test_runner():
    """Test the test runner module."""
    print_section("6. Testing Test Runner")

    import torch
    from quant_gemm.test_framework import discover_gemm_specs
    from quant_gemm.test_runner import GemmTestRunner, ErrorMetrics

    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return True

    # Discover specs
    try:
        specs = discover_gemm_specs()
    except FileNotFoundError:
        print("[SKIP] No JSON definitions found")
        return True

    if not specs:
        print("[SKIP] No specs discovered")
        return True

    # Test first available spec
    for spec in specs:
        print(f"\n[Testing spec: {spec.name}]")

        runner = GemmTestRunner(spec)

        # Check kernel availability
        if not runner.is_kernel_available():
            missing = runner.get_missing_components()
            print(f"  [SKIP] Missing components: {missing}")
            continue

        # Run shape test
        print("  Running shape test...")
        shape_result = runner.run_shape_test(M=8)
        print(f"    Shape test: {'PASS' if shape_result.passed else 'FAIL'}")

        # Run correctness test
        print("  Running correctness test (M=32)...")
        result = runner.run_test(M=32)
        print(f"    Correctness: {'PASS' if result.passed else 'FAIL'}")
        print(f"    NMSE: {result.nmse:.2e}")
        print(f"    Max error: {result.max_error:.4f}")

        # Run small benchmark
        print("  Running benchmark (M=128)...")
        bench = runner.run_benchmark(M=128, warmup=5, iterations=20)
        print(f"    Time: {bench.time_ms:.3f} ms")
        print(f"    GFLOPS: {bench.gflops:.2f}")

        print("  [OK] Test runner works")
        break  # Only test first available spec

    return True


def test_pytest_discovery():
    """Test that pytest can discover tests."""
    print_section("7. Testing Pytest Discovery")

    import subprocess
    import os

    test_file = "tests/test_gemm_generic.py"
    if not os.path.exists(test_file):
        print(f"[SKIP] {test_file} not found")
        return True

    print(f"  Running: pytest {test_file} --collect-only")
    result = subprocess.run(
        ["pytest", test_file, "--collect-only", "-q"],
        capture_output=True,
        text=True,
        timeout=30,
    )

    print(f"  Return code: {result.returncode}")
    if result.stdout:
        lines = result.stdout.strip().split('\n')
        print(f"  Discovered tests: {len([l for l in lines if '::' in l])}")
        # Show first few
        for line in lines[:5]:
            print(f"    {line}")
        if len(lines) > 5:
            print(f"    ... and {len(lines) - 5} more")

    if result.stderr:
        print(f"  Stderr: {result.stderr[:200]}")

    return result.returncode == 0


def run_quick_pytest():
    """Run a quick pytest on a single test."""
    print_section("8. Running Quick Pytest")

    import subprocess
    import os

    test_file = "tests/test_gemm_generic.py"
    if not os.path.exists(test_file):
        print(f"[SKIP] {test_file} not found")
        return True

    # Run just the registry test (fast)
    print("  Running: pytest tests/test_gemm_generic.py::TestRegistry -v")
    result = subprocess.run(
        ["pytest", test_file + "::TestRegistry", "-v", "--tb=short"],
        capture_output=True,
        text=True,
        timeout=60,
    )

    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)

    if result.returncode != 0 and result.stderr:
        print(f"\nStderr:\n{result.stderr[-500:]}")

    return result.returncode == 0


def main():
    print("=" * 70)
    print(" QUANT_GEMM FRAMEWORK TEST")
    print("=" * 70)
    print(f"\nPython: {sys.executable}")
    print(f"Version: {sys.version}")

    results = {}

    # Run tests
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Registry Module", test_registry),
        ("JSON Loading", test_json_loading),
        ("Low-Level API", test_low_level_api),
        ("Registry API", test_registry_api),
        ("Test Runner", test_test_runner),
        ("Pytest Discovery", test_pytest_discovery),
        ("Quick Pytest", run_quick_pytest),
    ]

    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n[ERROR] {name} failed with exception:")
            traceback.print_exc()
            results[name] = False

    # Summary
    print_section("SUMMARY")

    passed = sum(1 for v in results.values() if v)
    failed = len(results) - passed

    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\n  Total: {passed}/{len(results)} passed")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
