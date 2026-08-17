#!/usr/bin/env python3
"""
Test script for the universal operators framework.

This script demonstrates:
1. Operator discovery from directory structure
2. Registry initialization with pybind module
3. Running tests for all available operators
4. Benchmarking

Usage:
    conda activate KM-12.8
    cd /home/haiyan/Agent4Kernel/quant-gemm-from-scratch/python
    python test_operators_framework.py
"""

import sys
import traceback
from pathlib import Path

# Add operators to path
sys.path.insert(0, str(Path(__file__).parent))


def print_section(title):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def test_imports():
    """Test that all modules can be imported."""
    print_section("1. Testing Imports")

    try:
        import torch
        print(f"[OK] torch {torch.__version__}")
        print(f"     CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"     Device: {torch.cuda.get_device_name()}")
    except ImportError as e:
        print(f"[FAIL] torch: {e}")
        return False

    try:
        from operators import (
            OperatorRegistry,
            TestFramework,
            OperatorSpec,
            TensorSpec,
        )
        print("[OK] operators framework imported")
    except ImportError as e:
        print(f"[FAIL] operators: {e}")
        traceback.print_exc()
        return False

    try:
        from operators.common.types import (
            QuantType,
            QUANT_TYPES,
            get_quant_type,
        )
        print(f"[OK] common.types: {len(QUANT_TYPES)} quant types")
    except ImportError as e:
        print(f"[FAIL] common.types: {e}")
        return False

    return True


def test_quant_types():
    """Test quantization type definitions."""
    print_section("2. Testing Quant Types")

    from operators.common.types import (
        QUANT_TYPES,
        get_quant_type,
        get_block_size,
        get_block_bytes,
    )

    print("\nRegistered quantization types:")
    for name, qtype in QUANT_TYPES.items():
        print(f"  {name}:")
        print(f"    size_bytes={qtype.size_bytes}, block_size={qtype.block_size}, bits={qtype.bits}")

    # Test specific type
    q4_0 = get_quant_type("block_q4_0")
    assert q4_0 is not None
    assert q4_0.size_bytes == 18
    assert q4_0.block_size == 32
    print("\n[OK] Q4_0 type verified")

    # Test shape calculation
    shape = q4_0.get_quantized_shape((4, 1024))
    expected = (4, 32, 18)  # 1024/32 = 32 blocks
    assert shape == expected, f"Shape {shape} != expected {expected}"
    print(f"[OK] Shape calculation: (4, 1024) -> {shape}")

    return True


def test_registry_discovery():
    """Test operator discovery."""
    print_section("3. Testing Registry Discovery")

    from operators import OperatorRegistry

    registry = OperatorRegistry()
    registry.discover()

    print(f"\nDiscovered families: {registry.list_families()}")
    print(f"Discovered specs: {registry.list_specs()}")

    for family_name in registry.list_families():
        family = registry.get_family(family_name)
        print(f"\n  Family: {family_name}")
        print(f"    Variants: {family.list_variants()}")

    for spec_name in registry.list_specs():
        spec = registry.get_spec(spec_name)
        print(f"\n  Spec: {spec_name}")
        print(f"    Inputs: {list(spec.inputs.keys())}")
        print(f"    Outputs: {list(spec.outputs.keys())}")
        print(f"    Params: {list(spec.params.keys())}")
        print(f"    Test configs: {len(spec.test_configs)}")

    return True


def test_register_from_pybind():
    """Test registering kernels from pybind module."""
    print_section("4. Testing Pybind Registration")

    import torch
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return True

    from operators import OperatorRegistry

    registry = OperatorRegistry()
    registry.discover()

    try:
        # Import the existing quant_gemm module
        import quant_gemm._C as _C

        # Register from module
        registry.register_from_module(_C)

        print(f"Registered kernels: {registry.list_kernels()}")
        print(f"Registered quantizers: {registry.list_quantizers()}")

        # Verify registration
        assert registry.has_quantizer("block_q4_0"), "Q4_0 quantizer not registered"
        assert registry.has_quantizer("block_q8_1"), "Q8_1 quantizer not registered"
        print("[OK] Quantizers registered")

        # Check kernel
        kernel = registry.get_kernel("gemm_q4_0_q8_1")
        assert kernel is not None, "Q4_0 x Q8_1 kernel not registered"
        print("[OK] Kernel registered")

        return True

    except ImportError as e:
        print(f"[WARN] Could not import quant_gemm._C: {e}")
        print("       Try: python setup.py build_ext --inplace")
        return True  # Not a failure, just not compiled


def test_framework_run():
    """Test running tests through the framework."""
    print_section("5. Testing Framework Execution")

    import torch
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return True

    from operators import OperatorRegistry, TestFramework

    # Setup registry
    registry = OperatorRegistry()
    registry.discover()

    try:
        import quant_gemm._C as _C
        registry.register_from_module(_C)
    except ImportError:
        print("[SKIP] quant_gemm not compiled")
        return True

    # Create framework
    framework = TestFramework(registry)

    # Get available specs
    available = registry.get_available_specs()
    print(f"\nAvailable specs: {[s.name for s in available]}")

    if not available:
        print("[SKIP] No operators available")
        return True

    # Run tests for first available spec
    spec = available[0]
    print(f"\nRunning tests for: {spec.name}")

    results = framework.run_spec(spec)
    framework.print_results(results)

    # Check results
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"\n[{'OK' if passed == total else 'PARTIAL'}] {passed}/{total} tests passed")

    return passed > 0


def test_benchmark():
    """Test benchmarking."""
    print_section("6. Testing Benchmarks")

    import torch
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return True

    from operators import OperatorRegistry, TestFramework
    from operators.base import TestConfig

    registry = OperatorRegistry()
    registry.discover()

    try:
        import quant_gemm._C as _C
        registry.register_from_module(_C)
    except ImportError:
        print("[SKIP] quant_gemm not compiled")
        return True

    framework = TestFramework(registry)
    available = registry.get_available_specs()

    if not available:
        print("[SKIP] No operators available")
        return True

    spec = available[0]
    print(f"\nBenchmarking: {spec.name}")

    # Custom test configs for benchmark
    bench_configs = [
        TestConfig(name="M1", params={"M": 1, "N": 4096, "K": 4096}),
        TestConfig(name="M128", params={"M": 128, "N": 4096, "K": 4096}),
        TestConfig(name="M4096", params={"M": 4096, "N": 4096, "K": 4096}),
    ]

    results = framework.benchmark_spec(
        spec,
        configs=bench_configs,
        warmup=10,
        iterations=50,
    )

    framework.print_benchmarks(results)

    return len(results) > 0


def test_run_all():
    """Test running all operators."""
    print_section("7. Running All Available Tests")

    import torch
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return True

    from operators import OperatorRegistry, TestFramework

    registry = OperatorRegistry()
    registry.discover()

    try:
        import quant_gemm._C as _C
        registry.register_from_module(_C)
    except ImportError:
        print("[SKIP] quant_gemm not compiled")
        return True

    framework = TestFramework(registry)

    print("\nRegistry status:")
    registry.print_status()

    print("\nRunning all tests...")
    results = framework.run_all(skip_unavailable=True)
    framework.print_results(results)

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed and not r.skipped)
    skipped = sum(1 for r in results if r.skipped)

    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")

    return failed == 0


def main():
    print("=" * 70)
    print(" OPERATORS FRAMEWORK TEST")
    print("=" * 70)
    print(f"\nPython: {sys.executable}")
    print(f"Version: {sys.version.split()[0]}")

    tests = [
        ("Imports", test_imports),
        ("Quant Types", test_quant_types),
        ("Registry Discovery", test_registry_discovery),
        ("Pybind Registration", test_register_from_pybind),
        ("Framework Run", test_framework_run),
        ("Benchmark", test_benchmark),
        ("Run All", test_run_all),
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n[ERROR] {name} failed:")
            traceback.print_exc()
            results[name] = False

    # Summary
    print_section("SUMMARY")
    passed = sum(1 for v in results.values() if v)
    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\n  Total: {passed}/{len(results)} passed")

    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
