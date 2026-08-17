#!/usr/bin/env python3
"""
Example: Running generic GEMM tests from JSON specifications.

This script demonstrates how to use the quant_gemm test framework to:
1. Discover test specs from JSON definitions
2. Run correctness tests
3. Run benchmarks
4. Generate test reports

Usage:
    cd /home/haiyan/Agent4Kernel/quant-gemm-from-scratch/python
    python examples/run_generic_tests.py

    # With specific variant filter
    python examples/run_generic_tests.py --variant W4A8

    # With custom batch sizes
    python examples/run_generic_tests.py --batch-sizes 1,4,128,4096
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch


def check_cuda():
    """Check CUDA availability."""
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        print("This framework requires a CUDA-capable GPU.")
        sys.exit(1)
    print(f"CUDA Device: {torch.cuda.get_device_name()}")
    print(f"CUDA Version: {torch.version.cuda}")


def run_discovery_demo():
    """Demonstrate spec discovery."""
    print("\n" + "=" * 70)
    print("1. DISCOVERING TEST SPECIFICATIONS")
    print("=" * 70)

    from quant_gemm.test_framework import (
        discover_gemm_specs,
        list_available_variants,
        get_definitions_dir,
    )

    # Find definitions directory
    try:
        defs_dir = get_definitions_dir()
        print(f"\nDefinitions directory: {defs_dir}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return []

    # List variants
    variants = list_available_variants()
    print(f"\nAvailable variants: {variants}")

    # Discover all specs
    specs = discover_gemm_specs()
    print(f"\nDiscovered {len(specs)} test specifications:")
    for spec in specs:
        print(f"  - {spec.name}")
        print(f"      Variant: {spec.variant}")
        print(f"      Weight: {spec.weight_dtype}, Activation: {spec.activation_dtype}")
        print(f"      Dimensions: N={spec.N}, K={spec.K}")

    return specs


def run_registry_demo():
    """Demonstrate registry usage."""
    print("\n" + "=" * 70)
    print("2. REGISTRY DEMO")
    print("=" * 70)

    from quant_gemm.registry import (
        QuantTypeRegistry,
        GemmKernelRegistry,
    )

    # List quantization types
    print("\nRegistered quantization types:")
    for type_name in QuantTypeRegistry.list_types():
        info = QuantTypeRegistry.get(type_name)
        print(f"  - {type_name}: {info.size_bytes} bytes/block, {info.bits} bits")

    # List available kernels
    print("\nRegistered GEMM kernels:")
    for w_type, a_type in GemmKernelRegistry.list_kernels():
        print(f"  - {w_type} x {a_type}")

    # List quantizers
    print("\nRegistered quantizers:")
    for dtype in GemmKernelRegistry.list_quantizers():
        print(f"  - {dtype}")


def run_correctness_tests(specs, batch_sizes):
    """Run correctness tests for all specs."""
    print("\n" + "=" * 70)
    print("3. CORRECTNESS TESTS")
    print("=" * 70)

    from quant_gemm.test_runner import GemmTestRunner, TestSuite

    suite = TestSuite(specs)
    results = suite.run_all(M_values=batch_sizes)

    # Print results
    print(f"\nResults for batch sizes {batch_sizes}:")
    print("-" * 70)

    for result in results:
        if "SKIPPED" in result.message:
            status = "SKIP"
            color = "\033[33m"  # Yellow
        elif result.passed:
            status = "PASS"
            color = "\033[32m"  # Green
        else:
            status = "FAIL"
            color = "\033[31m"  # Red

        reset = "\033[0m"
        print(f"[{color}{status}{reset}] {result.name}: NMSE={result.nmse:.2e} {result.message}")

    # Summary
    suite.print_summary(results)

    return results


def run_benchmarks(specs, batch_sizes):
    """Run performance benchmarks."""
    print("\n" + "=" * 70)
    print("4. PERFORMANCE BENCHMARKS")
    print("=" * 70)

    from quant_gemm.test_runner import TestSuite

    suite = TestSuite(specs)
    results = suite.run_benchmarks(M_values=batch_sizes, warmup=10, iterations=50)

    if not results:
        print("\nNo benchmarks completed (kernels may not be available)")
        return

    print(f"\nBenchmark results:")
    print("-" * 70)
    print(f"{'Name':<40} {'M':>6} {'Time(ms)':>10} {'GFLOPS':>10}")
    print("-" * 70)

    for result in results:
        print(f"{result.name:<40} {result.M:>6} {result.time_ms:>10.3f} {result.gflops:>10.2f}")


def run_single_spec_demo(spec):
    """Demonstrate testing a single spec in detail."""
    print("\n" + "=" * 70)
    print(f"5. DETAILED TEST: {spec.name}")
    print("=" * 70)

    from quant_gemm.test_runner import GemmTestRunner

    runner = GemmTestRunner(spec)

    # Check if kernel is available
    if not runner.is_kernel_available():
        missing = runner.get_missing_components()
        print(f"\nKernel not available. Missing components: {missing}")
        return

    print(f"\nSpec details:")
    print(f"  Name: {spec.name}")
    print(f"  Variant: {spec.variant}")
    print(f"  Weight type: {spec.weight_dtype}")
    print(f"  Activation type: {spec.activation_dtype}")
    print(f"  Dimensions: N={spec.N}, K={spec.K}")
    print(f"  Formula: {spec.formula.get('dot_product', 'N/A')}")

    # Test with different batch sizes
    print("\nTesting with various batch sizes:")
    for M in [1, 4, 32, 128]:
        result = runner.run_test(M=M)
        print(f"  M={M:4d}: NMSE={result.nmse:.2e}, max_err={result.max_error:.4f}")

    # Benchmark
    print("\nBenchmark:")
    bench = runner.run_benchmark(M=128, warmup=10, iterations=100)
    print(f"  M=128: {bench.time_ms:.3f} ms, {bench.gflops:.2f} GFLOPS")


def main():
    parser = argparse.ArgumentParser(
        description="Run generic GEMM tests from JSON specifications"
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Filter by variant (e.g., W4A8, W4A16)"
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,4,128",
        help="Comma-separated batch sizes to test"
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Run only benchmarks, skip correctness tests"
    )

    args = parser.parse_args()

    # Parse batch sizes
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]

    print("=" * 70)
    print("QUANT_GEMM GENERIC TEST FRAMEWORK DEMO")
    print("=" * 70)

    # Check CUDA
    check_cuda()

    # Discover specs
    specs = run_discovery_demo()
    if not specs:
        print("No specs found, exiting")
        return

    # Filter by variant if specified
    if args.variant:
        specs = [s for s in specs if s.variant == args.variant]
        print(f"\nFiltered to {len(specs)} specs with variant={args.variant}")

    # Registry demo
    run_registry_demo()

    # Run tests
    if not args.benchmark_only:
        run_correctness_tests(specs, batch_sizes)

    # Run benchmarks
    run_benchmarks(specs, batch_sizes)

    # Detailed demo for first spec
    if specs:
        run_single_spec_demo(specs[0])

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
