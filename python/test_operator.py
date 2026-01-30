#!/usr/bin/env python3
"""
Simple operator testing tool.

Usage:
    python test_operator.py <name> <folder_path> [options]

Examples:
    # Test a specific operator variant
    python test_operator.py w4a8_q4_0_q8_1 operators/quant_gemm/variants/w4a8_q4_0_q8_1

    # Test with benchmark
    python test_operator.py w4a8_q4_0_q8_1 operators/quant_gemm/variants/w4a8_q4_0_q8_1 --benchmark

    # Test with custom pybind module
    python test_operator.py w4a8_q4_0_q8_1 operators/quant_gemm/variants/w4a8_q4_0_q8_1 --module quant_gemm._C
"""

import argparse
import importlib
import json
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import torch


@dataclass
class TestResult:
    name: str
    passed: bool
    metric_name: str
    metric_value: float
    threshold: float
    error: Optional[str] = None


def load_spec(folder_path: Path) -> dict:
    """Load spec.json from folder."""
    spec_file = folder_path / "spec.json"
    if not spec_file.exists():
        raise FileNotFoundError(f"spec.json not found in {folder_path}")

    with open(spec_file) as f:
        return json.load(f)


def run_reference(raw_inputs: Dict[str, torch.Tensor], params: dict) -> torch.Tensor:
    """
    Run FP32 reference implementation.

    For GEMM: output[M, N] = activation[M, K] @ weight[N, K].T
    """
    weight = raw_inputs.get("weight")
    activation = raw_inputs.get("activation")

    if weight is not None and activation is not None:
        return torch.matmul(activation, weight.T)

    raise RuntimeError("No reference implementation available")


def get_quantizer(module, dtype: str):
    """Get quantizer function from pybind module."""
    quantizer_map = {
        "block_q4_0": "quantize_q4_0",
        "block_q4_1": "quantize_q4_1",
        "block_q8_0": "quantize_q8_0",
        "block_q8_1": "quantize_q8_1",
    }

    func_name = quantizer_map.get(dtype)
    if func_name and hasattr(module, func_name):
        return getattr(module, func_name)
    return None


def get_kernel(module, spec: dict):
    """Get kernel function from pybind module."""
    # Try spec-defined kernel name
    kernel_info = spec.get("kernel", {})
    kernel_name = kernel_info.get("entry_point", spec["name"])

    # Try common naming patterns
    candidates = [
        kernel_name,
        f"gemm_{spec['name']}",
        spec["name"],
    ]

    # Also try based on input types
    inputs = spec.get("inputs", {})
    weight_dtype = inputs.get("weight", {}).get("dtype", "")
    act_dtype = inputs.get("activation", {}).get("dtype", "")

    if weight_dtype and act_dtype:
        # e.g., gemm_q4_0_q8_1
        w_short = weight_dtype.replace("block_", "")
        a_short = act_dtype.replace("block_", "").replace("float32", "fp32")
        candidates.append(f"gemm_{w_short}_{a_short}")

    for name in candidates:
        if hasattr(module, name):
            return getattr(module, name)

    return None


def generate_inputs(spec: dict, params: dict, device: str = "cuda") -> Dict[str, torch.Tensor]:
    """Generate test inputs based on spec."""
    inputs = {}
    M, N, K = params["M"], params["N"], params["K"]

    for name, tensor_info in spec["inputs"].items():
        dtype = tensor_info["dtype"]

        # Generate FP32 tensor first
        if "weight" in name.lower():
            shape = (N, K)
        elif "activation" in name.lower():
            shape = (M, K)
        else:
            # Parse shape from spec
            shape = []
            for dim in tensor_info["shape"]:
                if isinstance(dim, int):
                    shape.append(dim)
                elif dim == "M":
                    shape.append(M)
                elif dim == "N":
                    shape.append(N)
                elif dim == "K":
                    shape.append(K)
                elif "/" in str(dim):
                    parts = dim.split("/")
                    val = params.get(parts[0], M if parts[0] == "M" else (N if parts[0] == "N" else K))
                    shape.append(val // int(parts[1]))
            shape = tuple(shape)

        # Generate tensor
        if dtype == "float32":
            inputs[name] = torch.randn(shape, dtype=torch.float32, device=device)
        elif dtype == "float16":
            inputs[name] = torch.randn(shape, dtype=torch.float16, device=device)
        else:
            # Quantized types - generate FP32 first
            inputs[name] = torch.randn(shape, dtype=torch.float32, device=device)

    return inputs


def quantize_inputs(inputs: Dict[str, torch.Tensor], spec: dict, module) -> Dict[str, torch.Tensor]:
    """Quantize inputs according to spec."""
    quantized = {}

    for name, tensor in inputs.items():
        tensor_info = spec["inputs"][name]
        dtype = tensor_info["dtype"]

        if dtype.startswith("block_q"):
            quantizer = get_quantizer(module, dtype)
            if quantizer is None:
                raise RuntimeError(f"Quantizer not found for {dtype}")
            quantized[name] = quantizer(tensor)
        else:
            quantized[name] = tensor

    return quantized


def compute_nmse(output: torch.Tensor, reference: torch.Tensor) -> float:
    """Compute Normalized Mean Squared Error."""
    output = output.float().flatten()
    reference = reference.float().flatten()

    mse = torch.mean((output - reference) ** 2).item()
    ref_var = torch.var(reference).item()

    if ref_var < 1e-10:
        return 0.0 if mse < 1e-10 else float('inf')

    return mse / ref_var


def run_test(
    spec: dict,
    folder_path: Path,
    module,
    config: dict,
    device: str = "cuda"
) -> TestResult:
    """Run a single test configuration."""
    name = config.get("name", "test")
    params = {k: v for k, v in config.items() if k != "name"}

    # Fill in defaults
    for param_name, param_info in spec.get("params", {}).items():
        if param_name not in params and "default" in param_info:
            params[param_name] = param_info["default"]

    try:
        # Generate inputs
        raw_inputs = generate_inputs(spec, params, device)

        # Run reference (FP32 matmul)
        ref_output = run_reference(raw_inputs, params)
        if ref_output.device.type != device:
            ref_output = ref_output.to(device)

        # Quantize inputs
        quantized_inputs = quantize_inputs(raw_inputs, spec, module)

        # Get kernel
        kernel = get_kernel(module, spec)
        if kernel is None:
            return TestResult(
                name=name,
                passed=False,
                metric_name="nmse",
                metric_value=float('inf'),
                threshold=0.0,
                error="Kernel not found"
            )

        # Run kernel (handle GEMM dimension convention)
        M, N, K = params["M"], params["N"], params["K"]
        weight_q = quantized_inputs["weight"]
        activation_q = quantized_inputs["activation"]

        # Check if activation is quantized to determine calling convention
        act_dtype = spec["inputs"]["activation"]["dtype"]
        is_quantized_activation = act_dtype.startswith("block_q")

        if is_quantized_activation:
            # w4a8: Kernel uses C[kernel_M, kernel_N] = W[kernel_M, K] @ A[kernel_N, K]^T
            # Where kernel_M = N (output features), kernel_N = M (batch)
            output = kernel(weight_q, activation_q, N, M, K)
            output = output.T  # Transpose to get [M, N]
        else:
            # w4a16: Kernel uses C[M, N] = A[M, K] @ W[N, K]^T
            # Standard convention: M = batch, N = output features
            output = kernel(weight_q, activation_q, M, N, K)

        # Compute accuracy
        accuracy_spec = spec.get("accuracy", {"metric": "nmse", "threshold": 0.1})
        metric_name = accuracy_spec.get("metric", "nmse")
        threshold = accuracy_spec.get("threshold", 0.1)

        if metric_name == "nmse":
            metric_value = compute_nmse(output, ref_output)
        else:
            metric_value = compute_nmse(output, ref_output)

        passed = metric_value <= threshold

        return TestResult(
            name=name,
            passed=passed,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=threshold
        )

    except Exception as e:
        return TestResult(
            name=name,
            passed=False,
            metric_name="nmse",
            metric_value=float('inf'),
            threshold=0.0,
            error=str(e)
        )


def run_benchmark(
    spec: dict,
    folder_path: Path,
    module,
    config: dict,
    warmup: int = 10,
    iterations: int = 100,
    device: str = "cuda"
) -> dict:
    """Run benchmark for a configuration."""
    params = {k: v for k, v in config.items() if k != "name"}

    # Fill in defaults
    for param_name, param_info in spec.get("params", {}).items():
        if param_name not in params and "default" in param_info:
            params[param_name] = param_info["default"]

    M, N, K = params["M"], params["N"], params["K"]

    # Generate and quantize inputs
    raw_inputs = generate_inputs(spec, params, device)
    quantized_inputs = quantize_inputs(raw_inputs, spec, module)

    kernel = get_kernel(module, spec)
    weight_q = quantized_inputs["weight"]
    activation_q = quantized_inputs["activation"]

    # Check if activation is quantized to determine calling convention
    act_dtype = spec["inputs"]["activation"]["dtype"]
    is_quantized_activation = act_dtype.startswith("block_q")

    # Warmup
    for _ in range(warmup):
        if is_quantized_activation:
            _ = kernel(weight_q, activation_q, N, M, K)
        else:
            _ = kernel(weight_q, activation_q, M, N, K)
    torch.cuda.synchronize()

    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iterations):
        if is_quantized_activation:
            _ = kernel(weight_q, activation_q, N, M, K)
        else:
            _ = kernel(weight_q, activation_q, M, N, K)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end) / iterations

    # Compute FLOPS
    flops = 2 * M * N * K
    gflops = flops / (elapsed_ms * 1e-3) / 1e9

    return {
        "name": config.get("name", "bench"),
        "M": M,
        "N": N,
        "K": K,
        "time_ms": elapsed_ms,
        "gflops": gflops,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test an operator from spec.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python test_operator.py w4a8_q4_0_q8_1 operators/quant_gemm/variants/w4a8_q4_0_q8_1
    python test_operator.py w4a8_q4_0_q8_1 operators/quant_gemm/variants/w4a8_q4_0_q8_1 --benchmark
    python test_operator.py w4a8_q4_0_q8_1 operators/quant_gemm/variants/w4a8_q4_0_q8_1 --module quant_gemm._C
        """
    )

    parser.add_argument("name", help="Operator name (from spec.json)")
    parser.add_argument("folder", help="Path to operator folder containing spec.json")
    parser.add_argument("--module", "-m", default="quant_gemm._C",
                        help="Pybind module path (default: quant_gemm._C)")
    parser.add_argument("--benchmark", "-b", action="store_true",
                        help="Run benchmarks")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Benchmark warmup iterations")
    parser.add_argument("--iterations", "-i", type=int, default=100,
                        help="Benchmark iterations")
    parser.add_argument("--config", "-c", action="append",
                        help="Custom config in format 'M=1,N=4096,K=4096'")
    parser.add_argument("--device", "-d", default="cuda",
                        help="Device to use (default: cuda)")

    args = parser.parse_args()

    # Validate inputs
    folder_path = Path(args.folder)
    if not folder_path.exists():
        print(f"Error: Folder not found: {folder_path}")
        sys.exit(1)

    # Load spec
    try:
        spec = load_spec(folder_path)
    except Exception as e:
        print(f"Error loading spec: {e}")
        sys.exit(1)

    # Validate name
    if spec.get("name") != args.name:
        print(f"Warning: Spec name '{spec.get('name')}' != argument '{args.name}'")

    # Load pybind module
    try:
        module = importlib.import_module(args.module)
    except ImportError as e:
        print(f"Error importing module {args.module}: {e}")
        print("Try: python setup.py build_ext --inplace")
        sys.exit(1)

    # Check CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Error: CUDA not available")
        sys.exit(1)

    print("=" * 60)
    print(f" Testing: {args.name}")
    print("=" * 60)
    print(f"Folder: {folder_path}")
    print(f"Module: {args.module}")
    print(f"Device: {args.device}")

    # Parse custom configs
    configs = []
    if args.config:
        for cfg_str in args.config:
            cfg = {"name": f"custom_{len(configs)}"}
            for part in cfg_str.split(","):
                k, v = part.split("=")
                cfg[k.strip()] = int(v.strip())
            configs.append(cfg)
    else:
        configs = spec.get("test_configs", [])

    if not configs:
        configs = [{"name": "default", "M": 1, "N": 4096, "K": 4096}]

    print(f"\nConfigs: {len(configs)}")

    # Run tests
    print("\n" + "-" * 60)
    print(" Correctness Tests")
    print("-" * 60)

    results = []
    for config in configs:
        result = run_test(spec, folder_path, module, config, args.device)
        results.append(result)

        status = "PASS" if result.passed else "FAIL"
        if result.error:
            print(f"[{status}] {result.name}: {result.error}")
        else:
            print(f"[{status}] {result.name}: {result.metric_name}={result.metric_value:.4e} (threshold={result.threshold})")

    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed

    print(f"\nResults: {passed} passed, {failed} failed")

    # Run benchmarks
    if args.benchmark:
        print("\n" + "-" * 60)
        print(" Benchmarks")
        print("-" * 60)

        bench_results = []
        for config in configs:
            try:
                result = run_benchmark(
                    spec, folder_path, module, config,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    device=args.device
                )
                bench_results.append(result)
                print(f"{result['name']:20s} M={result['M']:5d} N={result['N']:5d} K={result['K']:5d} "
                      f"| {result['time_ms']:8.3f} ms | {result['gflops']:8.2f} GFLOPS")
            except Exception as e:
                print(f"{config.get('name', 'bench'):20s} ERROR: {e}")

    print("\n" + "=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
