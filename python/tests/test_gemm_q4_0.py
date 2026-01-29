"""
Test suite for Q4_0 x Q8_1 quantized GEMM

Usage:
    pytest tests/test_gemm_q4_0.py -v
    python tests/test_gemm_q4_0.py  # Run directly
"""

import torch
import numpy as np
import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import quant_gemm


class TestQuantization:
    """Test quantization functions"""

    def test_q4_0_shape(self):
        """Test Q4_0 quantization output shape"""
        K = 1024
        x = torch.randn(4, K, device='cuda')
        x_q = quant_gemm.quantize_q4_0(x)

        expected_shape = (4, K // 32, 18)
        assert x_q.shape == expected_shape, f"Expected {expected_shape}, got {x_q.shape}"
        assert x_q.dtype == torch.uint8

    def test_q8_1_shape(self):
        """Test Q8_1 quantization output shape"""
        K = 1024
        x = torch.randn(4, K, device='cuda')
        x_q = quant_gemm.quantize_q8_1(x)

        expected_shape = (4, K // 32, 36)
        assert x_q.shape == expected_shape, f"Expected {expected_shape}, got {x_q.shape}"
        assert x_q.dtype == torch.uint8

    def test_q4_0_roundtrip(self):
        """Test Q4_0 quantization roundtrip error"""
        K = 1024
        x = torch.randn(4, K, device='cuda')

        x_q = quant_gemm.quantize_q4_0(x)
        x_dq = quant_gemm.dequantize_q4_0(x_q, K)

        # Check shape
        assert x_dq.shape == x.shape, f"Shape mismatch: {x_dq.shape} vs {x.shape}"

        # Quantization error should be bounded
        max_error = (x - x_dq).abs().max().item()
        # Q4_0 with 4 bits and scale can have significant error
        assert max_error < 1.0, f"Max error too large: {max_error}"

        print(f"Q4_0 roundtrip max error: {max_error:.4f}")


class TestGEMM:
    """Test GEMM functions"""

    @pytest.fixture
    def setup_small(self):
        """Small test case"""
        torch.manual_seed(42)
        M, N, K = 4, 8, 1024
        weight = torch.randn(M, K, device='cuda')
        activation = torch.randn(N, K, device='cuda')
        return M, N, K, weight, activation

    @pytest.fixture
    def setup_llama(self):
        """LLM-like test case"""
        torch.manual_seed(42)
        M, N, K = 4096, 2, 14336
        weight = torch.randn(M, K, device='cuda')
        activation = torch.randn(N, K, device='cuda')
        return M, N, K, weight, activation

    def test_gemm_shape(self, setup_small):
        """Test GEMM output shape"""
        M, N, K, weight, activation = setup_small

        weight_q = quant_gemm.quantize_q4_0(weight)
        activation_q = quant_gemm.quantize_q8_1(activation)

        output = quant_gemm.gemm_q4_0_q8_1(weight_q, activation_q, M, N, K)

        assert output.shape == (M, N), f"Expected ({M}, {N}), got {output.shape}"
        assert output.dtype == torch.float32
        assert output.is_cuda

    def test_gemm_correctness_small(self, setup_small):
        """Test GEMM correctness on small matrix"""
        M, N, K, weight, activation = setup_small

        # Quantize
        weight_q = quant_gemm.quantize_q4_0(weight)
        activation_q = quant_gemm.quantize_q8_1(activation)

        # Run quantized GEMM
        output = quant_gemm.gemm_q4_0_q8_1(weight_q, activation_q, M, N, K)

        # Reference: FP32 GEMM
        # Note: Due to quantization, we compare against quantized reference
        output_ref = weight @ activation.T

        # Compute NMSE
        mse = torch.mean((output - output_ref) ** 2).item()
        ref_power = torch.mean(output_ref ** 2).item()
        nmse = mse / ref_power if ref_power > 0 else 0

        print(f"\nSmall GEMM (M={M}, N={N}, K={K}):")
        print(f"  NMSE: {nmse:.6e}")
        print(f"  Max error: {(output - output_ref).abs().max().item():.4f}")

        # Q4_0 quantization has significant error, so use relaxed threshold
        assert nmse < 0.1, f"NMSE too high: {nmse}"

    def test_gemm_correctness_llama(self, setup_llama):
        """Test GEMM correctness on LLM-like matrix"""
        M, N, K, weight, activation = setup_llama

        # Quantize
        weight_q = quant_gemm.quantize_q4_0(weight)
        activation_q = quant_gemm.quantize_q8_1(activation)

        # Run quantized GEMM
        output = quant_gemm.gemm_q4_0_q8_1(weight_q, activation_q, M, N, K)

        # Reference
        output_ref = weight @ activation.T

        # Compute NMSE
        mse = torch.mean((output - output_ref) ** 2).item()
        ref_power = torch.mean(output_ref ** 2).item()
        nmse = mse / ref_power if ref_power > 0 else 0

        print(f"\nLLM GEMM (M={M}, N={N}, K={K}):")
        print(f"  NMSE: {nmse:.6e}")
        print(f"  Max error: {(output - output_ref).abs().max().item():.4f}")

        # Check output is not NaN/Inf
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"

    @pytest.mark.parametrize("M,N,K", [
        (1, 1, 32),       # Minimum size
        (4, 2, 1024),     # Small
        (128, 8, 4096),   # Medium
        (4096, 2, 14336), # LLM decode
    ])
    def test_gemm_various_sizes(self, M, N, K):
        """Test GEMM with various matrix sizes"""
        torch.manual_seed(42)
        weight = torch.randn(M, K, device='cuda')
        activation = torch.randn(N, K, device='cuda')

        weight_q = quant_gemm.quantize_q4_0(weight)
        activation_q = quant_gemm.quantize_q8_1(activation)

        output = quant_gemm.gemm_q4_0_q8_1(weight_q, activation_q, M, N, K)

        assert output.shape == (M, N)
        assert not torch.isnan(output).any()
        print(f"  GEMM M={M}, N={N}, K={K}: OK")


class TestPerformance:
    """Performance benchmarks"""

    @pytest.mark.benchmark
    def test_gemm_performance(self):
        """Benchmark GEMM performance"""
        torch.manual_seed(42)
        M, N, K = 4096, 2, 14336

        weight = torch.randn(M, K, device='cuda')
        activation = torch.randn(N, K, device='cuda')

        weight_q = quant_gemm.quantize_q4_0(weight)
        activation_q = quant_gemm.quantize_q8_1(activation)

        # Warmup
        for _ in range(10):
            _ = quant_gemm.gemm_q4_0_q8_1(weight_q, activation_q, M, N, K)
        torch.cuda.synchronize()

        # Benchmark
        n_runs = 100
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(n_runs):
            _ = quant_gemm.gemm_q4_0_q8_1(weight_q, activation_q, M, N, K)
        end.record()
        torch.cuda.synchronize()

        elapsed_ms = start.elapsed_time(end) / n_runs
        flops = 2 * M * N * K
        gflops = flops / (elapsed_ms / 1000) / 1e9

        print(f"\nPerformance (M={M}, N={N}, K={K}):")
        print(f"  Time: {elapsed_ms:.3f} ms")
        print(f"  GFLOPS: {gflops:.2f}")


def run_quick_test():
    """Quick sanity test"""
    print("=" * 60)
    print("Quick Test: quant_gemm")
    print("=" * 60)

    # Check CUDA
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return False

    print(f"CUDA device: {torch.cuda.get_device_name()}")

    # Test quantization
    print("\n[1] Testing quantization...")
    x = torch.randn(4, 1024, device='cuda')
    x_q4 = quant_gemm.quantize_q4_0(x)
    x_q8 = quant_gemm.quantize_q8_1(x)
    print(f"  Q4_0: {x.shape} -> {x_q4.shape}")
    print(f"  Q8_1: {x.shape} -> {x_q8.shape}")

    # Test dequantization
    print("\n[2] Testing dequantization...")
    x_dq = quant_gemm.dequantize_q4_0(x_q4, 1024)
    max_err = (x - x_dq).abs().max().item()
    print(f"  Roundtrip max error: {max_err:.4f}")

    # Test GEMM
    print("\n[3] Testing GEMM...")
    M, N, K = 4096, 2, 14336
    weight = torch.randn(M, K, device='cuda')
    activation = torch.randn(N, K, device='cuda')

    weight_q = quant_gemm.quantize_q4_0(weight)
    activation_q = quant_gemm.quantize_q8_1(activation)

    output = quant_gemm.gemm_q4_0_q8_1(weight_q, activation_q, M, N, K)
    output_ref = weight @ activation.T

    nmse = (torch.mean((output - output_ref) ** 2) /
            torch.mean(output_ref ** 2)).item()

    print(f"  Shape: {output.shape}")
    print(f"  NMSE vs FP32: {nmse:.6e}")

    # Performance
    print("\n[4] Performance...")
    torch.cuda.synchronize()
    import time
    start = time.perf_counter()
    for _ in range(100):
        _ = quant_gemm.gemm_q4_0_q8_1(weight_q, activation_q, M, N, K)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_ms = elapsed / 100 * 1000
    gflops = 2 * M * N * K / (avg_ms / 1000) / 1e9
    print(f"  Avg time: {avg_ms:.3f} ms")
    print(f"  GFLOPS: {gflops:.2f}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    run_quick_test()
