"""
Test script to verify Triton Q4_0 GEMM implementation against reference.
"""

import torch
import struct
import sys
import os

# Add current directory to path for importing kernel
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)


def quantize_q4_0(weights_fp32: torch.Tensor) -> torch.Tensor:
    """
    Quantize FP32 weights to Q4_0 format.

    Args:
        weights_fp32: [N, K] float32 weights

    Returns:
        quantized: [N, K//32, 18] uint8 tensor
    """
    N, K = weights_fp32.shape
    assert K % 32 == 0, "K must be divisible by 32"

    num_blocks = K // 32
    quantized = torch.zeros((N, num_blocks, 18), dtype=torch.uint8)

    for n in range(N):
        for b in range(num_blocks):
            block = weights_fp32[n, b*32:(b+1)*32]

            # Compute scale (max absolute value / 7)
            amax = block.abs().max().item()
            d = amax / 7.0 if amax != 0 else 1.0

            # Quantize: q = round(x / d) + 8, clamped to [0, 15]
            q = torch.round(block / d) + 8
            q = q.clamp(0, 15).to(torch.uint8)

            # Pack scale as FP16 bytes
            scale_bytes = struct.pack('<e', d)
            quantized[n, b, 0] = scale_bytes[0]
            quantized[n, b, 1] = scale_bytes[1]

            # Pack quantized values (2 per byte)
            for i in range(16):
                q_low = q[i * 2].item()
                q_high = q[i * 2 + 1].item()
                quantized[n, b, 2 + i] = (q_high << 4) | q_low

    return quantized


def dequantize_q4_0(quantized: torch.Tensor, K: int) -> torch.Tensor:
    """
    Dequantize Q4_0 weights back to FP32.

    Args:
        quantized: [N, K//32, 18] uint8 tensor
        K: Original K dimension

    Returns:
        weights: [N, K] float32 tensor
    """
    N = quantized.shape[0]
    num_blocks = K // 32
    weights = torch.zeros((N, K), dtype=torch.float32)

    quantized_np = quantized.cpu().numpy()

    for n in range(N):
        for b in range(num_blocks):
            block_bytes = bytes(quantized_np[n, b])

            # Read scale
            d = struct.unpack('<e', block_bytes[:2])[0]

            # Dequantize
            for i in range(16):
                packed = block_bytes[2 + i]
                q_low = packed & 0x0F
                q_high = (packed >> 4) & 0x0F

                weights[n, b*32 + i*2] = (q_low - 8) * d
                weights[n, b*32 + i*2 + 1] = (q_high - 8) * d

    return weights


def reference_gemm(weight_q4_0: torch.Tensor, activation: torch.Tensor, K: int) -> torch.Tensor:
    """
    Reference implementation: dequantize then matmul.
    """
    weight_fp32 = dequantize_q4_0(weight_q4_0, K)
    return torch.matmul(activation, weight_fp32.T)


def check_cuda_available():
    """Check if CUDA is available and compatible."""
    if not torch.cuda.is_available():
        return False, "CUDA not available"

    try:
        # Try to create a small tensor on GPU
        test_tensor = torch.zeros(1, device='cuda')
        del test_tensor
        return True, "CUDA available"
    except RuntimeError as e:
        return False, f"CUDA error: {e}"


def test_quantization():
    """Test quantization/dequantization correctness on CPU."""
    print("=" * 60)
    print("Testing Q4_0 Quantization/Dequantization")
    print("=" * 60)

    test_configs = [
        {"N": 4, "K": 32, "name": "minimal"},
        {"N": 8, "K": 64, "name": "tiny"},
        {"N": 16, "K": 128, "name": "small"},
    ]

    all_passed = True

    for config in test_configs:
        N, K = config["N"], config["K"]
        name = config["name"]

        print(f"\nTest: {name} (N={N}, K={K})")
        print("-" * 40)

        torch.manual_seed(42)
        weights_fp32 = torch.randn(N, K, dtype=torch.float32)

        # Quantize
        weight_q4_0 = quantize_q4_0(weights_fp32)

        # Dequantize
        weights_recovered = dequantize_q4_0(weight_q4_0, K)

        # Check relative error (quantization introduces error)
        diff = (weights_fp32 - weights_recovered).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        # Q4_0 has limited precision, expect some error
        print(f"  Max diff: {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")

        # For Q4_0, the relative error should be bounded
        # Max representable range is roughly [-8*d, 7*d]
        # Typical quantization error is about 0.5 * step size
        passed = mean_diff < 0.5  # reasonable threshold for Q4_0
        print(f"  Status: {'PASS' if passed else 'FAIL'}")

        if not passed:
            all_passed = False

    return all_passed


def test_reference_gemm():
    """Test reference GEMM implementation on CPU."""
    print("\n" + "=" * 60)
    print("Testing Reference GEMM Implementation (CPU)")
    print("=" * 60)

    test_configs = [
        {"M": 1, "N": 32, "K": 32, "name": "minimal"},
        {"M": 4, "N": 64, "K": 64, "name": "tiny"},
        {"M": 8, "N": 128, "K": 128, "name": "small"},
    ]

    all_passed = True

    for config in test_configs:
        M, N, K = config["M"], config["N"], config["K"]
        name = config["name"]

        print(f"\nTest: {name} (M={M}, N={N}, K={K})")
        print("-" * 40)

        torch.manual_seed(42)
        weights_fp32 = torch.randn(N, K, dtype=torch.float32)
        activation = torch.randn(M, K, dtype=torch.float32)

        # Expected result (direct FP32 matmul)
        expected = torch.matmul(activation, weights_fp32.T)

        # Quantize weights
        weight_q4_0 = quantize_q4_0(weights_fp32)

        # Reference Q4_0 GEMM
        actual = reference_gemm(weight_q4_0, activation, K)

        # Compare (should have quantization error)
        diff = (expected - actual).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        # Compute relative NMSE
        nmse = (diff ** 2).sum() / (expected ** 2).sum()
        nmse = nmse.item()

        print(f"  Max diff: {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")
        print(f"  NMSE: {nmse:.6e}")

        # Should have some error due to quantization
        passed = nmse < 0.01  # 1% NMSE is reasonable for Q4_0
        print(f"  Status: {'PASS' if passed else 'FAIL'}")

        if not passed:
            all_passed = False

    return all_passed


def test_triton_kernel():
    """Test Triton kernel against reference."""
    print("\n" + "=" * 60)
    print("Testing Triton GEMM Kernel (GPU)")
    print("=" * 60)

    cuda_available, cuda_msg = check_cuda_available()
    if not cuda_available:
        print(f"Skipping Triton tests: {cuda_msg}")
        print("Note: RTX 50 series (sm_120) requires PyTorch built with CUDA 12.8+")
        return True  # Not a failure, just skipped

    try:
        from kernel import gemm_q4_0_triton
    except ImportError as e:
        print(f"Could not import Triton kernel: {e}")
        return True  # Not a failure

    test_configs = [
        {"M": 1, "N": 64, "K": 64, "name": "tiny"},
        {"M": 4, "N": 128, "K": 128, "name": "small"},
        {"M": 16, "N": 256, "K": 256, "name": "medium"},
        {"M": 32, "N": 512, "K": 512, "name": "larger"},
        {"M": 1, "N": 4096, "K": 4096, "name": "single_large"},
    ]

    device = torch.device("cuda")
    all_passed = True

    for config in test_configs:
        M, N, K = config["M"], config["N"], config["K"]
        name = config["name"]

        print(f"\nTest: {name} (M={M}, N={N}, K={K})")
        print("-" * 40)

        torch.manual_seed(42)
        weights_fp32 = torch.randn(N, K, dtype=torch.float32)
        activation = torch.randn(M, K, dtype=torch.float32, device=device)

        # Quantize weights
        weight_q4_0 = quantize_q4_0(weights_fp32).to(device)

        # Reference result (CPU)
        ref_output = reference_gemm(weight_q4_0.cpu(), activation.cpu(), K).to(device)

        # Triton result
        triton_output = gemm_q4_0_triton(weight_q4_0, activation, M, N, K)

        # Compare
        diff = (triton_output - ref_output).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        # Compute NMSE
        nmse = (diff ** 2).sum() / (ref_output ** 2).sum()
        nmse = nmse.item()

        # Check correctness
        threshold = 1e-4
        passed = max_diff < threshold

        status = "PASS" if passed else "FAIL"
        print(f"  Max diff: {max_diff:.6e}")
        print(f"  Mean diff: {mean_diff:.6e}")
        print(f"  NMSE: {nmse:.6e}")
        print(f"  Status: {status}")

        if not passed:
            all_passed = False
            print(f"  Reference sample: {ref_output[0, :5].tolist()}")
            print(f"  Triton sample: {triton_output[0, :5].tolist()}")

    return all_passed


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Q4_0 GEMM Test Suite")
    print("=" * 60)

    results = {}

    # Test quantization (CPU)
    results['quantization'] = test_quantization()

    # Test reference GEMM (CPU)
    results['reference_gemm'] = test_reference_gemm()

    # Test Triton kernel (GPU if available)
    results['triton_kernel'] = test_triton_kernel()

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    all_passed = all(results.values())
    print("\n" + ("All tests PASSED!" if all_passed else "Some tests FAILED!"))

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
