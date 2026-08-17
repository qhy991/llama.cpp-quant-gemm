"""
Reference implementation for W4A8 GEMM (Q4_0 weights x Q8_1 activations).

This is a pure Python implementation used for correctness verification.
"""

import torch
import struct
from typing import Tuple


def unpack_q4_0_block(block_bytes: bytes) -> Tuple[float, list]:
    """
    Unpack a Q4_0 block (18 bytes) to scale and 32 int values.

    Q4_0 format:
    - d: half (2 bytes) - scale factor
    - qs: uint8[16] (16 bytes) - 32 packed 4-bit values

    Returns:
        (scale, list of 32 integers in range [0, 15])
    """
    d = struct.unpack('<e', block_bytes[:2])[0]
    qs = block_bytes[2:18]

    values = []
    for i in range(16):
        q_low = qs[i] & 0x0F
        q_high = (qs[i] >> 4) & 0x0F
        values.append(q_low)
        values.append(q_high)

    return d, values


def unpack_q8_1_block(block_bytes: bytes) -> Tuple[float, float, list]:
    """
    Unpack a Q8_1 block (36 bytes) to scale, sum, and 32 int8 values.

    Q8_1 format:
    - ds: half2 (4 bytes) - d (scale) in low, s (sum) in high
    - qs: int8[32] (32 bytes) - 32 signed 8-bit values

    Returns:
        (scale, sum, list of 32 int8 values)
    """
    ds = struct.unpack('<ee', block_bytes[:4])
    d, s = ds[0], ds[1]
    qs = struct.unpack('<32b', block_bytes[4:36])

    return d, s, list(qs)


def vec_dot_q4_0_q8_1(w_block: bytes, a_block: bytes) -> float:
    """
    Compute dot product of Q4_0 weight block and Q8_1 activation block.

    Formula: result = d_w * (d_a * sumi - 8.0 * s_a)

    This compensates for Q4_0's +8 offset in the packed values.
    """
    # Unpack weight (Q4_0)
    d_w, w_qs = unpack_q4_0_block(w_block)

    # Unpack activation (Q8_1)
    d_a, s_a, a_qs = unpack_q8_1_block(a_block)

    # Integer dot product (no offset subtraction here)
    sumi = 0
    for i in range(16):
        # Low nibble: w_qs[i] corresponds to a_qs[i]
        # High nibble: w_qs[i+16] corresponds to a_qs[i+16]
        sumi += w_qs[i] * a_qs[i]
        sumi += w_qs[i + 16] * a_qs[i + 16]

    # Apply compensation formula
    return d_w * (d_a * sumi - 8.0 * s_a)


@torch.no_grad()
def run(weight: torch.Tensor, activation: torch.Tensor, **params) -> torch.Tensor:
    """
    Reference implementation for W4A8 GEMM.

    Args:
        weight: Q4_0 quantized weights [N, num_blocks, 18]
        activation: Q8_1 quantized activations [M, num_blocks, 36]
        **params: Additional parameters (M, N, K)

    Returns:
        output: [M, N] float32 tensor

    Note: This is a slow Python implementation for reference only.
    """
    # Get dimensions
    N = weight.shape[0]
    M = activation.shape[0]
    num_blocks = weight.shape[1]

    # Move to CPU for byte-level access
    weight_cpu = weight.cpu().numpy()
    activation_cpu = activation.cpu().numpy()

    # Compute output
    output = torch.zeros(M, N, dtype=torch.float32)

    for m in range(M):
        for n in range(N):
            acc = 0.0
            for b in range(num_blocks):
                w_block = bytes(weight_cpu[n, b])
                a_block = bytes(activation_cpu[m, b])
                acc += vec_dot_q4_0_q8_1(w_block, a_block)
            output[m, n] = acc

    return output


def run_fp32_reference(
    weight: torch.Tensor,
    activation: torch.Tensor,
    **params
) -> torch.Tensor:
    """
    Simple FP32 matmul reference.

    Args:
        weight: FP32 weight [N, K]
        activation: FP32 activation [M, K]

    Returns:
        output: [M, N] = activation @ weight.T
    """
    return torch.matmul(activation, weight.T)
