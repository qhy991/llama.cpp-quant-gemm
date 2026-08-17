"""
Reference implementation for W4_1A8 GEMM (Q4_1 weights x Q8_1 activations).
"""

import torch
import struct
from typing import Tuple


def unpack_q4_1_block(block_bytes: bytes) -> Tuple[float, float, list]:
    """
    Unpack a Q4_1 block (20 bytes).

    Q4_1 format:
    - d: half (2 bytes) - scale factor
    - m: half (2 bytes) - minimum value offset
    - qs: uint8[16] (16 bytes) - 32 packed 4-bit values

    Returns:
        (scale, min, list of 32 integers in range [0, 15])
    """
    d = struct.unpack('<e', block_bytes[:2])[0]
    m = struct.unpack('<e', block_bytes[2:4])[0]
    qs = block_bytes[4:20]

    values = []
    for i in range(16):
        q_low = qs[i] & 0x0F
        q_high = (qs[i] >> 4) & 0x0F
        values.append(q_low)
        values.append(q_high)

    return d, m, values


def unpack_q8_1_block(block_bytes: bytes) -> Tuple[float, float, list]:
    """Unpack a Q8_1 block (36 bytes)."""
    ds = struct.unpack('<ee', block_bytes[:4])
    d, s = ds[0], ds[1]
    qs = struct.unpack('<32b', block_bytes[4:36])
    return d, s, list(qs)


def vec_dot_q4_1_q8_1(w_block: bytes, a_block: bytes) -> float:
    """
    Compute dot product of Q4_1 weight block and Q8_1 activation block.

    Formula: result = d_w * d_a * sumi + m_w * s_a
    """
    # Unpack weight (Q4_1)
    d_w, m_w, w_qs = unpack_q4_1_block(w_block)

    # Unpack activation (Q8_1)
    d_a, s_a, a_qs = unpack_q8_1_block(a_block)

    # Integer dot product
    sumi = 0
    for i in range(16):
        sumi += w_qs[i] * a_qs[i]
        sumi += w_qs[i + 16] * a_qs[i + 16]

    # Apply Q4_1 formula
    return d_w * d_a * sumi + m_w * s_a


@torch.no_grad()
def run(weight: torch.Tensor, activation: torch.Tensor, **params) -> torch.Tensor:
    """Reference implementation for W4_1A8 GEMM."""
    N = weight.shape[0]
    M = activation.shape[0]
    num_blocks = weight.shape[1]

    weight_cpu = weight.cpu().numpy()
    activation_cpu = activation.cpu().numpy()

    output = torch.zeros(M, N, dtype=torch.float32)

    for m in range(M):
        for n in range(N):
            acc = 0.0
            for b in range(num_blocks):
                w_block = bytes(weight_cpu[n, b])
                a_block = bytes(activation_cpu[m, b])
                acc += vec_dot_q4_1_q8_1(w_block, a_block)
            output[m, n] = acc

    return output
