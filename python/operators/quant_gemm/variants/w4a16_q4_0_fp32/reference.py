"""
Reference implementation for W4A16 GEMM (Q4_0 weights x FP32 activations).
"""

import torch
import struct
from typing import List


def dequantize_q4_0_block(block_bytes: bytes) -> List[float]:
    """
    Dequantize Q4_0 block (18 bytes) to 32 float values.

    Formula: x = (q - 8) * d
    """
    d = struct.unpack('<e', block_bytes[:2])[0]
    qs = block_bytes[2:18]

    values = []
    for i in range(16):
        q_low = (qs[i] & 0x0F) - 8
        q_high = ((qs[i] >> 4) & 0x0F) - 8
        values.append(q_low * d)
        values.append(q_high * d)

    return values


@torch.no_grad()
def run(weight: torch.Tensor, activation: torch.Tensor, **params) -> torch.Tensor:
    """
    Reference implementation for W4A16 GEMM.

    Args:
        weight: Q4_0 quantized weights [N, num_blocks, 18]
        activation: FP32 activations [M, K]
        **params: M, N, K

    Returns:
        output: [M, N]
    """
    M, K = activation.shape
    N = weight.shape[0]
    num_blocks = K // 32

    # Dequantize weights to FP32
    weight_cpu = weight.cpu().numpy()
    weight_dequant = torch.zeros(N, K, dtype=torch.float32)

    for n in range(N):
        for b in range(num_blocks):
            block_bytes = bytes(weight_cpu[n, b])
            block_values = dequantize_q4_0_block(block_bytes)
            weight_dequant[n, b*32:(b+1)*32] = torch.tensor(block_values)

    # Standard GEMM
    output = torch.matmul(activation, weight_dequant.T)
    return output


def run_fp32_reference(
    weight: torch.Tensor,
    activation: torch.Tensor,
    **params
) -> torch.Tensor:
    """Simple FP32 matmul reference."""
    return torch.matmul(activation, weight.T)
