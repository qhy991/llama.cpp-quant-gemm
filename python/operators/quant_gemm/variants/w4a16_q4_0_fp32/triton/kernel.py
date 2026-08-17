"""
Triton implementation of W4A16 GEMM with Q4_0 quantization (llama.cpp style).

Q4_0 format:
- Block size: 32 elements
- Storage: 18 bytes per block (2 bytes scale + 16 bytes data)
- Scale: FP16 stored as first 2 bytes
- Data: 32 x 4-bit values packed into 16 bytes
- Dequantization: w[i] = (q[i] - 8) * d
"""

import torch
import triton
import triton.language as tl


@triton.jit
def gemm_q4_0_kernel(
    # Pointers
    weight_ptr,      # [N, num_blocks, 18] as uint8
    activation_ptr,  # [M, K] as float32
    output_ptr,      # [M, N] as float32
    # Dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_om, stride_on,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,  # Must be 32 (Q4_0 block size)
):
    """
    Triton kernel for Q4_0 GEMM.

    Each thread block computes a BLOCK_M x BLOCK_N tile of the output.
    """
    # Program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute starting indices
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    # Number of Q4_0 blocks along K dimension
    num_k_blocks = K // 32

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Create offsets for M and N dimensions
    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = n_start + tl.arange(0, BLOCK_N)

    # Masks for boundary checking
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Iterate over K dimension in blocks of 32
    for kb in range(num_k_blocks):
        # Load and dequantize weights for this block
        # Weight layout: [N, num_k_blocks, 18]
        # For each n in [n_start, n_start + BLOCK_N), load block kb

        # Base offset for weight block: (n * num_k_blocks + kb) * 18
        weight_block_offsets = (offs_n * num_k_blocks + kb) * 18

        # Load scale (first 2 bytes as uint16, then bitcast to float16)
        # We need to read 2 bytes and interpret as FP16
        scale_byte0 = tl.load(weight_ptr + weight_block_offsets, mask=mask_n, other=0)
        scale_byte1 = tl.load(weight_ptr + weight_block_offsets + 1, mask=mask_n, other=0)
        scale_uint16 = scale_byte0.to(tl.uint16) | (scale_byte1.to(tl.uint16) << 8)
        scale = scale_uint16.to(tl.float16, bitcast=True).to(tl.float32)

        # Process 32 elements in this block (2 elements per byte, 16 bytes)
        for i in range(16):
            # Load packed byte containing 2 x 4-bit values
            packed_byte = tl.load(weight_ptr + weight_block_offsets + 2 + i, mask=mask_n, other=0)

            # Extract low and high 4-bit values
            q_low = (packed_byte & 0x0F).to(tl.float32)
            q_high = (packed_byte >> 4).to(tl.float32)

            # Dequantize: w = (q - 8) * scale
            w_low = (q_low - 8.0) * scale   # Shape: [BLOCK_N]
            w_high = (q_high - 8.0) * scale  # Shape: [BLOCK_N]

            # Compute K indices for these two elements
            k_idx_low = kb * 32 + i * 2
            k_idx_high = kb * 32 + i * 2 + 1

            # Load activation values: A[m, k]
            # Activation layout: [M, K]
            a_low = tl.load(
                activation_ptr + offs_m[:, None] * stride_am + k_idx_low * stride_ak,
                mask=mask_m[:, None],
                other=0.0
            )  # Shape: [BLOCK_M, 1]

            a_high = tl.load(
                activation_ptr + offs_m[:, None] * stride_am + k_idx_high * stride_ak,
                mask=mask_m[:, None],
                other=0.0
            )  # Shape: [BLOCK_M, 1]

            # Accumulate: acc[m, n] += a[m] * w[n]
            acc += a_low * w_low[None, :]
            acc += a_high * w_high[None, :]

    # Store output
    output_offsets = offs_m[:, None] * stride_on + offs_n[None, :] * 1
    output_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(output_ptr + output_offsets, acc, mask=output_mask)


def gemm_q4_0_triton(
    weight: torch.Tensor,    # [N, num_blocks, 18] uint8
    activation: torch.Tensor,  # [M, K] float32
    M: int,
    N: int,
    K: int,
) -> torch.Tensor:
    """
    Wrapper function for Triton Q4_0 GEMM kernel.

    Args:
        weight: Q4_0 quantized weights [N, K/32, 18] as uint8
        activation: FP32 activations [M, K]
        M, N, K: Matrix dimensions

    Returns:
        output: [M, N] float32
    """
    # Ensure inputs are contiguous
    weight = weight.contiguous()
    activation = activation.contiguous()

    # Flatten weight to 1D for easier indexing
    weight_flat = weight.view(-1)

    # Allocate output
    output = torch.empty((M, N), dtype=torch.float32, device=activation.device)

    # Block sizes
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32  # Q4_0 block size

    # Grid dimensions
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    # Launch kernel
    gemm_q4_0_kernel[grid](
        weight_flat,
        activation,
        output,
        M, N, K,
        activation.stride(0), activation.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return output


# Entry point for the test framework
def run(weight: torch.Tensor, activation: torch.Tensor, **params) -> torch.Tensor:
    """
    Entry point compatible with the test framework.

    Args:
        weight: Q4_0 quantized weights [N, K/32, 18] as uint8
        activation: FP32 activations [M, K]
        **params: M, N, K parameters

    Returns:
        output: [M, N] float32
    """
    M = params.get('M', activation.shape[0])
    N = params.get('N', weight.shape[0])
    K = params.get('K', activation.shape[1])

    return gemm_q4_0_triton(weight, activation, M, N, K)
