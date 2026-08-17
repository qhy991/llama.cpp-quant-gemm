#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// 辅助宏：用于处理 block_q4_0 的特殊 18 字节偏移
#define Q40_BLOCK_SIZE 32
#define Q40_BYTE_SIZE 18

__global__ void gemm_q4_0_fp32(
    const uint8_t* weight,    // [N, K/32, 18]
    const float* activation,  // [M, K]
    float* output,            // [M, N]
    int M,
    int N,
    int K
) {
    // 计算当前线程负责的 Output 矩阵坐标 (m, n)
    int n = blockIdx.x * blockDim.x + threadIdx.x; // Output 的列索引 (Weight 的行索引)
    int m = blockIdx.y * blockDim.y + threadIdx.y; // Output 的行索引 (Activation 的行索引)

    if (m >= M || n >= N) return;

    float sum = 0.0f;
    int num_blocks = K / Q40_BLOCK_SIZE;

    // 遍历 K 维度上的所有 Block
    for (int kb = 0; kb < num_blocks; ++kb) {
        // 1. 定位当前 block 在 Weight 数组中的起始字节位置
        // Weight 布局是 [N, K/32, 18]，也就是行优先存储
        // Offset = (n * num_blocks + kb) * 18
        long long offset = (long long)(n * num_blocks + kb) * Q40_BYTE_SIZE;

        // 2. 读取 Scale (前 2 个字节)
        const uint8_t* block_ptr = weight + offset;
        __half scale_h = *reinterpret_cast<const __half*>(block_ptr);
        float scale = __half2float(scale_h);

        // 3. 遍历 block 内的 32 个元素
        // 数据从 block_ptr[2] 开始
        const uint8_t* data_ptr = block_ptr + 2;

        // 对应的 Activation 起始位置
        // Activation 布局 [M, K] -> activation[m * K + current_k]
        int k_start = kb * Q40_BLOCK_SIZE;

        for (int i = 0; i < Q40_BLOCK_SIZE; i += 2) {
            // 每个字节包含两个 4-bit 数值
            uint8_t packed = data_ptr[i / 2];

            // 低 4 位 (对应偶数索引 i)
            uint8_t q0 = packed & 0x0F;
            float w0 = (float(q0) - 8.0f) * scale;
            sum += activation[m * K + (k_start + i)] * w0;

            // 高 4 位 (对应奇数索引 i+1)
            uint8_t q1 = packed >> 4;
            float w1 = (float(q1) - 8.0f) * scale;
            sum += activation[m * K + (k_start + i + 1)] * w1;
        }
    }

    // 写回结果
    output[m * N + n] = sum;
}
