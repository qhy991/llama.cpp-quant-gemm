/**
 * test_simple.cu - 简单的调试测试
 *
 * 使用小矩阵测试，打印中间值
 */

#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include "../include/quant_types.h"
#include "../include/gemm_cuda_dp4a.cuh"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

// CPU 参考实现：直接计算量化点积
float compute_q4_q8_dot_cpu(const block_q4_0& w, const block_q8_1& a) {
    float d_w = __half2float(w.d);
    float d_a = __half2float(__low2half(a.ds));
    float s_a = __half2float(__high2half(a.ds));

    int32_t sumi = 0;

    // 计算量化值的点积
    for (int i = 0; i < QK4_0 / 2; i++) {
        uint8_t packed = w.qs[i];
        int q_w0 = (packed & 0x0F);  // 低 4 位
        int q_w1 = (packed >> 4);     // 高 4 位

        int8_t q_a0 = a.qs[2*i + 0];
        int8_t q_a1 = a.qs[2*i + 1];

        sumi += q_w0 * q_a0;
        sumi += q_w1 * q_a1;
    }

    // 应用补偿公式
    float result = d_w * (d_a * sumi - 8.0f * s_a);

    return result;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "简单调试测试" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // 创建一个简单的测试：1×1 矩阵，K=32
    const int M = 1;
    const int N = 1;
    const int K = 32;

    // 创建测试数据
    block_q8_1 h_A;
    block_q4_0 h_B;

    // 初始化 A (Q8_1)
    float d_a = 0.5f;
    float sum_a = 0.0f;
    for (int i = 0; i < QK8_1; i++) {
        h_A.qs[i] = i - 16;  // [-16, -15, ..., 14, 15]
        sum_a += (i - 16) * d_a;  // 原始值的和
    }
    h_A.ds = __halves2half2(__float2half(d_a), __float2half(sum_a));

    // 初始化 B (Q4_0)
    float d_b = 0.25f;
    h_B.d = __float2half(d_b);
    for (int i = 0; i < QK4_0 / 2; i++) {
        uint8_t q0 = i % 16;
        uint8_t q1 = (i + 1) % 16;
        h_B.qs[i] = (q1 << 4) | q0;
    }

    // 打印输入数据
    std::cout << "输入数据:" << std::endl;
    std::cout << "  A (Q8_1): d=" << d_a << ", s=" << sum_a << std::endl;
    std::cout << "    qs = [";
    for (int i = 0; i < 8; i++) std::cout << (int)h_A.qs[i] << " ";
    std::cout << "...]" << std::endl;

    std::cout << "  B (Q4_0): d=" << d_b << std::endl;
    std::cout << "    qs = [";
    for (int i = 0; i < 4; i++) {
        uint8_t packed = h_B.qs[i];
        std::cout << (int)(packed & 0x0F) << "," << (int)(packed >> 4) << " ";
    }
    std::cout << "...]" << std::endl;

    // CPU 参考结果
    float cpu_result = compute_q4_q8_dot_cpu(h_B, h_A);
    std::cout << "\nCPU 参考结果: " << cpu_result << std::endl;

    // GPU 计算
    block_q8_1* d_A;
    block_q4_0* d_B;
    float* d_C;

    CUDA_CHECK(cudaMalloc(&d_A, sizeof(block_q8_1)));
    CUDA_CHECK(cudaMalloc(&d_B, sizeof(block_q4_0)));
    CUDA_CHECK(cudaMalloc(&d_C, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, &h_A, sizeof(block_q8_1), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, &h_B, sizeof(block_q4_0), cudaMemcpyHostToDevice));

    // 运行 kernel
    dim3 block(1, 1);
    dim3 grid(1, 1);
    gemm_w4a8_dp4a_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 获取结果
    float gpu_result;
    CUDA_CHECK(cudaMemcpy(&gpu_result, d_C, sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "GPU 结果: " << gpu_result << std::endl;
    std::cout << "差异: " << std::abs(cpu_result - gpu_result) << std::endl;

    if (std::abs(cpu_result - gpu_result) < 0.01f) {
        std::cout << "\n✅ 测试通过！" << std::endl;
    } else {
        std::cout << "\n❌ 测试失败！" << std::endl;
    }

    // 清理
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
