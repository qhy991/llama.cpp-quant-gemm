/**
 * test_custom_kernel.cu - 测试自定义 kernel 集成
 *
 * 这个程序创建测试数据，调用我们的自定义 kernel，并验证结果的正确性。
 */

#include <iostream>
#include <random>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// 包含我们的自定义 kernel
#include "/home/haiyan/Agent4Kernel/quant-gemm-from-scratch/include/gemm_cuda_dp4a.cuh"

// CUDA 错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

// ============================================================================
// 量化函数
// ============================================================================

/**
 * 将 FP32 数据量化为 Q4_0 格式
 */
void quantize_q4_0(const float* src, block_q4_0* dst, int n) {
    const int nb = n / QK4_0;  // 块数量

    for (int b = 0; b < nb; b++) {
        const float* x = src + b * QK4_0;

        // 找到最大绝对值
        float amax = 0.0f;
        for (int i = 0; i < QK4_0; i++) {
            amax = std::max(amax, std::abs(x[i]));
        }

        // 计算 scale
        const float d = amax / 7.0f;
        const float id = d > 0.0f ? 1.0f / d : 0.0f;

        dst[b].d = __float2half(d);

        // 量化并打包
        for (int i = 0; i < QK4_0 / 2; i++) {
            int q0 = std::round(x[2*i + 0] * id) + 8;
            int q1 = std::round(x[2*i + 1] * id) + 8;

            // 限制到 [0, 15]
            q0 = std::max(0, std::min(15, q0));
            q1 = std::max(0, std::min(15, q1));

            // 打包：低 4 位是第一个值，高 4 位是第二个值
            dst[b].qs[i] = (q1 << 4) | q0;
        }
    }
}

/**
 * 将 FP32 数据量化为 Q8_1 格式
 */
void quantize_q8_1(const float* src, block_q8_1* dst, int n) {
    const int nb = n / QK8_1;

    for (int b = 0; b < nb; b++) {
        const float* x = src + b * QK8_1;

        // 找到最大绝对值
        float amax = 0.0f;
        for (int i = 0; i < QK8_1; i++) {
            amax = std::max(amax, std::abs(x[i]));
        }

        // 计算 scale
        const float d = amax / 127.0f;
        const float id = d > 0.0f ? 1.0f / d : 0.0f;

        // 计算 sum（用于补偿）
        float sum = 0.0f;
        for (int i = 0; i < QK8_1; i++) {
            const int8_t q = std::round(x[i] * id);
            dst[b].qs[i] = q;
            sum += x[i];
        }

        // 存储 d 和 s
        dst[b].ds = __halves2half2(__float2half(d), __float2half(sum));
    }
}

/**
 * Q4_0 反量化
 */
void dequantize_q4_0(const block_q4_0* src, float* dst, int n) {
    const int nb = n / QK4_0;

    for (int b = 0; b < nb; b++) {
        const float d = __half2float(src[b].d);

        for (int i = 0; i < QK4_0 / 2; i++) {
            const uint8_t packed = src[b].qs[i];
            const int q0 = (packed & 0x0F) - 8;
            const int q1 = (packed >> 4) - 8;

            dst[b * QK4_0 + 2*i + 0] = q0 * d;
            dst[b * QK4_0 + 2*i + 1] = q1 * d;
        }
    }
}

/**
 * Q8_1 反量化
 */
void dequantize_q8_1(const block_q8_1* src, float* dst, int n) {
    const int nb = n / QK8_1;

    for (int b = 0; b < nb; b++) {
        const float d = __half2float(__low2half(src[b].ds));

        for (int i = 0; i < QK8_1; i++) {
            dst[b * QK8_1 + i] = src[b].qs[i] * d;
        }
    }
}

// ============================================================================
// CPU 参考实现
// ============================================================================

/**
 * CPU 上的 FP32 GEMM（用于验证）
 */
void gemm_fp32_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[n * K + k];
            }
            C[m * N + n] = sum;
        }
    }
}

// ============================================================================
// 测试函数
// ============================================================================

/**
 * 计算 NMSE (Normalized Mean Squared Error)
 */
double compute_nmse(const float* ref, const float* test, int n) {
    double mse = 0.0;
    double ref_norm = 0.0;

    for (int i = 0; i < n; i++) {
        double diff = ref[i] - test[i];
        mse += diff * diff;
        ref_norm += ref[i] * ref[i];
    }

    return mse / (ref_norm + 1e-10);
}

/**
 * 主测试函数
 */
void test_custom_kernel(int M, int N, int K) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "测试配置: M=" << M << ", N=" << N << ", K=" << K << std::endl;
    std::cout << "========================================\n" << std::endl;

    // 确保 K 是 32 的倍数
    if (K % 32 != 0) {
        std::cerr << "错误: K 必须是 32 的倍数" << std::endl;
        return;
    }

    // ========================================================================
    // 1. 生成随机测试数据
    // ========================================================================
    std::cout << "1. 生成测试数据..." << std::endl;

    std::random_device rd;
    std::mt19937 gen(42);  // 固定种子以便复现
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    // 分配 host 内存
    float* h_A_fp32 = new float[M * K];
    float* h_B_fp32 = new float[N * K];
    float* h_C_ref = new float[M * N];
    float* h_C_test = new float[M * N];

    // 生成随机数据
    for (int i = 0; i < M * K; i++) h_A_fp32[i] = dis(gen);
    for (int i = 0; i < N * K; i++) h_B_fp32[i] = dis(gen);

    std::cout << "   ✓ 生成了 " << M*K << " 个激活值和 " << N*K << " 个权重值" << std::endl;

    // ========================================================================
    // 2. 量化数据
    // ========================================================================
    std::cout << "\n2. 量化数据..." << std::endl;

    const int nb_A = (M * K) / QK8_1;
    const int nb_B = (N * K) / QK4_0;

    block_q8_1* h_A_q8 = new block_q8_1[nb_A];
    block_q4_0* h_B_q4 = new block_q4_0[nb_B];

    // 量化
    for (int m = 0; m < M; m++) {
        quantize_q8_1(h_A_fp32 + m * K, h_A_q8 + m * (K / QK8_1), K);
    }
    for (int n = 0; n < N; n++) {
        quantize_q4_0(h_B_fp32 + n * K, h_B_q4 + n * (K / QK4_0), K);
    }

    std::cout << "   ✓ 激活量化为 Q8_1: " << nb_A << " 个块 ("
              << nb_A * sizeof(block_q8_1) << " 字节)" << std::endl;
    std::cout << "   ✓ 权重量化为 Q4_0: " << nb_B << " 个块 ("
              << nb_B * sizeof(block_q4_0) << " 字节)" << std::endl;

    // 验证量化质量
    float* h_A_dequant = new float[M * K];
    float* h_B_dequant = new float[N * K];
    for (int m = 0; m < M; m++) {
        dequantize_q8_1(h_A_q8 + m * (K / QK8_1), h_A_dequant + m * K, K);
    }
    for (int n = 0; n < N; n++) {
        dequantize_q4_0(h_B_q4 + n * (K / QK4_0), h_B_dequant + n * K, K);
    }

    double quant_error_A = compute_nmse(h_A_fp32, h_A_dequant, M * K);
    double quant_error_B = compute_nmse(h_B_fp32, h_B_dequant, N * K);

    std::cout << "   ✓ 激活量化误差 (NMSE): " << quant_error_A << std::endl;
    std::cout << "   ✓ 权重量化误差 (NMSE): " << quant_error_B << std::endl;

    // ========================================================================
    // 3. 计算 CPU 参考结果
    // ========================================================================
    std::cout << "\n3. 计算 CPU 参考结果..." << std::endl;

    gemm_fp32_cpu(h_A_dequant, h_B_dequant, h_C_ref, M, N, K);

    std::cout << "   ✓ CPU GEMM 完成" << std::endl;

    // ========================================================================
    // 4. 运行 GPU kernel
    // ========================================================================
    std::cout << "\n4. 运行 GPU kernel..." << std::endl;

    // 分配 device 内存
    block_q8_1* d_A;
    block_q4_0* d_B;
    float* d_C;

    CUDA_CHECK(cudaMalloc(&d_A, nb_A * sizeof(block_q8_1)));
    CUDA_CHECK(cudaMalloc(&d_B, nb_B * sizeof(block_q4_0)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

    // 拷贝数据到 GPU
    CUDA_CHECK(cudaMemcpy(d_A, h_A_q8, nb_A * sizeof(block_q8_1), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B_q4, nb_B * sizeof(block_q4_0), cudaMemcpyHostToDevice));

    // 配置 kernel
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);

    std::cout << "   Grid: (" << grid.x << ", " << grid.y << ")" << std::endl;
    std::cout << "   Block: (" << block.x << ", " << block.y << ")" << std::endl;

    // 创建 CUDA 事件用于计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    gemm_w4a8_dp4a_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 计时运行
    CUDA_CHECK(cudaEventRecord(start));
    gemm_w4a8_dp4a_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    // 拷贝结果回 CPU
    CUDA_CHECK(cudaMemcpy(h_C_test, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "   ✓ GPU kernel 完成" << std::endl;
    std::cout << "   ✓ 执行时间: " << milliseconds << " ms" << std::endl;

    // 计算性能
    double gflops = (2.0 * M * N * K) / (milliseconds * 1e6);
    std::cout << "   ✓ 性能: " << gflops << " GFLOPS" << std::endl;

    // ========================================================================
    // 5. 验证结果
    // ========================================================================
    std::cout << "\n5. 验证结果..." << std::endl;

    double nmse = compute_nmse(h_C_ref, h_C_test, M * N);

    std::cout << "   NMSE: " << nmse << std::endl;

    if (nmse < 1e-3) {
        std::cout << "   ✅ 测试通过！结果正确" << std::endl;
    } else {
        std::cout << "   ❌ 测试失败！误差过大" << std::endl;

        // 打印一些样本值用于调试
        std::cout << "\n   前 10 个值对比:" << std::endl;
        std::cout << "   Index | CPU参考值 | GPU结果 | 差异" << std::endl;
        std::cout << "   ------|-----------|---------|------" << std::endl;
        for (int i = 0; i < std::min(10, M * N); i++) {
            printf("   %5d | %9.4f | %7.4f | %7.4f\n",
                   i, h_C_ref[i], h_C_test[i], h_C_ref[i] - h_C_test[i]);
        }
    }

    // ========================================================================
    // 清理
    // ========================================================================
    delete[] h_A_fp32;
    delete[] h_B_fp32;
    delete[] h_C_ref;
    delete[] h_C_test;
    delete[] h_A_q8;
    delete[] h_B_q4;
    delete[] h_A_dequant;
    delete[] h_B_dequant;

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

// ============================================================================
// 主函数
// ============================================================================

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "自定义 Kernel 测试程序" << std::endl;
    std::cout << "========================================" << std::endl;

    // 检查 CUDA 设备
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "错误: 没有找到 CUDA 设备" << std::endl;
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout << "\nGPU 信息:" << std::endl;
    std::cout << "  名称: " << prop.name << std::endl;
    std::cout << "  计算能力: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  全局内存: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;

    // 运行不同大小的测试
    std::cout << "\n开始测试...\n" << std::endl;

    // 小矩阵测试
    test_custom_kernel(64, 64, 128);

    // 中等矩阵测试
    test_custom_kernel(256, 256, 512);

    // 大矩阵测试
    test_custom_kernel(1024, 1024, 2048);

    std::cout << "\n========================================" << std::endl;
    std::cout << "所有测试完成！" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
