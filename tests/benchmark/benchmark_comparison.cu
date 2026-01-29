/**
 * benchmark_comparison.cu
 *
 * 对比 quant-gemm-from-scratch 实现与 llama.cpp 风格的性能测试
 *
 * 使用方式:
 *   nvcc -O3 -arch=sm_80 benchmark_comparison.cu -o benchmark_comparison
 *   ./benchmark_comparison
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <curand.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>

// 包含你的实现
#include "../include/quant_types.h"
#include "../include/quantize.h"
#include "../include/gemm_cuda_naive.cuh"
#include "../include/gemm_cuda_tiled.cuh"
#include "../include/gemm_cuda_dp4a.cuh"

// ============================================================================
// 性能测试结构 (仿照 test-backend-ops.cpp)
// ============================================================================

struct benchmark_result {
    const char* name;
    const char* impl;
    int M, N, K;
    double avg_time_us;
    double gflops;
    double bandwidth_gb_s;
    int n_runs;
    bool passed;
};

// 计算 GEMM FLOPs
uint64_t gemm_flops(int M, int N, int K) {
    return 2ULL * M * N * K;  // 乘加运算
}

// ============================================================================
// CUDA 工具函数
// ============================================================================

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// GPU 计时器
class CudaTimer {
public:
    cudaEvent_t start, stop;

    CudaTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~CudaTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void record_start() {
        cudaEventRecord(start);
    }

    void record_stop() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    }

    float elapsed_ms() {
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

// ============================================================================
// 基准测试函数
// ============================================================================

/**
 * 运行性能测试 (仿照 llama.cpp test-backend-ops.cpp 的 eval_perf)
 *
 * @param kernel_fn   要测试的 kernel 函数
 * @param name        测试名称
 * @param M, N, K     矩阵维度
 * @param min_time_ms 最小运行时间 (毫秒)
 */
template<typename KernelFn>
benchmark_result run_benchmark(
    KernelFn kernel_fn,
    const char* name,
    const char* impl,
    int M, int N, int K,
    void* A, void* B, void* C,
    float min_time_ms = 1000.0f  // 运行至少 1 秒
) {
    CudaTimer timer;

    // Warmup
    kernel_fn(A, B, C, M, N, K);
    cudaDeviceSynchronize();

    // 性能测试循环
    int n_runs = 0;
    float total_time_ms = 0.0f;

    while (total_time_ms < min_time_ms) {
        timer.record_start();
        kernel_fn(A, B, C, M, N, K);
        timer.record_stop();

        total_time_ms += timer.elapsed_ms();
        n_runs++;
    }

    // 计算结果
    double avg_time_us = (total_time_ms * 1000.0) / n_runs;
    uint64_t flops = gemm_flops(M, N, K);
    double gflops = (flops * n_runs) / (total_time_ms / 1000.0) / 1e9;

    // 计算带宽 (对于量化 GEMM)
    // Q4_0: 18 bytes per 32 elements, Q8_1: 36 bytes per 32 elements
    size_t bytes_read = (size_t)N * K / 32 * 18 +  // Q4_0 weights
                        (size_t)M * K / 32 * 36 +  // Q8_1 activations
                        (size_t)M * N * 4;         // Output (FP32)
    double bandwidth_gb_s = (bytes_read * n_runs) / (total_time_ms / 1000.0) / 1e9;

    benchmark_result result = {
        .name = name,
        .impl = impl,
        .M = M,
        .N = N,
        .K = K,
        .avg_time_us = avg_time_us,
        .gflops = gflops,
        .bandwidth_gb_s = bandwidth_gb_s,
        .n_runs = n_runs,
        .passed = true
    };

    return result;
}

// ============================================================================
// 打印结果 (仿照 llama.cpp 格式)
// ============================================================================

void print_header() {
    printf("\n");
    printf("================================================================================\n");
    printf("  Quantized GEMM Performance Benchmark\n");
    printf("  (Compatible with llama.cpp test-backend-ops format)\n");
    printf("================================================================================\n");
    printf("\n");
    printf("%-30s %6s %6s %6s %12s %10s %10s %8s\n",
           "Test Case", "M", "N", "K", "Time(us)", "GFLOPS", "GB/s", "Runs");
    printf("--------------------------------------------------------------------------------\n");
}

void print_result(const benchmark_result& r) {
    printf("%-30s %6d %6d %6d %12.2f %10.2f %10.2f %8d\n",
           r.name, r.M, r.N, r.K, r.avg_time_us, r.gflops, r.bandwidth_gb_s, r.n_runs);
}

void print_comparison(const benchmark_result& baseline, const benchmark_result& optimized) {
    double speedup = baseline.avg_time_us / optimized.avg_time_us;
    printf("  -> Speedup vs %s: %.2fx\n", baseline.impl, speedup);
}

// ============================================================================
// 测试用例定义 (仿照 llama.cpp make_test_cases_perf)
// ============================================================================

struct test_case {
    int M, N, K;
    const char* description;
};

std::vector<test_case> make_test_cases() {
    return {
        // Llama-2 7B 典型尺寸
        {1,    4096, 4096, "Llama-7B decode bs=1"},
        {32,   4096, 4096, "Llama-7B decode bs=32"},
        {128,  4096, 4096, "Llama-7B prefill"},
        {512,  4096, 4096, "Llama-7B long prefill"},

        // Llama-2 13B
        {1,    5120, 5120, "Llama-13B decode bs=1"},
        {32,   5120, 5120, "Llama-13B decode bs=32"},

        // MLP layers (intermediate size)
        {1,    11008, 4096, "Llama-7B MLP up"},
        {1,    4096, 11008, "Llama-7B MLP down"},

        // 小尺寸测试
        {256,  256,  256,  "Small 256x256x256"},
        {1024, 1024, 1024, "Medium 1K x 1K x 1K"},
    };
}

// ============================================================================
// 主函数
// ============================================================================

int main(int argc, char** argv) {
    // 获取 GPU 信息
    int device;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    printf("GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Memory: %.1f GB\n", prop.totalGlobalMem / 1e9);

    print_header();

    auto test_cases = make_test_cases();

    for (const auto& tc : test_cases) {
        int M = tc.M, N = tc.N, K = tc.K;

        // 确保 K 是 32 的倍数 (量化块大小)
        if (K % 32 != 0) {
            printf("Skipping %s: K=%d not multiple of 32\n", tc.description, K);
            continue;
        }

        // 分配内存
        size_t size_A = M * K * sizeof(float);
        size_t size_B = N * K * sizeof(float);
        size_t size_C = M * N * sizeof(float);

        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, size_A));
        CUDA_CHECK(cudaMalloc(&d_B, size_B));
        CUDA_CHECK(cudaMalloc(&d_C, size_C));

        // 初始化随机数据
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, 1234);
        curandGenerateUniform(gen, d_A, M * K);
        curandGenerateUniform(gen, d_B, N * K);

        printf("\n--- %s ---\n", tc.description);

        // TODO: 这里添加你的实际 kernel 调用
        // 示例: 测试 FP32 naive GEMM
        auto fp32_naive = [](void* A, void* B, void* C, int M, int N, int K) {
            // 调用你的 naive FP32 GEMM kernel
            // gemm_fp32_naive<<<...>>>((float*)A, (float*)B, (float*)C, M, N, K);
        };

        // benchmark_result r1 = run_benchmark(fp32_naive, "FP32 Naive", "naive", M, N, K, d_A, d_B, d_C);
        // print_result(r1);

        // 清理
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        curandDestroyGenerator(gen);

        printf("(实现 kernel 调用后取消注释以运行基准测试)\n");
    }

    printf("\n================================================================================\n");
    printf("  Benchmark Complete\n");
    printf("================================================================================\n");

    return 0;
}
