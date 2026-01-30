/**
 * @file tests/benchmark_best.cu
 * @brief 只测试最佳的几个 kernel
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <vector>
#include <cmath>

#include "../kernels/gemm/gemm_quant_formats.cuh"
#include "../kernels/gemm/gemm_warp_optimized.cuh"
#include "../kernels/gemm/gemm_async_copy.cuh"
#include "../kernels/gemm/gemm_vectorized.cuh"

// ============================================================================
// 辅助函数
// ============================================================================

void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s - %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

// 初始化测试数据
void init_test_data(
    block_q4_0* weight, int M, int num_blocks_k,
    block_q8_1* activation, int N
) {
    for (int m = 0; m < M; m++) {
        for (int b = 0; b < num_blocks_k; b++) {
            block_q4_0& blk = weight[m * num_blocks_k + b];
            blk.d = __float2half(0.1f * ((rand() % 10) + 1));
            for (int i = 0; i < 16; i++) {
                blk.qs[i] = rand() & 0xFF;
            }
        }
    }

    for (int n = 0; n < N; n++) {
        for (int b = 0; b < num_blocks_k; b++) {
            block_q8_1& blk = activation[n * num_blocks_k + b];
            blk.ds = __halves2half2(__float2half(0.1f), __float2half(1.0f));
            for (int i = 0; i < 32; i++) {
                blk.qs[i] = (rand() % 256) - 128;
            }
        }
    }
}

// 验证结果正确性
bool verify_results(float* ref, float* test, int size, float tol = 1e-3f) {
    int errors = 0;
    float max_diff = 0.0f;

    for (int i = 0; i < size; i++) {
        float diff = fabs(ref[i] - test[i]);
        max_diff = fmax(max_diff, diff);

        if (diff > tol * fmax(fabs(ref[i]), 1.0f)) {
            if (errors < 5) {
                printf("  Mismatch at %d: ref=%.6f, test=%.6f, diff=%.6f\n",
                       i, ref[i], test[i], diff);
            }
            errors++;
        }
    }

    printf("  Max diff: %.6f, Errors: %d/%d\n", max_diff, errors, size);
    return errors == 0;
}

// ============================================================================
// Benchmark 函数
// ============================================================================

struct BenchmarkResult {
    const char* name;
    float time_ms;
    float gflops;
    bool correct;
};

template<typename KernelFunc>
BenchmarkResult benchmark_kernel(
    const char* name,
    KernelFunc kernel,
    const block_q4_0* d_weight,
    const block_q8_1* d_activation,
    float* d_output,
    float* d_reference,
    int M, int N, int K,
    int warmup_iters = 10,
    int bench_iters = 100
) {
    BenchmarkResult result;
    result.name = name;

    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        kernel(d_weight, d_activation, d_output, M, N, K, 0);
    }
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < bench_iters; i++) {
        kernel(d_weight, d_activation, d_output, M, N, K, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms;
    cudaEventElapsedTime(&total_ms, start, stop);
    result.time_ms = total_ms / bench_iters;

    // 计算 GFLOPS
    long long flops = 2LL * M * N * K;
    result.gflops = (flops / 1e9) / (result.time_ms / 1000.0);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // 验证正确性
    float* h_output = new float[M * N];
    float* h_reference = new float[M * N];
    cudaMemcpy(h_output, d_output, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_reference, d_reference, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\n[%s] Verifying...\n", name);
    result.correct = verify_results(h_reference, h_output, M * N);

    delete[] h_output;
    delete[] h_reference;

    return result;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    int M = 4096;
    int N = 2;
    int K = 14336;

    if (argc >= 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }

    printf("=======================================================================\n");
    printf("Best Kernels Benchmark\n");
    printf("=======================================================================\n");
    printf("Matrix size: M=%d, N=%d, K=%d\n", M, N, K);
    printf("FLOPs: %.2f GFLOP\n", 2.0 * M * N * K / 1e9);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("=======================================================================\n");

    // 分配内存
    int num_blocks_k = K / 32;

    block_q4_0* h_weight = new block_q4_0[M * num_blocks_k];
    block_q8_1* h_activation = new block_q8_1[N * num_blocks_k];

    srand(42);
    init_test_data(h_weight, M, num_blocks_k, h_activation, N);

    block_q4_0* d_weight;
    block_q8_1* d_activation;
    float* d_output;
    float* d_reference;

    check_cuda(cudaMalloc(&d_weight, M * num_blocks_k * sizeof(block_q4_0)), "alloc weight");
    check_cuda(cudaMalloc(&d_activation, N * num_blocks_k * sizeof(block_q8_1)), "alloc activation");
    check_cuda(cudaMalloc(&d_output, M * N * sizeof(float)), "alloc output");
    check_cuda(cudaMalloc(&d_reference, M * N * sizeof(float)), "alloc reference");

    check_cuda(cudaMemcpy(d_weight, h_weight, M * num_blocks_k * sizeof(block_q4_0), cudaMemcpyHostToDevice), "copy weight");
    check_cuda(cudaMemcpy(d_activation, h_activation, N * num_blocks_k * sizeof(block_q8_1), cudaMemcpyHostToDevice), "copy activation");

    // 计算参考结果
    printf("\nComputing reference result...\n");
    gemm_q4_0_q8_1(d_weight, d_activation, d_reference, M, N, K, 0);
    cudaDeviceSynchronize();

    // Benchmark 最佳 kernels
    std::vector<BenchmarkResult> results;

    results.push_back(benchmark_kernel(
        "Naive",
        gemm_q4_0_q8_1,
        d_weight, d_activation, d_output, d_reference,
        M, N, K
    ));

    results.push_back(benchmark_kernel(
        "Warp Multirow",
        gemm_q4_0_q8_1_warp_multirow,
        d_weight, d_activation, d_output, d_reference,
        M, N, K
    ));

    results.push_back(benchmark_kernel(
        "Shared Memory",
        gemm_q4_0_q8_1_smem,
        d_weight, d_activation, d_output, d_reference,
        M, N, K
    ));

    results.push_back(benchmark_kernel(
        "2D Tile (N=4)",
        gemm_q4_0_q8_1_tile2d,
        d_weight, d_activation, d_output, d_reference,
        M, N, K
    ));

    results.push_back(benchmark_kernel(
        "2D Tile (K=256)",
        gemm_q4_0_q8_1_tile2d_k256,
        d_weight, d_activation, d_output, d_reference,
        M, N, K
    ));

    // Test vectorized versions
    results.push_back(benchmark_kernel(
        "Vectorized (Safe)",
        gemm_q4_0_q8_1_vec_safe,
        d_weight, d_activation, d_output, d_reference,
        M, N, K
    ));

    results.push_back(benchmark_kernel(
        "Vectorized (Float4)",
        gemm_q4_0_q8_1_vec_float4,
        d_weight, d_activation, d_output, d_reference,
        M, N, K
    ));

    // Check compute capability for async copy (reuse prop from earlier)
    if (prop.major >= 8) {
        printf("\n[INFO] GPU supports async copy (SM %d.%d), testing async version...\n",
               prop.major, prop.minor);

        results.push_back(benchmark_kernel(
            "Async Copy + Pipeline",
            gemm_q4_0_q8_1_async,
            d_weight, d_activation, d_output, d_reference,
            M, N, K
        ));
    } else {
        printf("\n[INFO] GPU does not support async copy (SM %d.%d < 8.0), skipping async version\n",
               prop.major, prop.minor);
    }

    // 打印结果
    printf("\n=======================================================================\n");
    printf("                           RESULTS SUMMARY\n");
    printf("=======================================================================\n");
    printf("%-20s │ %10s │ %10s │ %10s │ %8s\n", "Kernel", "Time (ms)", "GFLOPS", "Speedup", "Correct");
    printf("─────────────────────┼────────────┼────────────┼────────────┼─────────\n");

    float base_time = results[0].time_ms;
    for (const auto& r : results) {
        printf("%-20s │ %10.4f │ %10.1f │ %9.2fx │ %8s\n",
               r.name, r.time_ms, r.gflops, base_time / r.time_ms,
               r.correct ? "✓" : "✗");
    }
    printf("=======================================================================\n");

    // 清理
    cudaFree(d_weight);
    cudaFree(d_activation);
    cudaFree(d_output);
    cudaFree(d_reference);
    delete[] h_weight;
    delete[] h_activation;

    return 0;
}
