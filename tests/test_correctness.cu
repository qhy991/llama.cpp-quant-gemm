/**
 * @file tests/test_correctness.cu
 * @brief 专门测试正确性
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

#include "../kernels/gemm/gemm_quant_formats.cuh"
#include "../kernels/gemm/gemm_warp_optimized.cuh"

void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s - %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

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

struct VerifyResult {
    int total;
    int errors;
    float max_diff;
    float avg_diff;
    float max_rel_error;
};

VerifyResult verify_detailed(float* ref, float* test, int size) {
    VerifyResult result = {0};
    result.total = size;
    result.errors = 0;
    result.max_diff = 0.0f;
    result.avg_diff = 0.0f;
    result.max_rel_error = 0.0f;

    double sum_diff = 0.0;

    for (int i = 0; i < size; i++) {
        float diff = fabs(ref[i] - test[i]);
        float rel_error = diff / fmax(fabs(ref[i]), 1e-6f);

        sum_diff += diff;
        result.max_diff = fmax(result.max_diff, diff);
        result.max_rel_error = fmax(result.max_rel_error, rel_error);

        // 相对误差容差 0.1%
        if (rel_error > 1e-3f) {
            if (result.errors < 10) {
                printf("  Error #%d at index %d: ref=%.6f, test=%.6f, diff=%.6f, rel=%.4f%%\n",
                       result.errors + 1, i, ref[i], test[i], diff, rel_error * 100);
            }
            result.errors++;
        }
    }

    result.avg_diff = sum_diff / size;
    return result;
}

void test_kernel(const char* name,
                 void (*kernel)(const block_q4_0*, const block_q8_1*, float*, int, int, int, cudaStream_t),
                 const block_q4_0* d_weight,
                 const block_q8_1* d_activation,
                 float* d_output,
                 float* d_reference,
                 float* h_output,
                 float* h_reference,
                 int M, int N, int K) {

    printf("\n=======================================================================\n");
    printf("Testing: %s\n", name);
    printf("=======================================================================\n");

    // 运行 kernel
    kernel(d_weight, d_activation, d_output, M, N, K, 0);
    cudaDeviceSynchronize();

    // 拷贝结果
    cudaMemcpy(h_output, d_output, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_reference, d_reference, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // 详细验证
    VerifyResult result = verify_detailed(h_reference, h_output, M * N);

    printf("\nResults:\n");
    printf("  Total elements:     %d\n", result.total);
    printf("  Errors:             %d (%.4f%%)\n", result.errors, 100.0f * result.errors / result.total);
    printf("  Max absolute diff:  %.6f\n", result.max_diff);
    printf("  Avg absolute diff:  %.6f\n", result.avg_diff);
    printf("  Max relative error: %.4f%%\n", result.max_rel_error * 100);

    if (result.errors == 0) {
        printf("\n✓ PASS - All elements correct!\n");
    } else {
        printf("\n✗ FAIL - %d elements incorrect\n", result.errors);
    }
}

int main() {
    printf("=======================================================================\n");
    printf("Correctness Test Suite\n");
    printf("=======================================================================\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);

    // 测试多种矩阵尺寸
    struct TestCase {
        int M, N, K;
        const char* desc;
    };

    TestCase cases[] = {
        {4096, 1, 4096, "Small square"},
        {4096, 2, 14336, "Llama typical"},
        {4096, 4, 14336, "Medium N"},
        {4096, 8, 14336, "Large N"},
        {8192, 2, 14336, "Large M"},
        {1024, 16, 4096, "Large N, small M"},
    };

    int num_cases = sizeof(cases) / sizeof(cases[0]);
    int total_pass = 0;

    for (int c = 0; c < num_cases; c++) {
        int M = cases[c].M;
        int N = cases[c].N;
        int K = cases[c].K;

        printf("\n\n");
        printf("###################################################################\n");
        printf("Test Case %d/%d: %s\n", c+1, num_cases, cases[c].desc);
        printf("Matrix size: M=%d, N=%d, K=%d\n", M, N, K);
        printf("###################################################################\n");

        int num_blocks_k = K / 32;

        // 分配内存
        block_q4_0* h_weight = new block_q4_0[M * num_blocks_k];
        block_q8_1* h_activation = new block_q8_1[N * num_blocks_k];
        float* h_output = new float[M * N];
        float* h_reference = new float[M * N];

        srand(42 + c);  // 每个测试用例不同的随机种子
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
        printf("\nComputing reference (Naive)...\n");
        gemm_q4_0_q8_1(d_weight, d_activation, d_reference, M, N, K, 0);
        cudaDeviceSynchronize();

        // 测试各个优化版本
        test_kernel("Warp Multirow", gemm_q4_0_q8_1_warp_multirow,
                    d_weight, d_activation, d_output, d_reference,
                    h_output, h_reference, M, N, K);

        test_kernel("Shared Memory", gemm_q4_0_q8_1_smem,
                    d_weight, d_activation, d_output, d_reference,
                    h_output, h_reference, M, N, K);

        test_kernel("2D Tile (N=4)", gemm_q4_0_q8_1_tile2d,
                    d_weight, d_activation, d_output, d_reference,
                    h_output, h_reference, M, N, K);

        if (N >= 8) {
            test_kernel("2D Tile (N=8)", gemm_q4_0_q8_1_tile2d_n8,
                        d_weight, d_activation, d_output, d_reference,
                        h_output, h_reference, M, N, K);
        }

        // 清理
        cudaFree(d_weight);
        cudaFree(d_activation);
        cudaFree(d_output);
        cudaFree(d_reference);
        delete[] h_weight;
        delete[] h_activation;
        delete[] h_output;
        delete[] h_reference;
    }

    printf("\n\n");
    printf("=======================================================================\n");
    printf("All tests completed!\n");
    printf("=======================================================================\n");

    return 0;
}
