/**
 * @file tests/profile_kernel.cu
 * @brief 用于 ncu profiling 的简单测试
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>

#include "../kernels/gemm/gemm_quant_formats.cuh"
#include "../kernels/gemm/gemm_warp_optimized.cuh"
#include "../kernels/gemm/gemm_async_copy.cuh"

void init_test_data(
    block_q4_0* weight, int M, int num_blocks_k,
    block_q8_1* activation, int N
) {
    for (int m = 0; m < M; m++) {
        for (int b = 0; b < num_blocks_k; b++) {
            block_q4_0& blk = weight[m * num_blocks_k + b];
            blk.d = __float2half(0.1f);
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

int main(int argc, char** argv) {
    int M = 4096;
    int N = 2;
    int K = 14336;

    if (argc >= 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }

    printf("Profiling kernel with M=%d, N=%d, K=%d\n", M, N, K);

    int num_blocks_k = K / 32;

    // 分配内存
    block_q4_0* h_weight = new block_q4_0[M * num_blocks_k];
    block_q8_1* h_activation = new block_q8_1[N * num_blocks_k];

    srand(42);
    init_test_data(h_weight, M, num_blocks_k, h_activation, N);

    block_q4_0* d_weight;
    block_q8_1* d_activation;
    float* d_output;

    cudaMalloc(&d_weight, M * num_blocks_k * sizeof(block_q4_0));
    cudaMalloc(&d_activation, N * num_blocks_k * sizeof(block_q8_1));
    cudaMalloc(&d_output, M * N * sizeof(float));

    cudaMemcpy(d_weight, h_weight, M * num_blocks_k * sizeof(block_q4_0), cudaMemcpyHostToDevice);
    cudaMemcpy(d_activation, h_activation, N * num_blocks_k * sizeof(block_q8_1), cudaMemcpyHostToDevice);

    // Warmup
    for (int i = 0; i < 10; i++) {
        gemm_q4_0_q8_1_tile2d(d_weight, d_activation, d_output, M, N, K, 0);
    }
    cudaDeviceSynchronize();

    printf("Running kernel for profiling...\n");

    // 运行 kernel (ncu 会捕获这个)
    gemm_q4_0_q8_1_tile2d(d_weight, d_activation, d_output, M, N, K, 0);
    cudaDeviceSynchronize();

    printf("Done!\n");

    // 清理
    cudaFree(d_weight);
    cudaFree(d_activation);
    cudaFree(d_output);
    delete[] h_weight;
    delete[] h_activation;

    return 0;
}
