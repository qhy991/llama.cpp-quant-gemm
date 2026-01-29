/**
 * llama.cpp MMQ vs Baseline 对比测试
 *
 * 基于 kerneleval 的 baseline.cu 实现
 * 使用相同的数据格式: INT8 数组 + 独立的 scale 因子
 *
 * 对比内容:
 * 1. Baseline naive kernel (与 kerneleval 一致)
 * 2. llama.cpp MMQ DP4A kernel (使用 Q8_0 block 格式)
 * 3. 数值一致性验证
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// 定义 CUDA_CHECK 宏
// ============================================================================
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// ============================================================================
// 数据结构 - 与 baseline 一致的格式
// ============================================================================

// 测试配置
struct TestConfig {
    int M, N, K;
    const char* name;
};

TestConfig test_configs[] = {
    {1, 4096, 4096, "M1_K4096_N4096"},
    {1, 4096, 256, "M1_K4096_N256_SMALL"},
    {1, 4096, 64, "M1_K4096_N64_SMALL"},
};

// 性能计时器
class GPUTimer {
    cudaEvent_t start_, stop_;
    cudaStream_t stream_;

public:
    void init(cudaStream_t s) {
        stream_ = s;
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }

    void start() {
        CUDA_CHECK(cudaEventRecord(start_, stream_));
    }

    float stop() {
        CUDA_CHECK(cudaEventRecord(stop_, stream_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

    ~GPUTimer() {
        if (start_) cudaEventDestroy(start_);
        if (stop_) cudaEventDestroy(stop_);
    }
};

// ============================================================================
// llama.cpp Q8_0 数据结构 (用于对比)
// ============================================================================

struct Q8_0_Block {
    float d;         // 缩放因子
    int8_t qs[32];   // 32 个 INT8 值
};

// ============================================================================
// Baseline W8A8 Kernel (与 kerneleval 一致)
// ============================================================================

/**
 * Baseline naive W8A8 GEMM
 * 与 /home/haiyan/Agent4Kernel/kerneleval/baselines/BW_5070/cuda/kernels/naive_gemm.cuh 中的实现相同
 *
 * 数据格式:
 * - activation: INT8 数组 (M, K), row-major
 * - weights: INT8 数组 (N, K), column-major
 * - act_scale: 标量, 所有激活值共用
 * - w_scale: 标量 (pertensor) 或 per-channel (perchannel)
 */
__global__ void baseline_w8a8_gemm_kernel(
    const int8_t* __restrict__ activation,  // (M, K) row-major
    const int8_t* __restrict__ weights,     // (N, K) column-major
    float* __restrict__ output,              // (M, N)
    float act_scale,                         // 激活值 scale (标量)
    const float* __restrict__ w_scale,       // 权重 scale (标量或 per-channel)
    int M, int K, int N,
    bool per_channel) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        // INT32 累加器
        int32_t acc = 0;

        // K 维度循环
        for (int k = 0; k < K; k++) {
            // activation[row * K + k]: row-major 访问
            // weights[col * K + k]: column-major 访问
            acc += (int32_t)activation[row * K + k] *
                   (int32_t)weights[col * K + k];
        }

        // 反量化: INT32 → FP32
        float scale = act_scale * (per_channel ? w_scale[col] : w_scale[0]);
        output[row * N + col] = (float)acc * scale;
    }
}

// ============================================================================
// llama.cpp MMQ DP4A Kernel (使用 Q8_0 block 格式)
// ============================================================================

/**
 * 将 plain INT8 数组转换为 Q8_0 block 格式
 *
 * 注意: 为了与 baseline 保持一致，这里不进行实际的量化
 * 而是直接存储 INT8 值，scale 设为 1.0
 */
__global__ void convert_to_q8_0_blocks_kernel(
    const int8_t* __restrict__ input,
    Q8_0_Block* __restrict__ output,
    int M, int K) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    int K_blocks = (K + 31) / 32;

    for (int kb = 0; kb < K_blocks; kb++) {
        // 计算 block 索引
        int block_idx = row * K_blocks + kb;

        // 不进行量化，直接存储 INT8 值
        // scale 设为 1.0，保持原始值
        output[block_idx].d = 1.0f;

        for (int i = 0; i < 32; i++) {
            int k = kb * 32 + i;
            if (k < K) {
                // 直接存储 INT8 值，不进行量化
                output[block_idx].qs[i] = input[row * K + k];
            } else {
                output[block_idx].qs[i] = 0;
            }
        }
    }
}

/**
 * llama.cpp MMQ DP4A Kernel
 *
 * 使用 Q8_0 block 格式，与 llama.cpp 的实现类似
 */
__global__ void mmq_dp4a_kernel(
    const Q8_0_Block* __restrict__ A_blocks,  // Q8_0 格式的激活值 (M, K/32)
    const int8_t* __restrict__ weights,        // INT8 格式的权重 (N, K) column-major
    float* __restrict__ C,                      // FP32 输出 (M, N)
    const float* __restrict__ w_scale,          // 权重 scale (per-channel)
    float act_scale,                            // 激活值 scale (baseline 格式)
    int M, int K, int N,
    int K_blocks) {

    // 与 baseline 相同的索引方式
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    // FP32 累加器 (直接使用 FP32 累加反量化后的值)
    float acc = 0.0f;

    // K 维度循环 (按 block 遍历)
    for (int kb = 0; kb < K_blocks; kb++) {
        const Q8_0_Block& A_blk = A_blocks[row * K_blocks + kb];
        float d_A = A_blk.d;

        // INT8×INT8 点积
        int32_t sumi = 0;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            int k = kb * 32 + i;
            if (k < K) {
                // A_blk.qs[i]: 量化后的激活值
                // weights[col * K + k]: 权重 (column-major)
                sumi += (int32_t)A_blk.qs[i] * (int32_t)weights[col * K + k];
            }
        }

        // 反量化并累加到 FP32
        // sumi * d_A: 反量化激活值 (INT8 → FP32)
        acc += (float)sumi * d_A;
    }

    // 最终输出: acc * act_scale * w_scale[col]
    // 与 baseline 保持一致：act_scale * w_scale
    C[row * N + col] = acc * act_scale * w_scale[col];
}

// ============================================================================
// CPU 参考实现
// ============================================================================

/**
 * CPU W8A8 GEMM - 用于验证
 */
void cpu_w8a8_gemm(
    const int8_t* __restrict__ activation,
    const int8_t* __restrict__ weights,
    float* __restrict__ output,
    float act_scale,
    const float* __restrict__ w_scale,
    int M, int K, int N,
    bool per_channel) {

    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++) {
                acc += (int32_t)activation[row * K + k] *
                       (int32_t)weights[col * K + k];
            }
            float scale = act_scale * (per_channel ? w_scale[col] : w_scale[0]);
            output[row * N + col] = (float)acc * scale;
        }
    }
}

// ============================================================================
// 数据初始化
// ============================================================================

/**
 * 初始化测试数据
 */
void init_test_data(
    int8_t* activation,
    int8_t* weights,
    float* w_scale,
    int M, int N, int K) {

    // 初始化激活值 (row-major)
    for (int i = 0; i < M * K; i++) {
        activation[i] = (int8_t)((i % 256) - 128);
    }

    // 初始化权重 (column-major)
    for (int col = 0; col < N; col++) {
        for (int k = 0; k < K; k++) {
            weights[col * K + k] = (int8_t)(((col * K + k) % 256) - 128);
        }
        // 每列的 scale
        w_scale[col] = 0.01f + (float)(col % 100) / 10000.0f;
    }
}

// ============================================================================
// 结果验证
// ============================================================================

/**
 * 验证两个结果是否一致
 */
bool verify_results(
    const float* ref,
    const float* test,
    int M, int N,
    float& max_rel_error,
    float& avg_rel_error,
    int& diff_count) {

    max_rel_error = 0.0f;
    avg_rel_error = 0.0f;
    diff_count = 0;

    for (int i = 0; i < M * N; i++) {
        float abs_error = fabsf(ref[i] - test[i]);
        float rel_error = abs_error;

        if (fabsf(ref[i]) > 1e-6f) {
            rel_error = abs_error / fabsf(ref[i]);
        }

        if (abs_error > 1e-5f) {
            diff_count++;
        }

        if (rel_error > max_rel_error) {
            max_rel_error = rel_error;
        }
        avg_rel_error += rel_error;
    }

    avg_rel_error /= (float)(M * N);

    return max_rel_error < 1e-4f && diff_count == 0;
}

// ============================================================================
// 测试主函数
// ============================================================================

void run_test(cudaStream_t stream, GPUTimer& timer, const TestConfig& config) {
    int M = config.M;
    int N = config.N;
    int K = config.K;

    printf("\n╔═════════════════════════════════════════════╗\n");
    printf("║  测试: %s\n", config.name);
    printf("║  矩阵: M=%d, K=%d, N=%d\n", M, K, N);
    printf("╚═════════════════════════════════════════════╝\n");

    // 分配 host 内存
    int8_t* h_activation = nullptr;
    int8_t* h_weights = nullptr;
    float* h_w_scale = nullptr;
    float* h_output_baseline = nullptr;
    float* h_output_mmq = nullptr;
    float* h_output_cpu = nullptr;

    h_activation = (int8_t*)malloc(M * K * sizeof(int8_t));
    h_weights = (int8_t*)malloc(N * K * sizeof(int8_t));
    h_w_scale = (float*)malloc(N * sizeof(float));
    h_output_baseline = (float*)malloc(M * N * sizeof(float));
    h_output_mmq = (float*)malloc(M * N * sizeof(float));
    h_output_cpu = (float*)malloc(M * N * sizeof(float));

    // 初始化数据
    init_test_data(h_activation, h_weights, h_w_scale, M, N, K);

    // CPU 参考计算
    float act_scale = 0.02f;
    cpu_w8a8_gemm(h_activation, h_weights, h_output_cpu,
                 act_scale, h_w_scale, M, K, N, false);

    // 分配 GPU 内存
    int8_t* d_activation = nullptr;
    int8_t* d_weights = nullptr;
    float* d_w_scale = nullptr;
    float* d_output_baseline = nullptr;
    float* d_output_mmq = nullptr;
    Q8_0_Block* d_A_blocks = nullptr;

    CUDA_CHECK(cudaMalloc(&d_activation, M * K * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_weights, N * K * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_w_scale, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output_baseline, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output_mmq, M * N * sizeof(float)));

    int K_blocks = (K + 31) / 32;
    CUDA_CHECK(cudaMalloc(&d_A_blocks, M * K_blocks * sizeof(Q8_0_Block)));

    // 拷贝数据到 GPU
    CUDA_CHECK(cudaMemcpyAsync(d_activation, h_activation,
                               M * K * sizeof(int8_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_weights, h_weights,
                               N * K * sizeof(int8_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_w_scale, h_w_scale,
                               N * sizeof(float),
                               cudaMemcpyHostToDevice, stream));

    // 转换为 Q8_0 格式
    int threads = 256;
    int blocks = (M + threads - 1) / threads;
    convert_to_q8_0_blocks_kernel<<<blocks, threads, 0, stream>>>(
        d_activation, d_A_blocks, M, K);
    CUDA_CHECK(cudaGetLastError());

    // ========== 配置 kernel 启动参数 ==========
    dim3 block(8, 32);
    dim3 grid((N + 8 - 1) / 8, (M + 32 - 1) / 32);

    // ========== 运行 Baseline Kernel ==========
    timer.start();
    float time_baseline = 0;
    for (int i = 0; i < 10; i++) {
        baseline_w8a8_gemm_kernel<<<grid, block, 0, stream>>>(
            d_activation, d_weights, d_output_baseline,
            act_scale, d_w_scale, M, K, N, false);
        CUDA_CHECK(cudaGetLastError());
        time_baseline += timer.stop();
    }

    CUDA_CHECK(cudaMemcpyAsync(h_output_baseline, d_output_baseline,
                               M * N * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    printf("║  Baseline:       %.4f ms (avg 10 runs)\n", time_baseline / 10.0f);

    // 验证 vs CPU
    float max_err_baseline, avg_err_baseline;
    int diff_cnt_baseline;
    bool baseline_valid = verify_results(h_output_cpu, h_output_baseline,
                                         M, N, max_err_baseline, avg_err_baseline, diff_cnt_baseline);
    printf("║    vs CPU:       %s (max_err: %.2e, diff: %d)\n",
           baseline_valid ? "✓ PASS" : "✗ FAIL", max_err_baseline, diff_cnt_baseline);

    // ========== 运行 MMQ Kernel ==========
    // 使用与 baseline 相同的配置
    dim3 block_mmq(8, 32);
    dim3 grid_mmq((N + 8 - 1) / 8, (M + 32 - 1) / 32);

    timer.start();
    float time_mmq = 0;
    for (int i = 0; i < 10; i++) {
        mmq_dp4a_kernel<<<grid_mmq, block_mmq, 0, stream>>>(
            d_A_blocks, d_weights, d_output_mmq,
            d_w_scale, act_scale, M, K, N, K_blocks);
        CUDA_CHECK(cudaGetLastError());
        time_mmq += timer.stop();
    }

    CUDA_CHECK(cudaMemcpyAsync(h_output_mmq, d_output_mmq,
                               M * N * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    printf("║  MMQ DP4A:       %.4f ms (avg 10 runs)\n", time_mmq / 10.0f);

    // 验证 vs CPU
    float max_err_mmq, avg_err_mmq;
    int diff_cnt_mmq;
    bool mmq_valid = verify_results(h_output_cpu, h_output_mmq,
                                    M, N, max_err_mmq, avg_err_mmq, diff_cnt_mmq);
    printf("║    vs CPU:       %s (max_err: %.2e, diff: %d)\n",
           mmq_valid ? "✓ PASS" : "✗ FAIL", max_err_mmq, diff_cnt_mmq);

    // 验证 Baseline vs MMQ
    float max_err_between, avg_err_between;
    int diff_cnt_between;
    bool between_valid = verify_results(h_output_baseline, h_output_mmq,
                                        M, N, max_err_between, avg_err_between, diff_cnt_between);
    printf("║  Baseline vs MMQ: %s (max_err: %.2e, diff: %d)\n",
           between_valid ? "✓ PASS" : "✗ FAIL", max_err_between, diff_cnt_between);

    printf("║  加速比: %.2fx (MMQ vs Baseline)\n", time_baseline / time_mmq);

    // 显示结果样本
    printf("║  结果样本 (前 8 个):\n");
    printf("║    CPU:      ");
    for (int i = 0; i < 8 && i < M * N; i++) {
        printf("%8.2f ", h_output_cpu[i]);
    }
    printf("\n║    Baseline: ");
    for (int i = 0; i < 8 && i < M * N; i++) {
        printf("%8.2f ", h_output_baseline[i]);
    }
    printf("\n║    MMQ:      ");
    for (int i = 0; i < 8 && i < M * N; i++) {
        printf("%8.2f ", h_output_mmq[i]);
    }
    printf("\n");

    printf("╚═════════════════════════════════════════════╝\n");

    // 清理
    free(h_activation);
    free(h_weights);
    free(h_w_scale);
    free(h_output_baseline);
    free(h_output_mmq);
    free(h_output_cpu);
    CUDA_CHECK(cudaFree(d_activation));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_w_scale));
    CUDA_CHECK(cudaFree(d_output_baseline));
    CUDA_CHECK(cudaFree(d_output_mmq));
    CUDA_CHECK(cudaFree(d_A_blocks));
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("╔══════════════════════════════════════════════╗\n");
    printf("║   llama.cpp MMQ vs Baseline 对比测试        ║\n");
    printf("║   基于 kerneleval baseline.cu 格式          ║\n");
    printf("╚══════════════════════════════════════════════╝\n\n");

    // 检查 CUDA 设备
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "错误: 未找到 CUDA 设备\n");
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("SM Count: %d\n", prop.multiProcessorCount);
    printf("Shared Memory per Block: %.2f KB\n\n",
           prop.sharedMemPerBlock / 1024.0);

    // 创建 CUDA stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // 运行测试
    GPUTimer timer;
    timer.init(stream);

    int num_tests = sizeof(test_configs) / sizeof(TestConfig);
    for (int i = 0; i < num_tests; i++) {
        run_test(stream, timer, test_configs[i]);
    }

    // 清理
    CUDA_CHECK(cudaStreamDestroy(stream));

    printf("\n✓ 所有测试完成！\n");
    return 0;
}
