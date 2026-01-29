/**
 * @file tests/unit/test_gemm_q4.cu
 * @brief GEMM Q4_0 x Q8_1 单元测试
 *
 * 使用测试框架验证 DP4A 量化 GEMM 的正确性
 */

#include "../framework/test_framework.cuh"

// 需要在包含 kernel 之前定义 quant_types.h 的路径
// 使用兼容层的类型定义
#include "../../compat/ggml_types.h"

// ============================================================================
// CPU 参考实现
// ============================================================================

/**
 * Q4_0 x Q8_1 GEMM CPU 参考实现
 *
 * 计算: C[M,N] = A[M,K] * B[N,K]^T
 * 其中:
 * - A (权重) 为 Q4_0 量化
 * - B (激活) 为 Q8_1 量化
 *
 * Q4_0 格式: d + qs[16] (32 个 4-bit 值, 偏移 8)
 * Q8_1 格式: (d, s) + qs[32] (32 个 8-bit 值, s = 原始值求和)
 *
 * 点积计算:
 *   result = d_w * (d_a * Σ(q_w - 8) * q_a - 8 * s_a)
 *          = d_w * (d_a * sumi - 8 * s_a)
 */
void cpu_gemm_q4_0_q8_1_reference(
    const block_q4_0* weight,  // [M, K/32] blocks
    const block_q8_1* activation,  // [N, K/32] blocks
    float* output,             // [M, N]
    int M, int N, int K
) {
    const int num_blocks = K / 32;

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;

            for (int b = 0; b < num_blocks; b++) {
                const block_q4_0& w_block = weight[m * num_blocks + b];
                const block_q8_1& a_block = activation[n * num_blocks + b];

                float d_w = __half2float(w_block.d);
                float d_a = __half2float(__low2half(a_block.ds));
                float s_a = __half2float(__high2half(a_block.ds));

                // 计算点积
                int sumi = 0;
                for (int i = 0; i < 16; i++) {
                    uint8_t packed = w_block.qs[i];
                    int8_t w0 = (packed & 0x0F) - 8;
                    int8_t w1 = ((packed >> 4) & 0x0F) - 8;

                    int8_t a0 = a_block.qs[i * 2 + 0];
                    int8_t a1 = a_block.qs[i * 2 + 1];

                    sumi += w0 * a0 + w1 * a1;
                }

                // 补偿公式: d_w * (d_a * sumi - 8 * s_a)
                sum += d_w * (d_a * sumi - 8.0f * s_a);
            }

            output[m * N + n] = sum;
        }
    }
}

// ============================================================================
// GPU Kernel
// ============================================================================

/**
 * 简化版 DP4A GEMM Kernel
 *
 * 每个 thread 处理一个输出元素
 * 这是教学版本，不是最优实现
 */
__global__ void kernel_gemm_q4_0_q8_1_simple(
    const block_q4_0* __restrict__ weight,
    const block_q8_1* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;

    if (m >= M || n >= N) return;

    const int num_blocks = K / 32;
    float sum = 0.0f;

    for (int b = 0; b < num_blocks; b++) {
        const block_q4_0& w_block = weight[m * num_blocks + b];
        const block_q8_1& a_block = activation[n * num_blocks + b];

        float d_w = __half2float(w_block.d);
        float d_a = __half2float(__low2half(a_block.ds));
        float s_a = __half2float(__high2half(a_block.ds));

        // 计算点积 (使用 DP4A)
        int sumi = 0;

        // 处理 32 个元素 (4 个 int32, 每个包含 8 个 4-bit 权重)
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            // 加载 4 字节 Q4_0 数据 (8 个 nibbles)
            int q4_packed = *((const int*)(w_block.qs + i * 4));

            // 加载 8 字节 Q8_1 数据 (8 个 int8)
            int q8_0 = *((const int*)(a_block.qs + i * 8));
            int q8_1 = *((const int*)(a_block.qs + i * 8 + 4));

            // 提取并展开 4-bit 到 8-bit
            // 低 nibbles: w0, w2, w4, w6
            // 高 nibbles: w1, w3, w5, w7
            int lo = (q4_packed >> 0) & 0x0F0F0F0F;
            int hi = (q4_packed >> 4) & 0x0F0F0F0F;

            // 交错排列以匹配激活顺序
            int w_lo = ((lo & 0x000000FF) <<  0) |
                       ((hi & 0x000000FF) <<  8) |
                       ((lo & 0x0000FF00) <<  8) |
                       ((hi & 0x0000FF00) << 16);

            int w_hi = ((lo & 0x00FF0000) >> 16) |
                       ((hi & 0x00FF0000) >>  8) |
                       ((lo & 0xFF000000) >>  8) |
                       ((hi & 0xFF000000) >>  0);

            // DP4A 计算 (带偏移 8)
            // 注意: DP4A 使用的是 unsigned-signed 版本
            // 这里简化使用解包后的计算
            char4 w0_4 = *reinterpret_cast<char4*>(&w_lo);
            char4 w1_4 = *reinterpret_cast<char4*>(&w_hi);
            char4 a0_4 = *reinterpret_cast<char4*>(&q8_0);
            char4 a1_4 = *reinterpret_cast<char4*>(&q8_1);

            sumi += (w0_4.x - 8) * a0_4.x + (w0_4.y - 8) * a0_4.y +
                    (w0_4.z - 8) * a0_4.z + (w0_4.w - 8) * a0_4.w;
            sumi += (w1_4.x - 8) * a1_4.x + (w1_4.y - 8) * a1_4.y +
                    (w1_4.z - 8) * a1_4.z + (w1_4.w - 8) * a1_4.w;
        }

        // 应用补偿公式
        sum += d_w * (d_a * sumi - 8.0f * s_a);
    }

    output[m * N + n] = sum;
}

// ============================================================================
// 测试用例
// ============================================================================

class GemmQ4Test : public testing::TestCase {
public:
    const char* name() const override {
        return "GEMM_Q4_0_Q8_1";
    }

    const char* description() const override {
        return "Q4_0 weights x Q8_1 activations matrix multiplication";
    }

    float threshold() const override {
        return 0.015f;  // 1.5% NMSE
    }

    void setup(const testing::TestConfig& config) override {
        M_ = config.M;
        N_ = config.N;
        K_ = config.K;

        int num_blocks = K_ / 32;
        output_size_ = M_ * N_;

        // 分配主机内存
        h_weight_f32_ = new float[M_ * K_];
        h_act_f32_ = new float[N_ * K_];
        h_weight_q4_ = new block_q4_0[M_ * num_blocks];
        h_act_q8_ = new block_q8_1[N_ * num_blocks];
        output_ref_ = new float[M_ * N_];
        output_test_ = new float[M_ * N_];

        // 生成随机数据
        testing::DataGenerator gen(config.seed);
        gen.generate(h_weight_f32_, M_ * K_,
                     testing::DataGenerator::NORMAL, 0.0f, 0.5f);
        gen.generate(h_act_f32_, N_ * K_,
                     testing::DataGenerator::NORMAL, 0.0f, 0.5f);

        // 量化
        testing::quantize::to_q4_0(h_weight_f32_, h_weight_q4_, M_ * K_);
        testing::quantize::to_q8_1(h_act_f32_, h_act_q8_, N_ * K_);

        // 分配设备内存
        size_t weight_bytes = M_ * num_blocks * sizeof(block_q4_0);
        size_t act_bytes = N_ * num_blocks * sizeof(block_q8_1);
        size_t output_bytes = M_ * N_ * sizeof(float);

        CUDA_CHECK(cudaMalloc(&d_weight_, weight_bytes));
        CUDA_CHECK(cudaMalloc(&d_act_, act_bytes));
        CUDA_CHECK(cudaMalloc(&d_output_, output_bytes));

        CUDA_CHECK(cudaMemcpy(d_weight_, h_weight_q4_, weight_bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_act_, h_act_q8_, act_bytes,
                              cudaMemcpyHostToDevice));

        printf("  Weight: [%d x %d] -> Q4_0 blocks: %d\n", M_, K_, M_ * num_blocks);
        printf("  Activation: [%d x %d] -> Q8_1 blocks: %d\n", N_, K_, N_ * num_blocks);
        printf("  Output: [%d x %d]\n", M_, N_);
    }

    void run_reference() override {
        cpu_gemm_q4_0_q8_1_reference(h_weight_q4_, h_act_q8_, output_ref_,
                                     M_, N_, K_);
    }

    void run_kernel() override {
        dim3 block(16, 16);
        dim3 grid((M_ + block.x - 1) / block.x,
                  (N_ + block.y - 1) / block.y);

        kernel_gemm_q4_0_q8_1_simple<<<grid, block>>>(
            d_weight_, d_act_, d_output_, M_, N_, K_
        );
        KERNEL_CHECK();

        CUDA_CHECK(cudaMemcpy(output_test_, d_output_, M_ * N_ * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    void cleanup() override {
        delete[] h_weight_f32_;
        delete[] h_act_f32_;
        delete[] h_weight_q4_;
        delete[] h_act_q8_;
        delete[] output_ref_;
        delete[] output_test_;

        cudaFree(d_weight_);
        cudaFree(d_act_);
        cudaFree(d_output_);
    }

private:
    int M_, N_, K_;

    float* h_weight_f32_ = nullptr;
    float* h_act_f32_ = nullptr;
    block_q4_0* h_weight_q4_ = nullptr;
    block_q8_1* h_act_q8_ = nullptr;

    block_q4_0* d_weight_ = nullptr;
    block_q8_1* d_act_ = nullptr;
    float* d_output_ = nullptr;
};

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    // 默认配置
    testing::TestConfig config;
    config.M = 4;
    config.N = 512;
    config.K = 1024;
    config.seed = 42;
    config.verbose = true;
    config.print_samples = true;
    config.num_samples = 10;

    // 简单命令行解析
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-M") == 0 && i + 1 < argc) {
            config.M = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-N") == 0 && i + 1 < argc) {
            config.N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-K") == 0 && i + 1 < argc) {
            config.K = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-seed") == 0 && i + 1 < argc) {
            config.seed = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-q") == 0) {
            config.verbose = false;
            config.print_samples = false;
        }
    }

    // 运行测试
    testing::TestRunner runner;
    GemmQ4Test test;
    runner.add(&test);
    runner.run_all(config);

    return 0;
}
