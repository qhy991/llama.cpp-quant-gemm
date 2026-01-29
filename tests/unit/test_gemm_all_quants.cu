/**
 * @file tests/unit/test_gemm_all_quants.cu
 * @brief 所有量化格式 GEMM 的综合测试
 *
 * 测试:
 * - Q4_0 × Q8_1
 * - Q4_1 × Q8_1
 * - Q5_0 × Q8_1
 * - Q5_1 × Q8_1
 * - Q8_0 × Q8_1
 */

#include "../framework/test_framework.cuh"
#include "../../kernels/gemm/gemm_quant_formats.cuh"

// ============================================================================
// CPU 参考实现
// ============================================================================

// Q4_0 × Q8_1 CPU 参考
// 注意: llama.cpp 的补偿公式是 d_w * (d_a * sumi - 8 * s_a)
// 其中 sumi = Σ q_w * q_a, q_w 是原始存储值 [0,15], 不减 8
void cpu_gemm_q4_0_q8_1(
    const block_q4_0* weight,
    const block_q8_1* activation,
    float* output,
    int M, int N, int K
) {
    const int num_blocks = K / 32;

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;

            for (int b = 0; b < num_blocks; b++) {
                const block_q4_0& w = weight[m * num_blocks + b];
                const block_q8_1& a = activation[n * num_blocks + b];

                float d_w = __half2float(w.d);
                float d_a = __half2float(__low2half(a.ds));
                float s_a = __half2float(__high2half(a.ds));

                // sumi 使用原始存储值，不减 8
                // Q4_0 布局: qs[i] 的低 nibble = x[i], 高 nibble = x[i+16]
                int sumi = 0;
                for (int i = 0; i < 16; i++) {
                    int w0 = (w.qs[i] & 0x0F);       // x[i]
                    int w1 = ((w.qs[i] >> 4) & 0x0F); // x[i+16]
                    sumi += w0 * a.qs[i] + w1 * a.qs[i + 16];
                }

                // 补偿公式处理 -8 的偏移
                sum += d_w * (d_a * sumi - 8.0f * s_a);
            }

            output[m * N + n] = sum;
        }
    }
}

// Q4_1 × Q8_1 CPU 参考
void cpu_gemm_q4_1_q8_1(
    const block_q4_1* weight,
    const block_q8_1* activation,
    float* output,
    int M, int N, int K
) {
    const int num_blocks = K / 32;

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;

            for (int b = 0; b < num_blocks; b++) {
                const block_q4_1& w = weight[m * num_blocks + b];
                const block_q8_1& a = activation[n * num_blocks + b];

                float d_w = __half2float(w.d);
                float m_w = __half2float(w.m);
                float d_a = __half2float(__low2half(a.ds));
                float s_a = __half2float(__high2half(a.ds));

                int sumi = 0;
                // Q4_1 布局: qs[i] 的低 nibble = x[i], 高 nibble = x[i+16]
                for (int i = 0; i < 16; i++) {
                    int w0 = w.qs[i] & 0x0F;
                    int w1 = (w.qs[i] >> 4) & 0x0F;
                    sumi += w0 * a.qs[i] + w1 * a.qs[i + 16];
                }

                sum += d_w * d_a * sumi + m_w * s_a / 4.0f;
            }

            output[m * N + n] = sum;
        }
    }
}

// Q5_0 × Q8_1 CPU 参考
void cpu_gemm_q5_0_q8_1(
    const block_q5_0* weight,
    const block_q8_1* activation,
    float* output,
    int M, int N, int K
) {
    const int num_blocks = K / 32;

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;

            for (int b = 0; b < num_blocks; b++) {
                const block_q5_0& w = weight[m * num_blocks + b];
                const block_q8_1& a = activation[n * num_blocks + b];

                float d_w = __half2float(w.d);
                float d_a = __half2float(__low2half(a.ds));
                float s_a = __half2float(__high2half(a.ds));

                uint32_t qh;
                memcpy(&qh, w.qh, sizeof(qh));

                // sumi 使用原始 5-bit 值 [0, 31], 不减 16
                // Q5_0 布局: qs[i] 的低 nibble = x[i], 高 nibble = x[i+16]
                // qh 布局: bits 0-15 对应 x[0..15], bits 16-31 对应 x[16..31]
                int sumi = 0;
                for (int i = 0; i < 16; i++) {
                    int w0 = (w.qs[i] & 0x0F) | (((qh >> i) & 1) << 4);
                    int w1 = ((w.qs[i] >> 4) & 0x0F) | (((qh >> (i + 16)) & 1) << 4);
                    sumi += w0 * a.qs[i] + w1 * a.qs[i + 16];
                }

                // 补偿公式处理 -16 的偏移
                sum += d_w * (d_a * sumi - 16.0f * s_a);
            }

            output[m * N + n] = sum;
        }
    }
}

// Q5_1 × Q8_1 CPU 参考
void cpu_gemm_q5_1_q8_1(
    const block_q5_1* weight,
    const block_q8_1* activation,
    float* output,
    int M, int N, int K
) {
    const int num_blocks = K / 32;

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;

            for (int b = 0; b < num_blocks; b++) {
                const block_q5_1& w = weight[m * num_blocks + b];
                const block_q8_1& a = activation[n * num_blocks + b];

                float d_w = __half2float(w.d);
                float m_w = __half2float(w.m);
                float d_a = __half2float(__low2half(a.ds));
                float s_a = __half2float(__high2half(a.ds));

                uint32_t qh;
                memcpy(&qh, w.qh, sizeof(qh));

                int sumi = 0;
                // Q5_1 布局: qs[i] 的低 nibble = x[i], 高 nibble = x[i+16]
                // qh 布局: bits 0-15 对应 x[0..15], bits 16-31 对应 x[16..31]
                for (int i = 0; i < 16; i++) {
                    int w0 = (w.qs[i] & 0x0F) | (((qh >> i) & 1) << 4);
                    int w1 = ((w.qs[i] >> 4) & 0x0F) | (((qh >> (i + 16)) & 1) << 4);
                    sumi += w0 * a.qs[i] + w1 * a.qs[i + 16];
                }

                sum += d_w * d_a * sumi + m_w * s_a / 4.0f;
            }

            output[m * N + n] = sum;
        }
    }
}

// Q8_0 × Q8_1 CPU 参考
void cpu_gemm_q8_0_q8_1(
    const block_q8_0* weight,
    const block_q8_1* activation,
    float* output,
    int M, int N, int K
) {
    const int num_blocks = K / 32;

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;

            for (int b = 0; b < num_blocks; b++) {
                const block_q8_0& w = weight[m * num_blocks + b];
                const block_q8_1& a = activation[n * num_blocks + b];

                float d_w = __half2float(w.d);
                float d_a = __half2float(__low2half(a.ds));

                int sumi = 0;
                for (int i = 0; i < 32; i++) {
                    sumi += w.qs[i] * a.qs[i];
                }

                sum += d_w * d_a * sumi;
            }

            output[m * N + n] = sum;
        }
    }
}

// ============================================================================
// 测试用例模板
// ============================================================================

template<typename BlockW>
class GemmQuantTest : public testing::TestCase {
public:
    GemmQuantTest(const char* name, const char* desc, float thresh)
        : name_(name), desc_(desc), thresh_(thresh) {}

    const char* name() const override { return name_; }
    const char* description() const override { return desc_; }
    float threshold() const override { return thresh_; }

    void setup(const testing::TestConfig& config) override {
        M_ = config.M;
        N_ = config.N;
        K_ = config.K;

        int num_blocks = K_ / 32;
        output_size_ = M_ * N_;

        // 分配内存
        h_weight_f32_ = new float[M_ * K_];
        h_act_f32_ = new float[N_ * K_];
        h_weight_q_ = new BlockW[M_ * num_blocks];
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
        quantize_weight(h_weight_f32_, h_weight_q_, M_ * K_);
        testing::quantize::to_q8_1(h_act_f32_, h_act_q8_, N_ * K_);

        // 分配 GPU 内存
        size_t weight_bytes = M_ * num_blocks * sizeof(BlockW);
        size_t act_bytes = N_ * num_blocks * sizeof(block_q8_1);

        CUDA_CHECK(cudaMalloc(&d_weight_, weight_bytes));
        CUDA_CHECK(cudaMalloc(&d_act_, act_bytes));
        CUDA_CHECK(cudaMalloc(&d_output_, M_ * N_ * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_weight_, h_weight_q_, weight_bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_act_, h_act_q8_, act_bytes,
                              cudaMemcpyHostToDevice));

        printf("  Dimensions: M=%d, N=%d, K=%d\n", M_, N_, K_);
    }

    void cleanup() override {
        delete[] h_weight_f32_;
        delete[] h_act_f32_;
        delete[] h_weight_q_;
        delete[] h_act_q8_;
        delete[] output_ref_;
        delete[] output_test_;

        cudaFree(d_weight_);
        cudaFree(d_act_);
        cudaFree(d_output_);
    }

protected:
    virtual void quantize_weight(const float* src, BlockW* dst, int n) = 0;

    const char* name_;
    const char* desc_;
    float thresh_;

    int M_, N_, K_;
    float* h_weight_f32_ = nullptr;
    float* h_act_f32_ = nullptr;
    BlockW* h_weight_q_ = nullptr;
    block_q8_1* h_act_q8_ = nullptr;

    BlockW* d_weight_ = nullptr;
    block_q8_1* d_act_ = nullptr;
    float* d_output_ = nullptr;
};

// ============================================================================
// 具体测试类
// ============================================================================

class GemmQ4_0Test : public GemmQuantTest<block_q4_0> {
public:
    GemmQ4_0Test() : GemmQuantTest("GEMM_Q4_0_Q8_1",
        "Q4_0 weights x Q8_1 activations", 0.015f) {}

    void quantize_weight(const float* src, block_q4_0* dst, int n) override {
        testing::quantize::to_q4_0(src, dst, n);
    }

    void run_reference() override {
        cpu_gemm_q4_0_q8_1(h_weight_q_, h_act_q8_, output_ref_, M_, N_, K_);
    }

    void run_kernel() override {
        gemm_q4_0_q8_1(d_weight_, d_act_, d_output_, M_, N_, K_);
        KERNEL_CHECK();
        CUDA_CHECK(cudaMemcpy(output_test_, d_output_, M_ * N_ * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }
};

class GemmQ4_1Test : public GemmQuantTest<block_q4_1> {
public:
    GemmQ4_1Test() : GemmQuantTest("GEMM_Q4_1_Q8_1",
        "Q4_1 weights x Q8_1 activations (asymmetric)", 0.015f) {}

    void quantize_weight(const float* src, block_q4_1* dst, int n) override {
        testing::quantize::to_q4_1(src, dst, n);
    }

    void run_reference() override {
        cpu_gemm_q4_1_q8_1(h_weight_q_, h_act_q8_, output_ref_, M_, N_, K_);
    }

    void run_kernel() override {
        gemm_q4_1_q8_1(d_weight_, d_act_, d_output_, M_, N_, K_);
        KERNEL_CHECK();
        CUDA_CHECK(cudaMemcpy(output_test_, d_output_, M_ * N_ * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }
};

class GemmQ5_0Test : public GemmQuantTest<block_q5_0> {
public:
    GemmQ5_0Test() : GemmQuantTest("GEMM_Q5_0_Q8_1",
        "Q5_0 weights x Q8_1 activations", 0.01f) {}

    void quantize_weight(const float* src, block_q5_0* dst, int n) override {
        testing::quantize::to_q5_0(src, dst, n);
    }

    void run_reference() override {
        cpu_gemm_q5_0_q8_1(h_weight_q_, h_act_q8_, output_ref_, M_, N_, K_);
    }

    void run_kernel() override {
        gemm_q5_0_q8_1(d_weight_, d_act_, d_output_, M_, N_, K_);
        KERNEL_CHECK();
        CUDA_CHECK(cudaMemcpy(output_test_, d_output_, M_ * N_ * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }
};

class GemmQ5_1Test : public GemmQuantTest<block_q5_1> {
public:
    GemmQ5_1Test() : GemmQuantTest("GEMM_Q5_1_Q8_1",
        "Q5_1 weights x Q8_1 activations (asymmetric)", 0.01f) {}

    void quantize_weight(const float* src, block_q5_1* dst, int n) override {
        testing::quantize::to_q5_1(src, dst, n);
    }

    void run_reference() override {
        cpu_gemm_q5_1_q8_1(h_weight_q_, h_act_q8_, output_ref_, M_, N_, K_);
    }

    void run_kernel() override {
        gemm_q5_1_q8_1(d_weight_, d_act_, d_output_, M_, N_, K_);
        KERNEL_CHECK();
        CUDA_CHECK(cudaMemcpy(output_test_, d_output_, M_ * N_ * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }
};

class GemmQ8_0Test : public GemmQuantTest<block_q8_0> {
public:
    GemmQ8_0Test() : GemmQuantTest("GEMM_Q8_0_Q8_1",
        "Q8_0 weights x Q8_1 activations", 0.005f) {}

    void quantize_weight(const float* src, block_q8_0* dst, int n) override {
        testing::quantize::to_q8_0(src, dst, n);
    }

    void run_reference() override {
        cpu_gemm_q8_0_q8_1(h_weight_q_, h_act_q8_, output_ref_, M_, N_, K_);
    }

    void run_kernel() override {
        gemm_q8_0_q8_1(d_weight_, d_act_, d_output_, M_, N_, K_);
        KERNEL_CHECK();
        CUDA_CHECK(cudaMemcpy(output_test_, d_output_, M_ * N_ * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }
};

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    testing::TestConfig config;
    config.M = 4;
    config.N = 512;
    config.K = 1024;
    config.seed = 42;
    config.verbose = true;
    config.print_samples = true;
    config.num_samples = 5;

    // 命令行解析
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

    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║     Quantized GEMM Test Suite - All Formats                   ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");

    testing::TestRunner runner;

    GemmQ4_0Test test_q4_0;
    GemmQ4_1Test test_q4_1;
    GemmQ5_0Test test_q5_0;
    GemmQ5_1Test test_q5_1;
    GemmQ8_0Test test_q8_0;

    runner.add(&test_q4_0);
    runner.add(&test_q4_1);
    runner.add(&test_q5_0);
    runner.add(&test_q5_1);
    runner.add(&test_q8_0);

    runner.run_all(config);

    return 0;
}
