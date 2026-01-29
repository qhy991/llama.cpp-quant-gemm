/**
 * @file tests/unit/test_softmax.cu
 * @brief Softmax 单元测试
 */

#include "../framework/test_framework.cuh"
#include "../../kernels/attention/softmax.cuh"

// ============================================================================
// 测试用例: Softmax FP32
// ============================================================================

class SoftmaxF32Test : public testing::TestCase {
public:
    const char* name() const override {
        return "SOFTMAX_F32";
    }

    const char* description() const override {
        return "Softmax attention scores (FP32)";
    }

    float threshold() const override {
        return 1e-5f;
    }

    void setup(const testing::TestConfig& config) override {
        n_rows_ = config.M;   // batch_size * n_heads
        n_cols_ = config.N;   // seq_len
        scale_ = 1.0f / sqrtf((float)config.K);  // 1/sqrt(d_k)
        output_size_ = n_rows_ * n_cols_;

        // 分配主机内存
        h_input_ = new float[n_rows_ * n_cols_];
        output_ref_ = new float[n_rows_ * n_cols_];
        output_test_ = new float[n_rows_ * n_cols_];

        // 生成随机数据 (注意力分数通常在 [-5, 5] 范围)
        testing::DataGenerator gen(config.seed);
        gen.generate(h_input_, n_rows_ * n_cols_,
                     testing::DataGenerator::NORMAL, 0.0f, 2.0f);

        // 分配设备内存
        CUDA_CHECK(cudaMalloc(&d_input_, n_rows_ * n_cols_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output_, n_rows_ * n_cols_ * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_input_, h_input_,
                              n_rows_ * n_cols_ * sizeof(float),
                              cudaMemcpyHostToDevice));

        printf("  Input shape: [%d, %d]\n", n_rows_, n_cols_);
        printf("  Scale: %.6f\n", scale_);
    }

    void run_reference() override {
        softmax_cpu_f32(h_input_, output_ref_, n_rows_, n_cols_, scale_);
    }

    void run_kernel() override {
        softmax_forward_f32(d_input_, d_output_, n_rows_, n_cols_, scale_);
        KERNEL_CHECK();

        CUDA_CHECK(cudaMemcpy(output_test_, d_output_,
                              n_rows_ * n_cols_ * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    void cleanup() override {
        delete[] h_input_;
        delete[] output_ref_;
        delete[] output_test_;

        cudaFree(d_input_);
        cudaFree(d_output_);
    }

private:
    int n_rows_;
    int n_cols_;
    float scale_;
    float* h_input_ = nullptr;
    float* d_input_ = nullptr;
    float* d_output_ = nullptr;
};

// ============================================================================
// 测试用例: Softmax 小序列
// ============================================================================

class SoftmaxSmallTest : public testing::TestCase {
public:
    const char* name() const override {
        return "SOFTMAX_F32_SMALL";
    }

    const char* description() const override {
        return "Softmax for small sequences (n <= 32)";
    }

    float threshold() const override {
        return 1e-5f;
    }

    void setup(const testing::TestConfig& config) override {
        n_rows_ = config.M * 8;  // 更多行
        n_cols_ = 32;            // 小序列长度
        scale_ = 1.0f;
        output_size_ = n_rows_ * n_cols_;

        h_input_ = new float[n_rows_ * n_cols_];
        output_ref_ = new float[n_rows_ * n_cols_];
        output_test_ = new float[n_rows_ * n_cols_];

        testing::DataGenerator gen(config.seed);
        gen.generate(h_input_, n_rows_ * n_cols_,
                     testing::DataGenerator::NORMAL, 0.0f, 2.0f);

        CUDA_CHECK(cudaMalloc(&d_input_, n_rows_ * n_cols_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output_, n_rows_ * n_cols_ * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_input_, h_input_,
                              n_rows_ * n_cols_ * sizeof(float),
                              cudaMemcpyHostToDevice));

        printf("  Input shape: [%d, %d] (small seq)\n", n_rows_, n_cols_);
    }

    void run_reference() override {
        softmax_cpu_f32(h_input_, output_ref_, n_rows_, n_cols_, scale_);
    }

    void run_kernel() override {
        softmax_forward_f32(d_input_, d_output_, n_rows_, n_cols_, scale_);
        KERNEL_CHECK();

        CUDA_CHECK(cudaMemcpy(output_test_, d_output_,
                              n_rows_ * n_cols_ * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    void cleanup() override {
        delete[] h_input_;
        delete[] output_ref_;
        delete[] output_test_;

        cudaFree(d_input_);
        cudaFree(d_output_);
    }

private:
    int n_rows_;
    int n_cols_;
    float scale_;
    float* h_input_ = nullptr;
    float* d_input_ = nullptr;
    float* d_output_ = nullptr;
};

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    testing::TestConfig config;
    config.M = 32;    // batch * n_heads
    config.N = 512;   // seq_len
    config.K = 64;    // head_dim (for scale calculation)
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
        } else if (strcmp(argv[i], "-seed") == 0 && i + 1 < argc) {
            config.seed = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-q") == 0) {
            config.verbose = false;
            config.print_samples = false;
        }
    }

    testing::TestRunner runner;
    SoftmaxF32Test test1;
    SoftmaxSmallTest test2;

    runner.add(&test1);
    runner.add(&test2);
    runner.run_all(config);

    return 0;
}
