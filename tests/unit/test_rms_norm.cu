/**
 * @file tests/unit/test_rms_norm.cu
 * @brief RMS Normalization 单元测试
 */

#include "../framework/test_framework.cuh"
#include "../../kernels/normalization/rms_norm.cuh"

// ============================================================================
// 测试用例: RMS Norm FP32
// ============================================================================

class RmsNormF32Test : public testing::TestCase {
public:
    const char* name() const override {
        return "RMS_NORM_F32";
    }

    const char* description() const override {
        return "RMS Normalization (FP32)";
    }

    float threshold() const override {
        return 1e-5f;  // 高精度要求
    }

    void setup(const testing::TestConfig& config) override {
        n_rows_ = config.M;  // batch_size * seq_len
        n_cols_ = config.K;  // hidden_size
        eps_ = 1e-5f;
        output_size_ = n_rows_ * n_cols_;

        // 分配主机内存
        h_input_ = new float[n_rows_ * n_cols_];
        h_weight_ = new float[n_cols_];
        output_ref_ = new float[n_rows_ * n_cols_];
        output_test_ = new float[n_rows_ * n_cols_];

        // 生成随机数据
        testing::DataGenerator gen(config.seed);
        gen.generate(h_input_, n_rows_ * n_cols_,
                     testing::DataGenerator::NORMAL, 0.0f, 1.0f);
        gen.generate(h_weight_, n_cols_,
                     testing::DataGenerator::UNIFORM, 0.5f, 1.5f);

        // 分配设备内存
        CUDA_CHECK(cudaMalloc(&d_input_, n_rows_ * n_cols_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_weight_, n_cols_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output_, n_rows_ * n_cols_ * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_input_, h_input_,
                              n_rows_ * n_cols_ * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_weight_, h_weight_,
                              n_cols_ * sizeof(float),
                              cudaMemcpyHostToDevice));

        printf("  Input shape: [%d, %d]\n", n_rows_, n_cols_);
        printf("  Epsilon: %e\n", eps_);
    }

    void run_reference() override {
        rms_norm_cpu_f32(h_input_, h_weight_, output_ref_,
                         n_rows_, n_cols_, eps_);
    }

    void run_kernel() override {
        rms_norm_forward_f32(d_input_, d_weight_, d_output_,
                             n_rows_, n_cols_, eps_);
        KERNEL_CHECK();

        CUDA_CHECK(cudaMemcpy(output_test_, d_output_,
                              n_rows_ * n_cols_ * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    void cleanup() override {
        delete[] h_input_;
        delete[] h_weight_;
        delete[] output_ref_;
        delete[] output_test_;

        cudaFree(d_input_);
        cudaFree(d_weight_);
        cudaFree(d_output_);
    }

private:
    int n_rows_;
    int n_cols_;
    float eps_;
    float* h_input_ = nullptr;
    float* h_weight_ = nullptr;
    float* d_input_ = nullptr;
    float* d_weight_ = nullptr;
    float* d_output_ = nullptr;
};

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    testing::TestConfig config;
    config.M = 32;    // batch_size * seq_len
    config.K = 4096;  // hidden_size (typical LLaMA)
    config.seed = 42;
    config.verbose = true;
    config.print_samples = true;
    config.num_samples = 10;

    // 简单命令行解析
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-M") == 0 && i + 1 < argc) {
            config.M = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-K") == 0 && i + 1 < argc) {
            config.K = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-seed") == 0 && i + 1 < argc) {
            config.seed = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-q") == 0) {
            config.verbose = false;
            config.print_samples = false;
        }
    }

    testing::TestRunner runner;
    RmsNormF32Test test;
    runner.add(&test);
    runner.run_all(config);

    return 0;
}
