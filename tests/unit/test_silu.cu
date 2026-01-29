/**
 * @file tests/unit/test_silu.cu
 * @brief SiLU 激活函数单元测试
 */

#include "../framework/test_framework.cuh"
#include "../../kernels/activation/silu.cuh"

// ============================================================================
// 测试用例: SiLU FP32
// ============================================================================

class SiluF32Test : public testing::TestCase {
public:
    const char* name() const override {
        return "SILU_F32";
    }

    const char* description() const override {
        return "SiLU activation function (FP32)";
    }

    float threshold() const override {
        return 1e-6f;  // 高精度要求
    }

    void setup(const testing::TestConfig& config) override {
        n_ = config.N * config.K;  // 使用 N*K 作为元素数量
        output_size_ = n_;

        // 分配主机内存
        h_input_ = new float[n_];
        output_ref_ = new float[n_];
        output_test_ = new float[n_];

        // 生成随机数据 (范围 [-5, 5])
        testing::DataGenerator gen(config.seed);
        gen.generate(h_input_, n_, testing::DataGenerator::UNIFORM, -5.0f, 5.0f);

        // 分配设备内存
        CUDA_CHECK(cudaMalloc(&d_input_, n_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output_, n_ * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_input_, h_input_, n_ * sizeof(float),
                              cudaMemcpyHostToDevice));

        printf("  Input size: %d elements\n", n_);
    }

    void run_reference() override {
        silu_cpu_f32(h_input_, output_ref_, n_);
    }

    void run_kernel() override {
        silu_forward_f32(d_input_, d_output_, n_);
        KERNEL_CHECK();

        CUDA_CHECK(cudaMemcpy(output_test_, d_output_, n_ * sizeof(float),
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
    int n_;
    float* h_input_ = nullptr;
    float* d_input_ = nullptr;
    float* d_output_ = nullptr;
};

// ============================================================================
// 测试用例: SiLU FP32 向量化
// ============================================================================

class SiluF32Vec4Test : public testing::TestCase {
public:
    const char* name() const override {
        return "SILU_F32_VEC4";
    }

    const char* description() const override {
        return "SiLU activation function (FP32, vectorized)";
    }

    float threshold() const override {
        return 1e-6f;
    }

    void setup(const testing::TestConfig& config) override {
        // 确保 n 是 4 的倍数
        n_ = ((config.N * config.K) / 4) * 4;
        output_size_ = n_;

        h_input_ = new float[n_];
        output_ref_ = new float[n_];
        output_test_ = new float[n_];

        testing::DataGenerator gen(config.seed);
        gen.generate(h_input_, n_, testing::DataGenerator::UNIFORM, -5.0f, 5.0f);

        CUDA_CHECK(cudaMalloc(&d_input_, n_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output_, n_ * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_input_, h_input_, n_ * sizeof(float),
                              cudaMemcpyHostToDevice));

        printf("  Input size: %d elements (4-aligned)\n", n_);
    }

    void run_reference() override {
        silu_cpu_f32(h_input_, output_ref_, n_);
    }

    void run_kernel() override {
        silu_forward_f32_vec4(d_input_, d_output_, n_);
        KERNEL_CHECK();

        CUDA_CHECK(cudaMemcpy(output_test_, d_output_, n_ * sizeof(float),
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
    int n_;
    float* h_input_ = nullptr;
    float* d_input_ = nullptr;
    float* d_output_ = nullptr;
};

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    testing::TestConfig config;
    config.N = 512;
    config.K = 1024;
    config.seed = 42;
    config.verbose = true;
    config.print_samples = true;
    config.num_samples = 10;

    // 简单命令行解析
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            config.N = atoi(argv[++i]);
            config.K = 1;
        } else if (strcmp(argv[i], "-seed") == 0 && i + 1 < argc) {
            config.seed = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-q") == 0) {
            config.verbose = false;
            config.print_samples = false;
        }
    }

    testing::TestRunner runner;
    SiluF32Test test1;
    SiluF32Vec4Test test2;

    runner.add(&test1);
    runner.add(&test2);
    runner.run_all(config);

    return 0;
}
