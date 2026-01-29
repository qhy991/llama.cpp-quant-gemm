/**
 * @file tests/unit/test_rope.cu
 * @brief RoPE (Rotary Position Embedding) 单元测试
 */

#include "../framework/test_framework.cuh"
#include "../../kernels/attention/rope.cuh"

// ============================================================================
// 测试用例: RoPE FP32
// ============================================================================

class RopeF32Test : public testing::TestCase {
public:
    const char* name() const override {
        return "ROPE_F32";
    }

    const char* description() const override {
        return "Rotary Position Embedding (FP32)";
    }

    float threshold() const override {
        return 1e-5f;
    }

    void setup(const testing::TestConfig& config) override {
        n_heads_ = config.M;
        head_dim_ = 128;  // 典型的 head_dim
        pos_ = 42;        // 测试位置
        base_ = 10000.0f;
        freq_scale_ = 1.0f;
        output_size_ = n_heads_ * head_dim_;

        // 分配主机内存
        h_input_ = new float[n_heads_ * head_dim_];
        h_ref_ = new float[n_heads_ * head_dim_];
        output_ref_ = new float[n_heads_ * head_dim_];
        output_test_ = new float[n_heads_ * head_dim_];

        // 生成随机数据
        testing::DataGenerator gen(config.seed);
        gen.generate(h_input_, n_heads_ * head_dim_,
                     testing::DataGenerator::NORMAL, 0.0f, 1.0f);

        // 复制到参考数组
        memcpy(h_ref_, h_input_, n_heads_ * head_dim_ * sizeof(float));

        // 分配设备内存
        CUDA_CHECK(cudaMalloc(&d_data_, n_heads_ * head_dim_ * sizeof(float)));

        printf("  Shape: [%d heads, %d dim]\n", n_heads_, head_dim_);
        printf("  Position: %d\n", pos_);
        printf("  Base: %.1f\n", base_);
    }

    void run_reference() override {
        // CPU 参考实现 (原地操作)
        rope_cpu_f32(h_ref_, n_heads_, head_dim_, pos_, base_, freq_scale_);
        memcpy(output_ref_, h_ref_, n_heads_ * head_dim_ * sizeof(float));
    }

    void run_kernel() override {
        // 复制输入到设备
        CUDA_CHECK(cudaMemcpy(d_data_, h_input_,
                              n_heads_ * head_dim_ * sizeof(float),
                              cudaMemcpyHostToDevice));

        // 运行 kernel (原地操作)
        rope_forward_f32(d_data_, n_heads_, head_dim_, pos_, base_, freq_scale_);
        KERNEL_CHECK();

        // 复制结果回主机
        CUDA_CHECK(cudaMemcpy(output_test_, d_data_,
                              n_heads_ * head_dim_ * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    void cleanup() override {
        delete[] h_input_;
        delete[] h_ref_;
        delete[] output_ref_;
        delete[] output_test_;

        cudaFree(d_data_);
    }

private:
    int n_heads_;
    int head_dim_;
    int pos_;
    float base_;
    float freq_scale_;
    float* h_input_ = nullptr;
    float* h_ref_ = nullptr;
    float* d_data_ = nullptr;
};

// ============================================================================
// 测试用例: RoPE 交错布局
// ============================================================================

class RopeInterleavedTest : public testing::TestCase {
public:
    const char* name() const override {
        return "ROPE_INTERLEAVED_F32";
    }

    const char* description() const override {
        return "RoPE with interleaved layout";
    }

    float threshold() const override {
        return 1e-5f;
    }

    void setup(const testing::TestConfig& config) override {
        n_heads_ = config.M;
        head_dim_ = 128;
        pos_ = 100;
        base_ = 10000.0f;
        freq_scale_ = 1.0f;
        output_size_ = n_heads_ * head_dim_;

        h_input_ = new float[n_heads_ * head_dim_];
        h_ref_ = new float[n_heads_ * head_dim_];
        output_ref_ = new float[n_heads_ * head_dim_];
        output_test_ = new float[n_heads_ * head_dim_];

        testing::DataGenerator gen(config.seed + 1);
        gen.generate(h_input_, n_heads_ * head_dim_,
                     testing::DataGenerator::NORMAL, 0.0f, 1.0f);

        memcpy(h_ref_, h_input_, n_heads_ * head_dim_ * sizeof(float));

        CUDA_CHECK(cudaMalloc(&d_data_, n_heads_ * head_dim_ * sizeof(float)));

        printf("  Shape: [%d heads, %d dim] (interleaved)\n", n_heads_, head_dim_);
        printf("  Position: %d\n", pos_);
    }

    void run_reference() override {
        rope_interleaved_cpu_f32(h_ref_, n_heads_, head_dim_, pos_, base_, freq_scale_);
        memcpy(output_ref_, h_ref_, n_heads_ * head_dim_ * sizeof(float));
    }

    void run_kernel() override {
        CUDA_CHECK(cudaMemcpy(d_data_, h_input_,
                              n_heads_ * head_dim_ * sizeof(float),
                              cudaMemcpyHostToDevice));

        rope_interleaved_forward_f32(d_data_, n_heads_, head_dim_, pos_, base_, freq_scale_);
        KERNEL_CHECK();

        CUDA_CHECK(cudaMemcpy(output_test_, d_data_,
                              n_heads_ * head_dim_ * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    void cleanup() override {
        delete[] h_input_;
        delete[] h_ref_;
        delete[] output_ref_;
        delete[] output_test_;

        cudaFree(d_data_);
    }

private:
    int n_heads_;
    int head_dim_;
    int pos_;
    float base_;
    float freq_scale_;
    float* h_input_ = nullptr;
    float* h_ref_ = nullptr;
    float* d_data_ = nullptr;
};

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    testing::TestConfig config;
    config.M = 32;  // n_heads
    config.seed = 42;
    config.verbose = true;
    config.print_samples = true;
    config.num_samples = 10;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-M") == 0 && i + 1 < argc) {
            config.M = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-seed") == 0 && i + 1 < argc) {
            config.seed = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-q") == 0) {
            config.verbose = false;
            config.print_samples = false;
        }
    }

    testing::TestRunner runner;
    RopeF32Test test1;
    RopeInterleavedTest test2;

    runner.add(&test1);
    runner.add(&test2);
    runner.run_all(config);

    return 0;
}
