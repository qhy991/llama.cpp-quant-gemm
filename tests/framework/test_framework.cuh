/**
 * @file tests/framework/test_framework.cuh
 * @brief 通用 Kernel 测试框架
 *
 * 提供统一的测试基础设施，包括:
 * - 误差度量 (MSE, NMSE, MAE)
 * - 数据生成 (均匀分布, 正态分布)
 * - 测试用例基类
 * - 测试运行器
 *
 * 使用方法:
 * 1. 继承 TestCase 类
 * 2. 实现必要的虚函数
 * 3. 使用 TestRunner 运行测试
 */

#pragma once

#include "../../compat/ggml_types.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <vector>
#include <string>

namespace testing {

// ============================================================================
// 误差度量
// ============================================================================

struct ErrorMetrics {
    float mse;       // Mean Squared Error
    float nmse;      // Normalized Mean Squared Error
    float max_err;   // Maximum Absolute Error
    float avg_err;   // Average Absolute Error
    int n;           // Number of elements

    void compute(const float* actual, const float* expected, int count) {
        n = count;
        double sum_sq_err = 0.0;
        double sum_sq_ref = 0.0;
        double sum_abs_err = 0.0;
        max_err = 0.0f;

        for (int i = 0; i < n; i++) {
            float diff = actual[i] - expected[i];
            float abs_diff = fabsf(diff);

            sum_sq_err += diff * diff;
            sum_sq_ref += expected[i] * expected[i];
            sum_abs_err += abs_diff;

            if (abs_diff > max_err) {
                max_err = abs_diff;
            }
        }

        mse = (float)(sum_sq_err / n);
        nmse = (sum_sq_ref > 0) ? (float)(sum_sq_err / sum_sq_ref) : 0.0f;
        avg_err = (float)(sum_abs_err / n);
    }

    bool check(float threshold) const {
        return nmse < threshold;
    }

    void print() const {
        printf("  Error Metrics:\n");
        printf("    MSE:      %.6e\n", mse);
        printf("    NMSE:     %.6e (%.4f%%)\n", nmse, nmse * 100);
        printf("    Max Err:  %.6e\n", max_err);
        printf("    Avg Err:  %.6e\n", avg_err);
    }
};

// ============================================================================
// 数据生成器
// ============================================================================

class DataGenerator {
public:
    enum Distribution {
        UNIFORM,    // 均匀分布 [min, max]
        NORMAL,     // 正态分布 (mean, stddev)
        XAVIER,     // Xavier 初始化
        HE          // He 初始化
    };

    DataGenerator(unsigned int seed = 0) {
        if (seed == 0) {
            seed = (unsigned int)time(NULL);
        }
        srand(seed);
        seed_ = seed;
    }

    void generate(float* data, int n, Distribution dist,
                  float param1 = 0.0f, float param2 = 1.0f) {
        switch (dist) {
            case UNIFORM:
                generate_uniform(data, n, param1, param2);
                break;
            case NORMAL:
                generate_normal(data, n, param1, param2);
                break;
            case XAVIER:
                generate_xavier(data, n, (int)param1);
                break;
            case HE:
                generate_he(data, n, (int)param1);
                break;
        }
    }

    unsigned int seed() const { return seed_; }

private:
    unsigned int seed_;

    void generate_uniform(float* data, int n, float min_val, float max_val) {
        float range = max_val - min_val;
        for (int i = 0; i < n; i++) {
            data[i] = min_val + range * ((float)rand() / RAND_MAX);
        }
    }

    void generate_normal(float* data, int n, float mean, float stddev) {
        // Box-Muller 变换
        for (int i = 0; i < n; i += 2) {
            float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
            float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
            float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
            float z1 = sqrtf(-2.0f * logf(u1)) * sinf(2.0f * M_PI * u2);
            data[i] = mean + stddev * z0;
            if (i + 1 < n) {
                data[i + 1] = mean + stddev * z1;
            }
        }
    }

    void generate_xavier(float* data, int n, int fan_in) {
        float stddev = sqrtf(2.0f / fan_in);
        generate_normal(data, n, 0.0f, stddev);
    }

    void generate_he(float* data, int n, int fan_in) {
        float stddev = sqrtf(2.0f / fan_in);
        generate_normal(data, n, 0.0f, stddev);
    }
};

// ============================================================================
// 量化工具
// ============================================================================

namespace quantize {

// Q4_0 量化
inline void to_q4_0(const float* src, block_q4_0* dst, int n) {
    const int block_size = 32;
    const int num_blocks = n / block_size;

    for (int b = 0; b < num_blocks; b++) {
        const float* block_src = src + b * block_size;

        float max_abs = 0.0f;
        for (int i = 0; i < block_size; i++) {
            float abs_val = fabsf(block_src[i]);
            if (abs_val > max_abs) max_abs = abs_val;
        }

        float scale = max_abs / 7.0f;
        float inv_scale = (scale > 0) ? (1.0f / scale) : 0.0f;

        dst[b].d = __float2half(scale);

        // llama.cpp 的 Q4_0 布局: 前半部分和后半部分交错
        // qs[i] 的低 nibble = block_src[i], 高 nibble = block_src[i+16]
        for (int i = 0; i < 16; i++) {
            int8_t v0 = (int8_t)roundf(block_src[i] * inv_scale);
            int8_t v1 = (int8_t)roundf(block_src[i + 16] * inv_scale);

            v0 = (v0 < -8) ? -8 : ((v0 > 7) ? 7 : v0);
            v1 = (v1 < -8) ? -8 : ((v1 > 7) ? 7 : v1);

            dst[b].qs[i] = ((v0 + 8) & 0x0F) | (((v1 + 8) & 0x0F) << 4);
        }
    }
}

// Q8_1 量化
inline void to_q8_1(const float* src, block_q8_1* dst, int n) {
    const int block_size = 32;
    const int num_blocks = n / block_size;

    for (int b = 0; b < num_blocks; b++) {
        const float* block_src = src + b * block_size;

        float max_abs = 0.0f;
        float sum = 0.0f;
        for (int i = 0; i < block_size; i++) {
            float abs_val = fabsf(block_src[i]);
            if (abs_val > max_abs) max_abs = abs_val;
            sum += block_src[i];
        }

        float scale = max_abs / 127.0f;
        float inv_scale = (scale > 0) ? (1.0f / scale) : 0.0f;

        int sum_q = 0;
        // llama.cpp 的 Q8_1 布局: 顺序存储 (不是交错的)
        for (int i = 0; i < block_size; i++) {
            int8_t v = (int8_t)roundf(block_src[i] * inv_scale);
            v = (v < -127) ? -127 : ((v > 127) ? 127 : v);
            dst[b].qs[i] = v;
            sum_q += v;
        }

        // 存储 scale 和 sum*scale (与 llama.cpp 一致)
        dst[b].ds = make_half2(__float2half(scale), __float2half(sum_q * scale));
    }
}

// Q8_0 量化
inline void to_q8_0(const float* src, block_q8_0* dst, int n) {
    const int block_size = 32;
    const int num_blocks = n / block_size;

    for (int b = 0; b < num_blocks; b++) {
        const float* block_src = src + b * block_size;

        float max_abs = 0.0f;
        for (int i = 0; i < block_size; i++) {
            float abs_val = fabsf(block_src[i]);
            if (abs_val > max_abs) max_abs = abs_val;
        }

        float scale = max_abs / 127.0f;
        float inv_scale = (scale > 0) ? (1.0f / scale) : 0.0f;

        dst[b].d = __float2half(scale);

        // llama.cpp 的 Q8_0 布局: 顺序存储
        for (int i = 0; i < block_size; i++) {
            int8_t v = (int8_t)roundf(block_src[i] * inv_scale);
            v = (v < -127) ? -127 : ((v > 127) ? 127 : v);
            dst[b].qs[i] = v;
        }
    }
}

// Q4_1 量化 (非对称)
inline void to_q4_1(const float* src, block_q4_1* dst, int n) {
    const int block_size = 32;
    const int num_blocks = n / block_size;

    for (int b = 0; b < num_blocks; b++) {
        const float* block_src = src + b * block_size;

        float min_val = block_src[0];
        float max_val = block_src[0];
        for (int i = 1; i < block_size; i++) {
            if (block_src[i] < min_val) min_val = block_src[i];
            if (block_src[i] > max_val) max_val = block_src[i];
        }

        float scale = (max_val - min_val) / 15.0f;
        float inv_scale = (scale > 0) ? (1.0f / scale) : 0.0f;

        dst[b].d = __float2half(scale);
        dst[b].m = __float2half(min_val);

        // llama.cpp 的 Q4_1 布局: 前半部分和后半部分交错
        for (int i = 0; i < 16; i++) {
            int q0 = (int)roundf((block_src[i] - min_val) * inv_scale);
            int q1 = (int)roundf((block_src[i + 16] - min_val) * inv_scale);

            q0 = (q0 < 0) ? 0 : ((q0 > 15) ? 15 : q0);
            q1 = (q1 < 0) ? 0 : ((q1 > 15) ? 15 : q1);

            dst[b].qs[i] = (q1 << 4) | q0;
        }
    }
}

// Q5_0 量化 (5-bit 对称)
inline void to_q5_0(const float* src, block_q5_0* dst, int n) {
    const int block_size = 32;
    const int num_blocks = n / block_size;

    for (int b = 0; b < num_blocks; b++) {
        const float* block_src = src + b * block_size;

        float max_abs = 0.0f;
        for (int i = 0; i < block_size; i++) {
            float abs_val = fabsf(block_src[i]);
            if (abs_val > max_abs) max_abs = abs_val;
        }

        float scale = max_abs / 15.0f;  // 5-bit: [-16, 15] -> /15 for positive
        float inv_scale = (scale > 0) ? (1.0f / scale) : 0.0f;

        dst[b].d = __float2half(scale);

        // llama.cpp 的 Q5_0 布局: 前半部分和后半部分交错
        // qh 布局: bits 0-15 对应 x[0..15], bits 16-31 对应 x[16..31]
        uint32_t qh = 0;
        for (int i = 0; i < 16; i++) {
            int q0 = (int)roundf(block_src[i] * inv_scale) + 16;
            int q1 = (int)roundf(block_src[i + 16] * inv_scale) + 16;

            q0 = (q0 < 0) ? 0 : ((q0 > 31) ? 31 : q0);
            q1 = (q1 < 0) ? 0 : ((q1 > 31) ? 31 : q1);

            // 低 4 bits 存入 qs
            dst[b].qs[i] = ((q1 & 0x0F) << 4) | (q0 & 0x0F);

            // 第 5 bit 存入 qh
            qh |= ((q0 >> 4) & 1) << i;
            qh |= ((q1 >> 4) & 1) << (i + 16);
        }
        memcpy(dst[b].qh, &qh, sizeof(qh));
    }
}

// Q5_1 量化 (5-bit 非对称)
inline void to_q5_1(const float* src, block_q5_1* dst, int n) {
    const int block_size = 32;
    const int num_blocks = n / block_size;

    for (int b = 0; b < num_blocks; b++) {
        const float* block_src = src + b * block_size;

        float min_val = block_src[0];
        float max_val = block_src[0];
        for (int i = 1; i < block_size; i++) {
            if (block_src[i] < min_val) min_val = block_src[i];
            if (block_src[i] > max_val) max_val = block_src[i];
        }

        float scale = (max_val - min_val) / 31.0f;
        float inv_scale = (scale > 0) ? (1.0f / scale) : 0.0f;

        dst[b].d = __float2half(scale);
        dst[b].m = __float2half(min_val);

        // llama.cpp 的 Q5_1 布局: 前半部分和后半部分交错
        // qh 布局: bits 0-15 对应 x[0..15], bits 16-31 对应 x[16..31]
        uint32_t qh = 0;
        for (int i = 0; i < 16; i++) {
            int q0 = (int)roundf((block_src[i] - min_val) * inv_scale);
            int q1 = (int)roundf((block_src[i + 16] - min_val) * inv_scale);

            q0 = (q0 < 0) ? 0 : ((q0 > 31) ? 31 : q0);
            q1 = (q1 < 0) ? 0 : ((q1 > 31) ? 31 : q1);

            dst[b].qs[i] = ((q1 & 0x0F) << 4) | (q0 & 0x0F);

            qh |= ((q0 >> 4) & 1) << i;
            qh |= ((q1 >> 4) & 1) << (i + 16);
        }
        memcpy(dst[b].qh, &qh, sizeof(qh));
    }
}

} // namespace quantize

// ============================================================================
// 测试配置
// ============================================================================

struct TestConfig {
    int M = 4;
    int N = 512;
    int K = 1024;
    float threshold = 0.01f;
    unsigned int seed = 42;
    bool verbose = true;
    bool print_samples = true;
    int num_samples = 10;

    void print() const {
        printf("Test Configuration:\n");
        printf("  Dimensions: M=%d, N=%d, K=%d\n", M, N, K);
        printf("  Threshold: %.6f (%.4f%%)\n", threshold, threshold * 100);
        printf("  Seed: %u\n", seed);
    }
};

// ============================================================================
// 测试用例基类
// ============================================================================

class TestCase {
public:
    virtual ~TestCase() = default;

    // 必须实现
    virtual const char* name() const = 0;
    virtual const char* description() const = 0;
    virtual void setup(const TestConfig& config) = 0;
    virtual void run_reference() = 0;
    virtual void run_kernel() = 0;
    virtual void cleanup() = 0;

    // 可选覆盖
    virtual float threshold() const { return 0.01f; }

    // 运行测试
    bool run(const TestConfig& config) {
        config_ = config;

        printf("\n");
        printf("═══════════════════════════════════════════════════════════════\n");
        printf("  Test: %s\n", name());
        printf("  Description: %s\n", description());
        printf("═══════════════════════════════════════════════════════════════\n");

        if (config_.verbose) {
            config_.print();
        }

        printf("\n[1/4] Setting up...\n");
        setup(config_);

        printf("[2/4] Running reference...\n");
        run_reference();

        printf("[3/4] Running kernel...\n");
        run_kernel();

        printf("[4/4] Verifying...\n");
        verify();

        printf("\n");
        metrics_.print();

        bool passed = metrics_.check(threshold());

        printf("\n");
        if (passed) {
            printf("✅ PASSED (NMSE %.6f < %.6f)\n", metrics_.nmse, threshold());
        } else {
            printf("❌ FAILED (NMSE %.6f >= %.6f)\n", metrics_.nmse, threshold());
        }

        cleanup();
        return passed;
    }

protected:
    TestConfig config_;
    ErrorMetrics metrics_;
    float* output_ref_ = nullptr;
    float* output_test_ = nullptr;
    int output_size_ = 0;

    void verify() {
        if (output_ref_ && output_test_ && output_size_ > 0) {
            metrics_.compute(output_test_, output_ref_, output_size_);

            if (config_.print_samples) {
                print_samples();
            }
        }
    }

    void print_samples() {
        printf("\n  Sample Values:\n");
        printf("  %-6s | %-12s | %-12s | %-12s\n", "Index", "Reference", "Kernel", "Diff");
        printf("  -------|--------------|--------------|-------------\n");

        int n = (config_.num_samples < output_size_) ? config_.num_samples : output_size_;
        for (int i = 0; i < n; i++) {
            float diff = output_test_[i] - output_ref_[i];
            printf("  %-6d | %12.6f | %12.6f | %+12.6f\n",
                   i, output_ref_[i], output_test_[i], diff);
        }
    }
};

// ============================================================================
// 测试运行器
// ============================================================================

class TestRunner {
public:
    void add(TestCase* test) {
        tests_.push_back(test);
    }

    void run_all(const TestConfig& config) {
        printf("\n");
        printf("╔═══════════════════════════════════════════════════════════════╗\n");
        printf("║        CUDA KERNEL TEST FRAMEWORK                             ║\n");
        printf("║        quant-gemm-from-scratch                                ║\n");
        printf("╚═══════════════════════════════════════════════════════════════╝\n");

        // Print GPU info
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("\nGPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);

        int passed = 0;
        int failed = 0;

        for (auto* test : tests_) {
            if (test->run(config)) {
                passed++;
                results_.push_back({test->name(), true});
            } else {
                failed++;
                results_.push_back({test->name(), false});
            }
        }

        print_summary(passed, failed);
    }

private:
    struct Result {
        std::string name;
        bool passed;
    };

    std::vector<TestCase*> tests_;
    std::vector<Result> results_;

    void print_summary(int passed, int failed) {
        printf("\n");
        printf("╔═══════════════════════════════════════════════════════════════╗\n");
        printf("║                      TEST SUMMARY                             ║\n");
        printf("╠═══════════════════════════════════════════════════════════════╣\n");

        for (const auto& r : results_) {
            const char* status = r.passed ? "✅ PASS" : "❌ FAIL";
            printf("║  %-50s %s  ║\n", r.name.c_str(), status);
        }

        printf("╠═══════════════════════════════════════════════════════════════╣\n");
        printf("║  Total: %d passed, %d failed                                   ║\n",
               passed, failed);
        printf("╚═══════════════════════════════════════════════════════════════╝\n");
    }
};

} // namespace testing
