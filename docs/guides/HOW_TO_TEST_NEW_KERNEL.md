# 如何测试新优化的算子实现

本文档专门为开发者编写，介绍当你优化了一个新的 CUDA Kernel 后，如何系统地进行测试。

---

## 目录

1. [测试流程概览](#1-测试流程概览)
2. [Step 1: 正确性测试](#2-step-1-正确性测试)
3. [Step 2: 性能测试](#3-step-2-性能测试)
4. [Step 3: 与基线对比](#4-step-3-与基线对比)
5. [Step 4: 边界情况测试](#5-step-4-边界情况测试)
6. [完整测试模板](#6-完整测试模板)
7. [快速检查清单](#7-快速检查清单)

---

## 1. 测试流程概览

```
┌─────────────────────────────────────────────────────────────────┐
│                    新 Kernel 测试流程                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Step 1: 正确性测试                                            │
│   ├── 编写 CPU 参考实现                                         │
│   ├── 对比 GPU 输出与 CPU 输出                                  │
│   └── 验证 NMSE < 阈值 (通常 1%)                                │
│                                                                 │
│   Step 2: 性能测试                                              │
│   ├── 测量执行时间                                              │
│   ├── 计算 GFLOPS / 带宽                                        │
│   └── 多次运行取平均                                            │
│                                                                 │
│   Step 3: 与基线对比                                            │
│   ├── 对比 Naive 实现                                           │
│   ├── 对比 llama.cpp (如果适用)                                 │
│   └── 计算加速比                                                │
│                                                                 │
│   Step 4: 边界情况测试                                          │
│   ├── 最小尺寸 (M=1, N=1)                                       │
│   ├── 非对齐尺寸                                                │
│   └── 极端值输入                                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Step 1: 正确性测试

### 2.1 核心思想

**将你的 Kernel 输出与可信的 CPU 参考实现对比**。

### 2.2 编写 CPU 参考实现

以 Q4_0 × Q8_1 GEMM 为例：

```cpp
// CPU 参考实现 - 必须100%正确，不考虑性能
void cpu_reference_q4_0_q8_1(
    const block_q4_0* weight,    // [M, K/32] 量化权重
    const block_q8_1* activation, // [N, K/32] 量化激活
    float* output,               // [M, N] 输出
    int M, int N, int K
) {
    const int num_blocks = K / 32;  // 每个 block 包含 32 个元素

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;

            for (int b = 0; b < num_blocks; b++) {
                const block_q4_0& w = weight[m * num_blocks + b];
                const block_q8_1& a = activation[n * num_blocks + b];

                // 提取 scale 和 sum
                float d_w = __half2float(w.d);
                float d_a = __half2float(__low2half(a.ds));
                float s_a = __half2float(__high2half(a.ds));

                // 计算整数点积
                int sumi = 0;
                for (int i = 0; i < 16; i++) {
                    // Q4_0 布局: qs[i] 的低 nibble = x[i], 高 nibble = x[i+16]
                    int w0 = (w.qs[i] & 0x0F);
                    int w1 = ((w.qs[i] >> 4) & 0x0F);
                    sumi += w0 * a.qs[i] + w1 * a.qs[i + 16];
                }

                // 补偿公式 (Q4_0 的 zero-point 是 8)
                sum += d_w * (d_a * sumi - 8.0f * s_a);
            }

            output[m * N + n] = sum;
        }
    }
}
```

### 2.3 计算误差指标

```cpp
#include <cmath>
#include <cstdio>

struct ErrorMetrics {
    float mse;      // Mean Squared Error
    float nmse;     // Normalized MSE (主要指标)
    float max_err;  // 最大绝对误差
    float avg_err;  // 平均绝对误差

    void compute(const float* actual, const float* expected, int n) {
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

    void print() const {
        printf("Error Metrics:\n");
        printf("  MSE:      %.6e\n", mse);
        printf("  NMSE:     %.6e (%.4f%%)\n", nmse, nmse * 100);
        printf("  Max Err:  %.6e\n", max_err);
        printf("  Avg Err:  %.6e\n", avg_err);
    }

    bool passed(float threshold = 0.01f) const {
        return nmse < threshold;
    }
};
```

### 2.4 正确性测试代码结构

```cpp
int test_correctness() {
    // 1. 配置参数
    const int M = 4096, N = 2, K = 14336;
    const int num_blocks = K / 32;
    const int output_size = M * N;

    // 2. 分配 Host 内存
    float* h_weight_f32 = new float[M * K];
    float* h_act_f32 = new float[N * K];
    block_q4_0* h_weight_q = new block_q4_0[M * num_blocks];
    block_q8_1* h_act_q = new block_q8_1[N * num_blocks];
    float* h_output_ref = new float[output_size];
    float* h_output_gpu = new float[output_size];

    // 3. 生成随机测试数据
    srand(42);  // 固定种子以便复现
    for (int i = 0; i < M * K; i++) {
        h_weight_f32[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    }
    for (int i = 0; i < N * K; i++) {
        h_act_f32[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    }

    // 4. 量化数据
    quantize_to_q4_0(h_weight_f32, h_weight_q, M * K);
    quantize_to_q8_1(h_act_f32, h_act_q, N * K);

    // 5. 运行 CPU 参考实现
    printf("Running CPU reference...\n");
    cpu_reference_q4_0_q8_1(h_weight_q, h_act_q, h_output_ref, M, N, K);

    // 6. 分配 Device 内存并拷贝数据
    block_q4_0* d_weight;
    block_q8_1* d_act;
    float* d_output;
    cudaMalloc(&d_weight, M * num_blocks * sizeof(block_q4_0));
    cudaMalloc(&d_act, N * num_blocks * sizeof(block_q8_1));
    cudaMalloc(&d_output, output_size * sizeof(float));
    cudaMemcpy(d_weight, h_weight_q, M * num_blocks * sizeof(block_q4_0),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_act, h_act_q, N * num_blocks * sizeof(block_q8_1),
               cudaMemcpyHostToDevice);

    // 7. 运行你的 GPU Kernel
    printf("Running GPU kernel...\n");
    my_optimized_kernel<<<grid, block>>>(d_weight, d_act, d_output, M, N, K);

    // 8. 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    cudaDeviceSynchronize();

    // 9. 拷贝结果回 Host
    cudaMemcpy(h_output_gpu, d_output, output_size * sizeof(float),
               cudaMemcpyDeviceToHost);

    // 10. 计算误差
    ErrorMetrics metrics;
    metrics.compute(h_output_gpu, h_output_ref, output_size);
    metrics.print();

    // 11. 打印样本对比
    printf("\nSample comparison (first 10 elements):\n");
    printf("%-6s | %-12s | %-12s | %-12s\n", "Index", "Reference", "GPU", "Diff");
    printf("-------|--------------|--------------|-------------\n");
    for (int i = 0; i < 10; i++) {
        float diff = h_output_gpu[i] - h_output_ref[i];
        printf("%-6d | %12.6f | %12.6f | %+12.6e\n",
               i, h_output_ref[i], h_output_gpu[i], diff);
    }

    // 12. 判定结果
    bool passed = metrics.passed(0.015f);  // Q4 量化用 1.5% 阈值
    printf("\n%s (NMSE %.6f %s threshold 0.015)\n",
           passed ? "✅ PASSED" : "❌ FAILED",
           metrics.nmse,
           passed ? "<" : ">=");

    // 13. 清理
    delete[] h_weight_f32;
    delete[] h_act_f32;
    delete[] h_weight_q;
    delete[] h_act_q;
    delete[] h_output_ref;
    delete[] h_output_gpu;
    cudaFree(d_weight);
    cudaFree(d_act);
    cudaFree(d_output);

    return passed ? 0 : 1;
}
```

---

## 3. Step 2: 性能测试

### 3.1 使用 CUDA Events 计时

```cpp
class CudaTimer {
public:
    cudaEvent_t start, stop;

    CudaTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~CudaTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void record_start() {
        cudaEventRecord(start);
    }

    void record_stop() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    }

    float elapsed_ms() {
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};
```

### 3.2 性能测试代码

```cpp
struct PerfResult {
    double avg_time_us;
    double gflops;
    double bandwidth_gb_s;
    int n_runs;
};

PerfResult benchmark_kernel(
    // kernel 参数...
    int M, int N, int K,
    float min_time_ms = 1000.0f  // 至少运行 1 秒
) {
    CudaTimer timer;

    // Warmup (重要！让 GPU 进入高频状态)
    for (int i = 0; i < 10; i++) {
        my_optimized_kernel<<<grid, block>>>(d_weight, d_act, d_output, M, N, K);
    }
    cudaDeviceSynchronize();

    // 正式测试
    int n_runs = 0;
    float total_time_ms = 0.0f;

    while (total_time_ms < min_time_ms) {
        timer.record_start();
        my_optimized_kernel<<<grid, block>>>(d_weight, d_act, d_output, M, N, K);
        timer.record_stop();

        total_time_ms += timer.elapsed_ms();
        n_runs++;
    }

    // 计算结果
    PerfResult result;
    result.n_runs = n_runs;
    result.avg_time_us = (total_time_ms * 1000.0) / n_runs;

    // GFLOPS = 2*M*N*K (乘加各算一次) / time_seconds / 1e9
    uint64_t flops = 2ULL * M * N * K;
    result.gflops = (flops / (result.avg_time_us / 1e6)) / 1e9;

    // 带宽计算 (以 Q4_0 x Q8_1 为例)
    // Q4_0: 18 bytes per 32 elements, Q8_1: 36 bytes per 32 elements
    size_t bytes_read = (size_t)M * (K / 32) * 18 +  // Q4_0 weights
                        (size_t)N * (K / 32) * 36 +  // Q8_1 activations
                        (size_t)M * N * 4;           // Output (FP32)
    result.bandwidth_gb_s = (bytes_read / (result.avg_time_us / 1e6)) / 1e9;

    return result;
}

void print_perf_result(const PerfResult& r, int M, int N, int K) {
    printf("Performance Results (M=%d, N=%d, K=%d):\n", M, N, K);
    printf("  Avg Time:  %.2f us\n", r.avg_time_us);
    printf("  GFLOPS:    %.2f\n", r.gflops);
    printf("  Bandwidth: %.2f GB/s\n", r.bandwidth_gb_s);
    printf("  Runs:      %d\n", r.n_runs);
}
```

### 3.3 多尺寸测试

```cpp
void benchmark_multiple_sizes() {
    // LLM 典型尺寸
    std::vector<std::tuple<int, int, int, const char*>> test_cases = {
        // Llama 3.2 decode 场景
        {4096, 1, 14336, "Llama decode N=1"},
        {4096, 2, 14336, "Llama decode N=2"},
        {4096, 4, 14336, "Llama decode N=4"},
        {4096, 8, 14336, "Llama decode N=8"},

        // Prefill 场景
        {4096, 128, 14336, "Llama prefill N=128"},
        {4096, 512, 14336, "Llama prefill N=512"},

        // 标准测试
        {4096, 4096, 4096, "Square 4K"},
    };

    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║              Performance Benchmark Results                    ║\n");
    printf("╠═══════════════════════════════════════════════════════════════╣\n");
    printf("║ %-25s %8s %8s %8s %10s ║\n",
           "Test Case", "M", "N", "K", "GFLOPS");
    printf("╠═══════════════════════════════════════════════════════════════╣\n");

    for (auto& [M, N, K, name] : test_cases) {
        // 分配内存、初始化...
        PerfResult result = benchmark_kernel(/*...*/ M, N, K);

        printf("║ %-25s %8d %8d %8d %10.2f ║\n",
               name, M, N, K, result.gflops);
    }

    printf("╚═══════════════════════════════════════════════════════════════╝\n");
}
```

---

## 4. Step 3: 与基线对比

### 4.1 对比 Naive 实现

```cpp
void compare_with_naive(int M, int N, int K) {
    // 运行 Naive 实现
    PerfResult naive_result = benchmark_kernel_naive(/*...*/);

    // 运行你的优化实现
    PerfResult optimized_result = benchmark_kernel_optimized(/*...*/);

    // 计算加速比
    double speedup = naive_result.avg_time_us / optimized_result.avg_time_us;

    printf("\nComparison (M=%d, N=%d, K=%d):\n", M, N, K);
    printf("  Naive:     %.2f us, %.2f GFLOPS\n",
           naive_result.avg_time_us, naive_result.gflops);
    printf("  Optimized: %.2f us, %.2f GFLOPS\n",
           optimized_result.avg_time_us, optimized_result.gflops);
    printf("  Speedup:   %.2fx\n", speedup);
}
```

### 4.2 对比 llama.cpp

如果项目链接了 llama.cpp，可以直接对比：

```cpp
#ifdef HAS_LLAMA_CPP
void compare_with_llama_cpp(int M, int N, int K) {
    // 运行 llama.cpp kernel
    // ... (需要正确设置 llama.cpp 的数据结构)

    // 运行你的 kernel
    PerfResult my_result = benchmark_kernel(/*...*/);

    printf("vs llama.cpp: %.2fx\n", llama_time_us / my_result.avg_time_us);
}
#endif
```

否则，可以手动运行 llama.cpp 的 `test-backend-ops` 并记录结果：

```bash
# 在 llama.cpp 目录运行
cd /path/to/llama.cpp/build/bin
./test-backend-ops perf -o MUL_MAT -b CUDA0 -p "type_a=q4_0.*m=4096.*n=2.*k=14336"
```

---

## 5. Step 4: 边界情况测试

### 5.1 测试用例列表

```cpp
void test_edge_cases() {
    std::vector<std::tuple<int, int, int, const char*>> edge_cases = {
        // 最小尺寸
        {1, 1, 32, "Minimum size"},
        {1, 1, 64, "Single element output"},

        // 单行/单列
        {1, 512, 1024, "Single row"},
        {512, 1, 1024, "Single column"},

        // 非 2 的幂
        {1000, 3, 2048, "Non-power-of-2 M"},
        {1024, 5, 2048, "Non-power-of-2 N"},

        // 大尺寸
        {8192, 8, 14336, "Large M"},
        {4096, 1024, 14336, "Large N"},
    };

    for (auto& [M, N, K, name] : edge_cases) {
        printf("Testing: %s (M=%d, N=%d, K=%d)... ", name, M, N, K);

        bool passed = test_correctness_for_size(M, N, K);

        printf("%s\n", passed ? "✅" : "❌");
    }
}
```

### 5.2 数值稳定性测试

```cpp
void test_numerical_stability() {
    printf("Testing numerical stability...\n");

    // 测试大值输入
    {
        // 生成大值数据 ([-100, 100])
        for (int i = 0; i < M * K; i++) {
            h_weight_f32[i] = ((float)rand() / RAND_MAX - 0.5f) * 200.0f;
        }
        // ... 量化并测试
        printf("  Large values: %s\n", passed ? "✅" : "❌");
    }

    // 测试小值输入
    {
        // 生成小值数据 ([-0.01, 0.01])
        for (int i = 0; i < M * K; i++) {
            h_weight_f32[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
        }
        // ... 量化并测试
        printf("  Small values: %s\n", passed ? "✅" : "❌");
    }

    // 测试稀疏输入 (大量零值)
    {
        for (int i = 0; i < M * K; i++) {
            h_weight_f32[i] = (rand() % 10 == 0) ?
                ((float)rand() / RAND_MAX - 0.5f) * 2.0f : 0.0f;
        }
        // ... 量化并测试
        printf("  Sparse input: %s\n", passed ? "✅" : "❌");
    }
}
```

---

## 6. 完整测试模板

将以下代码保存为 `tests/unit/test_my_kernel.cu`：

```cpp
/**
 * @file tests/unit/test_my_kernel.cu
 * @brief 测试我的优化 Kernel
 */

#include "../framework/test_framework.cuh"
#include "../../kernels/gemm/my_kernel.cuh"  // 你的 kernel 头文件

// ============================================================================
// CPU 参考实现
// ============================================================================
void cpu_reference(
    const block_q4_0* weight,
    const block_q8_1* activation,
    float* output,
    int M, int N, int K
) {
    // ... 完整的 CPU 实现
}

// ============================================================================
// 测试类
// ============================================================================
class MyKernelTest : public testing::TestCase {
public:
    const char* name() const override { return "MY_OPTIMIZED_KERNEL"; }
    const char* description() const override {
        return "测试我的优化 Kernel 实现";
    }
    float threshold() const override { return 0.015f; }  // NMSE 阈值

    void setup(const testing::TestConfig& config) override {
        M_ = config.M;
        N_ = config.N;
        K_ = config.K;

        int num_blocks = K_ / 32;
        output_size_ = M_ * N_;

        // 分配 Host 内存
        h_weight_f32_ = new float[M_ * K_];
        h_act_f32_ = new float[N_ * K_];
        h_weight_q_ = new block_q4_0[M_ * num_blocks];
        h_act_q_ = new block_q8_1[N_ * num_blocks];
        output_ref_ = new float[output_size_];   // 基类成员
        output_test_ = new float[output_size_];  // 基类成员

        // 生成随机数据
        testing::DataGenerator gen(config.seed);
        gen.generate(h_weight_f32_, M_ * K_,
                     testing::DataGenerator::NORMAL, 0.0f, 0.5f);
        gen.generate(h_act_f32_, N_ * K_,
                     testing::DataGenerator::NORMAL, 0.0f, 0.5f);

        // 量化
        testing::quantize::to_q4_0(h_weight_f32_, h_weight_q_, M_ * K_);
        testing::quantize::to_q8_1(h_act_f32_, h_act_q_, N_ * K_);

        // 分配 Device 内存
        size_t weight_bytes = M_ * num_blocks * sizeof(block_q4_0);
        size_t act_bytes = N_ * num_blocks * sizeof(block_q8_1);

        cudaMalloc(&d_weight_, weight_bytes);
        cudaMalloc(&d_act_, act_bytes);
        cudaMalloc(&d_output_, output_size_ * sizeof(float));

        cudaMemcpy(d_weight_, h_weight_q_, weight_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_act_, h_act_q_, act_bytes, cudaMemcpyHostToDevice);
    }

    void run_reference() override {
        cpu_reference(h_weight_q_, h_act_q_, output_ref_, M_, N_, K_);
    }

    void run_kernel() override {
        // ===== 在这里调用你的 Kernel =====
        my_optimized_kernel(d_weight_, d_act_, d_output_, M_, N_, K_);
        // =================================

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel error: %s\n", cudaGetErrorString(err));
        }

        cudaMemcpy(output_test_, d_output_, output_size_ * sizeof(float),
                   cudaMemcpyDeviceToHost);
    }

    void cleanup() override {
        delete[] h_weight_f32_;
        delete[] h_act_f32_;
        delete[] h_weight_q_;
        delete[] h_act_q_;
        delete[] output_ref_;
        delete[] output_test_;

        cudaFree(d_weight_);
        cudaFree(d_act_);
        cudaFree(d_output_);
    }

private:
    int M_, N_, K_;
    float *h_weight_f32_, *h_act_f32_;
    block_q4_0 *h_weight_q_, *d_weight_;
    block_q8_1 *h_act_q_, *d_act_;
    float *d_output_;
};

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    testing::TestConfig config;

    // 默认配置
    config.M = 4;
    config.N = 512;
    config.K = 1024;
    config.seed = 42;
    config.verbose = true;
    config.print_samples = true;
    config.num_samples = 5;

    // 解析命令行参数
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

    testing::TestRunner runner;
    MyKernelTest test;
    runner.add(&test);
    runner.run_all(config);

    return 0;
}
```

### 编译和运行

```bash
# 编译
nvcc -O3 -arch=sm_86 -std=c++17 \
    -I./include -I./compat -I./kernels -I./tests/framework \
    tests/unit/test_my_kernel.cu \
    -o bin/unit/test_my_kernel

# 运行默认测试
./bin/unit/test_my_kernel

# 使用自定义尺寸
./bin/unit/test_my_kernel -M 4096 -N 2 -K 14336

# 安静模式
./bin/unit/test_my_kernel -M 4096 -N 2 -K 14336 -q
```

---

## 7. 快速检查清单

### 发布前必须检查

- [ ] **正确性测试**
  - [ ] 小尺寸通过 (M=4, N=512, K=1024)
  - [ ] LLM 典型尺寸通过 (M=4096, N=2, K=14336)
  - [ ] NMSE < 阈值 (Q4: 1.5%, Q8: 0.5%)

- [ ] **性能测试**
  - [ ] 比 Naive 实现快
  - [ ] 记录各尺寸的 GFLOPS
  - [ ] 与 llama.cpp 对比（如果适用）

- [ ] **边界情况**
  - [ ] M=1, N=1 不崩溃
  - [ ] 非对齐尺寸正确处理
  - [ ] 大尺寸不超时/OOM

- [ ] **稳定性**
  - [ ] 多次运行结果一致
  - [ ] 不同随机种子都通过
  - [ ] 无 CUDA 错误 (`cudaGetLastError`)

### 可选检查

- [ ] 使用 `compute-sanitizer` 检查内存访问
- [ ] 使用 Nsight Compute 分析性能瓶颈
- [ ] 记录寄存器使用量 (`--ptxas-options=-v`)

---

## 附录: 误差阈值参考

| 量化格式 | 推荐 NMSE 阈值 | 说明 |
|---------|---------------|------|
| Q8_0 × Q8_1 | 0.005 (0.5%) | 高精度 |
| Q5_0 × Q8_1 | 0.01 (1%) | 中等精度 |
| Q5_1 × Q8_1 | 0.01 (1%) | 中等精度 |
| Q4_0 × Q8_1 | 0.015 (1.5%) | 低精度 |
| Q4_1 × Q8_1 | 0.015 (1.5%) | 低精度 |

---

**最后更新**: 2025-01-29
