/**
 * Comprehensive Kernel Tests for llama.cpp
 *
 * This file contains tests for multiple kernel implementations:
 * 1. MUL_MAT Q4_0 x Q8_1 (DP4A kernel)
 * 2. SILU activation
 * 3. RMS_NORM normalization
 * 4. ADD elementwise
 *
 * Usage:
 *   nvcc -o test_all_kernels test_all_kernels.cu -std=c++17 -O3 --gpu-architecture=sm_XX
 *   ./test_all_kernels
 */

#include "kernel_test_framework.cuh"
#include "../../../quant-gemm-from-scratch/include/gemm_cuda_dp4a.cuh"

// ============================================================================
// Test 1: MUL_MAT Q4_0 x Q8_1 (Our custom DP4A kernel)
// ============================================================================

class MulMatQ4_0_Test : public KernelTest {
public:
    const char* name() const override {
        return "MUL_MAT_Q4_0";
    }

    const char* description() const override {
        return "Matrix multiplication with Q4_0 weights and Q8_1 activations";
    }

    float nmse_threshold() const override {
        return 0.015f;  // 1.5% for Q4_0 quantized ops (typical range: 0.5%-1.5%)
    }

    void setup(const TestConfig& config) override {
        M_ = config.M;
        N_ = config.N;
        K_ = config.K;

        // Allocate host memory
        weight_fp32_ = new float[N_ * K_];
        activation_fp32_ = new float[M_ * K_];
        output_cpu_ = new float[M_ * N_];
        output_gpu_ = new float[M_ * N_];
        output_size_ = M_ * N_;

        // Generate random data
        DataGenerator gen(config.seed);
        gen.generate(weight_fp32_, N_ * K_, DataGenerator::NORMAL, 0.0f, 0.1f);
        gen.generate(activation_fp32_, M_ * K_, DataGenerator::NORMAL, 0.0f, 0.5f);

        // Quantize
        int weight_blocks = (N_ * K_) / 32;
        int activation_blocks = (M_ * K_) / 32;

        weight_q4_ = new block_q4_0[weight_blocks];
        activation_q8_ = new block_q8_1[activation_blocks];

        quantize::to_q4_0(weight_fp32_, weight_q4_, N_ * K_);
        quantize::to_q8_1(activation_fp32_, activation_q8_, M_ * K_);

        // Allocate GPU memory
        CUDA_CHECK(cudaMalloc(&d_weight_, weight_blocks * sizeof(block_q4_0)));
        CUDA_CHECK(cudaMalloc(&d_activation_, activation_blocks * sizeof(block_q8_1)));
        CUDA_CHECK(cudaMalloc(&d_output_, M_ * N_ * sizeof(float)));

        // Copy to GPU
        CUDA_CHECK(cudaMemcpy(d_weight_, weight_q4_,
                              weight_blocks * sizeof(block_q4_0),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_activation_, activation_q8_,
                              activation_blocks * sizeof(block_q8_1),
                              cudaMemcpyHostToDevice));

        printf("  Weight: [%d, %d] Q4_0 (%d blocks)\n", N_, K_, weight_blocks);
        printf("  Activation: [%d, %d] Q8_1 (%d blocks)\n", M_, K_, activation_blocks);
        printf("  Output: [%d, %d] FP32\n", M_, N_);
    }

    void run_cpu_reference() override {
        // FP32 GEMM (row-major)
        for (int m = 0; m < M_; m++) {
            for (int n = 0; n < N_; n++) {
                float sum = 0.0f;
                for (int k = 0; k < K_; k++) {
                    sum += activation_fp32_[m * K_ + k] * weight_fp32_[n * K_ + k];
                }
                output_cpu_[m * N_ + n] = sum;
            }
        }
    }

    void run_gpu_kernel() override {
        dim3 block(16, 16);
        dim3 grid((N_ + 15) / 16, (M_ + 15) / 16);

        gemm_w4a8_dp4a_kernel<<<grid, block>>>(
            d_activation_, d_weight_, d_output_, M_, N_, K_);

        KERNEL_LAUNCH_CHECK();

        CUDA_CHECK(cudaMemcpy(output_gpu_, d_output_,
                              M_ * N_ * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    void cleanup() override {
        delete[] weight_fp32_;
        delete[] activation_fp32_;
        delete[] output_cpu_;
        delete[] output_gpu_;
        delete[] weight_q4_;
        delete[] activation_q8_;
        cudaFree(d_weight_);
        cudaFree(d_activation_);
        cudaFree(d_output_);
    }

private:
    int M_, N_, K_;
    float* weight_fp32_;
    float* activation_fp32_;
    block_q4_0* weight_q4_;
    block_q8_1* activation_q8_;
    block_q4_0* d_weight_;
    block_q8_1* d_activation_;
    float* d_output_;
};

// ============================================================================
// Test 2: SILU Activation
// ============================================================================

// SILU kernel implementation
__global__ void silu_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        output[idx] = x / (1.0f + expf(-x));  // x * sigmoid(x)
    }
}

class SiluTest : public KernelTest {
public:
    const char* name() const override {
        return "SILU";
    }

    const char* description() const override {
        return "SiLU (Swish) activation function: x * sigmoid(x)";
    }

    float nmse_threshold() const override {
        return 1e-5f;  // FP32 precision
    }

    void setup(const TestConfig& config) override {
        n_ = config.M * config.K;  // Use M*K as vector size

        input_ = new float[n_];
        output_cpu_ = new float[n_];
        output_gpu_ = new float[n_];
        output_size_ = n_;

        DataGenerator gen(config.seed);
        gen.generate(input_, n_, DataGenerator::NORMAL, 0.0f, 1.0f);

        CUDA_CHECK(cudaMalloc(&d_input_, n_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output_, n_ * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_input_, input_, n_ * sizeof(float), cudaMemcpyHostToDevice));

        printf("  Vector size: %d\n", n_);
    }

    void run_cpu_reference() override {
        for (int i = 0; i < n_; i++) {
            float x = input_[i];
            output_cpu_[i] = x / (1.0f + expf(-x));
        }
    }

    void run_gpu_kernel() override {
        int block_size = 256;
        int grid_size = (n_ + block_size - 1) / block_size;

        silu_kernel<<<grid_size, block_size>>>(d_input_, d_output_, n_);

        KERNEL_LAUNCH_CHECK();

        CUDA_CHECK(cudaMemcpy(output_gpu_, d_output_, n_ * sizeof(float), cudaMemcpyDeviceToHost));
    }

    void cleanup() override {
        delete[] input_;
        delete[] output_cpu_;
        delete[] output_gpu_;
        cudaFree(d_input_);
        cudaFree(d_output_);
    }

private:
    int n_;
    float* input_;
    float* d_input_;
    float* d_output_;
};

// ============================================================================
// Test 3: RMS Normalization
// ============================================================================

// RMS Norm kernel (simplified single-block version for testing)
__global__ void rms_norm_kernel(const float* input, float* output,
                                 const float* weight, int n, float eps) {
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int idx = tid;

    // Compute sum of squares
    float sum_sq = 0.0f;
    while (idx < n) {
        sum_sq += input[idx] * input[idx];
        idx += blockDim.x;
    }

    // Reduce within block
    shared[tid] = sum_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    // Compute RMS
    float rms = sqrtf(shared[0] / n + eps);

    // Normalize
    idx = tid;
    while (idx < n) {
        output[idx] = (input[idx] / rms) * weight[idx];
        idx += blockDim.x;
    }
}

class RmsNormTest : public KernelTest {
public:
    const char* name() const override {
        return "RMS_NORM";
    }

    const char* description() const override {
        return "Root Mean Square Layer Normalization";
    }

    float nmse_threshold() const override {
        return 1e-4f;  // Allow small numerical differences
    }

    void setup(const TestConfig& config) override {
        n_ = config.K;  // Hidden size
        eps_ = 1e-5f;

        input_ = new float[n_];
        weight_ = new float[n_];
        output_cpu_ = new float[n_];
        output_gpu_ = new float[n_];
        output_size_ = n_;

        DataGenerator gen(config.seed);
        gen.generate(input_, n_, DataGenerator::NORMAL, 0.0f, 1.0f);
        gen.generate(weight_, n_, DataGenerator::NORMAL, 1.0f, 0.1f);  // Around 1.0

        CUDA_CHECK(cudaMalloc(&d_input_, n_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_weight_, n_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output_, n_ * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_input_, input_, n_ * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_weight_, weight_, n_ * sizeof(float), cudaMemcpyHostToDevice));

        printf("  Hidden size: %d\n", n_);
        printf("  Epsilon: %.2e\n", eps_);
    }

    void run_cpu_reference() override {
        // Compute sum of squares
        float sum_sq = 0.0f;
        for (int i = 0; i < n_; i++) {
            sum_sq += input_[i] * input_[i];
        }

        // Compute RMS
        float rms = sqrtf(sum_sq / n_ + eps_);

        // Normalize with weight
        for (int i = 0; i < n_; i++) {
            output_cpu_[i] = (input_[i] / rms) * weight_[i];
        }
    }

    void run_gpu_kernel() override {
        int block_size = 256;
        int shared_size = block_size * sizeof(float);

        rms_norm_kernel<<<1, block_size, shared_size>>>(
            d_input_, d_output_, d_weight_, n_, eps_);

        KERNEL_LAUNCH_CHECK();

        CUDA_CHECK(cudaMemcpy(output_gpu_, d_output_, n_ * sizeof(float), cudaMemcpyDeviceToHost));
    }

    void cleanup() override {
        delete[] input_;
        delete[] weight_;
        delete[] output_cpu_;
        delete[] output_gpu_;
        cudaFree(d_input_);
        cudaFree(d_weight_);
        cudaFree(d_output_);
    }

private:
    int n_;
    float eps_;
    float* input_;
    float* weight_;
    float* d_input_;
    float* d_weight_;
    float* d_output_;
};

// ============================================================================
// Test 4: Element-wise ADD
// ============================================================================

__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

class AddTest : public KernelTest {
public:
    const char* name() const override {
        return "ADD";
    }

    const char* description() const override {
        return "Element-wise addition: C = A + B";
    }

    float nmse_threshold() const override {
        return 1e-6f;  // Very tight for simple FP32 add
    }

    void setup(const TestConfig& config) override {
        n_ = config.M * config.N;

        input_a_ = new float[n_];
        input_b_ = new float[n_];
        output_cpu_ = new float[n_];
        output_gpu_ = new float[n_];
        output_size_ = n_;

        DataGenerator gen(config.seed);
        gen.generate(input_a_, n_, DataGenerator::NORMAL, 0.0f, 1.0f);
        gen.generate(input_b_, n_, DataGenerator::NORMAL, 0.0f, 1.0f);

        CUDA_CHECK(cudaMalloc(&d_a_, n_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_b_, n_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_c_, n_ * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_a_, input_a_, n_ * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b_, input_b_, n_ * sizeof(float), cudaMemcpyHostToDevice));

        printf("  Vector size: %d\n", n_);
    }

    void run_cpu_reference() override {
        for (int i = 0; i < n_; i++) {
            output_cpu_[i] = input_a_[i] + input_b_[i];
        }
    }

    void run_gpu_kernel() override {
        int block_size = 256;
        int grid_size = (n_ + block_size - 1) / block_size;

        add_kernel<<<grid_size, block_size>>>(d_a_, d_b_, d_c_, n_);

        KERNEL_LAUNCH_CHECK();

        CUDA_CHECK(cudaMemcpy(output_gpu_, d_c_, n_ * sizeof(float), cudaMemcpyDeviceToHost));
    }

    void cleanup() override {
        delete[] input_a_;
        delete[] input_b_;
        delete[] output_cpu_;
        delete[] output_gpu_;
        cudaFree(d_a_);
        cudaFree(d_b_);
        cudaFree(d_c_);
    }

private:
    int n_;
    float* input_a_;
    float* input_b_;
    float* d_a_;
    float* d_b_;
    float* d_c_;
};

// ============================================================================
// Test 5: GELU Activation
// ============================================================================

__global__ void gelu_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        const float SQRT_2_OVER_PI = 0.7978845608f;
        float x3 = x * x * x;
        float inner = SQRT_2_OVER_PI * (x + 0.044715f * x3);
        output[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

class GeluTest : public KernelTest {
public:
    const char* name() const override {
        return "GELU";
    }

    const char* description() const override {
        return "Gaussian Error Linear Unit activation";
    }

    float nmse_threshold() const override {
        return 1e-5f;
    }

    void setup(const TestConfig& config) override {
        n_ = config.M * config.K;

        input_ = new float[n_];
        output_cpu_ = new float[n_];
        output_gpu_ = new float[n_];
        output_size_ = n_;

        DataGenerator gen(config.seed);
        gen.generate(input_, n_, DataGenerator::NORMAL, 0.0f, 1.0f);

        CUDA_CHECK(cudaMalloc(&d_input_, n_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output_, n_ * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_input_, input_, n_ * sizeof(float), cudaMemcpyHostToDevice));

        printf("  Vector size: %d\n", n_);
    }

    void run_cpu_reference() override {
        const float SQRT_2_OVER_PI = 0.7978845608f;
        for (int i = 0; i < n_; i++) {
            float x = input_[i];
            float x3 = x * x * x;
            float inner = SQRT_2_OVER_PI * (x + 0.044715f * x3);
            output_cpu_[i] = 0.5f * x * (1.0f + tanhf(inner));
        }
    }

    void run_gpu_kernel() override {
        int block_size = 256;
        int grid_size = (n_ + block_size - 1) / block_size;

        gelu_kernel<<<grid_size, block_size>>>(d_input_, d_output_, n_);

        KERNEL_LAUNCH_CHECK();

        CUDA_CHECK(cudaMemcpy(output_gpu_, d_output_, n_ * sizeof(float), cudaMemcpyDeviceToHost));
    }

    void cleanup() override {
        delete[] input_;
        delete[] output_cpu_;
        delete[] output_gpu_;
        cudaFree(d_input_);
        cudaFree(d_output_);
    }

private:
    int n_;
    float* input_;
    float* d_input_;
    float* d_output_;
};

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    // Print GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("\n");
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Total Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    // Configure tests
    TestConfig config;
    config.M = 4;
    config.N = 512;
    config.K = 1024;
    config.verbose = true;
    config.print_samples = true;
    config.num_samples = 10;
    config.seed = 42;

    // Create test runner
    TestRunner runner;

    // Register tests
    MulMatQ4_0_Test mul_mat_test;
    SiluTest silu_test;
    RmsNormTest rms_norm_test;
    AddTest add_test;
    GeluTest gelu_test;

    runner.add_test(&mul_mat_test);
    runner.add_test(&silu_test);
    runner.add_test(&rms_norm_test);
    runner.add_test(&add_test);
    runner.add_test(&gelu_test);

    // Run all tests
    runner.run_all(config);

    return 0;
}
