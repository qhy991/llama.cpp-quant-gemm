/**
 * test_utils.h - Testing and Benchmarking Utilities
 *
 * This file provides utilities for:
 * 1. Random data generation
 * 2. Accuracy comparison
 * 3. Performance benchmarking
 * 4. CUDA error checking
 */

#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include "quant_types.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <string>
#include <vector>

// ============================================================================
// CUDA Error Checking
// ============================================================================

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

#define CURAND_CHECK(call)                                                  \
    do {                                                                    \
        curandStatus_t status = call;                                       \
        if (status != CURAND_STATUS_SUCCESS) {                              \
            fprintf(stderr, "cuRAND error at %s:%d - %d\n",                 \
                    __FILE__, __LINE__, status);                            \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// ============================================================================
// Random Data Generation
// ============================================================================

/**
 * Initialize random float array on GPU
 */
inline void generate_random_fp32(float* d_data, int64_t n,
                                  float scale = 1.0f, curandGenerator_t gen = nullptr)
{
    bool own_gen = (gen == nullptr);
    if (own_gen) {
        CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, 42));
    }

    CURAND_CHECK(curandGenerateUniform(gen, d_data, n));

    // Scale and shift to [-scale, scale]
    // We'll use a kernel for this
    // For simplicity, just generate [0, 1] first

    if (own_gen) {
        CURAND_CHECK(curandDestroyGenerator(gen));
    }
}

/**
 * Transform kernel: x = (x - 0.5) * 2 * scale
 */
__global__ void transform_uniform_kernel(float* data, int64_t n, float scale) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = (data[idx] - 0.5f) * 2.0f * scale;
    }
}

inline void transform_uniform(float* d_data, int64_t n, float scale,
                               cudaStream_t stream = 0)
{
    int block = 256;
    int grid = (n + block - 1) / block;
    transform_uniform_kernel<<<grid, block, 0, stream>>>(d_data, n, scale);
}

/**
 * Generate random matrix on GPU
 */
inline float* generate_random_matrix(int rows, int cols, float scale = 1.0f,
                                      cudaStream_t stream = 0)
{
    int64_t n = (int64_t)rows * cols;
    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(float)));

    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, 42));

    CURAND_CHECK(curandGenerateUniform(gen, d_data, n));
    transform_uniform(d_data, n, scale, stream);

    CURAND_CHECK(curandDestroyGenerator(gen));

    return d_data;
}

// ============================================================================
// Accuracy Metrics
// ============================================================================

/**
 * Compute Mean Squared Error between two arrays
 */
inline float compute_mse(const float* a, const float* b, int64_t n) {
    double sum = 0.0;
    for (int64_t i = 0; i < n; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return (float)(sum / n);
}

/**
 * Compute Normalized Mean Squared Error
 * NMSE = MSE / Var(reference)
 */
inline float compute_nmse(const float* test, const float* ref, int64_t n) {
    double sum_sq_diff = 0.0;
    double sum_ref = 0.0;
    double sum_ref_sq = 0.0;

    for (int64_t i = 0; i < n; i++) {
        double diff = test[i] - ref[i];
        sum_sq_diff += diff * diff;
        sum_ref += ref[i];
        sum_ref_sq += ref[i] * ref[i];
    }

    double mean_ref = sum_ref / n;
    double var_ref = sum_ref_sq / n - mean_ref * mean_ref;

    if (var_ref < 1e-10) return 0.0f;

    return (float)(sum_sq_diff / n / var_ref);
}

/**
 * Compute max absolute error
 */
inline float compute_max_abs_error(const float* test, const float* ref, int64_t n) {
    float max_err = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        float err = fabsf(test[i] - ref[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

/**
 * Compare two arrays and print accuracy metrics
 */
inline void compare_results(const char* name, const float* test, const float* ref,
                             int64_t n, bool verbose = true)
{
    float mse = compute_mse(test, ref, n);
    float nmse = compute_nmse(test, ref, n);
    float max_err = compute_max_abs_error(test, ref, n);

    if (verbose) {
        printf("%-30s MSE: %.6e  NMSE: %.6e  Max: %.6e\n",
               name, mse, nmse, max_err);
    }
}

// ============================================================================
// Benchmarking
// ============================================================================

struct BenchmarkResult {
    std::string name;
    float time_ms;
    float tflops;
    float memory_gb_s;
};

/**
 * Benchmark a CUDA kernel
 *
 * @param name      Kernel name for reporting
 * @param kernel    Lambda that launches the kernel
 * @param warmup    Number of warmup iterations
 * @param repeat    Number of timed iterations
 * @param flops     Total FLOPs for the operation (for TFLOPS calculation)
 * @param bytes     Total bytes transferred (for bandwidth calculation)
 */
template <typename KernelFunc>
inline BenchmarkResult benchmark_kernel(
    const std::string& name,
    KernelFunc kernel,
    int warmup = 5,
    int repeat = 20,
    double flops = 0,
    double bytes = 0)
{
    // Warmup
    for (int i = 0; i < warmup; i++) {
        kernel();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Create events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Timed runs
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeat; i++) {
        kernel();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float avg_ms = total_ms / repeat;

    BenchmarkResult result;
    result.name = name;
    result.time_ms = avg_ms;
    result.tflops = (flops > 0) ? (flops / (avg_ms * 1e9)) : 0;
    result.memory_gb_s = (bytes > 0) ? (bytes / (avg_ms * 1e6)) : 0;

    return result;
}

/**
 * Print benchmark results table
 */
inline void print_benchmark_table(const std::vector<BenchmarkResult>& results) {
    printf("\n");
    printf("%-35s %12s %12s %12s\n",
           "Kernel", "Time (ms)", "TFLOPS", "BW (GB/s)");
    printf("%-35s %12s %12s %12s\n",
           "------", "---------", "------", "---------");

    for (const auto& r : results) {
        printf("%-35s %12.3f %12.3f %12.2f\n",
               r.name.c_str(), r.time_ms, r.tflops, r.memory_gb_s);
    }
    printf("\n");
}

// ============================================================================
// Memory Management Helpers
// ============================================================================

template <typename T>
class CudaBuffer {
public:
    T* ptr = nullptr;
    size_t size = 0;

    CudaBuffer() = default;

    explicit CudaBuffer(size_t count) : size(count * sizeof(T)) {
        CUDA_CHECK(cudaMalloc(&ptr, size));
    }

    ~CudaBuffer() {
        if (ptr) {
            cudaFree(ptr);
        }
    }

    // Disable copy
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    // Enable move
    CudaBuffer(CudaBuffer&& other) noexcept : ptr(other.ptr), size(other.size) {
        other.ptr = nullptr;
        other.size = 0;
    }

    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr) cudaFree(ptr);
            ptr = other.ptr;
            size = other.size;
            other.ptr = nullptr;
            other.size = 0;
        }
        return *this;
    }

    void copyToHost(T* host_ptr) const {
        CUDA_CHECK(cudaMemcpy(host_ptr, ptr, size, cudaMemcpyDeviceToHost));
    }

    void copyFromHost(const T* host_ptr) {
        CUDA_CHECK(cudaMemcpy(ptr, host_ptr, size, cudaMemcpyHostToDevice));
    }

    void zero() {
        CUDA_CHECK(cudaMemset(ptr, 0, size));
    }
};

// ============================================================================
// Device Information
// ============================================================================

inline void print_device_info() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("\n=== Device Information ===\n");
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("SMs: %d\n", prop.multiProcessorCount);
    printf("Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("Memory Bandwidth: %.0f GB/s\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    printf("L2 Cache: %d KB\n", prop.l2CacheSize / 1024);
    printf("Max Threads/Block: %d\n", prop.maxThreadsPerBlock);
    printf("Shared Memory/Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("===========================\n\n");
}

#endif // TEST_UTILS_H
