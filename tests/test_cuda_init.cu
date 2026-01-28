/**
 * Simple CUDA initialization test
 */

#include <cstdio>
#include <cuda_runtime.h>

int main() {
    printf("Testing CUDA initialization...\n");
    fflush(stdout);

    // Test 1: Get device count
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        printf("ERROR: cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("Device count: %d\n", device_count);
    fflush(stdout);

    if (device_count == 0) {
        printf("ERROR: No CUDA devices found\n");
        return 1;
    }

    // Test 2: Get current device
    int device = 0;
    err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        printf("ERROR: cudaGetDevice failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("Current device: %d\n", device);
    fflush(stdout);

    // Test 3: Get device properties
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        printf("ERROR: cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("Device name: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    fflush(stdout);

    // Test 4: Allocate small buffer
    float* d_ptr = nullptr;
    err = cudaMalloc(&d_ptr, 1024 * sizeof(float));
    if (err != cudaSuccess) {
        printf("ERROR: cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("cudaMalloc succeeded\n");
    fflush(stdout);

    cudaFree(d_ptr);
    printf("All tests passed!\n");
    return 0;
}
