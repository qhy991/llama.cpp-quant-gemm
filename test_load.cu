#include "compat/ggml_types.h"
#include "tests/framework/test_framework.cuh"
#include <stdio.h>

__device__ __forceinline__ int load_int_b2(const void* x, int i32) {
    const uint16_t* x16 = (const uint16_t*)x;
    int x32  = x16[2*i32 + 0] <<  0;
    x32     |= x16[2*i32 + 1] << 16;
    return x32;
}

__device__ __forceinline__ int load_int_b4(const void* x, int i32) {
    return ((const int*)x)[i32];
}

__global__ void test_load_kernel(const block_q4_0* w, const block_q8_1* a, int* out_w, int* out_a) {
    // Load Q4_0
    for (int i = 0; i < 4; i++) {
        out_w[i] = load_int_b2(w->qs, i);
    }
    
    // Load Q8_1
    for (int i = 0; i < 8; i++) {
        out_a[i] = load_int_b4(a->qs, i);
    }
}

int main() {
    // Create test data
    float w_f32[32], a_f32[32];
    for (int i = 0; i < 32; i++) {
        w_f32[i] = (i % 16) - 8.0f;
        a_f32[i] = i - 16.0f;
    }
    
    // Quantize
    block_q4_0 w_q;
    block_q8_1 a_q;
    testing::quantize::to_q4_0(w_f32, &w_q, 32);
    testing::quantize::to_q8_1(a_f32, &a_q, 32);
    
    // Print CPU data
    printf("Q4_0 qs (CPU):\n");
    for (int i = 0; i < 16; i++) {
        printf("%02x ", w_q.qs[i]);
    }
    printf("\n");
    
    printf("Q8_1 qs (CPU):\n");
    for (int i = 0; i < 32; i++) {
        printf("%02x ", (uint8_t)a_q.qs[i]);
    }
    printf("\n");
    
    // GPU test
    block_q4_0 *d_w;
    block_q8_1 *d_a;
    int *d_out_w, *d_out_a;
    
    cudaMalloc(&d_w, sizeof(block_q4_0));
    cudaMalloc(&d_a, sizeof(block_q8_1));
    cudaMalloc(&d_out_w, 4 * sizeof(int));
    cudaMalloc(&d_out_a, 8 * sizeof(int));
    
    cudaMemcpy(d_w, &w_q, sizeof(block_q4_0), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a, &a_q, sizeof(block_q8_1), cudaMemcpyHostToDevice);
    
    test_load_kernel<<<1, 1>>>(d_w, d_a, d_out_w, d_out_a);
    
    int out_w[4], out_a[8];
    cudaMemcpy(out_w, d_out_w, 4 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_a, d_out_a, 8 * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("\nQ4_0 loaded (GPU):\n");
    for (int i = 0; i < 4; i++) {
        printf("%08x ", out_w[i]);
    }
    printf("\n");
    
    printf("Q8_1 loaded (GPU):\n");
    for (int i = 0; i < 8; i++) {
        printf("%08x ", out_a[i]);
    }
    printf("\n");
    
    cudaFree(d_w);
    cudaFree(d_a);
    cudaFree(d_out_w);
    cudaFree(d_out_a);
    
    return 0;
}
