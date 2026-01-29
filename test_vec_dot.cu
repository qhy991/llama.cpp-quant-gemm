#include "compat/ggml_types.h"
#include "tests/framework/test_framework.cuh"
#include <stdio.h>

__device__ __forceinline__ int dp4a(int a, int b, int c) {
#if __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    char4 va = *reinterpret_cast<char4*>(&a);
    char4 vb = *reinterpret_cast<char4*>(&b);
    return c + va.x*vb.x + va.y*vb.y + va.z*vb.z + va.w*vb.w;
#endif
}

__device__ __forceinline__ int load_int_b2(const void* x, int i32) {
    const uint16_t* x16 = (const uint16_t*)x;
    int x32  = x16[2*i32 + 0] <<  0;
    x32     |= x16[2*i32 + 1] << 16;
    return x32;
}

__device__ __forceinline__ int load_int_b4(const void* x, int i32) {
    return ((const int*)x)[i32];
}

__global__ void test_vec_dot_kernel(const block_q4_0* w, const block_q8_1* a, float* out, int* debug) {
    int sumi = 0;
    
    for (int i = 0; i < 4; i++) {
        int v = load_int_b2(w->qs, i);
        int vi0 = (v >> 0) & 0x0F0F0F0F;
        int vi1 = (v >> 4) & 0x0F0F0F0F;
        
        int u0 = load_int_b4(a->qs, 2*i + 0);
        int u1 = load_int_b4(a->qs, 2*i + 1);
        
        if (i == 0) {
            debug[0] = v;
            debug[1] = vi0;
            debug[2] = vi1;
            debug[3] = u0;
            debug[4] = u1;
        }
        
        sumi = dp4a(vi0, u0, sumi);
        sumi = dp4a(vi1, u1, sumi);
    }
    
    debug[5] = sumi;
    
    float d4 = __half2float(w->d);
    float d8 = __half2float(__low2half(a->ds));
    float s8 = __half2float(__high2half(a->ds));
    
    *out = d4 * (d8 * sumi - 8.0f * s8);
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
    
    // CPU computation
    int cpu_sumi = 0;
    for (int i = 0; i < 16; i++) {
        int w0 = (w_q.qs[i] & 0x0F);
        int w1 = ((w_q.qs[i] >> 4) & 0x0F);
        cpu_sumi += w0 * a_q.qs[i*2 + 0] + w1 * a_q.qs[i*2 + 1];
    }
    printf("CPU sumi: %d\n", cpu_sumi);
    
    // GPU test
    block_q4_0 *d_w;
    block_q8_1 *d_a;
    float *d_out;
    int *d_debug;
    
    cudaMalloc(&d_w, sizeof(block_q4_0));
    cudaMalloc(&d_a, sizeof(block_q8_1));
    cudaMalloc(&d_out, sizeof(float));
    cudaMalloc(&d_debug, 6 * sizeof(int));
    
    cudaMemcpy(d_w, &w_q, sizeof(block_q4_0), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a, &a_q, sizeof(block_q8_1), cudaMemcpyHostToDevice);
    
    test_vec_dot_kernel<<<1, 1>>>(d_w, d_a, d_out, d_debug);
    
    float result;
    int debug[6];
    cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(debug, d_debug, 6 * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("GPU sumi: %d\n", debug[5]);
    printf("GPU result: %f\n", result);
    printf("\nDebug (i=0):\n");
    printf("v:   %08x\n", debug[0]);
    printf("vi0: %08x\n", debug[1]);
    printf("vi1: %08x\n", debug[2]);
    printf("u0:  %08x\n", debug[3]);
    printf("u1:  %08x\n", debug[4]);
    
    cudaFree(d_w);
    cudaFree(d_a);
    cudaFree(d_out);
    cudaFree(d_debug);
    
    return 0;
}
