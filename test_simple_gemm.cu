#include "compat/ggml_types.h"
#include "tests/framework/test_framework.cuh"
#include "kernels/gemm/gemm_quant_formats.cuh"
#include <stdio.h>

int main() {
    // Simple 1x1x32 GEMM test
    const int M = 1, N = 1, K = 32;
    
    // Create simple test data
    float w_f32[32], a_f32[32];
    for (int i = 0; i < 32; i++) {
        w_f32[i] = (i % 16) - 8.0f;  // -8 to 7
        a_f32[i] = i - 16.0f;         // -16 to 15
    }
    
    // Compute reference
    float ref = 0.0f;
    for (int i = 0; i < 32; i++) {
        ref += w_f32[i] * a_f32[i];
    }
    printf("Reference: %f\n", ref);
    
    // Quantize
    block_q4_0 w_q;
    block_q8_1 a_q;
    testing::quantize::to_q4_0(w_f32, &w_q, 32);
    testing::quantize::to_q8_1(a_f32, &a_q, 32);
    
    // CPU reference
    float cpu_result = 0.0f;
    float d_w = __half2float(w_q.d);
    float d_a = __half2float(__low2half(a_q.ds));
    float s_a = __half2float(__high2half(a_q.ds));
    
    int sumi = 0;
    for (int i = 0; i < 16; i++) {
        int w0 = (w_q.qs[i] & 0x0F);
        int w1 = ((w_q.qs[i] >> 4) & 0x0F);
        sumi += w0 * a_q.qs[i*2 + 0] + w1 * a_q.qs[i*2 + 1];
    }
    cpu_result = d_w * (d_a * sumi - 8.0f * s_a);
    printf("CPU result: %f\n", cpu_result);
    printf("CPU error: %f\n", cpu_result - ref);
    
    // GPU test
    block_q4_0 *d_w_ptr;
    block_q8_1 *d_a_ptr;
    float *d_out_ptr;
    
    cudaMalloc(&d_w_ptr, sizeof(block_q4_0));
    cudaMalloc(&d_a_ptr, sizeof(block_q8_1));
    cudaMalloc(&d_out_ptr, sizeof(float));
    
    cudaMemcpy(d_w_ptr, &w_q, sizeof(block_q4_0), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_ptr, &a_q, sizeof(block_q8_1), cudaMemcpyHostToDevice);
    
    gemm_q4_0_q8_1(d_w_ptr, d_a_ptr, d_out_ptr, M, N, K);
    
    float gpu_result;
    cudaMemcpy(&gpu_result, d_out_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("GPU result: %f\n", gpu_result);
    printf("GPU error: %f\n", gpu_result - ref);
    
    cudaFree(d_w_ptr);
    cudaFree(d_a_ptr);
    cudaFree(d_out_ptr);
    
    return 0;
}
