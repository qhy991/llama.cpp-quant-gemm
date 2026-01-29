#include <stdio.h>
#include <stdint.h>

__device__ __forceinline__ int dp4a(int a, int b, int c) {
#if __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    char4 va = *reinterpret_cast<char4*>(&a);
    char4 vb = *reinterpret_cast<char4*>(&b);
    return c + va.x*vb.x + va.y*vb.y + va.z*vb.z + va.w*vb.w;
#endif
}

__global__ void test_dp4a_kernel(int a, int b, int* out) {
    *out = dp4a(a, b, 0);
}

int main() {
    // Test DP4A
    // a = 0x03020100 (bytes: 0, 1, 2, 3)
    // b = 0x07060504 (bytes: 4, 5, 6, 7)
    // Expected: 0*4 + 1*5 + 2*6 + 3*7 = 0 + 5 + 12 + 21 = 38
    
    int a = 0x03020100;
    int b = 0x07060504;
    
    int *d_out;
    cudaMalloc(&d_out, sizeof(int));
    
    test_dp4a_kernel<<<1, 1>>>(a, b, d_out);
    
    int result;
    cudaMemcpy(&result, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("DP4A result: %d (expected: 38)\n", result);
    
    // Test with Q4_0 style data
    // vi0 = 0x03020100 (nibbles: 0, 1, 2, 3 in bytes 0-3)
    // u0 = 0x81898991 (bytes: -127, -119, -111, -103)
    int vi0 = 0x03020100;
    int u0 = 0x99918981;  // Note: little endian
    
    test_dp4a_kernel<<<1, 1>>>(vi0, u0, d_out);
    cudaMemcpy(&result, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    
    // Expected: 0*(-127) + 1*(-119) + 2*(-111) + 3*(-103)
    //         = 0 - 119 - 222 - 309 = -650
    printf("DP4A with Q data: %d (expected: -650)\n", result);
    
    cudaFree(d_out);
    
    return 0;
}
