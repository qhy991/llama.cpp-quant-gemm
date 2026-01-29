#include <stdio.h>
#include <cuda_fp16.h>

int main() {
    // Test Q8_1 quantization
    float values[32];
    for (int i = 0; i < 32; i++) {
        values[i] = (i - 16) * 0.1f;  // -1.6 to 1.5
    }
    
    // Quantize
    float max_abs = 0.0f;
    for (int i = 0; i < 32; i++) {
        float abs_val = fabsf(values[i]);
        if (abs_val > max_abs) max_abs = abs_val;
    }
    
    float scale = max_abs / 127.0f;
    float inv_scale = 1.0f / scale;
    
    int sum_q = 0;
    int8_t qs[32];
    for (int i = 0; i < 32; i++) {
        qs[i] = (int8_t)roundf(values[i] * inv_scale);
        sum_q += qs[i];
    }
    
    float s = sum_q * scale;
    
    printf("scale = %f\n", scale);
    printf("sum_q = %d\n", sum_q);
    printf("s = sum_q * scale = %f\n", s);
    
    // Verify
    float sum_original = 0.0f;
    for (int i = 0; i < 32; i++) {
        sum_original += values[i];
    }
    printf("sum_original = %f\n", sum_original);
    
    return 0;
}
