#include <stdio.h>
#include <math.h>
#include <stdint.h>

int main() {
    // Simple test: 32 element dot product
    float w_vals[32], a_vals[32];
    
    // Initialize with simple values
    for (int i = 0; i < 32; i++) {
        w_vals[i] = (i % 16) - 8.0f;  // -8 to 7
        a_vals[i] = i - 16.0f;         // -16 to 15
    }
    
    // Compute reference dot product
    float ref_dot = 0.0f;
    for (int i = 0; i < 32; i++) {
        ref_dot += w_vals[i] * a_vals[i];
    }
    printf("Reference dot product: %f\n", ref_dot);
    
    // Quantize Q4_0
    float w_max = 0.0f;
    for (int i = 0; i < 32; i++) {
        float abs_val = fabsf(w_vals[i]);
        if (abs_val > w_max) w_max = abs_val;
    }
    float d_w = w_max / 7.0f;  // Q4_0: max value is 7 (15-8)
    float inv_d_w = 1.0f / d_w;
    
    uint8_t w_q[32];
    for (int i = 0; i < 32; i++) {
        int q = (int)roundf(w_vals[i] * inv_d_w) + 8;
        q = (q < 0) ? 0 : ((q > 15) ? 15 : q);
        w_q[i] = q;
    }
    
    // Quantize Q8_1
    float a_max = 0.0f;
    for (int i = 0; i < 32; i++) {
        float abs_val = fabsf(a_vals[i]);
        if (abs_val > a_max) a_max = abs_val;
    }
    float d_a = a_max / 127.0f;
    float inv_d_a = 1.0f / d_a;
    
    int8_t a_q[32];
    int sum_a_q = 0;
    for (int i = 0; i < 32; i++) {
        int8_t q = (int8_t)roundf(a_vals[i] * inv_d_a);
        a_q[i] = q;
        sum_a_q += q;
    }
    float s_a = sum_a_q * d_a;
    
    // Compute quantized dot product
    int sumi = 0;
    for (int i = 0; i < 32; i++) {
        sumi += w_q[i] * a_q[i];
    }
    
    float quant_dot = d_w * (d_a * sumi - 8.0f * s_a);
    
    printf("Quantized dot product: %f\n", quant_dot);
    printf("Error: %f\n", quant_dot - ref_dot);
    printf("\nDetails:\n");
    printf("d_w = %f, d_a = %f\n", d_w, d_a);
    printf("sum_a_q = %d, s_a = %f\n", sum_a_q, s_a);
    printf("sumi = %d\n", sumi);
    
    return 0;
}
