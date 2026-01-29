#include <stdio.h>
#include <stdint.h>
#include <math.h>

int main() {
    // Create test data
    float w_f32[32], a_f32[32];
    for (int i = 0; i < 32; i++) {
        w_f32[i] = (i % 16) - 8.0f;
        a_f32[i] = i - 16.0f;
    }
    
    // Compute reference
    float ref = 0.0f;
    for (int i = 0; i < 32; i++) {
        ref += w_f32[i] * a_f32[i];
    }
    printf("Reference: %f\n", ref);
    
    // Quantize Q4_0 (new layout)
    float w_max = 0.0f;
    for (int i = 0; i < 32; i++) {
        float abs_val = fabsf(w_f32[i]);
        if (abs_val > w_max) w_max = abs_val;
    }
    float d_w = w_max / 7.0f;
    float inv_d_w = 1.0f / d_w;
    
    uint8_t w_qs[16];
    for (int i = 0; i < 16; i++) {
        int8_t v0 = (int8_t)roundf(w_f32[i] * inv_d_w);
        int8_t v1 = (int8_t)roundf(w_f32[i + 16] * inv_d_w);
        v0 = (v0 < -8) ? -8 : ((v0 > 7) ? 7 : v0);
        v1 = (v1 < -8) ? -8 : ((v1 > 7) ? 7 : v1);
        w_qs[i] = ((v0 + 8) & 0x0F) | (((v1 + 8) & 0x0F) << 4);
    }
    
    // Quantize Q8_1 (new layout)
    float a_max = 0.0f;
    for (int i = 0; i < 32; i++) {
        float abs_val = fabsf(a_f32[i]);
        if (abs_val > a_max) a_max = abs_val;
    }
    float d_a = a_max / 127.0f;
    float inv_d_a = 1.0f / d_a;
    
    int8_t a_qs[32];
    int sum_a_q = 0;
    for (int i = 0; i < 16; i++) {
        int8_t v0 = (int8_t)roundf(a_f32[i] * inv_d_a);
        int8_t v1 = (int8_t)roundf(a_f32[i + 16] * inv_d_a);
        a_qs[i] = v0;
        a_qs[i + 16] = v1;
        sum_a_q += v0 + v1;
    }
    float s_a = sum_a_q * d_a;
    
    // CPU reference
    int sumi = 0;
    for (int i = 0; i < 16; i++) {
        int w0 = (w_qs[i] & 0x0F);
        int w1 = ((w_qs[i] >> 4) & 0x0F);
        sumi += w0 * a_qs[i] + w1 * a_qs[i + 16];
    }
    
    float cpu_result = d_w * (d_a * sumi - 8.0f * s_a);
    printf("CPU result: %f\n", cpu_result);
    printf("CPU error: %f\n", cpu_result - ref);
    
    printf("\nDetails:\n");
    printf("d_w = %f, d_a = %f\n", d_w, d_a);
    printf("sum_a_q = %d, s_a = %f\n", sum_a_q, s_a);
    printf("sumi = %d\n", sumi);
    
    // Print first few values
    printf("\nFirst 4 Q4_0 values:\n");
    for (int i = 0; i < 2; i++) {
        int w0 = (w_qs[i] & 0x0F);
        int w1 = ((w_qs[i] >> 4) & 0x0F);
        printf("qs[%d]=0x%02x: w0=%d (w_f32[%d]=%.1f), w1=%d (w_f32[%d]=%.1f)\n",
               i, w_qs[i], w0, i, w_f32[i], w1, i+16, w_f32[i+16]);
    }
    
    printf("\nFirst 4 Q8_1 values:\n");
    for (int i = 0; i < 4; i++) {
        printf("qs[%d]=%d (a_f32[%d]=%.1f)\n", i, a_qs[i], i, a_f32[i]);
    }
    
    return 0;
}
