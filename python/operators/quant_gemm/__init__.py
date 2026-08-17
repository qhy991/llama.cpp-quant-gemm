"""
Quantized GEMM operator family.

This family includes variants for different weight/activation quantization combinations:
- W4A8: 4-bit weights, 8-bit activations (Q4_0 x Q8_1)
- W4A16: 4-bit weights, FP32/FP16 activations
- W4_1A8: Asymmetric 4-bit weights (Q4_1 x Q8_1)
- W5A8: 5-bit weights, 8-bit activations
- W8A8: 8-bit weights and activations
"""
