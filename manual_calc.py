#!/usr/bin/env python3
"""
手动计算验证
"""

# A (Q8_1): d_a=0.5, s_a=-8
# qs = [-16, -15, -14, ..., 14, 15]
d_a = 0.5
s_a = -8.0
q_a = list(range(-16, 16))  # [-16, -15, ..., 14, 15]

# B (Q4_0): d_b=0.25
# qs = [0,1, 1,2, 2,3, 3,4, 4,5, 5,6, 6,7, 7,8, 8,9, 9,10, 10,11, 11,12, 12,13, 13,14, 14,15, 15,0]
d_b = 0.25
q_b = []
for i in range(16):
    q_b.append(i % 16)
    q_b.append((i + 1) % 16)

print("A (Q8_1):")
print(f"  d_a = {d_a}, s_a = {s_a}")
print(f"  q_a = {q_a[:8]} ...")
print(f"  原始值 = {[q * d_a for q in q_a[:8]]} ...")

print("\nB (Q4_0):")
print(f"  d_b = {d_b}")
print(f"  q_b = {q_b[:8]} ...")
print(f"  原始值 (q-8)*d = {[(q - 8) * d_b for q in q_b[:8]]} ...")

# 计算量化值的点积
sumi = sum(q_a[i] * q_b[i] for i in range(32))
print(f"\n量化值点积 sumi = {sumi}")

# 方法 1: 使用补偿公式
result1 = d_b * (d_a * sumi - 8.0 * s_a)
print(f"\n方法 1 (补偿公式): d_b * (d_a * sumi - 8 * s_a)")
print(f"  = {d_b} * ({d_a} * {sumi} - 8 * {s_a})")
print(f"  = {d_b} * ({d_a * sumi} - {8 * s_a})")
print(f"  = {d_b} * {d_a * sumi - 8 * s_a}")
print(f"  = {result1}")

# 方法 2: 直接计算原始值的点积
a_orig = [q * d_a for q in q_a]
b_orig = [(q - 8) * d_b for q in q_b]
result2 = sum(a_orig[i] * b_orig[i] for i in range(32))
print(f"\n方法 2 (直接计算): sum(a_orig * b_orig)")
print(f"  = {result2}")

print(f"\n差异: {abs(result1 - result2)}")
