#!/usr/bin/env python3
"""
详细分析 DP4A 计算
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

print("详细计算过程：\n")

# 模拟 GPU 的 DP4A 计算
sumi_gpu = 0
for i in range(4):
    # 加载 8 个激活值
    a0_vals = q_a[i*8 : i*8+4]
    a1_vals = q_a[i*8+4 : i*8+8]

    # 加载 8 个权重值（4 字节，每字节 2 个 4-bit 值）
    w_vals = q_b[i*8 : i*8+8]
    w_lo = [w_vals[j*2] for j in range(4)]  # 低 4 位
    w_hi = [w_vals[j*2+1] for j in range(4)]  # 高 4 位

    # DP4A
    dp0 = sum(a0_vals[j] * w_lo[j] for j in range(4))
    dp1 = sum(a1_vals[j] * w_hi[j] for j in range(4))

    print(f"迭代 {i}:")
    print(f"  a0 = {a0_vals}, w_lo = {w_lo}")
    print(f"  dp0 = {a0_vals} · {w_lo} = {dp0}")
    print(f"  a1 = {a1_vals}, w_hi = {w_hi}")
    print(f"  dp1 = {a1_vals} · {w_hi} = {dp1}")
    print(f"  sumi += {dp0} + {dp1} = {dp0 + dp1}")

    sumi_gpu += dp0 + dp1

print(f"\n总和 sumi = {sumi_gpu}")

# 应用补偿公式
result_gpu = d_b * (d_a * sumi_gpu - 8.0 * s_a)
print(f"\n结果 = {d_b} * ({d_a} * {sumi_gpu} - 8 * {s_a})")
print(f"     = {d_b} * ({d_a * sumi_gpu} - {8 * s_a})")
print(f"     = {d_b} * {d_a * sumi_gpu - 8 * s_a}")
print(f"     = {result_gpu}")

# 验证：直接计算
sumi_direct = sum(q_a[i] * q_b[i] for i in range(32))
result_direct = d_b * (d_a * sumi_direct - 8.0 * s_a)
print(f"\n直接计算: sumi = {sumi_direct}, result = {result_direct}")
print(f"差异: {abs(result_gpu - result_direct)}")
