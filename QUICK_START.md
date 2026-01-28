# 快速开始指南

## 🚀 5 分钟快速测试

### 1. 编译所有步骤

```bash
cd /home/haiyan/Agent4Kernel/quant-gemm-from-scratch

# 使用 conda 环境中的 nvcc
/home/haiyan/miniconda3/envs/KM-12.8/bin/nvcc -O3 -arch=sm_90 -std=c++17 \
  -I./include tests/step1_fp32_gemm.cu -o bin/step1_fp32_gemm -lcurand

/home/haiyan/miniconda3/envs/KM-12.8/bin/nvcc -O3 -arch=sm_90 -std=c++17 \
  -I./include tests/step2_quantization.cu -o bin/step2_quantization -lcurand

/home/haiyan/miniconda3/envs/KM-12.8/bin/nvcc -O3 -arch=sm_90 -std=c++17 \
  -I./include tests/step3_w4a16_gemm.cu -o bin/step3_w4a16_gemm -lcurand

/home/haiyan/miniconda3/envs/KM-12.8/bin/nvcc -O3 -arch=sm_90 -std=c++17 \
  -I./include tests/step4_w4a8_gemm.cu -o bin/step4_w4a8_gemm -lcurand
```

### 2. 运行测试

```bash
# Step 1: FP32 基准
./bin/step1_fp32_gemm

# Step 2: 量化介绍
./bin/step2_quantization

# Step 3: W4A16 GEMM
./bin/step3_w4a16_gemm

# Step 4: W4A8 GEMM（核心）
./bin/step4_w4a8_gemm
```

## 📊 预期结果

### Step 1 ✅
- 正确性: NMSE < 1e-13
- 性能: ~0.1 TFLOPS

### Step 2 ✅
- Q4_0 NMSE: ~4.6e-3
- Q8_0 NMSE: ~1.4e-5
- Sum 字段验证通过

### Step 3 ✅
- 量化误差: NMSE ~4.6e-3
- 内存节省: 7.1x

### Step 4 ⚠️
- 补偿公式演示: ✅ 成功
- Naive/Tiled: ✅ 通过
- DP4A: ❌ 内存对齐错误

## 📚 详细文档

- **完整教程**: `README.md`
- **测试结果**: `TEST_RESULTS.md`
- **项目总结**: `PROJECT_SUMMARY.md`
- **入门指南**: `docs/GETTING_STARTED.md`

## 🐛 常见问题

### nvcc 找不到？

```bash
# 使用完整路径
/home/haiyan/miniconda3/envs/KM-12.8/bin/nvcc --version
```

### GPU 架构不匹配？

```bash
# 检查你的 GPU
nvidia-smi --query-gpu=compute_cap --format=csv

# 使用正确的架构
# RTX 5070: sm_90
# RTX 4090: sm_89
# RTX 3090: sm_86
# A100: sm_80
```

## 🎯 核心学习点

1. **量化格式**: Q4_0, Q8_0, Q8_1
2. **补偿公式**: `result = d_w × (d_a × sumi - 8 × s_a)`
3. **为什么需要补偿**: Q4_0 存储 [0,15] 表示 [-8,7]

## 📈 下一步

1. 阅读 `README.md` 了解详细原理
2. 查看 `TEST_RESULTS.md` 了解性能数据
3. 修改代码进行实验
4. 尝试修复 DP4A 对齐问题

---

**祝学习愉快！** 🚀
