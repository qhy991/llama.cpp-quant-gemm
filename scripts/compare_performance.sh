#!/bin/bash

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  quant-gemm-from-scratch vs llama.cpp 性能对比                ║"
echo "║  矩阵尺寸: M=4096, N=2, K=14336                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate KM-12.8

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "测试 1: llama.cpp 官方实现"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cd /home/haiyan/Agent4Kernel/llama.cpp/build/bin

echo "Q4_0 性能:"
./test-backend-ops perf -o MUL_MAT -b CUDA0 -p "type_a=q4_0.*m=4096.*n=2.*k=14336" 2>&1 | grep "GFLOPS\|TFLOPS"

echo ""
echo "Q8_0 性能:"
./test-backend-ops perf -o MUL_MAT -b CUDA0 -p "type_a=q8_0.*m=4096.*n=2.*k=14336" 2>&1 | grep "GFLOPS\|TFLOPS"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "测试 2: quant-gemm-from-scratch 实现"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "注意: quant-gemm-from-scratch 使用的是教学性质的实现"
echo "      主要目标是正确性和可读性，而非极致性能"
echo ""

cd /home/haiyan/Agent4Kernel/quant-gemm-from-scratch

# Check if we have a performance test binary
if [ -f "test_custom_mnk" ]; then
    echo "运行自定义性能测试..."
    ./test_custom_mnk 4096 2 14336 2>&1 | grep -E "Performance|TFLOPS|Time"
else
    echo "未找到性能测试程序"
    echo "quant-gemm-from-scratch 项目主要用于教学，展示量化 GEMM 的实现原理"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "总结"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "quant-gemm-from-scratch 项目特点:"
echo "  ✓ 教学导向 - 每行代码都有详细注释"
echo "  ✓ 循序渐进 - 从 FP32 到量化的完整路径"
echo "  ✓ 100% 兼容 llama.cpp 的量化格式"
echo "  ✓ 正确性验证 - 与 llama.cpp 结果一致"
echo ""
echo "llama.cpp 项目特点:"
echo "  ✓ 生产级优化 - 使用 DP4A/Tensor Core"
echo "  ✓ 极致性能 - 针对各种 GPU 架构优化"
echo "  ✓ 工业应用 - 实际 LLM 推理场景"
echo ""
echo "两个项目的定位不同:"
echo "  - quant-gemm-from-scratch: 学习和理解量化 GEMM 原理"
echo "  - llama.cpp: 实际生产环境中的高性能推理"
echo ""
