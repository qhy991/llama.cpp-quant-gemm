#!/bin/bash

# 测试自定义 kernel 的脚本

set -e  # 遇到错误立即退出

echo "========================================"
echo "编译和测试自定义 Kernel"
echo "========================================"
echo ""

# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate KM-12.8

# 设置路径
PROJECT_DIR="/home/haiyan/Agent4Kernel/quant-gemm-from-scratch"
TEST_DIR="$PROJECT_DIR/tests"
BUILD_DIR="$PROJECT_DIR/build"

# 创建 build 目录
mkdir -p "$BUILD_DIR"

echo "1. 编译测试程序..."
echo ""

# 编译参数
NVCC_FLAGS="-O3 -std=c++17 -arch=sm_120a"
INCLUDE_FLAGS="-I$PROJECT_DIR/include"

# 编译
nvcc $NVCC_FLAGS $INCLUDE_FLAGS \
    "$TEST_DIR/test_custom_kernel.cu" \
    -o "$BUILD_DIR/test_custom_kernel"

if [ $? -eq 0 ]; then
    echo "✓ 编译成功！"
    echo ""
else
    echo "✗ 编译失败"
    exit 1
fi

echo "2. 运行测试..."
echo ""

# 运行测试
"$BUILD_DIR/test_custom_kernel"

echo ""
echo "========================================"
echo "测试完成！"
echo "========================================"
