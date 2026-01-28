#!/bin/bash
# 测试环境脚本

echo "=== 检查 CUDA 环境 ==="
which nvcc
nvcc --version

echo ""
echo "=== 检查 GPU ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv

echo ""
echo "=== 检查编译器 ==="
which g++
g++ --version | head -1

echo ""
echo "=== 当前目录 ==="
pwd
ls -la include/ tests/ | head -20
