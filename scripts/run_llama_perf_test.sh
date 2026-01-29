#!/bin/bash
#
# run_llama_perf_test.sh
#
# 使用 llama.cpp 的 test-backend-ops 进行性能测试
# 然后可以与自己的实现进行对比
#

set -e

LLAMA_CPP_DIR="/home/haiyan/Agent4Kernel/llama.cpp"
QUANT_GEMM_DIR="/home/haiyan/Agent4Kernel/quant-gemm-from-scratch"

echo "=============================================="
echo "  llama.cpp 量化 GEMM 性能测试"
echo "=============================================="

# 检查 llama.cpp 是否已编译
if [ ! -f "$LLAMA_CPP_DIR/build/bin/test-backend-ops" ]; then
    echo "编译 llama.cpp test-backend-ops..."
    cd "$LLAMA_CPP_DIR"
    mkdir -p build && cd build
    cmake .. -DGGML_CUDA=ON
    cmake --build . --target test-backend-ops -j$(nproc)
fi

cd "$LLAMA_CPP_DIR/build"

echo ""
echo "--- 测试 MUL_MAT (Q4_0 x F32) 性能 ---"
echo ""

# 运行性能测试 - Q4_0 量化矩阵乘法
# 这是 llama.cpp 中最常用的量化 GEMM 操作
./bin/test-backend-ops perf \
    -o MUL_MAT \
    -p "type_a=q4_0,type_b=f32" \
    -b CUDA0 \
    --output console 2>&1 | head -100

echo ""
echo "--- 测试其他量化类型 ---"
echo ""

# Q8_0
echo "Q8_0 x F32:"
./bin/test-backend-ops perf \
    -o MUL_MAT \
    -p "type_a=q8_0,type_b=f32,m=4096,n=1,k=4096" \
    -b CUDA0 \
    --output console 2>&1 | grep -E "MUL_MAT|us/run|GFLOPS" | head -20

echo ""
echo "=============================================="
echo "  测试完成"
echo "=============================================="
echo ""
echo "要测试你自己的实现，可以使用类似的测试框架:"
echo "  cd $QUANT_GEMM_DIR"
echo "  make step4"
echo "  ./bin/step4_w4a8_gemm"
echo ""
