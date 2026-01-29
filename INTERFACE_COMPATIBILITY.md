# 接口兼容性说明

## 重要说明

本项目的 kernel 实现与 llama.cpp **不是直接兼容**的。需要通过兼容层进行适配。

## 兼容性层次

### 层次 1: 数据类型 ✅ 完全兼容

```cpp
// llama.cpp (ggml-common.h)
typedef struct {
    half d;
    uint8_t qs[QK4_0/2];
} block_q4_0;

// 本项目 (compat/ggml_types.h)
typedef struct {
    half d;
    uint8_t qs[QK4_0/2];
} block_q4_0;

// 内存布局完全一致，可以直接互换
```

### 层次 2: Kernel 逻辑 ✅ 功能等效

本项目的 kernel 实现与 llama.cpp 的核心逻辑相同：

| 操作 | 本项目 | llama.cpp | 等效性 |
|------|--------|-----------|--------|
| Q4_0×Q8_1 补偿公式 | `d_w * (d_a * sumi - 8 * s)` | 相同 | ✅ |
| SiLU | `x / (1 + exp(-x))` | 相同 | ✅ |
| RMS Norm | `x * rsqrt(mean(x²) + ε)` | 相同 | ✅ |
| Softmax | `exp(x-max) / sum(exp)` | 相同 | ✅ |
| RoPE | 旋转矩阵 | 相同 | ✅ |

### 层次 3: 接口签名 ❌ 不兼容

**llama.cpp 接口:**
```cpp
void ggml_cuda_op_soft_max(
    ggml_backend_cuda_context & ctx,
    ggml_tensor * dst
);
```

**本项目接口:**
```cpp
void softmax_forward_f32(
    const float* x,
    float* y,
    int n_rows,
    int n_cols,
    float scale,
    cudaStream_t stream
);
```

**差异:**
- llama.cpp 使用 `ggml_tensor*` 封装数据和元信息
- llama.cpp 使用 `ggml_backend_cuda_context` 管理 GPU 资源
- 本项目使用原始指针和显式参数

## 如何集成到 llama.cpp

### 方法 1: 使用兼容层 (推荐)

```cpp
// 在 llama.cpp 中
#include "quant-gemm-from-scratch/compat/ggml_cuda_compat.cuh"

// 替换原有实现
void ggml_cuda_op_silu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    // 使用我们的实现
    ggml_cuda_custom::ggml_cuda_op_silu_custom(ctx, dst);
}
```

### 方法 2: 在 Kernel 级别替换

```cpp
// 在 llama.cpp/ggml/src/ggml-cuda/unary.cu 中
#include "quant-gemm-from-scratch/kernels/activation/silu.cuh"

static __global__ void silu_f32(const float * x, float * dst, const int k) {
    // 直接使用我们的单元素实现
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < k) {
        float val = x[i];
        dst[i] = val / (1.0f + expf(-val));  // 与我们的实现相同
    }
}
```

### 方法 3: 外部库调用

```cpp
// C 接口调用
extern "C" void cuda_silu_f32(const float* x, float* y, int n, void* stream);

// 在 llama.cpp 中
void my_silu(const float* x, float* y, int n, cudaStream_t stream) {
    cuda_silu_f32(x, y, n, (void*)stream);
}
```

## 测试兼容性验证

### 验证数据类型兼容

```cpp
#include "compat/ggml_types.h"
#include "ggml-common.h"  // llama.cpp

// 编译时验证
static_assert(sizeof(block_q4_0) == 18);
static_assert(sizeof(block_q8_1) == 36);
static_assert(offsetof(block_q4_0, d) == 0);
static_assert(offsetof(block_q4_0, qs) == 2);
```

### 验证计算结果兼容

```bash
# 运行本项目测试
cd quant-gemm-from-scratch
make test-gemm-q4

# 运行 llama.cpp 测试
cd ../llama.cpp
./build/bin/test-backend-ops -o MUL_MAT
```

## 兼容性矩阵

| 组件 | 兼容级别 | 说明 |
|------|----------|------|
| `block_q4_0` | ✅ 完全 | 内存布局一致 |
| `block_q8_1` | ✅ 完全 | 内存布局一致 |
| GEMM Q4_0×Q8_1 | ✅ 功能 | 补偿公式一致 |
| SiLU | ✅ 功能 | 算法一致 |
| GELU | ✅ 功能 | tanh 近似一致 |
| RMS Norm | ⚠️ 部分 | 缺少融合版本 |
| Softmax | ⚠️ 部分 | 缺少 mask/alibi |
| RoPE | ⚠️ 部分 | 缺少 neox 模式 |
| Add/Mul | ✅ 功能 | 元素操作一致 |
| ggml_tensor 接口 | ❌ 需适配 | 需要包装层 |

## 结论

**本项目适合:**
- 学习 CUDA kernel 开发
- 理解 llama.cpp 底层实现
- 作为独立组件使用
- 通过兼容层集成到 llama.cpp

**本项目不适合:**
- 直接 drop-in 替换 llama.cpp 文件
- 需要完整 ggml 功能的场景
