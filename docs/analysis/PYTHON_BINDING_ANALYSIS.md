# Python 绑定可行性分析

本文档分析将现有 CUDA kernel 封装为 Python 可调用模块的可行性和实现方案。

---

## 1. 可行性结论

**结论：完全可行**，且有多种实现方案可选。推荐使用 **PyTorch C++ Extension** 方案。

| 方案 | 难度 | 与 PyTorch 集成 | 性能 | 推荐场景 |
|------|------|----------------|------|---------|
| PyTorch C++ Extension | ⭐⭐ | 原生支持 | 最优 | **推荐** |
| pybind11 + CUDA | ⭐⭐⭐ | 需手动转换 | 优秀 | 独立库 |
| CuPy RawKernel | ⭐ | 不支持 | 良好 | 快速原型 |
| Triton | ⭐⭐ | 原生支持 | 优秀 | 重写 kernel |

---

## 2. 现有代码结构分析

### 2.1 Kernel 接口

```cpp
// 当前接口
void gemm_q4_0_q8_1(
    const block_q4_0* weight,    // GPU 指针
    const block_q8_1* activation, // GPU 指针
    float* output,               // GPU 指针
    int M, int N, int K,
    cudaStream_t stream = 0
);
```

### 2.2 数据类型

```cpp
// 量化块类型 (18 bytes)
typedef struct {
    half d;                 // 缩放因子
    uint8_t qs[16];        // 量化值
} block_q4_0;

// 激活块类型 (36 bytes)
typedef struct {
    half2 ds;               // d 和 s 打包
    int8_t qs[32];         // 量化值
} block_q8_1;
```

### 2.3 封装挑战

1. **自定义数据类型**: 需要在 Python 中表示 `block_q4_0` 等类型
2. **GPU 内存管理**: 需要处理 Host/Device 内存转换
3. **量化函数**: 需要暴露量化/反量化函数

---

## 3. 方案 A: PyTorch C++ Extension（推荐）

### 3.1 优势

- 与 PyTorch 张量无缝集成
- 自动处理 GPU 内存
- 支持自动微分（如果需要）
- 社区成熟，文档完善

### 3.2 实现架构

```
quant-gemm-from-scratch/
├── python/
│   ├── quant_gemm/
│   │   ├── __init__.py
│   │   ├── ops.py           # Python 接口
│   │   └── _C/              # C++ 扩展
│   ├── csrc/
│   │   ├── bindings.cpp     # pybind11 绑定
│   │   ├── gemm_ops.cu      # CUDA 封装
│   │   └── quantize_ops.cu  # 量化封装
│   └── setup.py
└── kernels/                  # 现有 kernel
```

### 3.3 核心代码示例

**csrc/bindings.cpp**
```cpp
#include <torch/extension.h>
#include <cuda_runtime.h>

// 声明 CUDA 函数
torch::Tensor gemm_q4_0_q8_1_forward(
    torch::Tensor weight_q,      // [M, K/32, 18] uint8
    torch::Tensor activation_q,  // [N, K/32, 36] uint8
    int M, int N, int K
);

torch::Tensor quantize_to_q4_0(torch::Tensor input);
torch::Tensor quantize_to_q8_1(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_q4_0_q8_1", &gemm_q4_0_q8_1_forward,
          "Quantized GEMM (Q4_0 x Q8_1)");
    m.def("quantize_q4_0", &quantize_to_q4_0,
          "Quantize FP32 to Q4_0");
    m.def("quantize_q8_1", &quantize_to_q8_1,
          "Quantize FP32 to Q8_1");
}
```

**csrc/gemm_ops.cu**
```cpp
#include <torch/extension.h>
#include <cuda_runtime.h>
#include "../../kernels/gemm/gemm_quant_formats.cuh"

torch::Tensor gemm_q4_0_q8_1_forward(
    torch::Tensor weight_q,
    torch::Tensor activation_q,
    int M, int N, int K
) {
    // 检查输入
    TORCH_CHECK(weight_q.is_cuda(), "weight must be CUDA tensor");
    TORCH_CHECK(activation_q.is_cuda(), "activation must be CUDA tensor");
    TORCH_CHECK(weight_q.dtype() == torch::kUInt8, "weight must be uint8");

    // 创建输出张量
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(weight_q.device());
    torch::Tensor output = torch::empty({M, N}, options);

    // 获取原始指针
    const block_q4_0* weight_ptr =
        reinterpret_cast<const block_q4_0*>(weight_q.data_ptr<uint8_t>());
    const block_q8_1* act_ptr =
        reinterpret_cast<const block_q8_1*>(activation_q.data_ptr<uint8_t>());
    float* output_ptr = output.data_ptr<float>();

    // 调用 kernel
    gemm_q4_0_q8_1(weight_ptr, act_ptr, output_ptr, M, N, K);

    return output;
}
```

**python/quant_gemm/ops.py**
```python
import torch
from . import _C

def gemm_q4_0_q8_1(weight_q: torch.Tensor, activation_q: torch.Tensor,
                    M: int, N: int, K: int) -> torch.Tensor:
    """
    执行量化 GEMM: C[M,N] = W[M,K] @ A[N,K]^T

    Args:
        weight_q: 量化权重, shape [M, K//32, 18], dtype uint8
        activation_q: 量化激活, shape [N, K//32, 36], dtype uint8
        M, N, K: 矩阵维度

    Returns:
        output: FP32 结果, shape [M, N]
    """
    return _C.gemm_q4_0_q8_1(weight_q, activation_q, M, N, K)


def quantize_q4_0(x: torch.Tensor) -> torch.Tensor:
    """将 FP32 张量量化为 Q4_0 格式"""
    return _C.quantize_q4_0(x)


def quantize_q8_1(x: torch.Tensor) -> torch.Tensor:
    """将 FP32 张量量化为 Q8_1 格式"""
    return _C.quantize_q8_1(x)
```

**setup.py**
```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='quant_gemm',
    version='0.1.0',
    packages=['quant_gemm'],
    ext_modules=[
        CUDAExtension(
            name='quant_gemm._C',
            sources=[
                'csrc/bindings.cpp',
                'csrc/gemm_ops.cu',
                'csrc/quantize_ops.cu',
            ],
            include_dirs=[
                '../kernels',
                '../compat',
                '../include',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '-arch=sm_86', '--use_fast_math'],
            },
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
```

### 3.4 使用示例

```python
import torch
import quant_gemm

# 创建测试数据
M, N, K = 4096, 2, 14336
weight_fp32 = torch.randn(M, K, device='cuda')
activation_fp32 = torch.randn(N, K, device='cuda')

# 量化
weight_q = quant_gemm.quantize_q4_0(weight_fp32)
activation_q = quant_gemm.quantize_q8_1(activation_fp32)

# 执行 GEMM
output = quant_gemm.gemm_q4_0_q8_1(weight_q, activation_q, M, N, K)

# 对比参考实现
output_ref = weight_fp32 @ activation_fp32.T
nmse = torch.mean((output - output_ref)**2) / torch.mean(output_ref**2)
print(f"NMSE: {nmse.item():.6e}")
```

---

## 4. 方案 B: pybind11 + CUDA（独立库）

### 4.1 适用场景

- 不依赖 PyTorch
- 使用 NumPy 进行测试
- 需要更细粒度的控制

### 4.2 实现架构

```cpp
// bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>

namespace py = pybind11;

py::array_t<float> gemm_q4_0_q8_1_numpy(
    py::array_t<uint8_t> weight_q,
    py::array_t<uint8_t> activation_q,
    int M, int N, int K
) {
    // 获取 NumPy 数组信息
    auto weight_buf = weight_q.request();
    auto act_buf = activation_q.request();

    // 分配 GPU 内存
    void *d_weight, *d_act;
    float *d_output;
    cudaMalloc(&d_weight, weight_buf.size);
    cudaMalloc(&d_act, act_buf.size);
    cudaMalloc(&d_output, M * N * sizeof(float));

    // 拷贝到 GPU
    cudaMemcpy(d_weight, weight_buf.ptr, weight_buf.size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_act, act_buf.ptr, act_buf.size,
               cudaMemcpyHostToDevice);

    // 调用 kernel
    gemm_q4_0_q8_1(
        reinterpret_cast<block_q4_0*>(d_weight),
        reinterpret_cast<block_q8_1*>(d_act),
        d_output, M, N, K
    );

    // 拷贝结果回 CPU
    auto result = py::array_t<float>({M, N});
    auto result_buf = result.request();
    cudaMemcpy(result_buf.ptr, d_output, M * N * sizeof(float),
               cudaMemcpyDeviceToHost);

    // 清理
    cudaFree(d_weight);
    cudaFree(d_act);
    cudaFree(d_output);

    return result;
}

PYBIND11_MODULE(quant_gemm_cuda, m) {
    m.def("gemm_q4_0_q8_1", &gemm_q4_0_q8_1_numpy,
          "Quantized GEMM using NumPy arrays");
}
```

### 4.3 Python 使用

```python
import numpy as np
import quant_gemm_cuda

# 准备数据
weight_q = np.zeros((M * (K // 32), 18), dtype=np.uint8)
activation_q = np.zeros((N * (K // 32), 36), dtype=np.uint8)

# 量化（需要单独实现）
# ...

# 调用
output = quant_gemm_cuda.gemm_q4_0_q8_1(weight_q, activation_q, M, N, K)
```

---

## 5. 方案 C: CuPy RawKernel（快速原型）

### 5.1 优势

- 无需编译，即时加载
- 适合快速验证

### 5.2 实现示例

```python
import cupy as cp

# 加载 CUDA 源码
with open('kernels/gemm/gemm_quant_formats.cuh', 'r') as f:
    kernel_code = f.read()

# 编译 kernel
module = cp.RawModule(code=kernel_code, options=('-std=c++17',))
gemm_kernel = module.get_function('gemm_quant_kernel')

# 调用
gemm_kernel(
    grid=(M // 16, N // 16),
    block=(16, 16),
    args=(d_weight, d_act, d_output, M, N, K)
)
```

### 5.3 局限性

- 需要修改 kernel 代码适配 CuPy
- 自定义类型处理复杂
- 不适合生产环境

---

## 6. 推荐实现路线图

### Phase 1: 基础封装（1-2 天）

1. 创建 `python/` 目录结构
2. 实现 `gemm_q4_0_q8_1` 的 PyTorch 封装
3. 实现基础的量化函数

### Phase 2: 完整接口（2-3 天）

1. 封装所有量化格式 (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0)
2. 添加 CPU 参考实现
3. 编写测试用例

### Phase 3: 测试框架（1-2 天）

1. 创建 pytest 测试套件
2. 实现正确性验证
3. 实现性能基准测试

---

## 7. Python 测试框架示例

### 7.1 项目结构

```
python/
├── quant_gemm/
│   ├── __init__.py
│   ├── ops.py
│   ├── reference.py      # CPU 参考实现
│   └── _C/
├── tests/
│   ├── conftest.py       # pytest 配置
│   ├── test_correctness.py
│   └── test_performance.py
├── setup.py
└── requirements.txt
```

### 7.2 测试代码示例

**tests/test_correctness.py**
```python
import pytest
import torch
import numpy as np
import quant_gemm
from quant_gemm.reference import cpu_gemm_q4_0_q8_1

class TestGemmQ4_0:
    """Q4_0 x Q8_1 GEMM 正确性测试"""

    @pytest.fixture
    def setup_data(self):
        """生成测试数据"""
        M, N, K = 4, 512, 1024
        torch.manual_seed(42)
        weight = torch.randn(M, K, device='cuda')
        activation = torch.randn(N, K, device='cuda')
        return M, N, K, weight, activation

    def test_small_matrix(self, setup_data):
        """测试小矩阵"""
        M, N, K, weight, activation = setup_data

        # 量化
        weight_q = quant_gemm.quantize_q4_0(weight)
        activation_q = quant_gemm.quantize_q8_1(activation)

        # GPU kernel
        output_gpu = quant_gemm.gemm_q4_0_q8_1(weight_q, activation_q, M, N, K)

        # CPU 参考
        output_cpu = cpu_gemm_q4_0_q8_1(
            weight_q.cpu().numpy(),
            activation_q.cpu().numpy(),
            M, N, K
        )
        output_cpu = torch.from_numpy(output_cpu).cuda()

        # 计算 NMSE
        nmse = torch.mean((output_gpu - output_cpu)**2) / torch.mean(output_cpu**2)
        assert nmse < 1e-6, f"NMSE {nmse:.6e} exceeds threshold"

    @pytest.mark.parametrize("M,N,K", [
        (1, 1, 32),        # 最小尺寸
        (4096, 2, 14336),  # LLM decode
        (4096, 128, 4096), # LLM prefill
    ])
    def test_various_sizes(self, M, N, K):
        """测试各种尺寸"""
        torch.manual_seed(42)
        weight = torch.randn(M, K, device='cuda')
        activation = torch.randn(N, K, device='cuda')

        weight_q = quant_gemm.quantize_q4_0(weight)
        activation_q = quant_gemm.quantize_q8_1(activation)

        # 应该不崩溃
        output = quant_gemm.gemm_q4_0_q8_1(weight_q, activation_q, M, N, K)
        assert output.shape == (M, N)
        assert not torch.isnan(output).any()


class TestQuantization:
    """量化函数测试"""

    def test_q4_0_roundtrip(self):
        """Q4_0 量化往返测试"""
        x = torch.randn(1024, device='cuda')
        x_q = quant_gemm.quantize_q4_0(x)
        x_dq = quant_gemm.dequantize_q4_0(x_q)

        # 量化误差应该有界
        max_error = (x - x_dq).abs().max()
        assert max_error < 0.5, f"Max error {max_error:.4f} too large"
```

**tests/test_performance.py**
```python
import pytest
import torch
import time
import quant_gemm

class TestPerformance:
    """性能基准测试"""

    @pytest.mark.benchmark
    @pytest.mark.parametrize("M,N,K,name", [
        (4096, 1, 14336, "decode_n1"),
        (4096, 2, 14336, "decode_n2"),
        (4096, 8, 14336, "decode_n8"),
        (4096, 128, 14336, "prefill_n128"),
    ])
    def test_gemm_performance(self, M, N, K, name):
        """测试 GEMM 性能"""
        # 准备数据
        weight = torch.randn(M, K, device='cuda')
        activation = torch.randn(N, K, device='cuda')
        weight_q = quant_gemm.quantize_q4_0(weight)
        activation_q = quant_gemm.quantize_q8_1(activation)

        # Warmup
        for _ in range(10):
            _ = quant_gemm.gemm_q4_0_q8_1(weight_q, activation_q, M, N, K)
        torch.cuda.synchronize()

        # Benchmark
        n_runs = 100
        start = time.perf_counter()
        for _ in range(n_runs):
            _ = quant_gemm.gemm_q4_0_q8_1(weight_q, activation_q, M, N, K)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        # 计算指标
        avg_time_us = elapsed / n_runs * 1e6
        flops = 2 * M * N * K
        gflops = flops / (elapsed / n_runs) / 1e9

        print(f"\n{name}: {avg_time_us:.2f} us, {gflops:.2f} GFLOPS")

        # 断言基本性能要求
        assert gflops > 10, f"Performance too low: {gflops:.2f} GFLOPS"
```

### 7.3 运行测试

```bash
# 安装依赖
pip install -e python/
pip install pytest pytest-benchmark

# 运行所有测试
pytest python/tests/ -v

# 只运行正确性测试
pytest python/tests/test_correctness.py -v

# 运行性能测试
pytest python/tests/test_performance.py -v --benchmark
```

---

## 8. 与 llama.cpp 对比测试

```python
import subprocess
import torch
import quant_gemm

def compare_with_llama_cpp(M, N, K, quant_type='q4_0'):
    """与 llama.cpp 对比"""

    # 运行我们的实现
    weight = torch.randn(M, K, device='cuda')
    activation = torch.randn(N, K, device='cuda')
    weight_q = quant_gemm.quantize_q4_0(weight)
    activation_q = quant_gemm.quantize_q8_1(activation)

    # Benchmark 我们的实现
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(100):
        _ = quant_gemm.gemm_q4_0_q8_1(weight_q, activation_q, M, N, K)
    end.record()
    torch.cuda.synchronize()

    our_time_ms = start.elapsed_time(end) / 100

    # 运行 llama.cpp (通过命令行)
    cmd = f"./test-backend-ops perf -o MUL_MAT -b CUDA0 " \
          f"-p 'type_a={quant_type}.*m={M}.*n={N}.*k={K}'"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    # 解析 llama.cpp 输出...

    print(f"Our implementation: {our_time_ms:.3f} ms")
    # print(f"llama.cpp: {llama_time_ms:.3f} ms")
```

---

## 9. 总结

### 推荐方案

**使用 PyTorch C++ Extension**，原因：

1. 与现有 PyTorch 生态无缝集成
2. 自动处理 GPU 内存管理
3. 易于编写测试用例
4. 社区支持好，文档完善

### 下一步行动

1. 在 `python/` 目录创建基础结构
2. 先封装一个 kernel (如 `gemm_q4_0_q8_1`)
3. 编写测试验证正确性
4. 逐步扩展到其他 kernel

---

**最后更新**: 2025-01-29
