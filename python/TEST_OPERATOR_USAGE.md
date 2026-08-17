# Operator Testing Tool

简单的命令行工具，用于测试符合 JSON schema 的算子。

## 基本用法

```bash
python test_operator.py <name> <folder_path>
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `name` | 算子名称（来自 spec.json） | 必需 |
| `folder` | 包含 spec.json 的文件夹路径 | 必需 |
| `--module, -m` | Pybind 模块路径 | `quant_gemm._C` |
| `--benchmark, -b` | 运行性能测试 | False |
| `--warmup` | Benchmark 预热迭代次数 | 10 |
| `--iterations, -i` | Benchmark 测试迭代次数 | 100 |
| `--config, -c` | 自定义配置（可多次使用） | 使用 spec 中的配置 |
| `--device, -d` | 设备 | `cuda` |

## 使用示例

### 1. 基本正确性测试

```bash
python test_operator.py w4a8_q4_0_q8_1 operators/quant_gemm/variants/w4a8_q4_0_q8_1
```

输出：
```
============================================================
 Testing: w4a8_q4_0_q8_1
============================================================
Folder: operators/quant_gemm/variants/w4a8_q4_0_q8_1
Module: quant_gemm._C
Device: cuda

Configs: 4

------------------------------------------------------------
 Correctness Tests
------------------------------------------------------------
[PASS] single: nmse=9.4378e-03 (threshold=0.1)
[PASS] small_batch: nmse=9.5081e-03 (threshold=0.1)
[PASS] medium_batch: nmse=9.6819e-03 (threshold=0.1)
[PASS] large_batch: nmse=9.6579e-03 (threshold=0.1)

Results: 4 passed, 0 failed
============================================================
```

### 2. 运行性能测试

```bash
python test_operator.py w4a8_q4_0_q8_1 operators/quant_gemm/variants/w4a8_q4_0_q8_1 --benchmark
```

输出：
```
------------------------------------------------------------
 Benchmarks
------------------------------------------------------------
single               M=    1 N= 4096 K= 4096 |    0.092 ms |   366.29 GFLOPS
small_batch          M=    4 N= 4096 K= 4096 |    0.182 ms |   737.76 GFLOPS
medium_batch         M=  128 N= 4096 K= 4096 |    3.588 ms |  1196.88 GFLOPS
large_batch          M= 4096 N= 4096 K= 4096 |  113.823 ms |  1207.48 GFLOPS
```

### 3. 自定义测试配置

```bash
# 单个自定义配置
python test_operator.py w4a8_q4_0_q8_1 operators/quant_gemm/variants/w4a8_q4_0_q8_1 \
    --config "M=1,N=2048,K=2048"

# 多个自定义配置
python test_operator.py w4a8_q4_0_q8_1 operators/quant_gemm/variants/w4a8_q4_0_q8_1 \
    --config "M=1,N=2048,K=2048" \
    --config "M=16,N=4096,K=4096" \
    --benchmark
```

### 4. 指定不同的 pybind 模块

```bash
python test_operator.py my_operator operators/my_op/variants/v1 \
    --module my_module._C
```

### 5. 调整 benchmark 参数

```bash
python test_operator.py w4a8_q4_0_q8_1 operators/quant_gemm/variants/w4a8_q4_0_q8_1 \
    --benchmark \
    --warmup 20 \
    --iterations 200
```

## 文件夹结构要求

算子文件夹必须包含 `spec.json`：

```
operators/quant_gemm/variants/w4a8_q4_0_q8_1/
├── spec.json          # 必需：算子规格定义
└── reference.py       # 可选：Python 参考实现（工具使用 FP32 matmul）
```

## spec.json 格式

```json
{
  "name": "w4a8_q4_0_q8_1",
  "family": "quant_gemm",
  "inputs": {
    "weight": {
      "dtype": "block_q4_0",
      "shape": ["N", "K/32", 18],
      "quantizer": "quantize_q4_0"
    },
    "activation": {
      "dtype": "block_q8_1",
      "shape": ["M", "K/32", 36],
      "quantizer": "quantize_q8_1"
    }
  },
  "outputs": {
    "output": {
      "dtype": "float32",
      "shape": ["M", "N"]
    }
  },
  "params": {
    "M": {"type": "int", "description": "Batch size"},
    "N": {"type": "int", "default": 4096},
    "K": {"type": "int", "default": 4096}
  },
  "kernel": {
    "entry_point": "gemm_q4_0_q8_1"
  },
  "test_configs": [
    {"name": "single", "M": 1, "N": 4096, "K": 4096},
    {"name": "large_batch", "M": 4096, "N": 4096, "K": 4096}
  ],
  "accuracy": {
    "metric": "nmse",
    "threshold": 0.1
  }
}
```

## 工作原理

1. **加载 spec.json**：读取算子定义
2. **导入 pybind 模块**：加载 kernel 和 quantizer 函数
3. **生成测试输入**：根据 spec 生成随机 FP32 张量
4. **量化输入**：使用指定的 quantizer 量化输入
5. **运行 kernel**：调用 CUDA kernel
6. **运行 reference**：使用 FP32 matmul 作为参考
7. **计算误差**：计算 NMSE (Normalized Mean Squared Error)
8. **判断通过**：与 threshold 比较

## 注意事项

- 工具使用 **FP32 matmul** 作为参考实现（快速）
- 不使用 spec 中定义的 Python reference（太慢）
- Kernel 必须在 pybind 模块中注册
- Quantizer 函数必须可用
- K 维度必须是 32 的倍数（对于 block 量化）

## 错误排查

### 错误：`ImportError: No module named 'quant_gemm._C'`

解决：重新编译 pybind 模块
```bash
python setup.py build_ext --inplace
```

### 错误：`Kernel not found`

检查：
1. Kernel 名称是否正确（spec.json 中的 `entry_point`）
2. Kernel 是否在 pybind 模块中注册
3. 尝试常见命名模式：`gemm_<name>`, `<name>`

### 错误：`Quantizer not found for block_q4_0`

检查：
1. Quantizer 函数是否在 pybind 模块中
2. 函数名称：`quantize_q4_0`, `quantize_q8_1` 等

### 测试失败：NMSE 过高

可能原因：
1. Kernel 实现有误
2. 量化精度损失过大
3. Threshold 设置过严格

调试：
```bash
# 使用小配置测试
python test_operator.py <name> <folder> --config "M=1,N=32,K=128"
```
