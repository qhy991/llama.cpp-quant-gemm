# Solution Schema

## 概述

`Solution` 描述了针对特定 `Definition` 的具体实现方案。

## JSON Schema 结构

### 顶层字段

| 字段 | 类型 | 必需 | 描述 |
|------|------|------|------|
| `name` | string | 是 | 方案名称: `{definition}_{language}_{optimization}` |
| `description` | string | 是 | 实现方案描述 |
| `definition` | string | 是 | 引用的 Definition name |
| `author` | string | 是 | 作者/来源 |
| `tags` | array[string] | 否 | 标签 |
| `spec` | object | 是 | 实现规范 |
| `performance` | object | 否 | 性能指标 |
| `optimizations` | array[string] | 否 | 优化技术列表 |
| `sources` | array[SourceFile] | 是 | 源代码文件 |

### spec 实现规范

```json
"spec": {
  "language": "cuda",
  "target_hardware": ["sm_61", "sm_70", "sm_75", "sm_80"],
  "entry_point": "include/gemm_cuda_naive.cuh::gemm_q4_0_q8_1_naive",
  "dependencies": ["cuda>=11.0", "curand"],
  "destination_passing_style": true,
  "min_compute_capability": "6.1"
}
```

| 字段 | 类型 | 描述 |
|------|------|------|
| `language` | string | 实现语言: `cuda`, `cpp`, `python`, `triton` |
| `target_hardware` | array[string] | 支持的硬件架构 |
| `entry_point` | string | 入口点: `{file_path}::{function_name}` |
| `dependencies` | array[string] | 依赖项 |
| `destination_passing_style` | bool | 是否使用目标传递风格 |
| `min_compute_capability` | string | 最低 GPU 计算能力 |

### performance 性能指标

```json
"performance": {
  "estimated_tflops": 1.2,
  "speedup_vs_naive": "8x",
  "memory_bound": false,
  "optimization_level": 3
}
```

### sources 源文件

```json
"sources": [
  {
    "path": "include/gemm_cuda_dp4a.cuh",
    "description": "DP4A optimized CUDA kernel"
  }
]
```

## 优化级别

本项目定义了 6 个优化级别:

| Level | 名称 | 技术 | 预期提升 |
|-------|------|------|----------|
| 1 | Naive | 基础实现 | 基线 |
| 2 | Tiled | 共享内存 | 2-3x |
| 3 | DP4A | SIMD 指令 | 8x |
| 4 | Vectorized DP4A | 向量化 | 1.5x |
| 5 | Tiled + DP4A | 组合优化 | 15-20x |
| 6 | MMQ | llama.cpp 级别 | 目标 |

## 示例

### Naive 实现

```json
{
  "name": "gemm_q4_0_q8_1_cuda_naive",
  "description": "Naive CUDA implementation",
  "definition": "gemm_q4_0_q8_1_w4a8",
  "author": "quant-gemm-from-scratch",
  "spec": {
    "language": "cuda",
    "target_hardware": ["sm_61", "sm_70", "sm_75", "sm_80"],
    "entry_point": "include/gemm_cuda_naive.cuh::gemm_q4_0_q8_1_naive"
  },
  "performance": {
    "estimated_tflops": 0.15,
    "optimization_level": 1
  }
}
```

### DP4A 优化实现

```json
{
  "name": "gemm_q4_0_q8_1_cuda_dp4a",
  "description": "DP4A optimized implementation",
  "definition": "gemm_q4_0_q8_1_w4a8",
  "author": "quant-gemm-from-scratch",
  "spec": {
    "language": "cuda",
    "target_hardware": ["sm_61", "sm_70", "sm_75", "sm_80"],
    "entry_point": "include/gemm_cuda_dp4a.cuh::gemm_q4_0_q8_1_dp4a",
    "min_compute_capability": "6.1"
  },
  "performance": {
    "estimated_tflops": 1.2,
    "speedup_vs_naive": "8x",
    "optimization_level": 3
  },
  "optimizations": [
    "__dp4a intrinsic: 4 multiply-add ops in single instruction",
    "INT8/INT32 compute path"
  ]
}
```
