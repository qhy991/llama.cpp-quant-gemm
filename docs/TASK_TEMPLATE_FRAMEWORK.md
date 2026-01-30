# 最小llama.cpp接口GEMM算子实现任务模板框架

本文档提供了实现最小llama.cpp兼容GEMM算子的完整任务描述模板，以及通用的任务框架设计，支持多种量化类型和配置组合。

---

## 目录

1. [最小llama.cpp兼容GEMM算子的文件结构](#最小llamacpp兼容gemm算子的文件结构)
2. [各文件详细职责](#各文件详细职责)
3. [数据流说明](#数据流说明)
4. [给LLM的任务描述模板](#给llm的任务描述模板)
5. [通用任务模板框架](#通用任务模板框架)

---

## 最小llama.cpp兼容GEMM算子的文件结构

### 核心文件清单

至少需要 **5个核心文件**：

| 文件 | 职责 | 必要性 |
|------|------|--------|
| `ggml_types.h` | 量化数据类型定义 | 必须 - 接口基础 |
| `quantize.h` | 量化/反量化函数 | 必须 - 数据转换 |
| `gemm_kernel.cuh` | CUDA GEMM kernel实现 | 必须 - 核心计算 |
| `gemm_api.h` | 对外API接口 | 必须 - 暴露功能 |
| `test_gemm.cu` | 验证测试 | 必须 - 确保正确性 |

---

## 各文件详细职责

### 1. ggml_types.h - 量化数据类型定义

**职责：**
- 定义与llama.cpp完全兼容的量化块结构
- `block_q4_0`: 4-bit权重 (18 bytes: 2B scale + 16B data)
- `block_q8_1`: 8-bit激活 (36 bytes: 4B ds + 32B data)
- 定义常量: `QK4_0=32`, `QK8_1=32`
- 提供辅助函数: 4-bit提取、打包

**关键结构：**
```cpp
struct block_q4_0 {
    half d;              // 缩放因子
    uint8_t qs[16];      // 4-bit量化数据 (32个元素，每个4-bit)
};

struct block_q8_1 {
    half2 ds;            // d (缩放) + s (求和)
    int8_t qs[32];       // 8-bit量化数据
};
```

### 2. quantize.h - 量化/反量化函数

**职责：**
- `quantize_row_q4_0()`: FP32 → Q4_0
- `quantize_row_q8_1()`: FP32 → Q8_1 (带求和)
- `dequantize_row_q4_0()`: Q4_0 → FP32
- `dequantize_row_q8_1()`: Q8_1 → FP32
- CPU参考实现，用于验证GPU结果

**量化公式：**
- **Q4_0**: `q = clamp(round(x / d) + 8, 0, 15)`, `d = max(|x|) / 7.0f`
- **Q8_1**: `q = round(x / d)`, `d = max(|x|) / 127.0f`, `s = Σx[i]`

### 3. gemm_kernel.cuh - CUDA GEMM Kernel

**职责：**
- 实现 `vec_dot_q4_0_q8_1()` 向量点积
- 实现 `gemm_w4a8_kernel()` GEMM核心
- 核心公式: `result = d_w * (d_a * sumi - 8 * s_a)`
- 支持GEMM约定: `C[M,N] = A[M,K] × B[N,K]^T`

**关键点积公式：**
```cpp
// Q4_0使用偏移量化: q_stored = q_real + 8
// 因此点积需要补偿:
result = d_w * (d_a * Σ(q_w * q_a) - 8 * s_a)
// 其中 s_a = Q8_1块中预计算的原始值求和
```

### 4. gemm_api.h - 对外API接口

**职责：**
- 封装kernel launch配置
- 提供易用的C/C++接口
- 处理维度验证和错误检查
- 可选: 支持ggml_tensor输入

**接口示例：**
```cpp
void gemm_w4a8(
    const block_q8_1* activation,   // Q8_1格式激活
    const block_q4_0* weight,       // Q4_0格式权重
    float* output,                  // FP32输出
    int M, int N, int K             // 维度
);
```

### 5. test_gemm.cu - 验证测试

**职责：**
- 验证量化/反量化正确性
- 验证GPU与CPU参考实现一致性
- 验证与llama.cpp原始实现兼容
- 不同规模的性能测试

**测试用例：**
- 测试1: 量化误差验证 (Q4_0误差<5%, Q8_1误差<1%)
- 测试2: 反量化正确性
- 测试3: GPU GEMM vs CPU参考实现 (相对误差<0.1%)
- 测试用例: M=64, N=128, K=256

---

## 数据流说明

### 关键理解：Weight vs Activation

**Weight（权重）—— 已经预量化**
- ✓ 传入时已经是 Q4_0 格式
- ✓ 不需要运行时量化
- ✓ 模型文件（GGUF）存储的就是量化后的权重

**Activation（激活）—— 取决于GEMM类型**

| GEMM类型 | 激活输入格式 | 是否需要运行时量化 |
|----------|-------------|------------------|
| W4A16    | FP32浮点     | 不需要，直接用     |
| W4A8     | FP32浮点     | 需要，量化为Q8_1   |

### 数据流示意图

**W4A16 (更简单)**
```
Weight:     GGUF文件 → Q4_0 (已量化) ─┐
                                    ├→ GEMM kernel → FP32输出
Activation: 上一层输出 → FP32 ───────┘
```

**W4A8 (需要激活量化)**
```
Weight:     GGUF文件 → Q4_0 (已量化) ─────────┐
                                            ├→ GEMM kernel → FP32输出
Activation: 上一层输出 → FP32 → 量化为Q8_1 ───┘
                        ↑
                  运行时执行
```

### 方案选择

#### 方案A: W4A16（入门推荐）
- 文件更少，逻辑更简单
- 不需要实现Q8_1量化
- 适合理解基本原理
- **需要4个文件**（不需要quantize.h）

#### 方案B: W4A8（完整实现）
- 需要实现激活量化（FP32→Q8_1）
- 性能更优（整数计算）
- 可使用DP4A指令加速
- **需要5个文件**（包含quantize.h）

### 接口设计

```cpp
// W4A16: 激活是FP32
void gemm_w4a16(
    const float* activation,        // FP32，直接使用
    const block_q4_0* weight,       // Q4_0，预量化好的
    float* output,
    int M, int N, int K
);

// W4A8: 激活需要先量化
void gemm_w4a8(
    const block_q8_1* activation,   // Q8_1，需要运行时量化
    const block_q4_0* weight,       // Q4_0，预量化好的
    float* output,
    int M, int N, int K
);

// 或者提供端到端接口（内部处理量化）
void gemm_w4a8_e2e(
    const float* activation_fp32,   // FP32输入
    const block_q4_0* weight,       // Q4_0权重
    float* output,
    int M, int N, int K
);  // 内部: FP32→Q8_1→GEMM
```

---

## 给LLM的任务描述模板

### 任务描述模板

```markdown
# 任务：实现最小llama.cpp兼容W4A8量化GEMM算子

## 目标
创建一个最小但完整的CUDA GEMM算子，兼容llama.cpp的量化格式。

## 核心约束

1. **数据类型必须完全兼容llama.cpp**:
   - block_q4_0: 18字节 (half d + uint8_t qs[16])
   - block_q8_1: 36字节 (half2 ds + int8_t qs[32])
   - 每块处理32个元素 (QK=32)

2. **GEMM计算约定**:
   - C[M, N] = A[M, K] × B[N, K]^T
   - A = 激活矩阵 (Q8_1格式)
   - B = 权重矩阵 (Q4_0格式)
   - C = 输出矩阵 (FP32)

3. **W4A8点积补偿公式** (关键！):
   Q4_0使用偏移量化: q_stored = q_real + 8
   因此点积需要补偿:
   result = d_w * (d_a * Σ(q_w * q_a) - 8 * s_a)
   其中 s_a = Q8_1块中预计算的原始值求和

## 需要创建的文件

### 文件1: ggml_types.h
实现:
- block_q4_0结构体 (18字节)
- block_q8_1结构体 (36字节)
- QK4_0, QK8_1常量 (=32)
- get_q4_0_low/high() 辅助函数
- get_q8_1_d/s() 获取缩放因子和求和

### 文件2: quantize.h
实现:
- quantize_row_q4_0_ref(): FP32→Q4_0
  - d = max(|x|) / 7.0f
  - q = round(x / d) + 8, 范围[0,15]
  
- quantize_row_q8_1_ref(): FP32→Q8_1
  - d = max(|x|) / 127.0f
  - s = Σx[i] (原始值求和，存入ds.y)
  - q = round(x / d), 范围[-127,127]
  
- 对应的反量化函数

### 文件3: gemm_kernel.cuh
实现:
- vec_dot_q4_0_q8_1(): 单块点积
- gemm_w4a8_naive_kernel(): 朴素GEMM kernel
- 每个线程计算C的一个元素
- 正确处理K维度分块 (K/32个块)

### 文件4: gemm_api.h
实现:
- gemm_w4a8(): 封装kernel launch
- 输入验证 (K必须是32的倍数)
- CUDA错误检查

### 文件5: test_gemm.cu
实现:
- 测试1: 量化误差验证 (Q4_0误差<5%, Q8_1误差<1%)
- 测试2: 反量化正确性
- 测试3: GPU GEMM vs CPU参考实现 (相对误差<0.1%)
- 测试用例: M=64, N=128, K=256

## 验收标准
1. 编译无错误无警告
2. 所有测试通过
3. 与CPU参考实现的NMSE < 0.001
4. 能处理任意M, N, K (K必须是32倍数)

## 参考公式汇总

Q4_0量化: q = clamp(round(x / d) + 8, 0, 15), d = max|x| / 7
Q8_1量化: q = round(x / d), d = max|x| / 127, s = Σx
点积: result = d_w * (d_a * sumi - 8 * s_a)
```

### 更简洁的描述版本（适合快速任务）

```markdown
# 实现llama.cpp兼容的W4A8 GEMM算子

需要5个文件:
1. ggml_types.h - 定义block_q4_0(18B)和block_q8_1(36B)
2. quantize.h - FP32↔量化格式转换
3. gemm_kernel.cuh - CUDA kernel，核心公式: result = d_w * (d_a * sumi - 8 * s_a)
4. gemm_api.h - C++接口封装
5. test_gemm.cu - 验证正确性

关键约束:
- block_q4_0: half d + uint8_t[16], 每块32元素
- block_q8_1: half2 ds (d+sum) + int8_t[32]
- Q4_0用偏移量化(q+8)，点积需补偿
- GEMM: C[M,N] = A_q8[M,K] × B_q4[N,K]^T
```

### 最小系统的关键点

1. **数据结构完全兼容** - 字节数必须精确匹配
2. **补偿公式正确** - Q4_0的偏移必须在点积时补偿
3. **CPU参考实现** - 用于验证GPU结果的正确性

---

## 通用任务模板框架

### 设计目标

设计一个通用的任务模板框架，能够支持多种量化类型组合（Q4_0、Q4_1、Q5_0、Q5_1、Q8_0、Q8_1等）和不同的优化级别（naive、tiled、dp4a、tensorcore）。

### 当前模板的局限性

| 局限性 | 问题 |
|--------|------|
| 硬编码类型 | 只描述了 Q4_0 × Q8_1 |
| 单一配置 | MNK 写死在描述中 |
| 公式绑定 | 补偿公式只适用于特定组合 |
| 无扩展性 | 新增类型需要重写模板 |

### 变化维度分析

可参数化的维度：

- **权重类型**: Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 | Q2_K...Q6_K
- **激活类型**: FP32 | FP16 | Q8_1 | Q8_0
- **输出类型**: FP32 | FP16
- **维度配置**: M, N, K (任意32的倍数)
- **优化级别**: naive | tiled | dp4a | tensorcore
- **目标架构**: sm_61 | sm_70 | sm_80 | sm_89 | sm_100

### 通用任务模板框架 V2

#### 1. 配置参数（填写此部分）

```yaml
task_config:
  # 量化类型配置
  weight_type: "{{WEIGHT_TYPE}}"      # Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0
  activation_type: "{{ACT_TYPE}}"     # FP32 | FP16 | Q8_1 | Q8_0
  output_type: "{{OUT_TYPE}}"         # FP32 | FP16
  
  # 维度约束
  block_size: {{BLOCK_SIZE}}          # 通常32，K-quant为256
  M_range: [1, 8192]                  # batch维度范围
  N_range: [1, 16384]                 # 输出特征维度
  K_range: [32, 16384]                # 输入特征维度
  K_alignment: {{BLOCK_SIZE}}         # K必须是此值的倍数
  
  # 实现级别
  optimization_level: "{{OPT_LEVEL}}" # naive | tiled | dp4a | tensorcore
  target_arch: "{{ARCH}}"             # sm_61 | sm_70 | sm_80 | sm_89
```

#### 2. 类型规格查找表

##### 2.1 数据结构规格

| 类型 | 块大小 | 字节数 | 结构 | 量化方式 |
|------|--------|--------|------|----------|
| Q4_0 | 32 | 18 | half d + uint8[16] | 对称,偏移+8 |
| Q4_1 | 32 | 20 | half d + half m + uint8[16] | 非对称 |
| Q5_0 | 32 | 22 | half d + uint8[4] + uint8[16] | 对称,5-bit |
| Q5_1 | 32 | 24 | half d + half m + uint8[4] + uint8[16] | 非对称 |
| Q8_0 | 32 | 34 | half d + int8[32] | 对称 |
| Q8_1 | 32 | 36 | half2 ds + int8[32] | 对称,带求和 |
| FP32 | 1 | 4 | float | 无量化 |
| FP16 | 1 | 2 | half | 无量化 |

##### 2.2 点积公式查找表

| 组合 | 公式 | 补偿项 |
|------|------|--------|
| Q4_0 × Q8_1 | `d_w * (d_a * sumi - 8 * s_a)` | 需要（偏移补偿） |
| Q4_1 × Q8_1 | `d_w * d_a * sumi + m_w * s_a` | 需要（最小值补偿） |
| Q5_0 × Q8_1 | `d_w * (d_a * sumi - 16 * s_a)` | 需要（偏移补偿） |
| Q5_1 × Q8_1 | `d_w * d_a * sumi + m_w * s_a` | 需要（最小值补偿） |
| Q8_0 × Q8_1 | `d_w * d_a * sumi` | 不需要 |
| Q4_0 × FP32 | `Σ (q_w - 8) * d_w * a_fp32` | 在线反量化 |
| FP32 × FP32 | `Σ a * b` | 无 |

#### 3. 文件生成规则

根据配置自动确定需要生成的文件：

##### 3.1 必需文件

1. **types_{{WEIGHT_TYPE}}_{{ACT_TYPE}}.h**
   - 定义 `block_{{WEIGHT_TYPE}}` 结构
   - 如果 ACT_TYPE 是量化类型，定义 `block_{{ACT_TYPE}}`
   - 常量: `QK_{{WEIGHT_TYPE}}`, sizeof验证

2. **quantize_{{ACT_TYPE}}.h** (仅当ACT_TYPE是量化类型时)
   - `quantize_row_{{ACT_TYPE}}()`: FP32 → {{ACT_TYPE}}
   - `dequantize_row_{{ACT_TYPE}}()`: {{ACT_TYPE}} → FP32

3. **gemm_{{WEIGHT_TYPE}}_{{ACT_TYPE}}_{{OPT_LEVEL}}.cuh**
   - `vec_dot_{{WEIGHT_TYPE}}_{{ACT_TYPE}}()`: 块点积
   - `gemm_kernel()`: CUDA kernel
   - 使用查找表中的正确公式

4. **gemm_api.h**
   - 统一接口封装
   - 参数验证

5. **test_gemm_{{WEIGHT_TYPE}}_{{ACT_TYPE}}.cu**
   - 正确性验证
   - 性能测试

##### 3.2 条件文件

- 如果 `ACT_TYPE == FP32|FP16`: 不需要激活量化文件
- 如果 `OPT_LEVEL == dp4a`: 需要DP4A辅助函数
- 如果 `OPT_LEVEL == tensorcore`: 需要WMMA接口

#### 4. 实现模板

##### 4.1 点积函数模板

```cpp
// 自动生成: vec_dot_{{WEIGHT_TYPE}}_{{ACT_TYPE}}
__device__ __forceinline__ float
vec_dot_{{WEIGHT_TYPE}}_{{ACT_TYPE}}(
    const block_{{WEIGHT_TYPE}}* w,
    const {{ACT_INPUT_TYPE}}* a  // block_{{ACT_TYPE}}* 或 float*
) {
    {{#if NEEDS_COMPENSATION}}
    // 使用查找表中的补偿公式
    float d_w = __half2float(w->d);
    {{#if ACT_TYPE == "Q8_1"}}
    float d_a = __half2float(__low2half(a->ds));
    float s_a = __half2float(__high2half(a->ds));
    {{/if}}
    
    int32_t sumi = 0;
    // ... 点积计算 ...
    
    return {{DOT_FORMULA}};  // 从查找表获取
    {{/else}}
    // 简单点积，无补偿
    {{/if}}
}
```

##### 4.2 GEMM Kernel模板

```cpp
__global__ void
gemm_{{WEIGHT_TYPE}}_{{ACT_TYPE}}_{{OPT_LEVEL}}_kernel(
    const block_{{WEIGHT_TYPE}}* __restrict__ weight,
    const {{ACT_INPUT_TYPE}}* __restrict__ activation,
    float* __restrict__ output,
    int M, int N, int K
) {
    // 线程索引计算（根据OPT_LEVEL变化）
    {{#if OPT_LEVEL == "naive"}}
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    {{/if}}
    {{#if OPT_LEVEL == "tiled"}}
    // 共享内存 tiling 逻辑
    {{/if}}
    
    // 点积计算
    const int nb = K / {{BLOCK_SIZE}};
    float sum = 0.0f;
    for (int b = 0; b < nb; b++) {
        sum += vec_dot_{{WEIGHT_TYPE}}_{{ACT_TYPE}}(
            &weight[col * nb + b],
            &activation[row * nb + b]
        );
    }
    output[row * N + col] = sum;
}
```

#### 5. 验收标准模板

```yaml
acceptance_criteria:
  correctness:
    - "GPU结果与CPU参考实现的相对误差 < {{ERROR_THRESHOLD}}"
    - "支持 K ∈ [{{K_MIN}}, {{K_MAX}}]，步长 {{BLOCK_SIZE}}"
    - "支持任意 M, N ∈ [1, {{MAX_DIM}}]"
  
  compatibility:
    - "block_{{WEIGHT_TYPE}} 大小 == {{WEIGHT_BYTES}} bytes"
    - "{{#if ACT_QUANTIZED}}block_{{ACT_TYPE}} 大小 == {{ACT_BYTES}} bytes{{/if}}"
    - "数据布局与llama.cpp完全一致"
  
  performance:  # 可选
    - "{{#if OPT_LEVEL != 'naive'}}性能 >= naive版本 {{SPEEDUP_TARGET}}x{{/if}}"
```

#### 6. 示例实例化

##### 示例1: W4A8 Naive

```yaml
weight_type: Q4_0
activation_type: Q8_1
output_type: FP32
block_size: 32
optimization_level: naive
```

自动推导:
- `dot_formula`: `"d_w * (d_a * sumi - 8 * s_a)"`
- `needs_activation_quant`: `true`
- `weight_bytes`: `18`
- `act_bytes`: `36`

##### 示例2: W5A16 Tiled

```yaml
weight_type: Q5_0
activation_type: FP32
output_type: FP32
block_size: 32
optimization_level: tiled
```

自动推导:
- `dot_formula`: `"Σ (q_w - 16) * d_w * a_fp32"`
- `needs_activation_quant`: `false`

##### 示例3: W8A8 DP4A

```yaml
weight_type: Q8_0
activation_type: Q8_1
output_type: FP32
block_size: 32
optimization_level: dp4a
```

自动推导:
- `dot_formula`: `"d_w * d_a * sumi"`
- `needs_compensation`: `false` (对称×对称)

### 模板通用化的核心思想

#### 1. 分离"规格"与"实现"

```
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│   配置参数       │ ───► │   规格查找表     │ ───► │   代码生成       │
│   (用户填写)     │      │   (预定义知识)   │      │   (自动推导)     │
└──────────────────┘      └──────────────────┘      └──────────────────┘
     weight_type              补偿公式                  vec_dot函数
     act_type                 结构体大小                kernel代码
     opt_level                量化公式                  测试用例
```

#### 2. 查找表驱动

将所有"领域知识"编码为查找表：
- 点积公式表
- 数据结构表
- 量化/反量化公式表
- 优化策略表

#### 3. 条件生成

根据配置自动决定：
- 是否需要激活量化文件
- 是否需要补偿项
- 使用哪种优化策略

### 简化版通用模板（给LLM用）

```markdown
# 通用量化GEMM实现任务

## 输入配置
- 权重类型: {{W_TYPE}}
- 激活类型: {{A_TYPE}}
- 优化级别: {{OPT}}

## 查找信息
1. 从类型表查: {{W_TYPE}}的结构、大小、量化公式
2. 从类型表查: {{A_TYPE}}的结构、大小、量化公式
3. 从点积表查: {{W_TYPE}} × {{A_TYPE}} 的计算公式和补偿项

## 生成文件
1. types.h - 定义所需的block结构
2. quantize.h - 如果{{A_TYPE}}需要运行时量化
3. gemm_{{W_TYPE}}_{{A_TYPE}}.cuh - kernel实现，使用查到的公式
4. api.h - 接口封装
5. test.cu - 测试

## 关键公式
点积: {{DOT_FORMULA}}  // 从查找表获取
```

这样，同一个模板框架可以支持 Q4_0×Q8_1、Q5_0×FP32、Q8_0×Q8_1 等任意组合，只需填写不同的配置参数即可。

---

## 总结

本文档提供了：

1. **最小系统文件结构** - 5个核心文件的职责和实现要点
2. **数据流说明** - 权重和激活的不同处理方式
3. **任务描述模板** - 可直接用于LLM的详细任务描述
4. **通用框架设计** - 支持多种量化类型组合的参数化模板

关键要点：
- 权重不需要运行时量化（已预量化）
- 激活是否需要量化取决于GEMM类型（W4A16 vs W4A8）
- 补偿公式是正确实现的关键
- 通用框架通过查找表驱动，实现配置参数化

---

## 参考

- [llama.cpp量化格式文档](https://github.com/ggerganov/llama.cpp)
- [项目架构文档](../PROJECT_ARCHITECTURE.md)
- [接口兼容性文档](../INTERFACE_COMPATIBILITY.md)
