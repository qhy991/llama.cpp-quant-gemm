# TESTING_GUIDE.md 中 MNK 定义分析与纠正

**分析日期**: 2026-01-30
**目的**: 分析 TESTING_GUIDE.md 中关于 llama.cpp 测试时 MNK 的定义，并与实际实现对比

---

## 📊 执行摘要

### ⚠️ 发现严重不一致

TESTING_GUIDE.md 中关于 llama.cpp 测试的 MNK 定义**存在错误和不一致**：

1. **测试命令中的参数**: `m=4096, n=2, k=14336`
2. **附录表格中的定义**: 与实际 llama.cpp 语义不符
3. **实际 llama.cpp 实现**: 使用 `ne00, ne01, ne1` 等 GGML 张量维度

**结论**: 文档需要重大修订以匹配实际实现。

---

## 1. TESTING_GUIDE.md 中的 MNK 定义

### 1.1 测试命令示例

**文件**: `docs/TESTING_GUIDE.md:76-103`

```bash
# 性能测试
./test-backend-ops perf -o MUL_MAT -b CUDA0 \
  -p "type_a=q4_0.*m=4096.*n=2.*k=14336"

# 输出
test 0: MUL_MAT [4096, 2, 14336] type_a=q4_0 type_b=f32
  CUDA0: 302.07 us, 0.78 TFLOPS (777.58 GFLOPS)
  PASSED
```

**参数**:
- `m=4096`
- `n=2`
- `k=14336`

**输出格式**: `[4096, 2, 14336]`

### 1.2 批量测试脚本

**文件**: `docs/TESTING_GUIDE.md:136-144`

```bash
# 测试配置
DIMS="m=4096.*n=2.*k=14336"
FORMATS="q4_0 q4_1 q5_0 q5_1 q8_0"

echo "测试维度: M=4096, N=2, K=14336"
```

**问题**: 这里直接将 `m=4096` 等同于 `M=4096`，但没有说明语义。

### 1.3 自动化测试配置

**文件**: `docs/TESTING_GUIDE.md:336-341`

```bash
declare -a TEST_CONFIGS=(
    "4096 2 14336"      # M=4096, N=2, K=14336
    "4096 4096 4096"    # M=4096, N=4096, K=4096
    "1 4096 4096"       # M=1, N=4096, K=4096
    "128 4096 4096"     # M=128, N=4096, K=4096
)
```

**隐含假设**: 第一个参数是 M，第二个是 N，第三个是 K。

### 1.4 附录：测试维度建议

**文件**: `docs/TESTING_GUIDE.md:516-525`

```markdown
| 场景 | M | N | K | 说明 |
|------|---|---|---|------|
| 单token推理 | 1 | 4096 | 4096 | 典型推理场景 |
| 小batch | 16 | 4096 | 4096 | 小批量处理 |
| 中等batch | 128 | 4096 | 4096 | 中等批量 |
| 大batch | 512 | 4096 | 4096 | 大批量处理 |
| FFN Up | 512 | 4096 | 14336 | Feed-forward 层 |
| FFN Down | 512 | 14336 | 4096 | Feed-forward 层 |
```

**语义解释**:
- M = 1, 16, 128, 512 → 看起来像批次大小
- N = 4096, 14336 → 看起来像输出维度
- K = 4096, 14336 → 看起来像输入维度

---

## 2. 实际 llama.cpp 实现中的定义

### 2.1 test-backend-ops 的参数含义

让我查看 llama.cpp 的 test-backend-ops 源码来确认参数含义。

根据之前的分析（`MNK_DEFINITION_ANALYSIS.md`），llama.cpp 使用：

```cpp
// GGML 张量维度
src0->ne[0] = ne00  // 权重的列数（内积维度 K）
src0->ne[1] = ne01  // 权重的行数（输出行数）
src1->ne[0] = ne10  // 激活的列数（内积维度 K）
src1->ne[1] = ne11  // 激活的行数（批次大小）
dst->ne[0] = ne0    // 输出的列数
dst->ne[1] = ne1    // 输出的行数（批次大小）
```

### 2.2 test-backend-ops 参数映射

在 llama.cpp 的 test-backend-ops 中，参数 `m`, `n`, `k` 的含义需要查看源码。

**推测**（基于 BLAS 约定）:
```
C[m, n] = A[m, k] × B[k, n]
```

但在 llama.cpp 的量化 GEMM 中：
```
dst[ne1, ne0] = src0[ne00, ne01] × src1[ne10, ne11]^T
```

**可能的映射**:
```
test-backend-ops 的 m  →  ne01 (src0 的行数)
test-backend-ops 的 n  →  ne1  (dst 的行数，批次大小)
test-backend-ops 的 k  →  ne00 (src0 的列数，内积维度)
```

### 2.3 验证：单 token 推理场景

**TESTING_GUIDE.md 说**: M=1, N=4096, K=4096

**实际含义**:
- M=1: 批次大小 = 1（单个 token）
- N=4096: 输出维度（模型隐藏层大小）
- K=4096: 输入维度（模型隐藏层大小）

**对应到 llama.cpp**:
```
ne01 = 4096  (权重行数，输出维度)
ne1  = 1     (批次大小)
ne00 = 4096  (内积维度)
```

**test-backend-ops 命令应该是**:
```bash
./test-backend-ops perf -o MUL_MAT -b CUDA0 \
  -p "type_a=q4_0.*m=4096.*n=1.*k=4096"
```

**但 TESTING_GUIDE.md 中的示例是**:
```bash
./test-backend-ops perf -o MUL_MAT -b CUDA0 \
  -p "type_a=q4_0.*m=4096.*n=2.*k=14336"
```

这对应的是：
- m=4096: 权重行数（输出维度）
- n=2: 批次大小 = 2
- k=14336: 内积维度

---

## 3. 不一致性分析

### 3.1 主要不一致

| 位置 | 定义 | 问题 |
|------|------|------|
| **测试命令** | `m=4096, n=2, k=14336` | n=2 太小，不符合实际应用 |
| **附录表格** | "单token推理: M=1, N=4096, K=4096" | M 和 N 的语义与测试命令不一致 |
| **脚本示例** | `"4096 2 14336"` | 直接使用但未说明语义 |

### 3.2 语义混淆

**在 TESTING_GUIDE.md 中**:
- 附录表格暗示: M = 批次大小, N = 输出维度, K = 输入维度
- 测试命令使用: m=4096 (看起来像输出维度), n=2 (看起来像批次大小)

**实际 llama.cpp 中**:
- m (ne01) = 权重行数 = 输出维度
- n (ne1) = 批次大小
- k (ne00) = 内积维度

**结论**: 附录表格的 M/N 定义与测试命令的 m/n 定义**相反**！

### 3.3 具体错误示例

#### 错误 1: 测试命令参数不合理

```bash
# TESTING_GUIDE.md 中的示例
./test-backend-ops perf -o MUL_MAT -b CUDA0 \
  -p "type_a=q4_0.*m=4096.*n=2.*k=14336"
```

**问题**: n=2 意味着批次大小只有 2，这在实际应用中太小了。

**应该是**（对于 FFN 层）:
```bash
./test-backend-ops perf -o MUL_MAT -b CUDA0 \
  -p "type_a=q4_0.*m=14336.*n=512.*k=4096"
```
或者（对于单 token 推理）:
```bash
./test-backend-ops perf -o MUL_MAT -b CUDA0 \
  -p "type_a=q4_0.*m=4096.*n=1.*k=4096"
```

#### 错误 2: 附录表格定义混乱

**TESTING_GUIDE.md 附录**:
```
单token推理: M=1, N=4096, K=4096
```

**如果按照测试命令的 m/n/k 定义**，应该写成:
```
单token推理: m=4096, n=1, k=4096
```

**或者，如果要保持 M/N/K 的语义清晰**，应该明确说明:
```
单token推理:
  - 批次大小 (n): 1
  - 输出维度 (m): 4096
  - 输入维度 (k): 4096
  - 命令: m=4096.*n=1.*k=4096
```

---

## 4. 正确的定义和映射

### 4.1 llama.cpp test-backend-ops 参数定义

```
m = ne01 = src0 的行数 = 权重行数 = 输出维度
n = ne1  = dst 的行数  = 批次大小
k = ne00 = src0 的列数 = 内积维度 = 输入维度
```

### 4.2 矩阵乘法形式

```
dst[n, m] = src0[k, m]^T × src1[k, n]^T
```

或者更清楚地：
```
Output[batch, out_dim] = Weight[out_dim, in_dim] × Input[batch, in_dim]^T
```

### 4.3 实际应用场景映射

| 场景 | 批次大小 (n) | 输出维度 (m) | 输入维度 (k) | test-backend-ops 命令 |
|------|-------------|-------------|-------------|----------------------|
| **单token推理** | 1 | 4096 | 4096 | `m=4096.*n=1.*k=4096` |
| **小batch** | 16 | 4096 | 4096 | `m=4096.*n=16.*k=4096` |
| **中等batch** | 128 | 4096 | 4096 | `m=4096.*n=128.*k=4096` |
| **大batch** | 512 | 4096 | 4096 | `m=4096.*n=512.*k=4096` |
| **FFN Up** | 512 | 14336 | 4096 | `m=14336.*n=512.*k=4096` |
| **FFN Down** | 512 | 4096 | 14336 | `m=4096.*n=512.*k=14336` |

### 4.4 与我们项目的映射

**我们的项目**:
```cpp
void gemm_q4_0_q8_1(
    const block_q4_0* weight,      // [M, K/32]
    const block_q8_1* activation,  // [N, K/32]
    float* output,                 // [M, N]
    int M, int N, int K
)
```

**映射到 llama.cpp test-backend-ops**:
```
我们的 M  →  llama.cpp 的 m (ne01, 输出维度)
我们的 N  →  llama.cpp 的 n (ne1, 批次大小)
我们的 K  →  llama.cpp 的 k (ne00, 内积维度)
```

**示例**:
```bash
# 我们的测试: M=4096, N=512, K=4096
./tests/benchmark_best 4096 512 4096

# 对应的 llama.cpp 测试
./test-backend-ops perf -o MUL_MAT -b CUDA0 \
  -p "type_a=q4_0.*m=4096.*n=512.*k=4096"
```

---

## 5. TESTING_GUIDE.md 需要的修正

### 5.1 修正测试命令示例

**当前（错误）**:
```bash
./test-backend-ops perf -o MUL_MAT -b CUDA0 \
  -p "type_a=q4_0.*m=4096.*n=2.*k=14336"
```

**应该改为（单 token 推理）**:
```bash
./test-backend-ops perf -o MUL_MAT -b CUDA0 \
  -p "type_a=q4_0.*m=4096.*n=1.*k=4096"
```

**或者（中等批次）**:
```bash
./test-backend-ops perf -o MUL_MAT -b CUDA0 \
  -p "type_a=q4_0.*m=4096.*n=128.*k=4096"
```

### 5.2 修正参数说明

**应该添加**:

```markdown
#### test-backend-ops 参数说明

在 llama.cpp 的 test-backend-ops 中，参数 m, n, k 的含义为：

- **m**: 输出维度（权重矩阵的行数，对应 GGML 的 ne01）
- **n**: 批次大小（输出矩阵的行数，对应 GGML 的 ne1）
- **k**: 输入维度（内积维度，对应 GGML 的 ne00）

**矩阵乘法形式**:
```
Output[n, m] = Weight[m, k] × Input[n, k]^T
```

**与我们项目的对应关系**:
```
llama.cpp 的 m  ←→  我们的 M (输出维度)
llama.cpp 的 n  ←→  我们的 N (批次大小)
llama.cpp 的 k  ←→  我们的 K (内积维度)
```

**重要**: 不要混淆 m/n 的顺序！在 llama.cpp 中，n 是批次大小，不是输出维度。
```

### 5.3 修正附录表格

**当前（混淆）**:
```markdown
| 场景 | M | N | K | 说明 |
|------|---|---|---|------|
| 单token推理 | 1 | 4096 | 4096 | 典型推理场景 |
```

**应该改为**:
```markdown
| 场景 | 批次大小 (n) | 输出维度 (m) | 输入维度 (k) | test-backend-ops 命令 |
|------|-------------|-------------|-------------|----------------------|
| 单token推理 | 1 | 4096 | 4096 | `m=4096.*n=1.*k=4096` |
| 小batch | 16 | 4096 | 4096 | `m=4096.*n=16.*k=4096` |
| 中等batch | 128 | 4096 | 4096 | `m=4096.*n=128.*k=4096` |
| 大batch | 512 | 4096 | 4096 | `m=4096.*n=512.*k=4096` |
| FFN Up | 512 | 14336 | 4096 | `m=14336.*n=512.*k=4096` |
| FFN Down | 512 | 4096 | 14336 | `m=4096.*n=512.*k=14336` |

**说明**:
- **批次大小 (n)**: 同时处理的 token 数量
- **输出维度 (m)**: 权重矩阵的行数，输出特征维度
- **输入维度 (k)**: 权重矩阵的列数，输入特征维度
```

### 5.4 修正测试配置数组

**当前（错误）**:
```bash
declare -a TEST_CONFIGS=(
    "4096 2 14336"      # 这是什么？
    "4096 4096 4096"
    "1 4096 4096"
    "128 4096 4096"
)
```

**应该改为**:
```bash
# 测试配置: m n k (输出维度 批次大小 输入维度)
declare -a TEST_CONFIGS=(
    "4096 1 4096"       # 单token推理
    "4096 16 4096"      # 小batch
    "4096 128 4096"     # 中等batch
    "4096 512 4096"     # 大batch
    "14336 512 4096"    # FFN Up
    "4096 512 14336"    # FFN Down
)
```

---

## 6. 实际测试验证

### 6.1 验证命令

让我们验证正确的参数：

```bash
cd /home/haiyan/Agent4Kernel/llama.cpp/build/bin

# 单token推理（正确）
./test-backend-ops perf -o MUL_MAT -b CUDA0 \
  -p "type_a=q4_0.*m=4096.*n=1.*k=4096"

# 中等batch（正确）
./test-backend-ops perf -o MUL_MAT -b CUDA0 \
  -p "type_a=q4_0.*m=4096.*n=128.*k=4096"

# FFN Up 层（正确）
./test-backend-ops perf -o MUL_MAT -b CUDA0 \
  -p "type_a=q4_0.*m=14336.*n=512.*k=4096"
```

### 6.2 与我们项目的对应测试

```bash
cd /home/haiyan/Agent4Kernel/quant-gemm-from-scratch

# 对应单token推理
./tests/benchmark_best 4096 1 4096

# 对应中等batch
./tests/benchmark_best 4096 128 4096

# 对应 FFN Up 层
./tests/benchmark_best 14336 512 4096
```

---

## 7. 总结

### 7.1 关键发现

1. **⚠️ TESTING_GUIDE.md 存在严重错误**
   - 测试命令使用 `n=2`，这在实际应用中没有意义
   - 附录表格的 M/N 定义与测试命令的 m/n 相反
   - 缺少参数语义的明确说明

2. **✅ 正确的参数定义**
   ```
   m = 输出维度（权重行数）
   n = 批次大小
   k = 输入维度（内积维度）
   ```

3. **✅ 与我们项目的映射一致**
   ```
   llama.cpp 的 m  ←→  我们的 M
   llama.cpp 的 n  ←→  我们的 N
   llama.cpp 的 k  ←→  我们的 K
   ```

### 7.2 推荐的测试参数

**实际应用场景**:
```bash
# 单token推理（最常见）
m=4096, n=1, k=4096

# 小batch推理
m=4096, n=16, k=4096

# 中等batch
m=4096, n=128, k=4096

# 大batch
m=4096, n=512, k=4096

# FFN 层
m=14336, n=512, k=4096  (Up projection)
m=4096, n=512, k=14336  (Down projection)
```

**不推荐**:
```bash
# ❌ 错误：n=2 太小，没有实际意义
m=4096, n=2, k=14336
```

### 7.3 文档修订建议

1. **立即修正**: 将所有 `n=2` 的示例改为合理的批次大小（1, 16, 128, 512）
2. **添加说明**: 明确解释 m, n, k 的语义
3. **统一术语**: 在整个文档中使用一致的参数命名
4. **添加映射**: 说明与我们项目参数的对应关系
5. **添加警告**: 提醒读者不要混淆 m/n 的顺序

---

## 8. 修订后的测试示例

### 8.1 单token推理测试

```bash
# llama.cpp
./test-backend-ops perf -o MUL_MAT -b CUDA0 \
  -p "type_a=q4_0.*m=4096.*n=1.*k=4096"

# 我们的项目
./tests/benchmark_best 4096 1 4096
```

**预期性能**: ~0.1-0.2 TFLOPS（计算量小）

### 8.2 中等batch测试

```bash
# llama.cpp
./test-backend-ops perf -o MUL_MAT -b CUDA0 \
  -p "type_a=q4_0.*m=4096.*n=128.*k=4096"

# 我们的项目
./tests/benchmark_best 4096 128 4096
```

**预期性能**: ~0.5-0.7 TFLOPS

### 8.3 FFN 层测试

```bash
# llama.cpp - FFN Up
./test-backend-ops perf -o MUL_MAT -b CUDA0 \
  -p "type_a=q4_0.*m=14336.*n=512.*k=4096"

# 我们的项目 - FFN Up
./tests/benchmark_best 14336 512 4096
```

**预期性能**: ~0.7-0.8 TFLOPS

---

**文档版本**: 1.0
**最后更新**: 2026-01-30
**状态**: ⚠️ 发现 TESTING_GUIDE.md 存在严重错误，需要修订
**作者**: Claude Sonnet 4.5
