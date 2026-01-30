# Testing Methods for quant-gemm-from-scratch

This document describes the testing infrastructure and methods available for validating quantized GEMM kernels.

## Quick Reference

```bash
# Activate environment
conda activate KM-12.8

# Quick test (fastest)
cd python
python -c "import torch, quant_gemm; ..."

# Spec-based test (recommended)
python test_operator.py w4a16_q4_0_fp32 operators/quant_gemm/variants/w4a16_q4_0_fp32

# With benchmarks
python test_operator.py w4a16_q4_0_fp32 operators/quant_gemm/variants/w4a16_q4_0_fp32 --benchmark

# Pytest
pytest tests/ -v
```

## Testing Methods

### 1. Quick Python Test (Development)

**Use case:** Fast iteration during development, debugging specific issues

**Example:**
```python
import torch
import quant_gemm

M, N, K = 1, 4096, 4096
weight_fp32 = torch.randn(N, K, device='cuda')
activation = torch.randn(M, K, device='cuda')

weight_q = quant_gemm.quantize_q4_0(weight_fp32)
output = quant_gemm.gemm_q4_0_fp32(weight_q, activation, M, N, K)
output_ref = torch.matmul(activation, weight_fp32.T)

nmse = torch.mean((output_ref - output) ** 2) / torch.mean(output_ref ** 2)
print(f'NMSE: {nmse.item():.6f} - {"PASS" if nmse < 0.05 else "FAIL"}')
```

**Pros:**
- Fastest method
- Easy to modify for debugging
- Good for testing specific scenarios

**Cons:**
- Manual setup required
- No automatic test discovery
- Limited reporting

### 2. Spec-Based Testing (Recommended)

**Use case:** Standard testing workflow, multiple configurations, benchmarking

**Command:**
```bash
python test_operator.py <variant_name> <variant_path> [--benchmark]
```

**Example:**
```bash
python test_operator.py w4a16_q4_0_fp32 operators/quant_gemm/variants/w4a16_q4_0_fp32 --benchmark
```

**Features:**
- Loads test configs from `spec.json`
- Multiple batch sizes (M=1, 4, 128, etc.)
- NMSE accuracy metric with configurable threshold
- Optional performance benchmarking
- Automatic kernel discovery

**Test Configuration (spec.json):**
```json
{
  "test_configs": [
    {"name": "single", "M": 1, "N": 4096, "K": 4096},
    {"name": "small_batch", "M": 4, "N": 4096, "K": 4096},
    {"name": "medium_batch", "M": 128, "N": 4096, "K": 4096}
  ],
  "accuracy": {
    "metric": "nmse",
    "threshold": 0.05
  }
}
```

**Pros:**
- Declarative configuration
- Consistent testing across variants
- Built-in benchmarking
- Good reporting

**Cons:**
- Requires spec.json setup
- Less flexible than direct Python

### 3. Framework-Based Testing (Comprehensive)

**Use case:** Advanced testing, registry-based discovery, comprehensive metrics

**Example:**
```python
from operators.registry import get_registry
from operators.test_framework import TestFramework
import quant_gemm

registry = get_registry()
registry.register_from_module(quant_gemm._C, prefix='gemm')

framework = TestFramework(registry)
spec = registry.get_spec('w4a16_q4_0_fp32')

# Run tests
results = framework.run_spec(spec)
framework.print_results(results)

# Run benchmarks
benchmarks = framework.benchmark_spec(spec)
framework.print_benchmarks(benchmarks)
```

**Features:**
- Registry-based kernel discovery
- Comprehensive error metrics (NMSE, MSE, max_error, cosine_sim)
- Automatic dimension convention handling
- Support for multiple operator families

**Pros:**
- Most comprehensive
- Automatic handling of conventions
- Rich error metrics
- Extensible architecture

**Cons:**
- More complex setup
- Requires understanding of registry system

### 4. Pytest Testing (CI/CD)

**Use case:** Continuous integration, automated testing, regression detection

**Commands:**
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_gemm_q4_0.py -v

# Run tests matching pattern
pytest tests/test_gemm_generic.py -v -k "w4a16"

# Run with benchmarks
pytest tests/ -v -m benchmark
```

**Features:**
- Standard pytest framework
- Parameterized tests
- Test discovery
- Fixtures for setup/teardown
- Integration with CI/CD pipelines

**Pros:**
- Industry standard
- Good CI/CD integration
- Rich plugin ecosystem
- Detailed reporting

**Cons:**
- Slower than direct testing
- Requires test file setup

## Accuracy Metrics

### NMSE (Normalized Mean Squared Error)

**Formula:**
```
NMSE = mean((output - reference)²) / mean(reference²)
```

**Interpretation:**
- NMSE = 0.00: Perfect match
- NMSE < 0.01: Excellent (< 1% error)
- NMSE < 0.05: Good (< 5% error) - typical threshold
- NMSE < 0.10: Acceptable for some quantized kernels
- NMSE > 0.10: Poor, likely has bugs

**Why NMSE?**
- Normalizes error relative to signal power
- Scale-invariant (works for small and large values)
- Easy to interpret as percentage

### Other Metrics

The framework also computes:
- **MSE**: Mean Squared Error
- **Max Error**: Maximum absolute difference
- **Avg Error**: Average absolute difference
- **Relative Error**: Mean relative error
- **Cosine Similarity**: Directional similarity

## Dimension Conventions

### w4a8 (Quantized Activation)

**Convention:** Transposed
```python
output = kernel(weight_q, activation_q, N, M, K)  # Swap M/N
output = output.T  # Transpose to [M, N]
```

**Rationale:** Kernel computes `C[N, M] = W[N, K] @ A[M, K]^T`

### w4a16 (FP32 Activation)

**Convention:** Standard
```python
output = kernel(weight_q, activation, M, N, K)  # Standard order
# Output is already [M, N]
```

**Rationale:** Kernel computes `C[M, N] = A[M, K] @ W[N, K]^T`

### Detection in test_operator.py

```python
act_dtype = spec["inputs"]["activation"]["dtype"]
is_quantized_activation = act_dtype.startswith("block_q")

if is_quantized_activation:
    output = kernel(weight_q, activation_q, N, M, K)
    output = output.T
else:
    output = kernel(weight_q, activation, M, N, K)
```

## Common Issues and Solutions

### Issue 1: "Weight shape mismatch"

**Symptom:**
```
Weight shape mismatch: expected 2304 elements, got 9437184
```

**Cause:** Dimension convention mismatch

**Solution:** Ensure test framework detects activation dtype correctly

### Issue 2: High NMSE (> 1.0)

**Symptom:**
```
NMSE: 1.822331  FAIL
```

**Cause:** Likely kernel bug (incorrect unpacking, wrong indexing)

**Solution:**
1. Test with simple fixed values
2. Verify quantization/dequantization roundtrip
3. Check memory access patterns
4. Validate against reference implementation

### Issue 3: NaN or Inf in output

**Symptom:**
```
Output contains NaN
```

**Cause:** Division by zero, overflow, or uninitialized memory

**Solution:**
1. Check scale computation (avoid division by zero)
2. Verify all memory is initialized
3. Check for integer overflow in accumulation

## Rebuilding After Changes

```bash
# Set CUDA_HOME
export CUDA_HOME=/home/haiyan/miniconda3/envs/KM-12.8

# Rebuild
cd python
python setup.py build_ext --inplace

# Or with explicit CUDA_HOME
CUDA_HOME=/home/haiyan/miniconda3/envs/KM-12.8 python setup.py build_ext --inplace
```

## Performance Benchmarking

### Using test_operator.py

```bash
python test_operator.py w4a16_q4_0_fp32 \
    operators/quant_gemm/variants/w4a16_q4_0_fp32 \
    --benchmark \
    --warmup 10 \
    --iterations 100
```

### Custom Benchmarking

```python
import torch
import quant_gemm

M, N, K = 128, 4096, 4096
weight_fp32 = torch.randn(N, K, device='cuda')
activation = torch.randn(M, K, device='cuda')

weight_q = quant_gemm.quantize_q4_0(weight_fp32)

# Warmup
for _ in range(10):
    _ = quant_gemm.gemm_q4_0_fp32(weight_q, activation, M, N, K)
torch.cuda.synchronize()

# Benchmark
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
for _ in range(100):
    _ = quant_gemm.gemm_q4_0_fp32(weight_q, activation, M, N, K)
end.record()
torch.cuda.synchronize()

elapsed_ms = start.elapsed_time(end) / 100
flops = 2 * M * N * K
gflops = flops / (elapsed_ms / 1000) / 1e9

print(f"Time: {elapsed_ms:.3f} ms")
print(f"GFLOPS: {gflops:.2f}")
```

## Best Practices

1. **Always test with multiple data patterns:**
   - Fixed values (0, 1, 0.5)
   - Positive random (torch.rand)
   - Mixed random (torch.randn)
   - Edge cases (very small, very large)

2. **Use appropriate thresholds:**
   - Q4_0: NMSE < 0.05 (5%)
   - Q8_1: NMSE < 0.01 (1%)
   - FP16: NMSE < 0.001 (0.1%)

3. **Verify quantization roundtrip:**
   ```python
   x_q = quantize(x)
   x_dq = dequantize(x_q)
   error = (x - x_dq).abs().max()
   ```

4. **Test multiple batch sizes:**
   - M=1 (single inference)
   - M=4 (small batch)
   - M=128 (medium batch)
   - M=4096 (large batch)

5. **Profile before optimizing:**
   - Use nsys or nvprof
   - Identify bottlenecks
   - Measure impact of changes

## References

- [TESTING_GUIDE.md](TESTING_GUIDE.md) - Original testing guide
- [spec.json](operators/quant_gemm/variants/w4a16_q4_0_fp32/spec.json) - Example spec
- [test_operator.py](python/test_operator.py) - Spec-based testing tool
- [test_framework.py](python/operators/test_framework.py) - Framework implementation
