# Enhanced Kernel Definition Schema

Based on [FlashInfer-Bench](https://flashinfer.ai/2025/10/21/flashinfer-bench.html), this schema extends the original format to support:

## New Fields

### 1. Model Architecture Tags
```json
{
  "model_architectures": ["llama", "llama-3.1", "deepseek", "qwen"],
  "use_case": "ffn-up | attention | lm-head | moe-router"
}
```

### 2. Op Type Classification
```json
{
  "op_type": "quant_gemm",
  "op_category": "matrix_multiply | attention | normalization | moe | sampling"
}
```

### 3. Workload Reference
```json
{
  "workload": {
    "source": "production | synthetic | benchmark",
    "dataset": "sharegpt | openorca",
    "trace_file": "traces/llama_3.1_8b_ffn_up.npz"
  }
}
```

### 4. Evaluation Results
```json
{
  "evaluation": {
    "baseline": "flashinfer | torch | cublas",
    "metrics": {
      "fast_p": 0.85,
      "speedup": 1.2,
      "latency_ms": 0.045
    },
    "hardware": "NVIDIA A100",
    "status": "verified | experimental | failed"
  }
}
```

### 5. Solution Metadata
```json
{
  "solution": {
    "author": "claude-opus-4 | gpt-4 | human",
    "language": "cuda | triton | python",
    "target": "sm80 | sm90"
  }
}
```

## Example Enhanced Definition

```json
{
  "name": "w4a16_q4_0_fp32_n4096_k4096",
  "op_type": "quant_gemm",
  "op_category": "matrix_multiply",

  "model_architectures": ["llama-3.1", "llama-3", "mistral"],
  "use_case": "ffn-up",

  "description": "...",

  "workload": {
    "source": "production",
    "description": "Llama 3.1 8B FFN up-projection, captured from ShareGPT dataset"
  },

  "evaluation": {
    "baseline": "flashinfer",
    "metrics": {
      "fast_p": 0.92,
      "speedup": 1.05,
      "latency_ms": 0.042
    },
    "hardware": "NVIDIA A100-80GB",
    "tested_at": "2025-01-15",
    "status": "verified"
  },

  "solution": {
    "author": "claude-opus-4",
    "version": "1.0.0",
    "language": "cuda",
    "target_arch": "sm80,sm90"
  },

  "...": "existing fields"
}
```

## Directory Structure

```
definitions/
├── quant_gemm/
│   ├── w4a16_q4_0_fp32_n4096_k4096.json
│   └── workload_data/
│       └── llama_3.1_8b_ffn_up.npz
├── attention/
├── moe/
├── normalization/
└── sampling/
```

## Leaderboard Data Structure

```json
{
  "leaderboard": {
    "op_type": "quant_gemm",
    "kernels": [
      {
        "name": "w4a16_q4_0_fp32_n4096_k4096",
        "solutions": [
          {
            "author": "claude-opus-4",
            "fast_p": 0.92,
            "speedup": 1.05
          },
          {
            "author": "gpt-4",
            "fast_p": 0.88,
            "speedup": 0.98
          }
        ]
      }
    ]
  }
}
```
