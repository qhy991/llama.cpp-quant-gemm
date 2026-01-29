# 教程 01: CUDA 编程基础

## 学习目标

- 理解 CUDA 编程模型
- 掌握内存管理和数据传输
- 编写第一个 CUDA kernel
- 理解线程层次结构

## CUDA 编程模型

### Host vs Device

```
┌─────────────────────────────────────────────────────────────┐
│                        Host (CPU)                           │
│  ┌──────────────┐                    ┌──────────────────┐   │
│  │ Main Memory  │ ←── PCIe 总线 ───→ │ Device (GPU)     │   │
│  │ (DRAM)       │                    │  ┌────────────┐  │   │
│  └──────────────┘                    │  │ GPU Memory │  │   │
│                                      │  │ (HBM/GDDR) │  │   │
│                                      │  └────────────┘  │   │
│                                      └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

- **Host**: CPU 及其内存
- **Device**: GPU 及其内存
- 数据需要在 Host 和 Device 之间传输

### 线程层次结构

```
Grid (所有线程)
├── Block 0
│   ├── Thread 0
│   ├── Thread 1
│   ├── ...
│   └── Thread N
├── Block 1
│   ├── Thread 0
│   ├── ...
│   └── Thread N
└── ...

关键概念:
- gridDim: Grid 中的 block 数量
- blockDim: Block 中的 thread 数量
- blockIdx: 当前 block 的索引
- threadIdx: 当前 thread 在 block 中的索引
```

### 内存层次

```
延迟     容量
  ↑        ↓
  │  ┌────────────┐
  │  │ Registers  │ ← 最快，每线程私有
  │  ├────────────┤
  │  │ Shared Mem │ ← 快，block 内共享
  │  ├────────────┤
  │  │ L1/L2 Cache│ ← 自动缓存
  │  ├────────────┤
  │  │ Global Mem │ ← 最慢，所有线程可见
  ↓  └────────────┘
```

## 第一个 CUDA 程序

### 向量加法

```cpp
// vector_add.cu

#include <cuda_runtime.h>
#include <stdio.h>

// CUDA kernel: 每个线程处理一个元素
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    // 计算全局线程索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int N = 1024;
    size_t bytes = N * sizeof(float);

    // 1. 分配 Host 内存
    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    float* h_c = (float*)malloc(bytes);

    // 初始化数据
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // 2. 分配 Device 内存
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // 3. Host -> Device 传输
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // 4. 启动 kernel
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    vector_add<<<grid_size, block_size>>>(d_a, d_b, d_c, N);

    // 5. Device -> Host 传输
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // 6. 验证结果
    for (int i = 0; i < 10; i++) {
        printf("c[%d] = %.1f\n", i, h_c[i]);
    }

    // 7. 释放内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
```

### 编译和运行

```bash
nvcc -o vector_add vector_add.cu
./vector_add
```

## 常用 CUDA API

### 内存管理

```cpp
// 分配设备内存
cudaMalloc(void** ptr, size_t size);

// 释放设备内存
cudaFree(void* ptr);

// 内存拷贝
cudaMemcpy(dst, src, size, direction);
// direction: cudaMemcpyHostToDevice / cudaMemcpyDeviceToHost
```

### 错误检查

```cpp
// 检查 API 调用
cudaError_t err = cudaMalloc(&ptr, size);
if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
}

// 检查 kernel 执行
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("Kernel error: %s\n", cudaGetErrorString(err));
}
```

### 设备信息

```cpp
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);  // 获取设备 0 的属性

printf("Device: %s\n", prop.name);
printf("Compute capability: %d.%d\n", prop.major, prop.minor);
printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
printf("Shared memory per block: %zu\n", prop.sharedMemPerBlock);
```

## 练习

1. 修改向量加法程序，计算 `c[i] = a[i] * b[i] + a[i]`
2. 实现向量点积 (需要使用原子操作或归约)
3. 测量不同 block size 对性能的影响

## 下一步

- [教程 02: 量化基础](../02-quantization-basics/README.md)
