#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace py = pybind11;

// External CUDA kernel declaration
extern __global__ void gemm_q4_0_fp32(
    const uint8_t* weight,
    const float* activation,
    float* output,
    int M,
    int N,
    int K
);

PYBIND11_MODULE(w4a16_q4_0_fp32_binding, m) {
    m.def("gemm_q4_0_fp32",
        [](py::array_t<uint8_t> weight, py::array_t<float> activation, py::array_t<float> output) {

            // 1. 获取 Buffer Info
            py::buffer_info w_buf = weight.request();
            py::buffer_info a_buf = activation.request();
            py::buffer_info o_buf = output.request();

            // 2. 解析维度
            // Activation shape: [M, K]
            int M = a_buf.shape[0];
            int K = a_buf.shape[1];

            // Weight shape: [N, K/32, 18] -> 这里的 N 是 output features
            int N = w_buf.shape[0];

            // 3. 校验 K 维度 (Block 对齐)
            if (K % 32 != 0) {
                throw std::runtime_error("K must be a multiple of 32");
            }
            if (w_buf.shape[1] != K / 32) {
                throw std::runtime_error("Weight block dimension mismatch");
            }

            // 4. 获取数据指针
            const uint8_t* w_ptr = static_cast<const uint8_t*>(w_buf.ptr);
            const float* a_ptr = static_cast<const float*>(a_buf.ptr);
            float* o_ptr = static_cast<float*>(o_buf.ptr);

            // 5. 释放 GIL 并启动 Kernel
            py::gil_scoped_release release;

            // 定义 Grid/Block
            dim3 blockDim(32, 32);
            dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

            gemm_q4_0_fp32<<<gridDim, blockDim>>>(w_ptr, a_ptr, o_ptr, M, N, K);

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                throw std::runtime_error(cudaGetErrorString(err));
            }
            cudaDeviceSynchronize();
        },
        py::arg("weight"), py::arg("activation"), py::arg("output"),
        "Kernel implementation for w4a16_q4_0_fp32"
    );
}
