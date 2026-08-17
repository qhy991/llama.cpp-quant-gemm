import torch
from torch.utils.cpp_extension import CUDAExtension, setup

setup(
    name="w4a16_q4_0_fp32",
    ext_modules=[
        CUDAExtension(
            name="w4a16_q4_0_fp32_binding",
            sources=["bindings.cpp", "kernel.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "--use_fast_math"]},
        )
    ],
)
