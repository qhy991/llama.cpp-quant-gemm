from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import torch

ext_modules = [
    Pybind11Extension(
        "w4a16_q4_0_fp32_binding",
        [
            "bindings.cpp",
            "kernel.cu",
        ],
        extra_compile_args=["-O3"],
        extra_link_args=[],
        include_dirs=[torch.utils.cpp_extension.CUDA_HOME + "/include"],
        library_dirs=[torch.utils.cpp_extension.CUDA_HOME + "/lib64"],
        libraries=["cudart"],
        language="c++",
    ),
]

setup(
    name="w4a16_q4_0_fp32",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
