"""
quant_gemm: Python bindings for quantized GEMM kernels
"""

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Get the directory containing this file
here = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(here)

setup(
    name='quant_gemm',
    version='0.1.0',
    description='Python bindings for quantized GEMM CUDA kernels',
    author='quant-gemm-from-scratch',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='quant_gemm._C',
            sources=[
                'quant_gemm/csrc/bindings.cpp',
                'quant_gemm/csrc/gemm_ops.cu',
            ],
            include_dirs=[
                os.path.join(project_root, 'kernels'),
                os.path.join(project_root, 'compat'),
                os.path.join(project_root, 'include'),
                project_root,
            ],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    '-std=c++17',
                    '--use_fast_math',
                    '-gencode=arch=compute_80,code=sm_80',  # A100
                    '-gencode=arch=compute_86,code=sm_86',  # RTX 3090
                    '-gencode=arch=compute_89,code=sm_89',  # RTX 4090
                    '-gencode=arch=compute_120,code=sm_120',  # RTX 5070/5080/5090
                ],
            },
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
    install_requires=[
        'torch>=2.0.0',
        'numpy',
    ],
    python_requires='>=3.8',
)
