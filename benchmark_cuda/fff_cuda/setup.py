from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fff_cuda',
    ext_modules=[
        CUDAExtension('fff_cuda', [
            'fff_cuda.cpp',
            'fff_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
