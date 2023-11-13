from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='ff_cuda',
    ext_modules=[
        CUDAExtension('ff_cuda', [
            'ff_cuda.cpp',
            'ff_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
