from torch.utils.cpp_extension import load
fff_cuda = load(
    'ff_cuda', ['ff_cuda.cpp', 'ff_cuda_kernel.cu'], verbose=True)
help(fff_cuda)
