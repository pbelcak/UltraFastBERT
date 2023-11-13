from torch.utils.cpp_extension import load
fff_cuda = load(
    'fff_cuda', ['fff_cuda.cpp', 'fff_cuda_kernel.cu'], verbose=True)
help(fff_cuda)
