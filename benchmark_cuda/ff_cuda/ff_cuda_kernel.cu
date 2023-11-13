#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>

#include <vector>

namespace {
template <typename scalar_t>
__device__ __forceinline__ scalar_t gelu(scalar_t z) {
  return z * normcdff(z);
}

template <typename scalar_t>
__global__ void ff_cuda_forward_kernel(
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> output,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> x,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> in_weight,
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> in_bias,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> out_weight,
    const int width,
    const int hidden_width
  ) {
  // compute which row of inputs we're dealing with
  const int row_index = blockIdx.x * blockDim.x + threadIdx.x;

  // zero the output
  for (int i = 0; i < width; ++i) {
    output[row_index][i] = 0;
  }

  for (int current_node = 0; current_node < hidden_width;++current_node) {
      scalar_t acc = 0;
      for (int j = 0; j < width;++j) {
          acc += x[row_index][j] * in_weight[current_node][j];
      }
      acc += in_bias[current_node];

      // compute the activation
      scalar_t activation = gelu(acc);

      // compute the output contribution due to the current node
      for (int k = 0; k < width; ++k) {
          output[row_index][k] += activation * out_weight[current_node][k];
      }
  }
}
} // namespace

torch::Tensor ff_cuda_forward(
	torch::Tensor x,
	torch::Tensor in_weight,
	torch::Tensor in_bias,
	torch::Tensor out_weight
) {

  auto output = torch::empty(
    {x.size(0), out_weight.size(1)},
    torch::TensorOptions()
      .dtype(torch::kFloat32)
      .device(x.device())
  );

  const int batch_size = x.size(0);

  const int threads = 1024;
  const int blocks = (batch_size + threads - 1) / threads;

  // fprintf(stdout, "Launching a CUDA kernel");
  AT_DISPATCH_FLOATING_TYPES(in_weight.type(), "ff_forward_cuda", ([&] {
    ff_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        x.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        in_weight.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        in_bias.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
        out_weight.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        x.size(1),
        in_bias.size(0)
    );
  }));
  // fprintf(stdout, "CUDA kernel launched");

  cudaError_t err;
  err = cudaGetLastError();
  if (cudaSuccess != err) {
      fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
  }

  cudaError_t cudaStatus;
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
  }

  return output;
}