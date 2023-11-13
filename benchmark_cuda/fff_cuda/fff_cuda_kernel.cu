#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {
template <typename scalar_t>
__device__ __forceinline__ scalar_t gelu(scalar_t z) {
  return z * normcdff(z);
}

template <typename scalar_t>
__global__ void fff_cuda_forward_kernel(
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> output,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> x,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> in_weight,
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> in_bias,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> out_weight,
    const unsigned int width,
    const unsigned int depth,
    const unsigned int n_nodes
  ) {
  // compute which row of inputs we're dealing with
  const int row_index = blockIdx.x * blockDim.x + threadIdx.x;

  // zero the output
  for (int i = 0; i < width; ++i) {
    output[row_index][i] = 0;
  }

  if (row_index < x.size(0)) {
    int current_node = 0;
    for (int current_depth = 0; current_depth <= depth; ++current_depth) {
        scalar_t acc = 0;
        for (int i = 0; i < width;++i) {
            acc += x[row_index][i] * in_weight[current_node][i];
        }
        acc += in_bias[current_node];

        // compute the activation
        scalar_t activation = gelu(acc);

        // compute the output contribution due to the current node
        for (int i = 0; i < width; ++i) {
            output[row_index][i] += activation * out_weight[current_node][i];
        }

        // decide where to move to
        current_node = (current_node<<1) + 1 + (acc > 0 ? 1 : 0);
    }
  }
}
} // namespace

torch::Tensor fff_cuda_forward(
	torch::Tensor x,
	torch::Tensor in_weight,
	torch::Tensor in_bias,
	torch::Tensor out_weight,
	const unsigned int width,
	const unsigned int depth,
	const unsigned int parallel_size,
	const unsigned int n_nodes
) {

  auto output = torch::empty(
    {x.size(0), width},
    torch::TensorOptions()
      .dtype(torch::kFloat32)
      .device(x.device())
  );

  const int batch_size = x.size(0);

  const int threads = 1024;
  const int blocks = (batch_size + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(in_weight.type(), "fff_forward_cuda", ([&] {
    fff_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        x.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        in_weight.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        in_bias.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
        out_weight.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        width,
        depth,
        n_nodes
    );
  }));

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