#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

torch::Tensor ff_cuda_forward(
	torch::Tensor x,
	torch::Tensor in_weight,
	torch::Tensor in_bias,
	torch::Tensor out_weight
);

/*std::vector<torch::Tensor> ff_cuda_backward(
		torch::Tensor inputs
);*/

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.device().type() == torch::kCUDA, #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_IDENTITY(x) AT_ASSERTM(x==1, #x " must be 1")
#define CHECK_POSITIVE(x) AT_ASSERTM(x>0, #x " must be positive")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor ff_forward(
		torch::Tensor x,
		torch::Tensor in_weight,
		torch::Tensor in_bias,
		torch::Tensor out_weight
	) {
	CHECK_INPUT(x);
	CHECK_INPUT(in_weight);
	CHECK_INPUT(in_bias);
	CHECK_INPUT(out_weight);

	return ff_cuda_forward(x, in_weight, in_bias, out_weight);
}

std::vector<torch::Tensor> ff_backward(
		torch::Tensor inputs
) {
	CHECK_INPUT(inputs);

	return { };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward", &ff_forward, "FF forward (CUDA)");
	m.def("backward", &ff_backward, "FF backward (CUDA)");
}
