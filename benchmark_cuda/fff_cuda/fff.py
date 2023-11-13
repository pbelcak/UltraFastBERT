import math
from torch import nn
from torch.autograd import Function
import torch

import fff_cuda

torch.manual_seed(42)

class FFF(nn.Module):
	def __init__(self, input_width, output_width, depth, parallel_size, activation=nn.GELU):
		super(FFF, self).__init__()
		self.input_width = input_width
		self.output_width = output_width
		self.depth = depth
		self.parallel_size = parallel_size
		self.n_nodes = 2 ** (self.depth + 1) - 1

		self.linear_in_weight =  nn.Parameter(torch.empty((self.parallel_size * self.n_nodes, self.input_width)), requires_grad=True)
		self.linear_in_bias =  nn.Parameter(torch.empty((self.parallel_size * self.n_nodes)), requires_grad=True)
		self.linear_out_weight = nn.Parameter(torch.empty((self.parallel_size * self.n_nodes, self.output_width)), requires_grad=True)

		self.reset_parameters()

	def reset_parameters(self):
		init_k = math.sqrt(1.0 / self.input_width)
		init_k2 = math.sqrt(1.0 / ((self.depth+1) * self.parallel_size))
		self.linear_in_weight.data.uniform_(-init_k, +init_k)
		self.linear_in_bias.data.uniform_(-init_k, +init_k)
		self.linear_out_weight.data.uniform_(-init_k2, +init_k2)

	def forward(self, input):
		return FFFFunction.apply(
			input,
			self.linear_in_weight,
			self.linear_in_bias,
			self.linear_out_weight,
			self.input_width,
			self.depth,
			self.parallel_size,
			self.n_nodes
		)

class FFFFunction(Function):
	@staticmethod
	def forward(ctx, oldx, in_weight, in_bias, out_weight, width, depth, parallel_size, n_nodes):
		# oldx has shape (..., width)
		x = oldx.reshape(-1, width)
		# x has shape (batch_size, width)

		new_logits = fff_cuda.forward(x, in_weight, in_bias, out_weight, width, depth, parallel_size, n_nodes)

		ret = new_logits.reshape_as(oldx)
		return ret

	@staticmethod
	def backward(ctx, grad_of_output):
		pass