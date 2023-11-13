import math
from torch import nn
from torch.autograd import Function
import torch

import ff_cuda

torch.manual_seed(42)

class FF(nn.Module):
	def __init__(self, input_width, hidden_width, output_width):
		super(FF, self).__init__()
		self.input_width = input_width
		self.hidden_width = hidden_width
		self.output_width = output_width
		

		self.linear_in_weight =  nn.Parameter(torch.empty((self.hidden_width, self.input_width)), requires_grad=True)
		self.linear_in_bias =  nn.Parameter(torch.empty((self.hidden_width)), requires_grad=True)
		self.linear_out_weight = nn.Parameter(torch.empty((self.hidden_width, self.output_width)), requires_grad=True)

		print("Hidden width: ", self.hidden_width)

		self.reset_parameters()

	def reset_parameters(self):
		init_k = math.sqrt(1.0 / self.input_width)
		init_k2 = math.sqrt(1.0 / self.hidden_width)
		self.linear_in_weight.data.uniform_(-init_k, +init_k)
		self.linear_in_bias.data.uniform_(-init_k, +init_k)
		self.linear_out_weight.data.uniform_(-init_k2, +init_k2)

	def forward(self, input):
		return FFFunction.apply(
			input,
			self.linear_in_weight,
			self.linear_in_bias,
			self.linear_out_weight
		)

class FFFunction(Function):
	@staticmethod
	def forward(ctx, oldx, in_weight, in_bias, out_weight):
		# oldx has shape (..., width)
		x = oldx.reshape(-1, in_weight.shape[1])
		# x has shape (batch_size, width)

		new_logits = ff_cuda.forward(x, in_weight, in_bias, out_weight)

		ret = new_logits.reshape_as(oldx)
		return ret
		

	@staticmethod
	def backward(ctx, grad_of_output):
		pass