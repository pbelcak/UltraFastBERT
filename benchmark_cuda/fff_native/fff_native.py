import torch
import torch.nn as nn

class FFF(nn.Module):
	def __init__(self, input_width, output_width, depth):
		super().__init__()

		self.input_width = input_width
		self.output_width = output_width
		self.depth = depth

		self.linear_in = nn.Linear(input_width, depth+1, bias=True)
		self.linear_out = nn.Linear(depth+1, output_width, bias=False)


	def forward(self, oldx: torch.Tensor) -> torch.Tensor:
		# x has shape (..., input_width)
		x = oldx.reshape(-1, self.input_width)
		# x has shape (batch_size, input_width)
		batch_size = x.shape[0]

		logits = self.linear_in(x) # (batch_size, depth+1)
		activations = torch.nn.functional.gelu(logits)
		new_logits = self.linear_out(activations)

		ret = new_logits.reshape_as(oldx)
		return ret
