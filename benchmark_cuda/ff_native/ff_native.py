import torch
import torch.nn as nn

class FF(nn.Module):
	def __init__(self, input_width, hidden_width, output_width,):
		super().__init__()

		self.input_width = input_width
		self.output_width = output_width

		self.linear_in = nn.Linear(input_width, hidden_width, bias=True)
		self.linear_out = nn.Linear(hidden_width, output_width, bias=False)

	def forward(self, oldx: torch.Tensor) -> torch.Tensor:
		# x has shape (..., input_width)
		x = oldx.reshape(-1, self.input_width)
		# x has shape (batch_size, input_width)
		batch_size = x.shape[0]

		logits = self.linear_in(x)
		activations = torch.nn.functional.gelu(logits) 
		new_logits = self.linear_out(activations)

		ret = new_logits.reshape_as(oldx)
		return ret
