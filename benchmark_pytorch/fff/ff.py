import torch

from torch import nn
import math

class FF(nn.Module):
	def __init__(self, input_width, hidden_width, output_width):
		super().__init__()
		self.input_width = input_width
		self.hidden_width = hidden_width
		self.output_width = output_width

		l1_init_factor = 1.0 / math.sqrt(self.input_width)
		l2_init_factor = 1.0 / math.sqrt(self.hidden_width)
		self.w1s = nn.Parameter(torch.empty((input_width, hidden_width), dtype=torch.float).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)
		self.w2s = nn.Parameter(torch.empty((hidden_width, output_width), dtype=torch.float).uniform_(-l2_init_factor, +l2_init_factor), requires_grad=True)
		
	def forward(self, x):
		# x has shape (batch_size, input_width)
		batch_size = x.shape[0]

		logits = torch.bmm(
			x.unsqueeze(1),
			self.w1s.unsqueeze(0).expand(batch_size, self.input_width, self.hidden_width)
		) 											# (batch_size, 1, hidden_width)
		logits = torch.nn.functional.gelu(logits)	# (batch_size, 1, hidden_width)
		logits = torch.bmm(
			logits,
			self.w2s.unsqueeze(0).expand(batch_size, self.hidden_width, self.output_width)
		)							# (batch_size, 1, output_width)

		return logits.squeeze(1)	# (batch_size, output_width)
