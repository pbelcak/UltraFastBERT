import torch

from torch import nn
import math

class FFF_Sparse(nn.Module):
	def __init__(self, input_width, output_width, depth):
		super().__init__()
		self.input_width = input_width
		self.output_width = output_width

		self.depth = depth
		self.n_nodes = 2 ** (depth + 1) - 1

		l1_init_factor = 1.0 / math.sqrt(self.input_width)
		l2_init_factor = 1.0 / math.sqrt(self.n_nodes)
		self.w1s = nn.Parameter(torch.empty((input_width * self.n_nodes, 1), dtype=torch.float).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)
		self.w2s = nn.Parameter(torch.empty((self.n_nodes, output_width), dtype=torch.float).uniform_(-l2_init_factor, +l2_init_factor), requires_grad=True)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		batch_size = x.shape[0]
		current_nodes = torch.zeros((batch_size,), dtype=torch.long, device=x.device)
		all_nodes = torch.zeros((batch_size, self.depth+1), dtype=torch.long, device=x.device)
		all_logits = torch.empty((batch_size, self.depth+1), dtype=torch.float, device=x.device)

		x_coordinates = torch.arange(batch_size, dtype=torch.long, device=x.device).repeat(self.input_width)	# (batch_size * self.input_width,)
		y_coordinate_offsets = torch.arange(self.input_width, dtype=torch.long, device=x.device).unsqueeze(0)	# (1, input_width)

		for i in range(self.depth+1):
			all_nodes[:, i] = current_nodes
			y_coordinate_bases = (current_nodes * self.input_width).unsqueeze(-1)	# (batch_size, 1)
			y_coordinates = y_coordinate_bases + y_coordinate_offsets				# (batch_size, input_width)
			indices = torch.stack((x_coordinates, y_coordinates.flatten()), dim=0)	# (2, batch_size * input_width)
			x_sparse = torch.sparse_coo_tensor(
				indices=indices,
				values=x.flatten(),
				size=(batch_size, self.n_nodes * self.input_width),
				device=x.device
			)	# (batch_size, self.n_nodes * self.input_width)
			logits = torch.sparse.mm(x_sparse, self.w1s).squeeze(-1)	# (batch_size,)

			all_logits[:, i] = logits				# (batch_size,)
			plane_choices = (logits >= 0).long()	# (batch_size,)

			current_nodes = current_nodes * 2 + plane_choices + 1	# (batch_size,)

		all_logits = torch.nn.functional.gelu(all_logits)	# (batch_size, self.depth)

		x_coordinates = torch.arange(batch_size, dtype=torch.long, device=x.device).repeat(self.depth+1)# (batch_size * (self.depth+1),)
		indices = torch.stack((x_coordinates, all_nodes.flatten()), dim=0)	# (2, batch_size * (self.depth+1))
		hidden_sparse = torch.sparse_coo_tensor(
			indices=indices,
			values=all_logits.flatten(),
			size=(batch_size, self.n_nodes),
			device=x.device
		)	# (batch_size, self.n_nodes)
		new_logits = torch.sparse.mm(
			hidden_sparse,
			self.w2s
		)	# (batch_size, self.output_width)

		return new_logits	# (batch_size, self.output_width)
