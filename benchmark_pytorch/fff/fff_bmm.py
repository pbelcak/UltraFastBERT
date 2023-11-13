import torch
from torch import nn
import math

class FFF(nn.Module):
	def __init__(self, input_width: int, depth: int, output_width: int, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.input_width = input_width
		self.depth = depth
		self.output_width = output_width

		self.n_nodes = 2 ** (depth + 1) - 1
		self.initialise_weights()

	def initialise_weights(self):
		init_factor_l1 = 1.0 / math.sqrt(self.input_width)
		init_factor_l2 = 1.0 / math.sqrt(self.depth + 1)
		self.w1s = nn.Parameter(torch.empty(self.n_nodes, self.input_width).uniform_(-init_factor_l1, +init_factor_l1), requires_grad=True)
		self.w2s = nn.Parameter(torch.empty(self.n_nodes, self.output_width).uniform_(-init_factor_l2, +init_factor_l2), requires_grad=True)

	def forward(self, x):
		# the shape of x is (batch_size, input_width)
		# retrieve the indices of the relevant elements
		batch_size = x.shape[0]
		current_nodes = torch.zeros((batch_size,), dtype=torch.long, device=x.device)
		all_nodes = torch.zeros(batch_size, self.depth+1, dtype=torch.long, device=x.device)
		all_logits = torch.empty((batch_size, self.depth+1), dtype=torch.float, device=x.device)

		for i in range(self.depth+1):
			all_nodes[:, i] = current_nodes
			plane_coeffs = self.w1s.index_select(dim=0, index=current_nodes)			# (batch_size, input_width)
			plane_coeff_score = torch.bmm(x.unsqueeze(1), plane_coeffs.unsqueeze(-1))	# (batch_size, 1, 1)
			plane_score = plane_coeff_score.squeeze(-1).squeeze(-1) 					# (batch_size,)
			all_logits[:, i] = plane_score
			plane_choices = (plane_score >= 0).long()									# (batch_size,)

			current_nodes = current_nodes * 2 + plane_choices + 1						# (batch_size,)

		# get the weights
		selected_w2s = self.w2s.index_select(0, index=all_nodes.flatten()) \
			.view(batch_size, self.depth+1, self.output_width)	# (batch_size, depth+1, output_width)

		# forward pass
		mlp1 = torch.nn.functional.gelu(all_logits)				# (batch_size, depth+1)
		mlp2 = torch.bmm(mlp1.unsqueeze(1), selected_w2s) 		# (batch_size, output_width)
		
		# done
		return mlp2
	