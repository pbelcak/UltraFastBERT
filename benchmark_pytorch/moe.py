import torch
from torch import nn

class MoENetwork(nn.Module):
    def __init__(self, input_width: int, block_size: int, output_width: int, n_blocks: int, k: int = 1):
        super().__init__()
        self.input_width = input_width
        self.block_size = block_size
        self.output_width = output_width
        self.n_blocks = n_blocks
        self.k = k

        self.w1s = nn.Parameter(torch.empty((n_blocks, input_width, block_size), dtype=torch.float).uniform_(-1.0, +1.0), requires_grad=True)
        self.b1s = nn.Parameter(torch.empty((n_blocks, block_size), dtype=torch.float).uniform_(-1.0, +1.0), requires_grad=True)
        self.w2s = nn.Parameter(torch.empty((n_blocks, block_size, output_width), dtype=torch.float).uniform_(-1.0, +1.0), requires_grad=True)
        self.b2s = nn.Parameter(torch.empty((n_blocks, output_width), dtype=torch.float).uniform_(-1.0, +1.0), requires_grad=True)

        self.mixer = nn.Parameter(torch.zeros((input_width, n_blocks), dtype=torch.float), requires_grad=True)
        self.noise_controller = nn.Parameter(torch.zeros((input_width, n_blocks), dtype=torch.float), requires_grad=True)

    def forward_wide(self, x):
        # x has shape (batch_size, input_width)
        batch_size = x.shape[0]

        # compute the mixing coefficients
        prod = torch.matmul(x, self.mixer) # (batch_size, n_blocks)
        h_x = prod # (batch_size, n_blocks)
        scores = torch.softmax(h_x, dim=-1) # (batch_size, n_blocks)

        # retrieve the relevant parameters
        w1s = self.w1s.unsqueeze(0) # (1, n_blocks, input_width, block_size)
        b1s = self.b1s.unsqueeze(0) # (1, n_blocks, block_size)
        w2s = self.w2s.unsqueeze(0) # (1, n_blocks, block_size, output_width)
        b2s = self.b2s.unsqueeze(0) # (1, n_blocks, output_width)

        # compute the forward pass
        val = torch.matmul(x.unsqueeze(1).unsqueeze(1), w1s) + b1s.unsqueeze(-2) # (batch_size, n_blocks, 1, block_size)
        val = torch.nn.functional.relu(val) # (batch_size, n_blocks, 1, block_size)
        val = torch.matmul(val, w2s).squeeze(-2) + b2s # (batch_size, n_blocks, output_width)

        mixed_result = torch.sum(val * scores.unsqueeze(-1), dim=1) # (batch_size, output_width)

        return mixed_result
    
    def forward(self, x):
        # x has shape (batch_size, input_width)
        batch_size = x.shape[0]

        # compute the mixing coefficients
        mixer_prod = torch.matmul(x, self.mixer)
        noise_prod = torch.matmul(x, self.noise_controller)
        h_x = mixer_prod \
             + torch.randn(mixer_prod.shape, device=x.device) * torch.nn.functional.softplus(noise_prod) # (batch_size, n_blocks)
        h_x_topk_values, h_x_topk_indices = torch.topk(h_x, self.k, dim=-1) # (batch_size, k), (batch_size, k)
        scores = torch.softmax(h_x_topk_values, dim=-1) # (batch_size, k)

        # retrieve the relevant parameters
        w1s = self.w1s.index_select(
            dim=0,
            index=h_x_topk_indices.flatten()
        ).reshape(*h_x_topk_indices.shape, self.w1s.shape[1],  self.w1s.shape[2]) # (batch_size, k, input_width, block_size)
        b1s = self.b1s.index_select(
            dim=0,
            index=h_x_topk_indices.flatten()
        ).reshape(*h_x_topk_indices.shape, self.b1s.shape[1]) # (batch_size, k, block_size)
        w2s = self.w2s.index_select(
            dim=0,
            index=h_x_topk_indices.flatten()
        ).reshape(*h_x_topk_indices.shape, self.w2s.shape[1],  self.w2s.shape[2]) # (batch_size, k, block_size, output_width)
        b2s = self.b2s.index_select(
            dim=0,
            index=h_x_topk_indices.flatten()
        ).reshape(*h_x_topk_indices.shape, self.b2s.shape[1]) # (batch_size, k, output_width)

        # compute the forward pass
        val = torch.matmul(x.unsqueeze(1).unsqueeze(1), w1s) + b1s.unsqueeze(-2) # (batch_size, k, 1, block_size)
        val = torch.nn.functional.relu(val) # (batch_size, k, 1, block_size)
        val = torch.matmul(val, w2s).squeeze(-2) + b2s # (batch_size, k, output_width)

        mixed_result = torch.sum(val * scores.unsqueeze(-1), dim=1) # (batch_size, output_width)

        return mixed_result

    