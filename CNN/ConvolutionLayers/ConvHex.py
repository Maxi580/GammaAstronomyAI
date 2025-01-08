import math

import torch
import torch.nn as nn
from torch.nn import init

from CNN.ConvolutionLayers.neighbor import get_neighbor_tensor


class ConvHex(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=1, bias=True):
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")

        neighbor_info = get_neighbor_tensor(kernel_size)
        self.register_buffer('neighbors', neighbor_info.tensor)

        total_inputs = neighbor_info.max_neighbors + 1
        self.weight = nn.Parameter(
            torch.empty((out_channels, in_channels // groups, total_inputs))
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.total_inputs = total_inputs

        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        fan_in = self.in_channels * self.total_inputs // self.groups
        bound = 1.0 / math.sqrt(fan_in)
        init.uniform_(self.weight, -bound, bound)

        if self.bias is not None:
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        batch_size, in_channels, num_hex = x.shape

        if in_channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels but got {in_channels}")

        # Input normalization
        x = x / (x.std() + 1e-5)

        # Get valid mask and indices (Not all hexagons have max_neighbors, these values are padded)
        valid_mask = (self.neighbors >= 0)
        neighbor_indices = self.neighbors.clamp(min=0)  # Set padded Values to 0

        # Center values
        center_values = x.unsqueeze(3)

        # Get neighbor values
        neighbor_values = x[:, :, neighbor_indices]

        # Set all padded Values to 0
        neighbor_values = neighbor_values * valid_mask.unsqueeze(0).unsqueeze(1)

        # Concatenate center and neighbor values
        all_values = torch.cat([center_values, neighbor_values], dim=3)

        # Reshape for group convolution
        all_values = all_values.view(batch_size, self.groups, in_channels // self.groups, num_hex, -1)
        weight = self.weight.view(self.groups, self.out_channels // self.groups,
                                  in_channels // self.groups, -1)

        # Apply convolution
        out = torch.einsum('bgihk,goik->bgoh', all_values, weight)
        out = out.reshape(batch_size, self.out_channels, num_hex)

        total_valid = valid_mask.sum(dim=1).unsqueeze(0).unsqueeze(1) + 1
        out = out / (total_valid + 1e-6)

        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1)

        out = self.batch_norm(out)

        return out
