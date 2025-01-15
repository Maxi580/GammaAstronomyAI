from typing import Optional

import torch
import torch.nn as nn
from torch.nn import init

from CNN.ConvolutionLayers.neighbor import get_neighbor_tensor


class ConvHex(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pooling, pooling_kernel_size, pooling_cnt,
                 groups=1, bias=True):
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")

        neighbor_info = get_neighbor_tensor(kernel_size, pooling, pooling_kernel_size, pooling_cnt)
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

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='relu')

        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, x):
        batch_size, in_channels, num_hex = x.shape

        if in_channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels but got {in_channels}")

        # Get valid mask and indices (Not all hexagons have max_neighbors, these values are padded)
        valid_mask = (self.neighbors >= 0)
        neighbor_indices = self.neighbors.clamp(min=0)  # Set padded Values to 0

        # [batch_size, in_channels, num_hex] => [batch_size, in_channels, num_hex, 1]
        center_values = x.unsqueeze(3)

        # Get neighbor values
        neighbor_values = x[:, :, neighbor_indices]

        # Calculate average Value
        expanded_mask = valid_mask.unsqueeze(0).unsqueeze(1)
        valid_sum = torch.where(
            expanded_mask,
            neighbor_values,
            torch.zeros_like(neighbor_values)
        ).sum(dim=3, keepdim=True)
        total_sum = valid_sum + center_values
        valid_count = expanded_mask.sum(dim=3, keepdim=True).float() + 1.0
        avg_values = total_sum / valid_count

        # Replace padding values with average
        neighbor_values = torch.where(
            expanded_mask,
            neighbor_values,
            avg_values.expand(-1, -1, -1, neighbor_values.size(3))
        )

        # Concatenate center and neighbor values
        all_values = torch.cat([center_values, neighbor_values], dim=3)

        # Reshape for group convolution
        all_values = all_values.view(batch_size, self.groups, in_channels // self.groups, num_hex, -1)
        weight = self.weight.view(self.groups, self.out_channels // self.groups,
                                  in_channels // self.groups, -1)

        # Apply convolution
        out = torch.einsum('bgihk,goik->bgoh', all_values, weight)
        out = out.reshape(batch_size, self.out_channels, num_hex)

        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1)

        return out
