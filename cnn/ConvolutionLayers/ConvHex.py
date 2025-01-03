import torch
import torch.nn as nn

from cnn.ConvolutionLayers.Neighbor import get_neighbor_tensor


class ConvHex(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        neighbor_info = get_neighbor_tensor(kernel_size)
        self.register_buffer('neighbors', neighbor_info.tensor)

        total_inputs = neighbor_info.max_neighbors + 1

        # Shape [out_channels, in_channels, number_of_hexagons]
        self.weights = nn.Parameter(
            torch.randn(out_channels, in_channels, total_inputs) /
            torch.sqrt(torch.tensor(in_channels * total_inputs))
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        # Before Transpose [Batch_size, in_channel, num_hex]
        x = x.transpose(1, 2)
        # After Transpose [Batch_size, num_hex, in_channel]

        # Expand Size to match with neighbors, because we are gonna merge them
        center_values = x.unsqueeze(2)
        # Center values shape [Batch_size, number_of_hexagons, 1, in_channel] (1 later holds all values that exist)

        # [True, True, False...] etc. greats a mask to save valid/ padding indices
        valid_mask = (self.neighbors >= 0)
        # Valid Mask shape [num_hex, max_neighbors]

        # Temporarily sets [-1] padding values to 0, to get valid neighbor values
        neighbor_indices = self.neighbors.clamp(min=0)
        # Valid Mask shape [num_hex, max_neighbors]

        # Get the corresponding values, note that for -1 padding there are the values of hex 0
        neighbor_values = x[:, neighbor_indices]
        # Neighbor Value Shape [Batch size, num_hex, max_neighbor, input_channel]
        # Valid Mask Shape (valid_mask.unsqueeze(0).unsqueeze(-1).shape) \
        # [1 size, num_hex, max_neighbor, 1]

        # Set the values that were invalid to 0, so they don't affect the outcome/ weights
        neighbor_values = neighbor_values * valid_mask.unsqueeze(0).unsqueeze(-1)
        # Masked neighbor values shape [Batch size, num_hex, max_neighbor, input_channel]

        all_values = torch.cat([center_values, neighbor_values], dim=2)
        # Masked neighbor values shape [Batch size, num_hex, max_neighbor + 1, input_channel]
        # Weights shape: [Out_channel, In_channel, max_neighbor + 1]

        all_values = all_values.transpose(2, 3)
        # Masked neighbor values shape [Batch size, num_hex, input_channel, max_neighbor + 1]

        out = torch.einsum('bhit,oit->bho', all_values, self.weights)
        # Output shape: [Batch_size,  max_neighbor + 1, out_channels]

        total_valid = valid_mask.sum(dim=1).unsqueeze(1) + 1
        # Total Valid shape: torch.Size([num_hex, 1]

        out = out / total_valid + self.bias
        return out.transpose(1, 2)
