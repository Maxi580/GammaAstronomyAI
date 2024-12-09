import torch
import torch.nn as nn

from arrayClassification.HexLayers.neighbor import get_neighbor_tensor


class ConvHex(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        neighbor_info = get_neighbor_tensor(kernel_size)
        self.register_buffer('neighbors', neighbor_info.tensor)  # (Makes it accessible under self.neighbors)

        total_inputs = neighbor_info.max_neighbors + 1  # + 1 for center
        self.weights = nn.Parameter(
            torch.randn(out_channels, in_channels, total_inputs) /
            torch.sqrt(torch.tensor(in_channels * total_inputs))
        )

        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        x = x.transpose(1, 2)  # [batch, hexagons, channels]

        # Adds dimension: [batch, hex, 1, channels] (Because we want to combine with neighbors)
        center_values = x.unsqueeze(2)

        # self.neighbors has indices like [6,2,-1,4,-1] for each hexagon
        valid_mask = (self.neighbors >= 0)  # Creates [True,True,False,True,False]
        neighbor_indices = self.neighbors.clamp(min=0)  # Gets all the valid indices + 0 for invalid ones
        # Now we get the values and zero the ones that were not valid before
        neighbor_values = x[:, neighbor_indices] * valid_mask.unsqueeze(0).unsqueeze(-1)

        all_values = torch.cat([center_values, neighbor_values], dim=2)
        all_values = all_values.transpose(2, 3)

        out = torch.einsum('bhit,oit->bho', all_values, self.weights)
        total_valid = valid_mask.sum(dim=1).unsqueeze(1) + 1  # +1 for center
        out = out / total_valid + self.bias

        return out.transpose(1, 2)
