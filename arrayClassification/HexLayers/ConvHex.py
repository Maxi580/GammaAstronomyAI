import torch
import torch.nn as nn

from arrayClassification.HexLayers.neighbor import get_neighbor_tensor


class ConvHex(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size

        self.in_channels = in_channels
        self.out_channels = out_channels

        neighbor_info = get_neighbor_tensor(kernel_size)
        self.register_buffer('neighbors', neighbor_info.tensor)  # (Makes it accessible under self.neighbors)

        # Create weight matrix for central hexagons
        # [out_channels, in_channels]
        self.weight_center = nn.Parameter(
            torch.randn(out_channels, in_channels) / torch.sqrt(torch.tensor(in_channels))
        )

        # Create weight matrix for neighbor hexagons
        # Shape will be [out_channels, in_channels, max_neighbors]
        self.weight_neighbors = nn.Parameter(
            torch.randn(out_channels, in_channels, neighbor_info.max_neighbors) /
            torch.sqrt(torch.tensor(in_channels * neighbor_info.max_neighbors))
        )

        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        # [batch_size, in_channels, num_hexagons]
        batch_size, in_channels, num_hexagons = x.shape

        # Center Contribution
        # [batch_size, num_hexagons, in_channels] [32, 1039, 1]
        x = x.transpose(1, 2)
        # Weight.T: [in_channels, out_channels] [1, 16]
        # Contrib form will be: # [batch_size, num_hexagon, Out_channel] [32, 1039, 16]
        center_contrib = torch.matmul(x, self.weight_center.t())

        # Neighbor Contribution
        neighbor_contrib = torch.zeros_like(center_contrib)
        for hex_idx in range(num_hexagons):
            neighbor_indices = self.neighbors[hex_idx]

            # Every valid neighbor gets multiplied with the corresponding weight
            for pos, neighbor_idx in enumerate(neighbor_indices):
                # Filter out padding
                if neighbor_idx >= 0:
                    neighbor_value = x[:, neighbor_idx]
                    neighbor_weights = self.weight_neighbors[:, :, pos].t()
                    curr_contrib = torch.matmul(neighbor_value, neighbor_weights)
                    neighbor_contrib[:, hex_idx] += curr_contrib

        # Normalize by total number of valid neighbors + center
        total_valid = (self.neighbors[0] >= 0).sum() + 1
        out = (center_contrib + neighbor_contrib) / total_valid + self.bias

        # Return in the expected shape [batch_size, out_channels, num_hexagons]
        return out.transpose(1, 2)
