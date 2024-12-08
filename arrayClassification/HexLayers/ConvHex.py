import torch
import torch.nn as nn
from arrayClassification.HexLayers.neighbor import get_neighbor_tensor


class ConvHex(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super().__init__()

        assert kernel_size >= 1, "Kernel size must be greater than 0"
        self.kernel_size = kernel_size

        self.in_channels = in_channels
        self.out_channels = out_channels

        neighbor_info = get_neighbor_tensor(kernel_size)
        self.register_buffer('neighbors', neighbor_info.tensor)  # (Makes it accessible under self.neighbors)

        # Create weight matrix for central hexagons
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
        batch_size = x.shape[0]
        num_hexagons = x.shape[2]

        out = torch.zeros(batch_size, self.out_channels, num_hexagons, device=x.device)

        for hex_idx in range(num_hexagons):
            # Get central hexagon values
            center = x[:, :, hex_idx]  # Shape: [batch_size, in_channels]

            # Central contribution
            center_contrib = torch.matmul(center, self.weight_center.t())  # [batch_size, out_channels]

            # Get neighbor indices for current hexagon
            neighbor_indices = self.neighbors[hex_idx]  # [3, 4, -1 , -1]
            valid_neighbors = neighbor_indices >= 0  # [True, True, False, False]
            valid_neighbor_indices = neighbor_indices[valid_neighbors]  # [3, 4]

            # Neighbor contribution
            neighbor_contrib = torch.zeros_like(center_contrib)
            for n_idx, neighbor_idx in enumerate(valid_neighbor_indices):
                neighbor = x[:, :, neighbor_idx]  # [batch_size, in_channels]

                n_contrib = torch.matmul(neighbor, self.weight_neighbors[:, :, n_idx].t())
                neighbor_contrib += n_contrib

            # Combine contributions and add bias
            total_valid = len(valid_neighbor_indices) + 1  # number of valid neighbors plus center
            out[:, :, hex_idx] = (center_contrib + neighbor_contrib) / total_valid + self.bias

        return out
