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
            # Get all valid neighbors (indices that are not -1/ padding)
            neighbor_indices = self.neighbors[hex_idx]
            valid_neighbors = neighbor_indices >= 0
            valid_neighbor_indices = neighbor_indices[valid_neighbors]

            # [batch_size, num_valid_neighbors, in_channels]
            neighbor_values = x[:, valid_neighbor_indices]

            # Get weights for each valid neighbor
            # [out_channels, in_channels, num_valid_neighbors]
            valid_weights = self.weight_neighbors[:, :, :len(valid_neighbor_indices)]

            # For each neighbor:
            # [batch_size, num_valid_neighbors, in_channels] @ [in_channels, out_channels]
            n_contrib = 0
            for i in range(len(valid_neighbor_indices)):
                # Get weights for this neighbor position
                # [out_channels, in_channels] -> [in_channels, out_channels]
                neighbor_weights = valid_weights[:, :, i].t()

                # Get values for this neighbor
                # [batch_size, in_channels]
                neighbor_value = neighbor_values[:, i]

                # Calculate contribution
                # [batch_size, out_channels]
                curr_contrib = torch.matmul(neighbor_value, neighbor_weights)
                n_contrib += curr_contrib

            neighbor_contrib[:, hex_idx] = n_contrib

        # Normalize by total number of valid neighbors + center
        total_valid = (self.neighbors[0] >= 0).sum() + 1  # Add 1 for center
        out = (center_contrib + neighbor_contrib) / total_valid + self.bias

        # Return in the expected shape [batch_size, out_channels, num_hexagons]
        return out.transpose(1, 2)
