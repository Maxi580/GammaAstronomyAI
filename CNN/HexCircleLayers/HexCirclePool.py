import torch
import torch.nn as nn
from CNN.HexCircleLayers.pooling import get_clusters

class HexCirclePool(nn.Module):
    def __init__(self, kernel_size: int, n_pixels: int, mode: str = 'avg'):
        """
        A simple pooling layer for hexagonal grids.
        
        Parameters:
          - n_pixels: pixel count of original hex circle
          - mode: either 'avg' pooling or 'max' pooling.
        """
        
        super().__init__()
        
        if kernel_size <= 0:
            raise ValueError("kernel_size must a positive integer")
        if n_pixels <= 0:
            raise ValueError("n_pixels must be a positive integer")
        if mode not in ('avg', 'max'):
            raise ValueError("mode must be either 'avg' or 'max'")

        self.kernel_size = kernel_size
        self.n_pixels = n_pixels
        self.mode = mode
        
        # Generate clusters list for given pixels and kernel size and save it
        self.clusters = get_clusters(n_pixels, kernel_size)

    def extra_repr(self):
        return (
            "kernel_size={kernel_size}, n_pixels={n_pixels}, mode={mode}".format(**self.__dict__)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Input:
          x: Tensor of shape (B, channels, N) for N hexagon nodes.
        Output:
          A tensor of shape (B, channels, N_pooled) where N_pooled is the number of clusters.
        """
        B, C, N = x.shape
        
        if N != self.n_pixels:
            raise ValueError(f"Passed data has shape (..., n_pixels={N}), but expected shape is (..., n_pixels={self.n_pixels})")
        
        pooled_features = []
        # For each pooling region (cluster), aggregate the features.
        for cluster in self.clusters:
            # cluster is a list of node indices belonging to that pooling region.
            # Convert it to a tensor (on the correct device).
            # Also filter out -1 indices.
            indices = torch.tensor([x for x in cluster if x >= 0], dtype=torch.long, device=x.device)
            
            # Gather features for these indices. The result has shape (B, C, len(cluster)).
            cluster_features = x.index_select(dim=2, index=indices)
            
            if self.mode == 'avg':
                pooled = cluster_features.mean(dim=2)  # shape: (B, C)
            elif self.mode == 'max':
                pooled, _ = cluster_features.max(dim=2)
            pooled_features.append(pooled)

        # Stack the pooled features along the node dimension.
        # Final shape: (B, C, number_of_clusters)
        return torch.stack(pooled_features, dim=2)