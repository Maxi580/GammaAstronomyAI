import math
import torch
import torch.nn as nn

from CNN.HexCircleLayers.neighbors import get_neighbor_tensor


def calc_kernel_pixels(kernel_size: int):
    return 1 + sum(range(1, kernel_size+1)) * 6

class HexCircleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        bias: bool = True,
        # padding_mode: str = "zeros", # TODO
        device=None,
        dtype=None,
    ):
        """
        HexConv applies a convolution on a circular hexagonal grid.
        """
        
        
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        
        if in_channels <= 0:
            raise ValueError("in_channels must be a positive integer")
        if out_channels <= 0:
            raise ValueError("out_channels must be a positive integer")
        if kernel_size < 0:
            raise ValueError("kernel_size must be zero or a positive integer")
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_pixels = None
        
        total_kernel_pixels = calc_kernel_pixels(kernel_size)
        self.weight = nn.Parameter(torch.empty((out_channels, in_channels, total_kernel_pixels), **factory_kwargs))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        
        self.reset_parameters()
    
    def extra_repr(self):
        return (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}, n_pixels={n_pixels}".format(**self.__dict__)
        )

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Input:
        x: Tensor of shape (batch_size, in_channels, N) where N is the number of hexagons.
        Output:
        A tensor of shape (batch_size, out_channels, N).
        """
        B, Cin, N = x.shape
        
        if not self.n_pixels:
            self.n_pixels = N
            # Generate neighbors list for given pixels and kernel size and save it in buffer
            neighbor_list = get_neighbor_tensor(self.n_pixels, self.kernel_size).to(device=x.get_device())
            self.register_buffer('neighbors', neighbor_list)
        
        expected_N, K = self.neighbors.shape
        
        if N != expected_N or Cin != self.in_channels:
            raise ValueError(f"Passed data has shape (in_channels={Cin}, n_pixels={N}), but expected shape is (in_channels={self.in_channels}, n_pixels={expected_N})")
        
        # Expand neighbors tensor to batch dimension.
        # self.neighbors has shape (N, K) and contains -1 for missing neighbors on the edges.
        # We want to gather along the hexagon (N) dimension of x.
        neighbors = self.neighbors.unsqueeze(0).expand(B, N, K)  # shape: (B, N, K)
        
        # Create a mask for valid indices.
        # Will be True for all valid indices, else False.
        valid_mask = (neighbors != -1)  # shape: (B, N, K)
        
        # Replace -1 with 0 (so that we can safely gather without indexing errors).
        # Uses the previously generated mask for that.
        neighbors_safe = neighbors.clone()
        neighbors_safe[~valid_mask] = 0  # now all indices are in range [0, N-1]
        
        # Gather neighbor features from x.
        # x is (B, Cin, N). We expand it to (B, Cin, N, K).
        x_exp = x.unsqueeze(-1).expand(B, Cin, N, K)
        
        # Now, we need to gather along dimension 2.
        # Expand neighbors_safe to shape (B, Cin, N, K) so that each channel is gathered.
        neighbors_safe_exp = neighbors_safe.unsqueeze(1).expand(B, Cin, N, K)
        # Gather features: neighbor_features will have shape (B, Cin, N, K).
        neighbor_features = torch.gather(x_exp, dim=2, index=neighbors_safe_exp)
        
        # Zero out features from missing neighbors using the valid_mask.
        # .float() converts all Booleans into 1 or 0.
        valid_mask_exp = valid_mask.unsqueeze(1).expand(B, Cin, N, K)
        neighbor_features = neighbor_features * valid_mask_exp.float()
        
        # Now apply the convolution: we have a weight tensor of shape (out_channels, in_channels, K).
        # We want to compute for each hexagon n and output channel o:
        #     out[b, o, n] = sum_{cin, k} self.weight[o, cin, k] * neighbor_features[b, cin, n, k]
        out = torch.einsum('oik, bink -> bon', self.weight, neighbor_features)
        
        # Add bias if present.
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1)
        
        return out