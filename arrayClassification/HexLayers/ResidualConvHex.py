import torch.nn as nn
from arrayClassification.HexLayers.ConvHex import ConvHex
import torch.nn.functional as F


class ResidualConvHex(nn.Module):
    """Trying to solve "gradient vanishing problem".
       this is essentially achieved by skipping connections
       Probably not needed as we don't have 100 convolution layers right now"""
    def __init__(self, channels, kernel_size):
        super().__init__()
        self.conv1 = ConvHex(channels, channels, kernel_size=kernel_size)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = ConvHex(channels, channels, kernel_size=kernel_size)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity

        out = F.relu(out)
        return out
