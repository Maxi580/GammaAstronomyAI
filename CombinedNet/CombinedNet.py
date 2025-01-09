import torch
import torch.nn as nn
from CNN.ConvolutionLayers.ConvHex import ConvHex

NUM_OF_HEXAGONS = 1039
NUM_FEATURES = 59


def resize_input(image):
    """Arrays are 1183 long, however the last 144 are always 0"""
    return image[:, :, :NUM_OF_HEXAGONS]


class CombinedNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.m1_cnn = nn.Sequential(
            ConvHex(1, 2, kernel_size=3),
            nn.BatchNorm1d(2),
            nn.ReLU(),
            nn.Dropout1d(0.2),

            ConvHex(2, 4, kernel_size=2),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Dropout1d(0.2),
        )

        self.m2_cnn = nn.Sequential(
            ConvHex(1, 2, kernel_size=3),
            nn.BatchNorm1d(2),
            nn.ReLU(),
            nn.Dropout1d(0.2),

            ConvHex(2, 4, kernel_size=2),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Dropout1d(0.2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(8 * NUM_OF_HEXAGONS, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, m1_image, m2_image):
        # First add channel dimension (1 = in_channels)
        m1_image = m1_image.unsqueeze(1)  # Shape becomes [batch_size, 1, num_hexagons]
        m2_image = m2_image.unsqueeze(1)
        m1_image = resize_input(m1_image)
        m2_image = resize_input(m2_image)
        m1_cnn_features = self.m1_cnn(m1_image)
        m2_cnn_features = self.m2_cnn(m2_image)
        m1_cnn_features = m1_cnn_features.flatten(1)
        m2_cnn_features = m2_cnn_features.flatten(1)

        combined = torch.cat([m1_cnn_features, m2_cnn_features], dim=1)

        return self.classifier(combined)
