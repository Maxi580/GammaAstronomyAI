import torch
import torch.nn as nn
from CNN.ConvolutionLayers.ConvHex import ConvHex

NUM_OF_HEXAGONS = 1039
CNN_REDUCE_OUTPUT_FEATURES = 2048


def resize_input(image):
    """Arrays are 1183 long, however the last 144 are always 0"""
    return image[:, :, :NUM_OF_HEXAGONS]


class CombinedNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            ConvHex(in_channels=1, out_channels=4, kernel_size=3),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Dropout1d(0.4),

            ConvHex(in_channels=4, out_channels=8, kernel_size=2),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout1d(0.4),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * NUM_OF_HEXAGONS * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, m1_image, m2_image):
        # First add channel dimension (1 = in_channels)
        m1_image = m1_image.unsqueeze(1)  # Shape becomes [batch_size, 1, num_hexagons]
        m2_image = m2_image.unsqueeze(1)

        m1_image = resize_input(m1_image)
        m2_image = resize_input(m2_image)

        # Process images
        m1_features = self.cnn(m1_image)
        m2_features = self.cnn(m2_image)

        combined = torch.cat([m1_features, m2_features], dim=1)

        return self.classifier(combined)

