import torch
import torch.nn as nn

from CNN.HexLayers.ConvHex import ConvHex

NUM_OF_HEXAGONS = 1039
NUM_FEATURES = 59


def resize_input(image):
    """Arrays are 1183 long, however the last 144 are always 0"""
    return image[:, :, :NUM_OF_HEXAGONS]


class TelescopeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            ConvHex(1, 4, kernel_size=3),
            nn.GroupNorm(2, 4),
            nn.ReLU(),
            nn.Dropout1d(0.2),

            ConvHex(4, 8, kernel_size=2),
            nn.GroupNorm(4, 8),
            nn.ReLU(),
            nn.Dropout1d(0.2),
        )

    def forward(self, x):
        return self.cnn(x)


class CombinedNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.m1_cnn = TelescopeCNN()
        self.m2_cnn = TelescopeCNN()

        self.classifier = nn.Sequential(
            nn.Linear(8 * (1039 // (2 ** 0)) * 2, 768),
            nn.GroupNorm(32, 768),
            nn.ReLU(),
            nn.Dropout(0.4936012148325484),

            nn.Linear(768, 384),
            nn.GroupNorm(16, 384),
            nn.ReLU(),
            nn.Dropout(0.5340424416693821),

            nn.Linear(384, 128),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Dropout(0.21686133182097764),

            nn.Linear(128, 2)
            )

    def forward(self, m1_image, m2_image, measurement_features):
        # Add Channel Dimension
        m1_image = m1_image.unsqueeze(1)
        m2_image = m2_image.unsqueeze(1)
        # Throw away 144 0 Values at the end
        m1_image = resize_input(m1_image)
        m2_image = resize_input(m2_image)
        m1_features = self.m1_cnn(m1_image)
        m2_features = self.m2_cnn(m2_image)
        m1_features = m1_features.flatten(1)
        m2_features = m2_features.flatten(1)

        combined = torch.cat([m1_features, m2_features], dim=1)

        return self.classifier(combined)