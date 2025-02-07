import torch
import torch.nn as nn

from CNN.HexCircleLayers.HexCircleConv import HexCircleConv
from CNN.HexCircleLayers.HexCirclePool import HexCirclePool


class HexCircleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            HexCircleConv(1, 8, kernel_size=3, n_pixels=1039),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            # HexCirclePool(1, 1039, mode='max'),
            nn.Dropout1d(0.2),
            
            
            HexCircleConv(8, 16, kernel_size=1, n_pixels=1039),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            HexCirclePool(1, 1039, mode='max'),
            nn.Dropout1d(0.2),

            HexCircleConv(16, 16, kernel_size=1, n_pixels=163),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            HexCirclePool(1, 163, mode='max'),
            nn.Dropout1d(0.2),

            # HexConv(18, 36, kernel_size=1, n_pixels=1039),
            # nn.GroupNorm(18, 36),
            # nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2),
            # nn.Dropout1d(0.35031261571845096),
        )

    def forward(self, x):
        return self.cnn(x)


class HexCircleNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.m1_cnn = HexCircleCNN()
        self.m2_cnn = HexCircleCNN()

        self.classifier = nn.Sequential(
            nn.Linear(16 * 31 * 2, 256),
            # nn.GroupNorm(32, 768),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 32),
            # nn.GroupNorm(16, 384),
            nn.ReLU(),
            nn.Dropout(0.2),

            # nn.Linear(384, 128),
            # nn.GroupNorm(8, 128),
            # nn.ReLU(),
            # nn.Dropout(0.21686133182097764),

            nn.Linear(32, 2)
        )

    def forward(self, m1_image, m2_image, measurement_features):
        m1_image = m1_image.unsqueeze(1)
        m2_image = m2_image.unsqueeze(1)
        m1_cnn_features = (self.m1_cnn(m1_image)).flatten(1)
        m2_cnn_features = (self.m2_cnn(m2_image)).flatten(1)

        combined = torch.cat([m1_cnn_features, m2_cnn_features], dim=1)

        return self.classifier(combined)