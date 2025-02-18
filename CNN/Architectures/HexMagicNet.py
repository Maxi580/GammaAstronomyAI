import torch
import torch.nn as nn
from CNN.MagicConv.MagicConv import MagicConv


class TelescopeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            MagicConv(1, 16, kernel_size=3),
            nn.GroupNorm(8, 16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout1d(p=0.2),

            MagicConv(16, 32, kernel_size=2),
            nn.GroupNorm(16, 32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout1d(p=0.2),
        )

    def forward(self, x):
        return self.cnn(x)


class HexMagicNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.m1_cnn = TelescopeCNN()
        self.m2_cnn = TelescopeCNN()

        self.classifier = nn.Sequential(
            nn.Linear(32 * (1039 // (2**2)) * 2, 4096),
            nn.GroupNorm(128, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(4096, 512),
            nn.GroupNorm(32, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(512, 128),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),

            nn.Linear(128, 2)
        )

    def forward(self, m1_image, m2_image, measurement_features):
        m1_image = m1_image.unsqueeze(1)
        m2_image = m2_image.unsqueeze(1)

        m1_cnn_features = (self.m1_cnn(m1_image)).flatten(1)
        m2_cnn_features = (self.m2_cnn(m2_image)).flatten(1)

        combined = torch.cat([m1_cnn_features, m2_cnn_features], dim=1)

        return self.classifier(combined)
