import torch
import torch.nn as nn
from CNN.MagicConv.MagicConv import MagicConv


class TelescopeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            MagicConv(1, 8, kernel_size=2),
            nn.GroupNorm(1, 14),
            nn.ReLU(),
            nn.Dropout1d(p=0.3),
        )

    def forward(self, x):
        return self.cnn(x)


class HexMagicNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.m1_cnn = TelescopeCNN()
        self.m2_cnn = TelescopeCNN()

        self.classifier = nn.Sequential(
            nn.Linear(8 * 1039 * 2, 1024),
            nn.GroupNorm(8, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 2),
        )

    def forward(self, m1_image, m2_image, measurement_features):
        m1_image = m1_image.unsqueeze(1)
        m2_image = m2_image.unsqueeze(1)

        m1_cnn_features = (self.m1_cnn(m1_image)).flatten(1)
        m2_cnn_features = (self.m2_cnn(m2_image)).flatten(1)

        combined = torch.cat([m1_cnn_features, m2_cnn_features], dim=1)

        return self.classifier(combined)