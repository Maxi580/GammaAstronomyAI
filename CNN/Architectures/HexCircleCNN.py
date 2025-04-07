import torch
import torch.nn as nn

from CNN.HexCircleLayers.HexCircleConv import HexCircleConv
from CNN.HexCircleLayers.HexCirclePool import HexCirclePool


class HexCircleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            HexCircleConv(1, 16, kernel_size=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            HexCirclePool(1, mode='max'),
            nn.Dropout1d(0.25),

            HexCircleConv(16, 32, kernel_size=5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout1d(0.05),
            
            HexCircleConv(32, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            HexCirclePool(3, mode='avg'),
            nn.Dropout1d(0.15),
        )

    def forward(self, x):
        return self.cnn(x)


class HexCircleNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.m1_cnn = HexCircleCNN()
        self.m2_cnn = HexCircleCNN()

        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 2, 448),
            nn.ReLU(),
            nn.Dropout(0.55),

            nn.Linear(448, 2)
        )

    def forward(self, m1_image, m2_image, measurement_features):
        m1_image = m1_image.unsqueeze(1)
        m2_image = m2_image.unsqueeze(1)
        m1_cnn_features = (self.m1_cnn(m1_image)).flatten(1)
        m2_cnn_features = (self.m2_cnn(m2_image)).flatten(1)

        combined = torch.cat([m1_cnn_features, m2_cnn_features], dim=1)

        return self.classifier(combined)