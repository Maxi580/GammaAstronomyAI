import torch
import torch.nn as nn

from CNN.HexCircleLayers.HexCircleConv import HexCircleConv
from CNN.HexCircleLayers.HexCirclePool import HexCirclePool


class HexCircleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            HexCircleConv(1, 8, kernel_size=5, n_pixels=1039),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            HexCirclePool(1, 1039, mode='avg'),
            nn.Dropout1d(0.06664789059481752),

            HexCircleConv(8, 40, kernel_size=2, n_pixels=163),
            nn.BatchNorm1d(40),
            nn.ReLU(),
            HexCirclePool(3, 163, mode='max'),
            nn.Dropout1d(0.08472856027638453),
        )

    def forward(self, x):
        return self.cnn(x)


class HexCircleNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.m1_cnn = HexCircleCNN()
        self.m2_cnn = HexCircleCNN()

        self.classifier = nn.Sequential(
            nn.Linear(40 * 7 * 2, 446),
            nn.ReLU(),
            nn.Dropout(0.40370243392755745),

            nn.Linear(446, 330),
            nn.ReLU(),
            nn.Dropout(0.25437384755300974),

            nn.Linear(330, 192),
            nn.ReLU(),
            nn.Dropout(0.37123397581958945),

            nn.Linear(192, 2)
        )

    def forward(self, m1_image, m2_image, measurement_features):
        m1_image = m1_image.unsqueeze(1)
        m2_image = m2_image.unsqueeze(1)
        m1_cnn_features = (self.m1_cnn(m1_image)).flatten(1)
        m2_cnn_features = (self.m2_cnn(m2_image)).flatten(1)

        combined = torch.cat([m1_cnn_features, m2_cnn_features], dim=1)

        return self.classifier(combined)