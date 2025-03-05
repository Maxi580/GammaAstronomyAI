import torch
import torch.nn as nn

from CNN.HexCircleLayers.HexCircleConv import HexCircleConv
from CNN.HexCircleLayers.HexCirclePool import HexCirclePool


class HexCircleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            HexCircleConv(1, 8, kernel_size=2),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout1d(0.05027208056732827),

            HexCircleConv(8, 20, kernel_size=5),
            nn.BatchNorm1d(20),
            nn.ReLU(),
            HexCirclePool(2, mode='max'),
            nn.Dropout1d(0.21308784431475414),
            
            HexCircleConv(20, 47, kernel_size=3),
            nn.BatchNorm1d(47),
            nn.ReLU(),
            HexCirclePool(1, mode='max'),
            nn.Dropout1d(0.09910993376674002),
        )

    def forward(self, x):
        return self.cnn(x)


class HexCircleNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.m1_cnn = HexCircleCNN()
        self.m2_cnn = HexCircleCNN()

        self.classifier = nn.Sequential(
            nn.Linear(47 * 13 * 2, 398),
            nn.ReLU(),
            nn.Dropout(0.4283984386317653),

            nn.Linear(398, 62),
            nn.ReLU(),
            nn.Dropout(0.20237090903445126),

            nn.Linear(62, 9),
            nn.ReLU(),
            nn.Dropout(0.40145111076242346),

            nn.Linear(9, 2)
        )

    def forward(self, m1_image, m2_image, measurement_features):
        m1_image = m1_image.unsqueeze(1)
        m2_image = m2_image.unsqueeze(1)
        m1_cnn_features = (self.m1_cnn(m1_image)).flatten(1)
        m2_cnn_features = (self.m2_cnn(m2_image)).flatten(1)

        combined = torch.cat([m1_cnn_features, m2_cnn_features], dim=1)

        return self.classifier(combined)