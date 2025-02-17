import torch
import torch.nn as nn
from CNN.MagicConv.MagicConv import MagicConv


class TelescopeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            MagicConv(1, 15, kernel_size=3),
            nn.GroupNorm(3, 15),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout1d(p=0.29336358072696656),

            MagicConv(15, 17, kernel_size=3, pooling=True, pooling_kernel_size=2, pooling_cnt=1),
            nn.GroupNorm(1, 17),
            nn.ReLU(),
            nn.Dropout1d(p=0.23143340912286325),

            MagicConv(17, 42, kernel_size=3, pooling=True, pooling_kernel_size=2, pooling_cnt=1),
            nn.GroupNorm(3, 42),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout1d(p=0.07056170312260002),
        )

    def forward(self, x):
        return self.cnn(x)


class HexMagicNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.m1_cnn = TelescopeCNN()
        self.m2_cnn = TelescopeCNN()

        self.classifier = nn.Sequential(
            nn.Linear(21756, 512),
            nn.GroupNorm(64, 512),
            nn.ReLU(),
            nn.Dropout(p=0.15421243805878218),

            nn.Linear(512, 512),
            nn.GroupNorm(64, 512),
            nn.ReLU(),
            nn.Dropout(p=0.25303929962985794),

            nn.Linear(512, 256),
            nn.GroupNorm(32, 256),
            nn.ReLU(),
            nn.Dropout(p=0.25303929962985794),

            nn.Linear(256, 2)
        )

    def forward(self, m1_image, m2_image, measurement_features):
        m1_image = m1_image.unsqueeze(1)
        m2_image = m2_image.unsqueeze(1)

        m1_cnn_features = (self.m1_cnn(m1_image)).flatten(1)
        m2_cnn_features = (self.m2_cnn(m2_image)).flatten(1)

        combined = torch.cat([m1_cnn_features, m2_cnn_features], dim=1)

        return self.classifier(combined)
