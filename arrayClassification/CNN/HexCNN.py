import torch.nn as nn
from arrayClassification.HexLayers.ConvHex import ConvHex
from arrayClassification.HexLayers.ResidualConvHex import ResidualConvHex

class HexCNN(nn.Module):
    def __init__(self):
        super(HexCNN, self).__init__()

        self.features = nn.Sequential(
            ConvHex(in_channels=1, out_channels=32, kernel_size=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout1d(0.1),

            ResidualConvHex(32),

            ConvHex(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout1d(0.1),

            ResidualConvHex(64),

            ConvHex(in_channels=64, out_channels=128, kernel_size=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout1d(0.1),

            ResidualConvHex(128),

            ConvHex(128, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout1d(0.1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(128 * 1039, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
