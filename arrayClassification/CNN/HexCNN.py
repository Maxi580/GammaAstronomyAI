import torch.nn as nn
from arrayClassification.HexLayers.ConvHex import ConvHex


class HexCNN(nn.Module):
    def __init__(self):
        super(HexCNN, self).__init__()

        self.features = nn.Sequential(
            ConvHex(in_channels=1, out_channels=16, kernel_size=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout1d(0.3),

            ConvHex(in_channels=16, out_channels=32, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout1d(0.4),

            ConvHex(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout1d(0.5),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(66496, 1024),  # 64*1039
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
