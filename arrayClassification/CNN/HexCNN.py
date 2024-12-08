import torch.nn as nn
from arrayClassification.HexLayers.ConvHex import ConvHex


class HexCNN(nn.Module):
    def __init__(self):
        super(HexCNN, self).__init__()

        self.features = nn.Sequential(
            ConvHex(in_channels=1, out_channels=16, kernel_size=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout1d(0.6),

            ConvHex(in_channels=16, out_channels=32, kernel_size=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout1d(0.6),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(33248, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
