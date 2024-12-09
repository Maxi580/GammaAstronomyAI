import torch.nn as nn
from arrayClassification.HexLayers.ConvHex import ConvHex


class HexCNN(nn.Module):
    def __init__(self):
        super(HexCNN, self).__init__()

        self.features = nn.Sequential(
            ConvHex(in_channels=1, out_channels=8, kernel_size=3),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout1d(0.5),

            ConvHex(in_channels=8, out_channels=16, kernel_size=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout1d(0.5),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*1039, 4*1039),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(4*1039, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
