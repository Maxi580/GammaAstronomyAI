import torch.nn as nn
from CNN.ConvolutionLayers.ConvHex import ConvHex


class HexCNN(nn.Module):
    def __init__(self):
        super(HexCNN, self).__init__()

        self.features = nn.Sequential(
            ConvHex(in_channels=1, out_channels=8, kernel_size=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout1d(0.061173084865384815),

            ConvHex(in_channels=8, out_channels=16, kernel_size=5),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout1d(0.026358832448413604),

            ConvHex(in_channels=16, out_channels=32, kernel_size=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout1d(0.09316383648273836),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(32 * 1039, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(2048, 256),
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
