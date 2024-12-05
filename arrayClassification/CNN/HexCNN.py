import torch.nn as nn

from arrayClassification.HexLayers.ConvHex import ConvHex


class HexCNN(nn.Module):
    def __init__(self):
        super(HexCNN, self).__init__()

        self.features = nn.Sequential(
            ConvHex(in_channels=1, out_channels=16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout1d(0.1),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout1d(0.15),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16576, 32),  # 64*129 (1039 / 2 / 2 = 129)
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
