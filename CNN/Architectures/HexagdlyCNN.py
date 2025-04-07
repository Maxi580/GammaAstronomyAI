import torch
import torch.nn as nn
import hexagdly


class HexagdlyCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            hexagdly.Conv2d(1, 16, kernel_size=4),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            hexagdly.MaxPool2d(1, 2),
            nn.Dropout2d(0.2),
            
            
            hexagdly.Conv2d(16, 48, kernel_size=5),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            hexagdly.MaxPool2d(1, 3),
            nn.Dropout2d(0.35),
        )

    def forward(self, x):
        return self.cnn(x)


class HexagdlyNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.m1_cnn = HexagdlyCNN()
        self.m2_cnn = HexagdlyCNN()

        self.classifier = nn.Sequential(
            nn.Linear((6 * 7) * 48 * 2, 288),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(288, 192),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(192, 96),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(96, 2)
        )

    def forward(self, m1_image, m2_image, measurement_features):
        m1_image = m1_image.unsqueeze(1)
        m2_image = m2_image.unsqueeze(1)
        m1_cnn_features = (self.m1_cnn(m1_image)).flatten(1)
        m2_cnn_features = (self.m2_cnn(m2_image)).flatten(1)

        combined = torch.cat([m1_cnn_features, m2_cnn_features], dim=1)

        return self.classifier(combined)