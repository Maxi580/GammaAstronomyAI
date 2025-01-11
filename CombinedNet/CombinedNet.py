import pandas as pd
import torch
import torch.nn as nn

NUM_OF_HEXAGONS = 1039
NUM_FEATURES = 59


def resize_input(image):
    """Arrays are 1183 long, however the last 144 are always 0"""
    return image[:, :, :NUM_OF_HEXAGONS]


class TelescopeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=3),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Dropout1d(0.2),

            nn.Conv1d(4, 8, kernel_size=2),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout1d(0.2),
        )

    def forward(self, x):
        return self.cnn(x)


class CombinedNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.m1_cnn = TelescopeCNN()
        self.m2_cnn = TelescopeCNN()

        self.classifier = nn.Sequential(
            nn.Linear(8 * 1036 * 2, 1792),
            nn.BatchNorm1d(1792),
            nn.ReLU(),
            nn.Dropout(0.09210354957166011),

            nn.Linear(1792, 448),
            nn.BatchNorm1d(448),
            nn.ReLU(),
            nn.Dropout(0.2126585523934971),

            nn.Linear(448, 2)
        )

    def forward(self, m1_image, m2_image, measurement_features):
        # Add Channel Dimension
        m1_image = m1_image.unsqueeze(1)
        m2_image = m2_image.unsqueeze(1)
        # Throw away 144 0 Values at the end
        m1_image = resize_input(m1_image)
        m2_image = resize_input(m2_image)
        m1_features = self.m1_cnn(m1_image)
        m2_features = self.m2_cnn(m2_image)
        m1_features = m1_features.flatten(1)
        m2_features = m2_features.flatten(1)

        combined = torch.cat([m1_features, m2_features], dim=1)

        return self.classifier(combined)
