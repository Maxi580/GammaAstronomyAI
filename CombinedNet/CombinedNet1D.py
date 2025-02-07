import torch
import torch.nn as nn

from CNN.ConvolutionLayers.ConvHex import ConvHex

NUM_OF_HEXAGONS = 1039
NUM_FEATURES = 59


def resize_input(image):
    """Arrays are 1183 long, however the last 144 are always 0"""
    return image[:, :, :NUM_OF_HEXAGONS]


class TelescopeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 15, kernel_size=1),
            nn.GroupNorm(3, 15),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout1d(0.29336358072696656),

            nn.Conv1d(15, 17, kernel_size=3),
            nn.GroupNorm(1, 17),
            nn.ReLU(),
            nn.Dropout1d(0.23143340912286325),

            nn.Conv1d(17, 42, kernel_size=1),
            nn.GroupNorm(3, 42),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout1d(0.07056170312260002),
        )

    def forward(self, x):
        return self.cnn(x)


class CombinedNet1D(nn.Module):
    def __init__(self):
        super().__init__()

        self.m1_cnn = TelescopeCNN()
        self.m2_cnn = TelescopeCNN()

        self.classifier = nn.Sequential(
            nn.Linear(2016, 512),
            nn.GroupNorm(64, 512),
            nn.ReLU(),
            nn.Dropout(0.15421243805878218),

            nn.Linear(512, 512),
            nn.GroupNorm(64, 512),
            nn.ReLU(),
            nn.Dropout(0.25303929962985794),

            nn.Linear(512, 256),
            nn.GroupNorm(32, 256),
            nn.ReLU(),
            nn.Dropout(0.25303929962985794),

            nn.Linear(256, 2)
            )

    def forward(self, m1_image, m2_image, measurement_features):
        # Add Channel Dimension
        m1_image = m1_image.unsqueeze(1)
        m2_image = m2_image.unsqueeze(1)
        m1_features = self.m1_cnn(m1_image)
        m2_features = self.m2_cnn(m2_image)
        m1_features = m1_features.flatten(1)
        m2_features = m2_features.flatten(1)

        combined = torch.cat([m1_features, m2_features], dim=1)
        
        return self.classifier(combined)