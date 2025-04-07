import torch
import torch.nn as nn


class Simple1dCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=4),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout1d(0.05),

            nn.Conv1d(16, 32, kernel_size=8),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AvgPool1d(8),
            nn.Dropout1d(0.25),
        )

    def forward(self, x):
        return self.cnn(x)


class Simple1dNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.m1_cnn = Simple1dCNN()
        self.m2_cnn = Simple1dCNN()

        self.classifier = nn.Sequential(
            nn.Linear(32 * 128 * 2, 656),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(656, 2)
        )

    def forward(self, m1_image, m2_image, measurement_features):
        m1_image = m1_image.unsqueeze(1)
        m2_image = m2_image.unsqueeze(1)
        m1_cnn_features = (self.m1_cnn(m1_image)).flatten(1)
        m2_cnn_features = (self.m2_cnn(m2_image)).flatten(1)

        combined = torch.cat([m1_cnn_features, m2_cnn_features], dim=1)

        return self.classifier(combined)