import torch
import torch.nn as nn


class Simple1dCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Dropout1d(0.2),

            nn.Conv1d(8, 16, kernel_size=5),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Dropout1d(0.2),
        )

    def forward(self, x):
        return self.cnn(x)


class Simple1dNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.m1_cnn = Simple1dCNN()
        self.m2_cnn = Simple1dCNN()

        self.classifier = nn.Sequential(
            nn.Linear(16 * 115 * 21, 512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 2)
        )

    def forward(self, m1_image, m2_image, measurement_features):
        m1_image = m1_image.unsqueeze(1)
        m2_image = m2_image.unsqueeze(1)
        m1_cnn_features = (self.m1_cnn(m1_image)).flatten(1)
        m2_cnn_features = (self.m2_cnn(m2_image)).flatten(1)

        combined = torch.cat([m1_cnn_features, m2_cnn_features], dim=1)

        return self.classifier(combined)