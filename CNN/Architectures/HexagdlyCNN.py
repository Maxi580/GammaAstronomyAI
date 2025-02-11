import torch
import torch.nn as nn
import hexagdly


class HexagdlyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            
            
            nn.Conv2d(8, 16, kernel_size=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            nn.Conv2d(16, 32, kernel_size=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # HexCirclePool(1, 163, mode='max'),
            nn.Dropout2d(0.2),

            # HexConv(18, 36, kernel_size=1, n_pixels=1039),
            # nn.GroupNorm(18, 36),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
            # nn.Dropout2d(0.35031261571845096),
        )

    def forward(self, x):
        return self.cnn(x)


class HexagdlyNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.m1_cnn = HexagdlyCNN()
        self.m2_cnn = HexagdlyCNN()

        self.classifier = nn.Sequential(
            nn.Linear(8 * 8 * 32 * 2, 1024),
            # nn.GroupNorm(32, 768),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(1024, 256),
            # nn.GroupNorm(16, 384),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 64),
            # nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 2)
        )

    def forward(self, m1_image, m2_image, measurement_features):
        m1_image = m1_image.unsqueeze(1)
        m2_image = m2_image.unsqueeze(1)
        m1_cnn_features = (self.m1_cnn(m1_image)).flatten(1)
        m2_cnn_features = (self.m2_cnn(m2_image)).flatten(1)

        combined = torch.cat([m1_cnn_features, m2_cnn_features], dim=1)

        return self.classifier(combined)