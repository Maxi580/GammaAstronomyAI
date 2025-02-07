import torch
import torch.nn as nn

NUM_FEATURES = 59


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(2 * 1039, 1536),
            nn.GroupNorm(32, 1536),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(1536, 1024),
            nn.GroupNorm(32, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(1024, 768),
            nn.GroupNorm(32, 768),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(768, 512),
            nn.GroupNorm(16, 512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.GroupNorm(8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 2)
        )

    def forward(self, m1_image, m2_image, measurement_features):
        output = torch.cat([m1_image, m2_image], dim=1)

        return self.classifier(output)
