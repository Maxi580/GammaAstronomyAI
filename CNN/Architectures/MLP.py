import torch
import torch.nn as nn

NUM_FEATURES = 59


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(2 * 1039, 768),
            nn.GroupNorm(32, 768),
            nn.ReLU(),
            nn.Dropout(0.4936012148325484),

            nn.Linear(768, 384),
            nn.GroupNorm(16, 384),
            nn.ReLU(),
            nn.Dropout(0.5340424416693821),

            nn.Linear(384, 128),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Dropout(0.21686133182097764),

            nn.Linear(128, 2)
        )

    def forward(self, m1_image, m2_image, measurement_features):
        output = torch.cat([m1_image, m2_image], dim=1)

        return self.classifier(output)
