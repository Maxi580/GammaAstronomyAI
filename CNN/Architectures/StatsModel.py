import torch
import torch.nn as nn


def get_stats(img):
    return torch.tensor([
        img.mean(),
        img.std(),
        (img < 0).float().mean(),
        img.min(),
        img.max(),
        (img ** 2).mean(),
        torch.quantile(img, 0.25),
        torch.quantile(img, 0.5),
        torch.quantile(img, 0.75)
    ], device=img.device)


class StatsMagicNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(18, 9),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(9, 2),
        )

    def forward(self, m1_image, m2_image, measurement_features):
        m1_stats = get_stats(m1_image)
        m2_stats = get_stats(m2_image)
        combined = torch.cat([m1_stats, m2_stats])
        return self.classifier(combined.unsqueeze(0))
