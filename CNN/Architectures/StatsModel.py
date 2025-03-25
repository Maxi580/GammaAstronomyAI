import torch
import torch.nn as nn


def get_batch_stats(img_batch):
    return torch.stack([
        img_batch.mean(dim=1),
        img_batch.std(dim=1),
        (img_batch < 0).float().mean(dim=1),
        img_batch.min(dim=1).values,
        img_batch.max(dim=1).values,
        (img_batch ** 2).mean(dim=1),
        (img_batch != 0).float().mean(dim=1),
        torch.quantile(img_batch, 0.25, dim=1),
        torch.quantile(img_batch, 0.5, dim=1),
        torch.quantile(img_batch, 0.75, dim=1)
    ], dim=1)


class StatsMagicNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(18, 32),
            nn.Linear(20, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Dropout(0.4),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Dropout(0.4),

            nn.Linear(16, 2)
        )

    def forward(self, m1_image, m2_image, measurement_features):
        m1_stats = get_batch_stats(m1_image)
        m2_stats = get_batch_stats(m2_image)
        return self.classifier(torch.cat([m1_stats, m2_stats], dim=1))