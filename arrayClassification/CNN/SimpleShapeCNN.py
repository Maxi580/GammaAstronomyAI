import torch.nn as nn


class SimpleShapeCNN(nn.Module):
    def __init__(self):
        super(SimpleShapeCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=0),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout1d(0.1),
            nn.Conv1d(16, 32, kernel_size=5, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout1d(0.15),
            nn.Conv1d(32, 64, kernel_size=3, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout1d(0.2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8128, 128),  # 64*129 because of pooling
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            # TODO: the out features need to be adapted according to number of labels
            nn.Linear(32, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
