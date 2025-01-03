import torch.nn as nn


class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=0),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout1d(0.1),
            nn.Conv1d(32, 64, kernel_size=3, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout1d(0.15),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8192, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            # TODO: the out features need to be adapted according to number of labels
            nn.Linear(16, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
