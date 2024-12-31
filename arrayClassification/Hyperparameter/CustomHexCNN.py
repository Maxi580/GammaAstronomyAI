import torch.nn as nn
from arrayClassification.HexLayers.ConvHex import ConvHex


class CustomHexCNN(nn.Module):
    def __init__(
            self,
            channels1=8,
            channels2=16,
            channels3=32,
            kernel_size1=4,
            kernel_size2=3,
            kernel_size3=2,
            dropout_conv1=0.1,
            dropout_conv2=0.1,
            dropout_conv3=0.1,
            linear1_size=1024,
            linear2_size=256,
            linear3_size=64,
            dropout_linear1=0.2,
            dropout_linear2=0.2,
            dropout_linear3=0.1
    ):
        super(CustomHexCNN, self).__init__()

        self.features = nn.Sequential(
            ConvHex(in_channels=1, out_channels=8, kernel_size=kernel_size1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout1d(dropout_conv1),

            ConvHex(in_channels=8, out_channels=16, kernel_size=kernel_size2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout1d(dropout_conv2),

            ConvHex(in_channels=16, out_channels=32, kernel_size=kernel_size3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout1d(dropout_conv3),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 1039, linear1_size),
            nn.ReLU(),
            nn.Dropout(dropout_linear1),

            nn.Linear(linear1_size, linear2_size),
            nn.ReLU(),
            nn.Dropout(dropout_linear2),

            nn.Linear(linear2_size, linear3_size),
            nn.ReLU(),
            nn.Dropout(dropout_linear3),

            nn.Linear(linear3_size, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
