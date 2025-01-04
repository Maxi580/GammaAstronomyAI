import torch
import torch.nn as nn
from CNN.ConvolutionLayers.ConvHex import ConvHex

NUM_OF_HEXAGONS = 1183
CNN_REDUCE_OUTPUT_FEATURES = 2048
NON_IMAGE_FEATURES = 59


class CombinedNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Gets 32 * Num_of_hexagons features
        self.cnn_features = nn.Sequential(
            ConvHex(in_channels=1, out_channels=8, kernel_size=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout1d(0.061173084865384815),

            ConvHex(in_channels=8, out_channels=16, kernel_size=5),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout1d(0.026358832448413604),

            ConvHex(in_channels=16, out_channels=32, kernel_size=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout1d(0.09316383648273836),
        )

        self.cnn_reducer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * NUM_OF_HEXAGONS, CNN_REDUCE_OUTPUT_FEATURES),  # Reduce cnn output to 2048 features
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Combine all features
        self.classifier = nn.Sequential(
            nn.Linear(CNN_REDUCE_OUTPUT_FEATURES * 2 + NON_IMAGE_FEATURES, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 2)
        )

    def forward(self, m1_image, m2_image, features):
        # Process images
        m1_features = self.cnn_reducer(self.cnn_features(m1_image))
        m2_features = self.cnn_reducer(self.cnn_features(m2_image))

        # Process numerical features
        feature_output = self.feature_net(features)

        # Combine all features
        combined = torch.cat([m1_features, m2_features, feature_output], dim=1)

        return self.classifier(combined)
