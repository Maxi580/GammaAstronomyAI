import os
import torch
import torch.nn as nn
from CNN.MagicConv.MagicConv import MagicConv
import pickle

from random_forest.random_forest import train_random_forest_classifier, plot_results

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RF_MODEL_DIR = os.path.join(BASE_DIR, "models")
RF_MODEL_PATH = os.path.join(RF_MODEL_DIR, "rf_model.pkl")
PROTON_FILE = os.path.join(BASE_DIR, "../magic-protons.parquet")
GAMMA_FILE = os.path.join(BASE_DIR, "../magic-gammas-new.parquet")


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            MagicConv(1, 4, kernel_size=2),
            nn.GroupNorm(2, 4),
            nn.ReLU(),
            nn.Dropout1d(p=0.3),
        )

    def forward(self, x):
        return self.cnn(x)


class HybridNet(nn.Module):
    def __init__(self, num_rf_outputs=1):
        super().__init__()

        if os.path.exists(RF_MODEL_PATH):
            print(f"Random Forest model already exists at {RF_MODEL_PATH}")
            with open(RF_MODEL_PATH, 'rb') as f:
                self.rf_model = pickle.load(f)
        else:
            print(f"Random Forest model not found at {RF_MODEL_PATH}, training new model...")
            results = train_random_forest_classifier(
                proton_file=PROTON_FILE,
                gamma_file=GAMMA_FILE,
                path=RF_MODEL_PATH,
                test_size=0.3,
            )
            self.rf_model = results['model']
            plot_results(results)
            print("RF Training and evaluation complete!")

        self.num_rf_outputs = num_rf_outputs

        self.m1_cnn = CNN()
        self.m2_cnn = CNN()

        self.cnn_classifier = nn.Sequential(
            nn.Linear(4 * 1039 * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 2),
        )

        self.ensemble_layer = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(8, 2)
        )

    def forward(self, m1_image, m2_image, features):
        m1_image = m1_image.unsqueeze(1)
        m2_image = m2_image.unsqueeze(1)

        m1_cnn_features = (self.m1_cnn(m1_image)).flatten(1)
        m2_cnn_features = (self.m2_cnn(m2_image)).flatten(1)
        combined_cnn = torch.cat([m1_cnn_features, m2_cnn_features], dim=1)
        cnn_prediction = self.cnn_classifier(combined_cnn)

        features_np = features.cpu().detach().numpy()
        rf_probabilities = self.rf_model.predict_proba(features_np)
        rf_prediction = torch.tensor(rf_probabilities, dtype=torch.float32, device=m1_image.device)

        combined_predictions = torch.cat([cnn_prediction, rf_prediction], dim=1)
        final_output = self.ensemble_layer(combined_predictions)

        return final_output
