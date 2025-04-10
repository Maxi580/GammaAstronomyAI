import os
import time
import torch
import torch.nn as nn
import pickle

from CNN.Architectures import HexMagicNet
from TrainingPipeline.TrainingSupervisor import TrainingSupervisor
from TrainingPipeline.Datasets import MagicDataset
from random_forest.random_forest import train_random_forest_classifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HYBRID_DIR = os.path.join(BASE_DIR, "HybridSystem")
CNN_MODEL_PATH = os.path.join(HYBRID_DIR, "cnn_model.pth")
RF_MODEL_PATH = os.path.join(HYBRID_DIR, "rf_model.pkl")
ENSEMBLE_MODEL_PATH = os.path.join(HYBRID_DIR, "ensemble_model.pth")

PROTON_FILE = "magic-protons.parquet"
GAMMA_FILE = "magic-gammas-new.parquet"

os.makedirs(HYBRID_DIR, exist_ok=True)


def train_cnn_model(epochs=30):
    print("Training CNN component...")
    cnn_nametag = f"CNN_Component_{time.strftime('%d-%m-%Y_%H-%M-%S')}"
    cnn_output_dir = os.path.join(HYBRID_DIR, cnn_nametag)

    dataset = MagicDataset(PROTON_FILE, GAMMA_FILE)
    supervisor = TrainingSupervisor("hexmagicnet", dataset, cnn_output_dir,
                                    debug_info=True, save_model=True,
                                    save_debug_data=True, early_stopping=False)

    print(f"CNN model has {supervisor._count_trainable_weights()} trainable weights.")
    supervisor.LEARNING_RATE = 5.269632147047427e-06
    supervisor.WEIGHT_DECAY = 0.00034049323130326087
    supervisor.BATCH_SIZE = 64
    supervisor.GRAD_CLIP_NORM = 0.7168560391358462

    supervisor.train_model(epochs)

    # Save the model to the standardized path
    torch.save(supervisor.model.state_dict(), CNN_MODEL_PATH)
    print(f"CNN model saved to {CNN_MODEL_PATH}")

    return CNN_MODEL_PATH


def train_rf_model():
    print("Training Random Forest component...")
    results = train_random_forest_classifier(
        proton_file=PROTON_FILE,
        gamma_file=GAMMA_FILE,
        path=RF_MODEL_PATH,
        test_size=0.3,
    )
    print(f"Random Forest model saved to {RF_MODEL_PATH}")
    return RF_MODEL_PATH


class EnsembleModel(nn.Module):
    def __init__(self, cnn_model_path, rf_model_path, device='gpu'):
        super().__init__()

        self.cnn_model = HexMagicNet()
        self.cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=device))

        for param in self.cnn_model.parameters():
            param.requires_grad = False

        with open(rf_model_path, 'rb') as f:
            self.rf_model = pickle.load(f)
            self.rf_model.verbose = 0

        self.ensemble_layer = nn.Sequential(
            nn.Linear(4, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(32, 2)
        )

        self.cnn_weight = nn.Parameter(torch.tensor([0.5]), requires_grad=True)

    def forward(self, m1_image, m2_image, features):
        self.cnn_model.eval()
        with torch.no_grad():
            cnn_pred = self.cnn_model(m1_image, m2_image, features)

        features_np = features.cpu().detach().numpy()
        rf_probs = self.rf_model.predict_proba(features_np)
        rf_pred = torch.tensor(rf_probs, dtype=torch.float32, device=features.device)

        rf_weight = 1.0 - self.cnn_weight

        combined_preds = torch.cat([
            cnn_pred * self.cnn_weight,
            rf_pred * rf_weight
        ], dim=1)

        return self.ensemble_layer(combined_preds)


def train_ensemble_model(cnn_path, rf_path, epochs=5):
    print("Training ensemble combiner model...")
    ensemble_nametag = f"Ensemble_Combiner_{time.strftime('%d-%m-%Y_%H-%M-%S')}"
    ensemble_output_dir = os.path.join(HYBRID_DIR, ensemble_nametag)

    dataset = MagicDataset(PROTON_FILE, GAMMA_FILE)

    ensemble = EnsembleModel(cnn_path, rf_path)
    supervisor = TrainingSupervisor("custom", dataset, ensemble_output_dir,
                                    debug_info=True, save_model=True,
                                    save_debug_data=True, early_stopping=False)

    supervisor.model = ensemble.to(supervisor.device)

    supervisor.LEARNING_RATE = 1e-3
    supervisor.WEIGHT_DECAY = 1e-3
    supervisor.GRAD_CLIP_NORM = 1.0

    print(f"Ensemble model has {sum(p.numel() for p in ensemble.ensemble_layer.parameters() if p.requires_grad)} "
          f"trainable weights.")
    supervisor.train_model(epochs)

    torch.save(ensemble.state_dict(), ENSEMBLE_MODEL_PATH)
    print(f"Ensemble model saved to {ENSEMBLE_MODEL_PATH}")

    return ENSEMBLE_MODEL_PATH


def main():
    if os.path.exists(CNN_MODEL_PATH):
        print(f"CNN model already exists at {CNN_MODEL_PATH}")
        cnn_path = CNN_MODEL_PATH
    else:
        cnn_path = train_cnn_model()

    if os.path.exists(RF_MODEL_PATH):
        print(f"Random Forest model already exists at {RF_MODEL_PATH}")
        rf_path = RF_MODEL_PATH
    else:
        rf_path = train_rf_model()

    ensemble_path = train_ensemble_model(cnn_path, rf_path)

    print("Complete hybrid system trained successfully!")
    print(f"CNN Model: {cnn_path}")
    print(f"RF Model: {rf_path}")
    print(f"Ensemble Model: {ensemble_path}")


if __name__ == "__main__":
    main()
