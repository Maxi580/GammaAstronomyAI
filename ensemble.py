import os
import time
import torch
import torch.nn as nn
import pickle

from CNN.Architectures import HexMagicNet
from TrainingPipeline.TrainingSupervisor import TrainingSupervisor
from TrainingPipeline.Datasets import MagicDataset
from dataset_splitter import split_parquet_files
from random_forest.random_forest import train_random_forest_classifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HYBRID_DIR = os.path.join(BASE_DIR, "HybridSystem")
CNN_MODEL_PATH = os.path.join(HYBRID_DIR, "cnn_model.pth")
RF_MODEL_PATH = os.path.join(HYBRID_DIR, "rf_model.pkl")
ENSEMBLE_MODEL_PATH = os.path.join(HYBRID_DIR, "ensemble_model.pth")
os.makedirs(HYBRID_DIR, exist_ok=True)

PROTON_FILE = "magic-protons.parquet"
GAMMA_FILE = "magic-gammas-new.parquet"
RANDOM_SEED = 42


def train_cnn_model(dataset, epochs=30):
    cnn_supervisor = TrainingSupervisor("hexmagicnet", dataset, HYBRID_DIR,
                                        debug_info=True, save_model=True, val_split=0.1,
                                        save_debug_data=True, early_stopping=False)

    cnn_supervisor.LEARNING_RATE = 5.269632147047427e-06
    cnn_supervisor.WEIGHT_DECAY = 0.00034049323130326087
    cnn_supervisor.BATCH_SIZE = 64
    cnn_supervisor.GRAD_CLIP_NORM = 0.7168560391358462

    cnn_supervisor.train_model(epochs)
    torch.save(cnn_supervisor.model.state_dict(), CNN_MODEL_PATH)

    return HYBRID_DIR


def train_rf_model(dataset):
    print("Training Random Forest component...")
    train_random_forest_classifier(
        dataset=dataset,
        path=RF_MODEL_PATH,
        test_size=0.1,
    )
    print(f"Random Forest model saved to {RF_MODEL_PATH}")
    return RF_MODEL_PATH


class EnsembleModel(nn.Module):
    def __init__(self, cnn_model_path, rf_model_path):
        super().__init__()

        self.cnn_model = HexMagicNet()
        self.cnn_model.load_state_dict(torch.load(cnn_model_path, weights_only=True))

        for param in self.cnn_model.parameters():
            param.requires_grad = False

        with open(rf_model_path, 'rb') as f:
            self.rf_model = pickle.load(f)
            self.rf_model.verbose = 0

        self.ensemble_layer = nn.Sequential(
            nn.Linear(4, 211),
            nn.ReLU(),
            nn.Dropout(p=0.043191866396247024),

            nn.Linear(211, 234),
            nn.BatchNorm1d(234),
            nn.LeakyReLU(negative_slope=0.08605475199145621),
            nn.Dropout(p=0.22001088958156756),

            nn.Linear(234, 180),
            nn.ReLU(),
            nn.Dropout(p=0.12375715238499085),

            nn.Linear(180, 2)
        )

    def forward(self, m1_image, m2_image, features):
        self.cnn_model.eval()
        with torch.no_grad():
            cnn_pred = self.cnn_model(m1_image, m2_image, features)

        features_np = features.cpu().detach().numpy()
        rf_probs = self.rf_model.predict_proba(features_np)
        rf_pred = torch.tensor(rf_probs, dtype=torch.float32, device=features.device)

        combined_preds = torch.cat([
            cnn_pred,
            rf_pred
        ], dim=1)

        return self.ensemble_layer(combined_preds)


def train_ensemble_model(cnn_path, rf_path, dataset, epochs=10):
    print("Training ensemble combiner model...")
    ensemble_nametag = f"Ensemble_Combiner_{time.strftime('%d-%m-%Y_%H-%M-%S')}"
    ensemble_output_dir = os.path.join(HYBRID_DIR, ensemble_nametag)

    ensemble = EnsembleModel(cnn_path, rf_path)
    supervisor = TrainingSupervisor("custom", dataset, ensemble_output_dir,
                                    debug_info=True, save_model=True, val_split=0.1,
                                    save_debug_data=True, early_stopping=False)

    supervisor.model = ensemble.to(supervisor.device)

    supervisor.LEARNING_RATE = 2.156799027646505e-05
    supervisor.WEIGHT_DECAY = 3.5287444954790375e-05
    supervisor.GRAD_CLIP_NORM = 0.6637686241010907

    print(f"Ensemble model has {sum(p.numel() for p in ensemble.ensemble_layer.parameters() if p.requires_grad)} "
          f"trainable weights.")
    supervisor.train_model(epochs)

    torch.save(ensemble.state_dict(), ENSEMBLE_MODEL_PATH)
    print(f"Ensemble model saved to {ENSEMBLE_MODEL_PATH}")

    return ENSEMBLE_MODEL_PATH


def train_hybrid_system(ensemble_epochs=5, val_split=0.3):

    print("Splitting datasets into train/validation sets...")
    data_dir = os.path.join(HYBRID_DIR, "data")
    file_paths = split_parquet_files(
        PROTON_FILE,
        GAMMA_FILE,
        data_dir,
        val_split=val_split,
        random_seed=RANDOM_SEED
    )

    ensemble_dataset = MagicDataset(
        file_paths['val']['proton'],
        file_paths['val']['gamma']
    )

    if os.path.exists(CNN_MODEL_PATH):
        print(f"CNN model already exists at {CNN_MODEL_PATH}. Skipping training.")
        cnn_path = CNN_MODEL_PATH
    else:
        print(f"Didnt find cnn")
        return

    if os.path.exists(RF_MODEL_PATH):
        print(f"Random Forest model already exists at {RF_MODEL_PATH}. Skipping training.")
        rf_path = RF_MODEL_PATH
    else:
        print(f"Didnt find rf ")
        return
    print("Training ensemble model...")
    train_ensemble_model(cnn_path, rf_path, ensemble_dataset, epochs=ensemble_epochs)


def main():
    train_hybrid_system(
        ensemble_epochs=10,
    )


if __name__ == "__main__":
    main()
