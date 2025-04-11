import os
import time
import torch
import torch.nn as nn
import pickle
import numpy as np

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

TEST_SIZE = 0.2
RANDOM_SEED = 42


def split_magic_dataset(proton_file, gamma_file, test_size=0.2, random_state=42, **dataset_kwargs):
    """
    Simple function to split a MagicDataset into training and test sets.

    Args:
        proton_file: Path to proton file
        gamma_file: Path to gamma file
        test_size: Fraction of dataset to use for testing (default: 0.2)
        random_state: Random seed for reproducibility
        **dataset_kwargs: Additional arguments for MagicDataset constructor

    Returns:
        train_dataset, test_dataset (both are MagicDataset instances)
    """
    # Create the base dataset
    base_dataset = MagicDataset(proton_file, gamma_file, **dataset_kwargs)

    # Set random seed for reproducibility
    np.random.seed(random_state)

    # Generate random indices for protons
    proton_indices = np.random.permutation(base_dataset.n_protons)
    proton_split = int(base_dataset.n_protons * (1 - test_size))

    train_proton_indices = proton_indices[:proton_split]
    test_proton_indices = proton_indices[proton_split:]

    # Generate random indices for gammas
    gamma_indices = np.random.permutation(base_dataset.n_gammas)
    gamma_split = int(base_dataset.n_gammas * (1 - test_size))

    train_gamma_indices = gamma_indices[:gamma_split]
    test_gamma_indices = gamma_indices[gamma_split:]

    # Create train dataset with selected samples
    train_dataset = MagicDataset(
        proton_file,
        gamma_file,
        **dataset_kwargs
    )

    # Replace the proton and gamma data with the selected samples
    train_dataset.proton_data = base_dataset.proton_data.iloc[train_proton_indices].reset_index(drop=True)
    train_dataset.gamma_data = base_dataset.gamma_data.iloc[train_gamma_indices].reset_index(drop=True)
    train_dataset.n_protons = len(train_dataset.proton_data)
    train_dataset.n_gammas = len(train_dataset.gamma_data)
    train_dataset.length = train_dataset.n_protons + train_dataset.n_gammas

    # Create test dataset with selected samples
    test_dataset = MagicDataset(
        proton_file,
        gamma_file,
        **dataset_kwargs
    )

    # Replace the proton and gamma data with the selected samples
    test_dataset.proton_data = base_dataset.proton_data.iloc[test_proton_indices].reset_index(drop=True)
    test_dataset.gamma_data = base_dataset.gamma_data.iloc[test_gamma_indices].reset_index(drop=True)
    test_dataset.n_protons = len(test_dataset.proton_data)
    test_dataset.n_gammas = len(test_dataset.gamma_data)
    test_dataset.length = test_dataset.n_protons + test_dataset.n_gammas

    # Log some information about the split
    print(f"Train dataset: {train_dataset.length} samples "
          f"({train_dataset.n_protons} protons, {train_dataset.n_gammas} gammas)")
    print(f"Test dataset: {test_dataset.length} samples "
          f"({test_dataset.n_protons} protons, {test_dataset.n_gammas} gammas)")

    return train_dataset, test_dataset


def train_cnn_model(dataset, epochs=30):
    print("Training CNN component...")
    cnn_nametag = f"CNN_Component_{time.strftime('%d-%m-%Y_%H-%M-%S')}"
    cnn_output_dir = os.path.join(HYBRID_DIR, cnn_nametag)

    supervisor = TrainingSupervisor("hexmagicnet", dataset, cnn_output_dir,
                                    debug_info=True, save_model=True,
                                    save_debug_data=True, early_stopping=False)

    print(f"CNN model has {supervisor._count_trainable_weights()} trainable weights.")
    supervisor.LEARNING_RATE = 5.269632147047427e-06
    supervisor.WEIGHT_DECAY = 0.00034049323130326087
    supervisor.BATCH_SIZE = 64
    supervisor.GRAD_CLIP_NORM = 0.7168560391358462

    supervisor.train_model(epochs)

    torch.save(supervisor.model.state_dict(), CNN_MODEL_PATH)
    print(f"CNN model saved to {CNN_MODEL_PATH}")

    return CNN_MODEL_PATH


def train_rf_model(dataset):
    print("Training Random Forest component...")
    train_random_forest_classifier(
        dataset=dataset,
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


def train_ensemble_model(cnn_path, rf_path, dataset, epochs=10):
    print("Training ensemble combiner model...")
    ensemble_nametag = f"Ensemble_Combiner_{time.strftime('%d-%m-%Y_%H-%M-%S')}"
    ensemble_output_dir = os.path.join(HYBRID_DIR, ensemble_nametag)

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


def train_simple_hybrid_system(cnn_epochs=30, ensemble_epochs=5, test_size=0.2, random_state=42):
    print(f"\n{'=' * 60}")
    print(f"Training hybrid system with {test_size:.0%} held out for ensemble training")
    print(f"{'=' * 60}\n")

    run_timestamp = time.strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(HYBRID_DIR, f"hybrid_run_{run_timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print("Splitting dataset...")
    model_dataset, ensemble_dataset = split_magic_dataset(
        PROTON_FILE,
        GAMMA_FILE,
        test_size=test_size,
        random_state=random_state
    )
    
    train_cnn_model(model_dataset, epochs=cnn_epochs)
    train_rf_model(model_dataset)
    
    train_ensemble_model(CNN_MODEL_PATH, RF_MODEL_PATH, ensemble_dataset, epochs=ensemble_epochs)

    summary_path = os.path.join(run_dir, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Hybrid System Training Summary\n")
        f.write(f"Timestamp: {run_timestamp}\n")
        f.write(f"CNN epochs: {cnn_epochs}\n")
        f.write(f"Ensemble epochs: {ensemble_epochs}\n")
        f.write(f"Test size: {test_size:.0%}\n")
        f.write(f"Random state: {random_state}\n\n")
        f.write(f"CNN Model: {CNN_MODEL_PATH}\n")
        f.write(f"RF Model: {RF_MODEL_PATH}\n")
        f.write(f"Ensemble Model: {ENSEMBLE_MODEL_PATH}\n")

    print(f"\nHybrid system training complete!")
    print(f"Models saved to {run_dir}")
    print(f"Summary saved to {summary_path}")

    return {
        'cnn_model': CNN_MODEL_PATH,
        'rf_model': RF_MODEL_PATH,
        'ensemble_model': ENSEMBLE_MODEL_PATH,
        'summary': summary_path
    }


def main():
    train_simple_hybrid_system(
        cnn_epochs=30,
        ensemble_epochs=10,
        test_size=0.3,
        random_state=RANDOM_SEED
    )


if __name__ == "__main__":
    main()
