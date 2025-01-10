import optuna
import os
import time
import gc
import torch
from CombinedNet.TrainingSupervisor import TrainingSupervisor
import torch.nn as nn

from CombinedNet.magicDataset import MagicDataset

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
NUM_OF_HEXAGONS = 1039

def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()
    # Force a sync with GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def create_model_with_params(trial):
    def resize_input(image):
        """Arrays are 1183 long, however the last 144 are always 0"""
        return image[:, :, :NUM_OF_HEXAGONS]

    class TelescopeCNN(nn.Module):
        def __init__(self):
            super().__init__()
            channels1 = trial.suggest_int('cnn_channels1', 4, 16)
            channels2 = trial.suggest_int('cnn_channels2', 8, 32)
            dropout_cnn_1 = trial.suggest_float('dropout_cnn', 0.05, 0.5)
            dropout_cnn_2 = trial.suggest_float('dropout_cnn', 0.05, 0.5)

            self.cnn = nn.Sequential(
                nn.Conv1d(1, channels1, kernel_size=3),
                nn.BatchNorm1d(channels1),
                nn.ReLU(),
                nn.Dropout1d(dropout_cnn_1),

                nn.Conv1d(channels1, channels2, kernel_size=2),
                nn.BatchNorm1d(channels2),
                nn.ReLU(),
                nn.Dropout1d(dropout_cnn_2),
            )

        def forward(self, x):
            return self.cnn(x)

    class CustomCombinedNet(nn.Module):
        def __init__(self):
            super().__init__()

            self.m1_cnn = TelescopeCNN()
            self.m2_cnn = TelescopeCNN()

            channels2 = trial.params['cnn_channels2']
            linear_input_size = channels2 * 1036 * 2 + 59

            # Let optuna choose linear layer sizes
            linear1_size = trial.suggest_int('linear1_size', 512, 4096, step=256)
            linear2_size = trial.suggest_int('linear2_size', 128, 1024, step=64)
            dropout_linear_1 = trial.suggest_float('dropout_linear', 0.05, 0.5)
            dropout_linear_2 = trial.suggest_float('dropout_linear', 0.05, 0.5)

            self.classifier = nn.Sequential(
                nn.Linear(linear_input_size, linear1_size),
                nn.BatchNorm1d(linear1_size),
                nn.ReLU(),
                nn.Dropout(dropout_linear_1),

                nn.Linear(linear1_size, linear2_size),
                nn.BatchNorm1d(linear2_size),
                nn.ReLU(),
                nn.Dropout(dropout_linear_2),

                nn.Linear(linear2_size, 2)
            )

        def forward(self, m1_image, m2_image, measurement_features):
            m1_image = m1_image.unsqueeze(1)
            m2_image = m2_image.unsqueeze(1)
            m1_image = resize_input(m1_image)
            m2_image = resize_input(m2_image)
            m1_features = self.m1_cnn(m1_image)
            m2_features = self.m2_cnn(m2_image)
            m1_features = m1_features.flatten(1)
            m2_features = m2_features.flatten(1)

            combined = torch.cat([m1_features, m2_features, measurement_features], dim=1)
            return self.classifier(combined)

    return CustomCombinedNet()


def objective(trial, proton_file: str, gamma_file: str, study_name, epochs: int):
    supervisor = None
    try:
        nametag = f"{study_name}_WTF_{time.strftime('%Y-%m-%d_%H-%M-%S')}_trial_{trial.number}"
        output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  f"parameter_tuning/{study_name}", nametag)

        dataset = MagicDataset(proton_file, gamma_file, max_samples=20000, debug_info=False)
        supervisor = TrainingSupervisor("combinednet", dataset, output_dir, debug_info=False, save_model=False)

        supervisor.model = create_model_with_params(trial).to(supervisor.device)

        supervisor.LEARNING_RATE = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        supervisor.WEIGHT_DECAY = trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True)
        supervisor.GRAD_CLIP_NORM = trial.suggest_float('grad_clip_norm', 0.1, 5.0)

        supervisor.train_model(epochs)

        last_n_accuracies = [metrics['accuracy'] for metrics in supervisor.validation_metrics[-3:]]
        avg_accuracy = sum(last_n_accuracies) / len(last_n_accuracies)
        print(f"\nTrial {trial.number} got an accuracy of {avg_accuracy}%")

        print("Parameters:")
        for param_name, param_value in trial.params.items():
            print(f"  {param_name}: {param_value}")
        print("-" * 50)

        return avg_accuracy

    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        print(f"Trial {trial.number} failed with error: {str(e)}")
        raise optuna.exceptions.TrialPruned()

    finally:
        if supervisor is not None:
            clean_memory()


def start_or_resume_study(proton_file: str, gamma_file: str, study_name: str, epochs: int, n_trials: int):
    try:
        study = optuna.load_study(
            study_name=study_name,
            storage="sqlite:///optuna_study.db"
        )
        print("Resuming existing study")

    except KeyError:
        study = optuna.create_study(
            study_name=study_name,
            storage="sqlite:///optuna_study.db",
            direction="maximize"
        )
        print("Creating new study")

    study.optimize(
        lambda trial: objective(trial, proton_file, gamma_file, study_name, epochs),
        n_trials=n_trials
    )

    return study


def main(proton_file: str, gamma_file: str, epochs: int, n_trials: int):
    study_name = "combinednet_optimization"
    study = start_or_resume_study(proton_file, gamma_file, study_name, epochs, n_trials)

    print("Best trial:")
    print(f" Value (Val Accuracy): {study.best_trial.value}")
    print(" Params:")
    for key, value in study.best_trial.params.items():
        print(f" {key}: {value}")


if __name__ == "__main__":
    proton_file = "magic-protons.parquet"
    gamma_file = "magic-gammas.parquet"

    main(
        proton_file=proton_file,
        gamma_file=gamma_file,
        epochs=10,
        n_trials=150
    )
