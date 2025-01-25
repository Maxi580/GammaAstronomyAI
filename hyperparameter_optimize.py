import optuna
import os
import time
import gc
import torch
from CombinedNet.TrainingSupervisor import TrainingSupervisor
from CNN.ConvolutionLayers.ConvHex import ConvHex
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

    def calculate_group_size(num_channels, target_groups_per_channel=4):
        target_groups = num_channels // target_groups_per_channel

        for groups in range(target_groups, 0, -1):
            if num_channels % groups == 0:
                return groups

        return 1

    class TelescopeCNN(nn.Module):
        def __init__(self, prefix):
            super().__init__()
            self.prefix = prefix

            pooling_pattern = [
                trial.suggest_categorical(f'pooling_layer_{i}', [True, False])
                for i in range(3)
            ]

            channels = [
                1,
                trial.suggest_int(f'cnn_channels1', 2, 16),
                trial.suggest_int(f'cnn_channels2', 4, 32),
                trial.suggest_int(f'cnn_channels3', 8, 48)
            ]

            layers = []
            pooling_count = 0
            has_previous_pooling = False

            for i in range(3):
                needs_pooling = has_previous_pooling or pooling_pattern[i]

                layers.append(
                    ConvHex(
                        channels[i],
                        channels[i + 1],
                        kernel_size=trial.suggest_int(f'kernel_size{i + 1}', 1, 5),
                        pooling=needs_pooling,
                        pooling_cnt=pooling_count,
                        pooling_kernel_size=2
                    )
                )

                layers.extend([
                    nn.GroupNorm(calculate_group_size(channels[i+1], trial.suggest_int(f'cnn_group_size_{i + 1}',
                                                                                       1, 16)), channels[i + 1]),
                    nn.ReLU(),
                ])

                if pooling_pattern[i]:
                    layers.append(nn.MaxPool1d(kernel_size=2))
                    pooling_count += 1
                    has_previous_pooling = True

                layers.append(nn.Dropout1d(
                    trial.suggest_float(f'dropout_cnn_{i + 1}', 0.05, 0.6)
                ))

            self.cnn = nn.Sequential(*layers)

        def forward(self, x):
            return self.cnn(x)

    class CustomCombinedNet(nn.Module):
        def __init__(self):
            super().__init__()

            self.m1_cnn = TelescopeCNN("m1_")
            self.m2_cnn = TelescopeCNN("m2_")

            channels3 = trial.params['cnn_channels3']
            num_pooling = sum(1 for i in range(3) if trial.params[f'pooling_layer_{i}'])
            input_size = channels3 * (1039 // (2 ** num_pooling)) * 2

            linear1_size = trial.suggest_int('linear1_size', 512, 2048, step=256)
            linear2_size = trial.suggest_int('linear2_size', 128, 512, step=64)
            linear3_size = trial.suggest_int('linear3_size', 64, 256, step=32)
            dropout_linear_1 = trial.suggest_float('dropout_linear_1', 0.05, 0.6)
            dropout_linear_2 = trial.suggest_float('dropout_linear_2', 0.05, 0.6)
            dropout_linear_3 = trial.suggest_float('dropout_linear_3', 0.05, 0.6)

            self.classifier = nn.Sequential(
                nn.Linear(input_size, linear1_size),
                nn.GroupNorm(calculate_group_size(linear1_size, trial.suggest_int(f'mlp_group_size_1', 1, 32)),
                             linear1_size),
                nn.ReLU(),
                nn.Dropout(dropout_linear_1),

                nn.Linear(linear1_size, linear2_size),
                nn.GroupNorm(calculate_group_size(linear2_size, trial.suggest_int(f'mlp_group_size_2', 1, 32)),
                             linear2_size),
                nn.ReLU(),
                nn.Dropout(dropout_linear_2),

                nn.Linear(linear2_size, linear3_size),
                nn.GroupNorm(calculate_group_size(linear3_size, trial.suggest_int(f'mlp_group_size_3', 1, 32)),
                             linear3_size),
                nn.ReLU(),
                nn.Dropout(dropout_linear_3),

                nn.Linear(linear3_size, 2)
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

            combined = torch.cat([m1_features, m2_features], dim=1)
            return self.classifier(combined)

    return CustomCombinedNet()


def objective(trial, dataset, study_name, epochs: int):
    supervisor = None
    try:
        nametag = f"{study_name}_WTF_{time.strftime('%Y-%m-%d_%H-%M-%S')}_trial_{trial.number}"
        output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  f"parameter_tuning/{study_name}", nametag)

        supervisor = TrainingSupervisor("combinednet", dataset, output_dir, debug_info=False, save_model=False)

        supervisor.model = create_model_with_params(trial).to(supervisor.device)

        supervisor.LEARNING_RATE = trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True)
        supervisor.WEIGHT_DECAY = trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True)
        supervisor.GRAD_CLIP_NORM = trial.suggest_float('grad_clip_norm', 0.1, 5.0)

        supervisor.train_model(epochs)

        best_accuracy = max(metrics['accuracy'] for metrics in supervisor.validation_metrics)
        print(f"\nBest Accuracy of Trial {trial.number} is {best_accuracy}%")

        print("Parameters:")
        for param_name, param_value in trial.params.items():
            print(f"  {param_name}: {param_value}")
        print("-" * 50)

        return best_accuracy

    except Exception as e:
        print(f"Trial {trial.number} failed with error: {str(e)}")
        raise optuna.exceptions.TrialPruned()

    finally:
        if supervisor is not None:
            clean_memory()


def start_or_resume_study(dataset, study_name: str, epochs: int, n_trials: int):
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
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(),
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        print("Creating new study")

    study.optimize(
        lambda trial: objective(trial, dataset, study_name, epochs),
        n_trials=n_trials
    )

    return study


def main(proton: str, gamma: str, epochs: int, n_trials: int):
    study_name = "OptimizeHexCNN"
    dataset = MagicDataset(proton, gamma, max_samples=100000, debug_info=False)
    study = start_or_resume_study(dataset, study_name, epochs, n_trials)

    print("Best trial:")
    print(f" Value (Val Accuracy): {study.best_trial.value}")
    print(" Params:")
    for key, value in study.best_trial.params.items():
        print(f" {key}: {value}")


if __name__ == "__main__":
    proton_file = "magic-protons.parquet"
    gamma_file = "magic-gammas.parquet"

    main(
        proton_file,
        gamma_file,
        epochs=10,
        n_trials=250
    )
