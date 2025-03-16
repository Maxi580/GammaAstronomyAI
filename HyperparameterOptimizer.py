import optuna
import os
import time
import gc
import torch
import sys

from ParameterTuning.HexMagicCNN import parameterize_hex_magicnet
from TrainingPipeline.MagicDataset import MagicDataset
from TrainingPipeline.TrainingSupervisor import TrainingSupervisor

from ParameterTuning import *

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'


def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                del obj
        except:
            pass


def objective(trial: optuna.Trial, model: str, dataset, study_name, epochs: int):
    supervisor = None
    try:
        nametag = f"{study_name}_{time.strftime('%Y-%m-%d_%H-%M-%S')}_trial_{trial.number}"
        output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  f"parameter_tuning/{study_name}", nametag)

        supervisor = TrainingSupervisor(model, dataset, output_dir, debug_info=False, save_model=False, save_debug_data=False)
        
        match model.lower():
            case "basicmagicnet":
                parameterize_func = parameterize_BasicMagicNet
            case "hexcirclenet":
                parameterize_func = parameterize_HexCircleNet
            case "hexmagicnet":
                parameterize_func = parameterize_hex_magicnet
            case _:
                raise ValueError(f"Invalid Modelname for parameterization: '{model}'")

        supervisor.model = parameterize_func(trial).to(supervisor.device)

        supervisor.LEARNING_RATE = trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True)
        supervisor.WEIGHT_DECAY = trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True)
        supervisor.GRAD_CLIP_NORM = trial.suggest_float('grad_clip_norm', 0.1, 5.0)

        supervisor.train_model(epochs)

        last_n_accuracies = [metrics['accuracy'] for metrics in supervisor.validation_metrics[-3:]]
        avg_accuracy = sum(last_n_accuracies) / len(last_n_accuracies)

        return avg_accuracy

    except Exception as e:
        print(f"Trial {trial.number} failed with error:", e)
        raise optuna.exceptions.TrialPruned()

    finally:
        if supervisor is not None:
            clean_memory()


def start_or_resume_study(dataset, model: str, study_name: str, epochs: int, n_trials: int):
    try:
        study = optuna.create_study(
            study_name=study_name,
            storage="sqlite:///optuna_study.db",
            load_if_exists=True,
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(),
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        print("Study loaded/created")

    except Exception as e:
        print(f"Error creating/loading study: {e}")
        return None

    study.optimize(
        lambda trial: objective(trial, model, dataset, study_name, epochs),
        n_trials=n_trials
    )

    return study


def main(model: str, proton: str, gamma: str, epochs: int, n_trials: int):
    study_name = f"Optimize_{model}"
    dataset = MagicDataset(proton, gamma, max_samples=100000, debug_info=False)
    study = start_or_resume_study(dataset, model, study_name, epochs, n_trials)

    print("Best trial:")
    print(f" Value (Val Accuracy): {study.best_trial.value}")
    print(" Params:")
    for key, value in study.best_trial.params.items():
        print(f" {key}: {value}")


if __name__ == "__main__":
    model_name = "hexmagicnet"
    proton_file = "magic-protons.parquet"
    gamma_file = "magic-gammas-new.parquet"

    main(
        model_name,
        proton_file,
        gamma_file,
        epochs=10,
        n_trials=250
    )
