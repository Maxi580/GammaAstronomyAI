import optuna
import os
import time
import gc
import torch

from TrainingPipeline.Datasets import *
from TrainingPipeline.TrainingSupervisor import TrainingSupervisor

from ParameterTuning import *


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'


def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()
    # Force a sync with GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def objective(trial: optuna.Trial, model: str, dataset, study_name, epochs: int):
    supervisor = None
    try:
        nametag = f"{study_name}_WTF_{time.strftime('%Y-%m-%d_%H-%M-%S')}_trial_{trial.number}"
        output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  f"parameter_tuning/{study_name}", nametag)

        supervisor = TrainingSupervisor(model, dataset, output_dir, debug_info=False, save_model=False, save_debug_data=False)
        
        match model.lower():
            case "basicmagicnet":
                parameterize_func = parameterize_BasicMagicNet
            case "hexcirclenet":
                parameterize_func = parameterize_HexCircleNet
            case "hexmagicnet":
                parameterize_func = parameterize_HexMagicNet
            case "hexagdlynet":
                parameterize_func = parameterize_HexagdlyNet
            case "simple1dnet":
                parameterize_func = parameterize_Simple1dNet
            case _:
                raise ValueError(f"Invalid Modelname for parameterization: '{model}'")

        supervisor.model = parameterize_func(trial).to(supervisor.device)
        weights = supervisor._count_trainable_weights()
        
        # Throw away trial if it has too many weights
        if weights > 20_000_000:
            raise optuna.exceptions.TrialPruned(f"Too many weights: {weights}")

        supervisor.LEARNING_RATE = learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-1, log=True)
        supervisor.WEIGHT_DECAY = trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True)
        supervisor.GRAD_CLIP_NORM = trial.suggest_float('grad_clip_norm', 0.1, 5.0, step=0.1)
        supervisor.SCHEDULER_CYCLE_MOMENTUM = trial.suggest_categorical('scheduler_cycle_momentum', [True, False])
        supervisor.SCHEDULER_STEP_SIZE = trial.suggest_int('scheduler_step_size', 3, 6)
        supervisor.SCHEDULER_BASE_LR = trial.suggest_float('scheduler_base_lr', 1e-6, learning_rate, log=True)
        supervisor.SCHEDULER_MAX_LR = trial.suggest_float('scheduler_max_lr', learning_rate, learning_rate * 10, log=True)

        supervisor.train_model(epochs)

        last_n_accuracies = [metrics['accuracy'] for metrics in supervisor.validation_metrics[-3:]]
        avg_accuracy = sum(last_n_accuracies) / len(last_n_accuracies)

        return avg_accuracy, supervisor._count_trainable_weights()

    except Exception as e:
        print(f"Trial {trial.number} failed with error:", e)
        raise optuna.exceptions.TrialPruned()

    finally:
        if supervisor is not None:
            clean_memory()


def start_or_resume_study(dataset, model: str, study_name: str, epochs: int, n_trials: int):
    storage = "sqlite:///optuna_study.db"
    
    try:
        study = optuna.load_study(
            study_name=study_name,
            storage=storage
        )
        print("Resuming existing study")

    except KeyError:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            directions=["maximize", "minimize"],
            pruner=optuna.pruners.MedianPruner(),
            sampler=optuna.samplers.TPESampler(
                seed=42,
                n_startup_trials=(n_trials // 5), # Use 20% of trials with random sampler to explor more options first.
            )
        )
        print("Creating new study")

    study.optimize(
        lambda trial: objective(trial, model, dataset, study_name, epochs),
        n_trials=n_trials
    )

    return study


def main(model: str, proton: str, gamma: str, epochs: int, n_trials: int):
    study_name = f"Optimize_{model}"
    
    dataset_class: MagicDataset = MagicDatasetHexagdly if model.lower() == 'hexagdlynet' else MagicDataset

    dataset = dataset_class(proton, gamma, max_samples=100000, debug_info=False)

    study = start_or_resume_study(dataset, model, study_name, epochs, n_trials)

    print("Best 3 trials:")
    for trial in study.best_trials[:3]:
        print(f"Trial #{trial.number}")
        print(f"  Values: {trial.values}")
        print("  Params:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")


if __name__ == "__main__":
    model_name = "Simple1DNet"
    proton_file = "magic-protons.parquet"
    gamma_file = "magic-gammas-new-1.parquet"

    main(
        model_name,
        proton_file,
        gamma_file,
        epochs=10,
        n_trials=300
    )
