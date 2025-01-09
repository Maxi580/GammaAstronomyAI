import optuna
import os
import gc
import time
import torch
from CombinedNet.trainingSupervisor import TrainingSupervisor
from CNN.HyperparameterTuning import CustomHexCNN

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'


def create_model_with_params(trial):
    return CustomHexCNN(
        kernel_size1=trial.suggest_int('kernel_size1', 1, 5),
        kernel_size2=trial.suggest_int('kernel_size2', 1, 5),
        kernel_size3=trial.suggest_int('kernel_size3', 1, 5),
        dropout_conv1=trial.suggest_float('dropout_conv1', 0.025, 0.5),
        dropout_conv2=trial.suggest_float('dropout_conv2', 0.025, 0.5),
        dropout_conv3=trial.suggest_float('dropout_conv3', 0.025, 0.5),
        linear1_size=trial.suggest_int('linear1_size', 256, 4096, step=256),
        linear2_size=trial.suggest_int('linear2_size', 128, 1024, step=128),
        linear3_size=trial.suggest_int('linear3_size', 64, 512, step=64),
        dropout_linear1=trial.suggest_float('dropout_linear1', 0.025, 0.5),
        dropout_linear2=trial.suggest_float('dropout_linear2', 0.025, 0.5),
        dropout_linear3=trial.suggest_float('dropout_linear3', 0.025, 0.3)
    )


def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()
    # Force a sync with GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def objective(trial, dataset: str, study_name, epochs: int):
    supervisor = None
    try:
        nametag = f"{study_name}_{dataset}_{time.strftime('%Y-%m-%d_%H-%M-%S')}_trial_{trial.number}"
        dataset_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "datasets", dataset)
        output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"parameter_tuning/{study_name}",
                                  nametag)

        supervisor = TrainingSupervisor("hexcnn", dataset_dir, output_dir, debug_info=False, save_model=False)

        supervisor.model = create_model_with_params(trial).to(supervisor.device)

        supervisor.LEARNING_RATE = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        beta1 = trial.suggest_float('adam_beta1', 0.8, 0.95)
        supervisor.ADAM_BETA_1 = beta1
        supervisor.ADAM_BETA_2 = trial.suggest_float('adam_beta2', beta1 + 0.04, 0.999)
        supervisor.WEIGHT_DECAY = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
        supervisor.SCHEDULER_MIN_LR = trial.suggest_float('scheduler_min_lr', 1e-6, 1e-4, log=True)
        supervisor.SCHEDULER_MAX_LR = trial.suggest_float('scheduler_max_lr', 1e-4, 1e-2, log=True)
        supervisor.GRAD_CLIP_NORM = trial.suggest_float('grad_clip_norm', 0.1, 5.0)

        supervisor.train_model(epochs)
        accuracy = supervisor.validation_metrics[-1]['accuracy']

        print(f"\nTrial {trial.number} got an accuracy of {accuracy}%")
        print("Parameters:")
        for param_name, param_value in trial.params.items():
            print(f"  {param_name}: {param_value}")
        print("-" * 50)

        return accuracy

    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        print(f"Trial {trial.number} failed with error: {str(e)}")
        raise optuna.exceptions.TrialPruned()

    finally:
        if supervisor is not None:
            # Clean up DataLoaders
            if hasattr(supervisor, 'TRAINING_DATA_LOADER'):
                del supervisor.TRAINING_DATA_LOADER

            if hasattr(supervisor, 'VALIDATION_DATA_LOADER'):
                del supervisor.VALIDATION_DATA_LOADER

            # Clean up the model
            if hasattr(supervisor, 'model'):
                supervisor.model.cpu()
                del supervisor.model

            del supervisor

        clean_memory()


def start_or_resume_study(dataset: str, study_name: str, epochs: int, n_trials: int):
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

    study.optimize(lambda trial: objective(trial, dataset, study_name, epochs), n_trials=n_trials)

    return study


def main(dataset: str, epochs: int, n_trials: int):
    study_name = "hexcnn_optimization"
    study = start_or_resume_study(dataset, study_name, epochs, n_trials)
    print("Best trial:")
    print(f" Value (Val Accuracy): {study.best_trial.value}")
    print(" Params:")
    for key, value in study.best_trial.params.items():
        print(f" {key}: {value}")


if __name__ == "__main__":
    main("TuneDataset20k", 10, 500)
