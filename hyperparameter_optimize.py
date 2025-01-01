import optuna
import os
import time
import torch
from arrayClassification.trainingSupervisor import TrainingSupervisor
from arrayClassification.Hyperparameter.CustomHexCNN import CustomHexCNN


def create_model_with_params(trial):
    return CustomHexCNN(
        kernel_size1=trial.suggest_int('kernel_size1', 1, 5),
        kernel_size2=trial.suggest_int('kernel_size2', 1, 5),
        kernel_size3=trial.suggest_int('kernel_size3', 1, 5),
        dropout_conv1=trial.suggest_float('dropout_conv1', 0.025, 0.5),
        dropout_conv2=trial.suggest_float('dropout_conv2', 0.025, 0.5),
        dropout_conv3=trial.suggest_float('dropout_conv3', 0.025, 0.5),
        linear1_size=trial.suggest_int('linear1_size', 512, 38400, step=1, log=True),
        linear2_size=trial.suggest_int('linear2_size', 128, 5096, step=1, log=True),
        linear3_size=trial.suggest_int('linear3_size', 32, 2048, step=1, log=True),
        dropout_linear1=trial.suggest_float('dropout_linear1', 0.025, 0.5),
        dropout_linear2=trial.suggest_float('dropout_linear2', 0.025, 0.5),
        dropout_linear3=trial.suggest_float('dropout_linear3', 0.025, 0.3)
    )


def objective(trial, dataset: str, study_name, epochs: int):
    try:
        nametag = f"{study_name}_{dataset}_{time.strftime('%Y-%m-%d_%H-%M-%S')}_trial_{trial.number}"
        dataset_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "datasets", dataset)
        output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"parameter_tuning/{study_name}",
                                  nametag)
        os.makedirs(output_dir, exist_ok=True)

        supervisor = TrainingSupervisor("hexcnn", dataset_dir, output_dir, debug_info=False)

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

        # Clean up
        del supervisor.model
        del supervisor
        torch.cuda.empty_cache()

        return accuracy

    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        # Clean up on error
        if 'supervisor' in locals():
            del supervisor.model
            del supervisor
        torch.cuda.empty_cache()

        print(f"Trial {trial.number} failed with error: {str(e)}")
        # Tell Optuna to prune this trial
        raise optuna.exceptions.TrialPruned()


def start_or_resume_study(dataset: str, study_name: str, epochs: int, n_trials: int):
    try:
        study = optuna.load_study(
            study_name= study_name,
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
    main("Dataset20k", 1, 5)
