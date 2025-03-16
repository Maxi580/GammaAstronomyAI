import optuna
import os
import time
import gc
import torch
import traceback
import random
import numpy as np

from ParameterTuning.HexMagicCNN import parameterize_hex_magicnet
from TrainingPipeline.MagicDataset import MagicDataset
from TrainingPipeline.TrainingSupervisor import TrainingSupervisor

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def reset_cuda_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        try:
            x = torch.zeros(1, device=device)
            y = torch.ones(1, device=device)
            z = x + y
            del x, y, z
        except:
            pass
        gc.collect()


def create_supervisor(model, dataset, output_dir):
    try:
        return TrainingSupervisor(model, dataset, output_dir,
                                  debug_info=False, save_model=False, save_debug_data=False)
    except Exception as e:
        print(f"Error creating supervisor: {e}")
        traceback.print_exc()
        reset_cuda_device()
        return None


def objective(trial, model, dataset, study_name, epochs):
    supervisor = None

    set_seeds(42 + trial.number)
    clean_memory()

    try:
        nametag = f"{study_name}_{time.strftime('%Y-%m-%d_%H-%M-%S')}_trial_{trial.number}"
        output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  f"parameter_tuning/{study_name}", nametag)

        supervisor = create_supervisor(model, dataset, output_dir)
        if supervisor is None:
            raise optuna.exceptions.TrialPruned()

        try:
            if model.lower() == "basicmagicnet":
                from ParameterTuning.BasicMagicCNN import parameterize_BasicMagicNet
                parameterize_func = parameterize_BasicMagicNet
            elif model.lower() == "hexcirclenet":
                from ParameterTuning.HexCircleNet import parameterize_HexCircleNet
                parameterize_func = parameterize_HexCircleNet
            elif model.lower() == "hexmagicnet":
                parameterize_func = parameterize_hex_magicnet
            else:
                raise ValueError(f"Invalid Modelname for parameterization: '{model}'")
        except ImportError as e:
            print(f"Error importing parameter functions: {e}")
            raise optuna.exceptions.TrialPruned()

        try:
            model_instance = parameterize_func(trial)
            supervisor.model = model_instance.to(supervisor.device)
        except torch.cuda.OutOfMemoryError:
            print(f"CUDA OOM during model creation for trial {trial.number}")
            reset_cuda_device()
            raise optuna.exceptions.TrialPruned()
        except Exception as e:
            print(f"Error creating model for trial {trial.number}: {e}")
            traceback.print_exc()
            reset_cuda_device()
            raise optuna.exceptions.TrialPruned()

        supervisor.LEARNING_RATE = trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True)
        supervisor.WEIGHT_DECAY = trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True)
        supervisor.GRAD_CLIP_NORM = trial.suggest_float('grad_clip_norm', 0.1, 5.0)

        try:
            supervisor.train_model(epochs)
        except torch.cuda.OutOfMemoryError:
            print(f"CUDA OOM during training for trial {trial.number}")
            reset_cuda_device()
            raise optuna.exceptions.TrialPruned()
        except Exception as e:
            print(f"Error during training for trial {trial.number}: {e}")
            traceback.print_exc()
            reset_cuda_device()
            raise optuna.exceptions.TrialPruned()

        if not supervisor.validation_metrics or len(supervisor.validation_metrics) < 3:
            print(f"Trial {trial.number} has insufficient validation metrics")
            return 0.0

        last_n_accuracies = [metrics['accuracy'] for metrics in supervisor.validation_metrics[-3:]]
        avg_accuracy = sum(last_n_accuracies) / len(last_n_accuracies)

        return avg_accuracy

    except torch.cuda.OutOfMemoryError:
        print(f"Trial {trial.number} failed with CUDA out of memory error")
        traceback.print_exc()
        reset_cuda_device()
        return 0.0

    except optuna.exceptions.TrialPruned:
        raise

    except Exception as e:
        print(f"Trial {trial.number} failed with error: {str(e)}")
        traceback.print_exc()
        reset_cuda_device()
        return 0.0

    finally:
        if supervisor is not None:
            if hasattr(supervisor, 'model') and supervisor.model is not None:
                try:
                    supervisor.model = supervisor.model.to('cpu')
                except:
                    pass
                try:
                    del supervisor.model
                    supervisor.model = None
                except:
                    pass

            for attr in ['training_data_loader', 'val_data_loader', 'train_dataset', 'val_dataset']:
                if hasattr(supervisor, attr):
                    try:
                        delattr(supervisor, attr)
                    except:
                        pass

            try:
                del supervisor
            except:
                pass

        clean_memory()

        time.sleep(0.5)


def objective_wrapper(trial, model, dataset, study_name, epochs):
    clean_memory()
    reset_cuda_device()
    try:
        result = objective(trial, model, dataset, study_name, epochs)
        return result
    except Exception as e:
        print(f"Unhandled exception in objective_wrapper: {e}")
        traceback.print_exc()
        reset_cuda_device()
        return 0.0


def start_or_resume_study(dataset, model, study_name, epochs, n_trials):
    try:
        study = optuna.create_study(
            study_name=study_name,
            storage="sqlite:///optuna_study.db",
            load_if_exists=True,
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        print("Study loaded/created")

    except Exception as e:
        print(f"Error creating/loading study: {e}")
        traceback.print_exc()
        return None

    try:
        study.optimize(
            lambda trial: objective_wrapper(trial, model, dataset, study_name, epochs),
            n_trials=n_trials,
            catch=(Exception,)  # Catch all exceptions to prevent the study from terminating
        )
    except KeyboardInterrupt:
        print("Study interrupted by user.")
    except Exception as e:
        print(f"Study optimization error: {e}")
        traceback.print_exc()

    return study


def main(model, proton, gamma, epochs, n_trials):
    set_seeds(42)

    study_name = f"Optimize_{model.lower()}"

    max_samples = 80000

    try:
        dataset = MagicDataset(proton, gamma, max_samples=max_samples, debug_info=False)
    except Exception as e:
        print(f"Error creating dataset: {e}")
        traceback.print_exc()
        return

    study = start_or_resume_study(dataset, model, study_name, epochs, n_trials)

    if study:
        if hasattr(study, 'best_trial') and study.best_trial:
            print("\nBest trial:")
            print(f"Value (Val Accuracy): {study.best_trial.value}")
            print("Params:")
            for key, value in study.best_trial.params.items():
                print(f"  {key}: {value}")
        else:
            print("\nNo successful trials found.")


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
