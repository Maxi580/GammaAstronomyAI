import optuna
import os
import time
import sys
import subprocess
import json
import traceback
import torch
import torch.nn as nn
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CNN.Architectures import HexMagicNet
from TrainingPipeline.Datasets import MagicDataset
from TrainingPipeline.TrainingSupervisor import TrainingSupervisor
from dataset_splitter import split_parquet_files
from random_forest.random_forest import train_random_forest_classifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OPTUNA_DB = "sqlite:///optuna_ensemble_study.db"

PROTON_FILE = "../magic-protons.parquet"
GAMMA_FILE = "../magic-gammas-new.parquet"
RANDOM_SEED = 42


def run_ensemble_trial_subprocess(trial_id, study_name, proton_file, gamma_file, cnn_path, rf_path, epochs, run_dir):
    """
    Run a single trial as a completely separate Python process.
    Uses the ensemble dataset for training the ensemble model.
    """
    print(f"\n==== Starting Ensemble Trial {trial_id} ====")

    script_content = f"""
import optuna
import os
import sys
import torch
import torch.nn as nn
import gc
import json
import time
import traceback
import pickle

# Import your modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from CNN.Architectures import HexMagicNet
    from TrainingPipeline.Datasets import MagicDataset
    from TrainingPipeline.TrainingSupervisor import TrainingSupervisor
except ImportError as e:
    print(json.dumps({{"trial_id": trial_id, "error": f"Import error: {{e}}", "status": "failed", "traceback": traceback.format_exc()}}))
    sys.exit(1)

# Set environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# Define Ensemble Model class here so we can modify it with trial parameters
class EnsembleModel(nn.Module):
    def __init__(self, cnn_model_path, rf_model_path, trial):
        super().__init__()

        self.cnn_model = HexMagicNet()
        self.cnn_model.load_state_dict(torch.load(cnn_model_path))

        for param in self.cnn_model.parameters():
            param.requires_grad = False

        with open(rf_model_path, 'rb') as f:
            self.rf_model = pickle.load(f)
            self.rf_model.verbose = 0

        self.cnn_weight = nn.Parameter(
            torch.tensor([trial.suggest_float('initial_cnn_weight', 0.1, 0.9)]), 
            requires_grad=True
        )

        num_layers = trial.suggest_int('num_layers', 1, 3)
        input_size = 4

        layers = []
        prev_size = input_size

        for i in range(num_layers):
            size = trial.suggest_int(f'hidden_size_{{i}}', 8, 256)
            layers.append(nn.Linear(prev_size, size))

            # Optional batch normalization
            if trial.suggest_categorical(f"use_batchnorm_{{i}}", [True, False]):
                layers.append(nn.BatchNorm1d(size))

            # Activation function
            activation = trial.suggest_categorical(f'activation_{{i}}', ['relu', 'leaky_relu', 'elu'])
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(trial.suggest_float(f'leaky_relu_slope_{{i}}', 0.01, 0.3)))
            elif activation == 'elu':
                layers.append(nn.ELU())

            # Dropout
            dropout_rate = trial.suggest_float(f'dropout_{{i}}', 0.0, 0.7)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            prev_size = size

        layers.append(nn.Linear(prev_size, 2))
        self.ensemble_layer = nn.Sequential(*layers)

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

try:
    storage = optuna.storages.RDBStorage(url="{OPTUNA_DB}")
    study = optuna.load_study(study_name="{study_name}", storage=storage)
except Exception as e:
    print(json.dumps({{"trial_id": trial_id, "error": f"Study load error: {{e}}", "status": "failed", "traceback": traceback.format_exc()}}))
    sys.exit(1)

# Get a frozen trial
try:
    frozen_trial = study.ask()
    trial_id = frozen_trial.number
except Exception as e:
    print(json.dumps({{"trial_id": trial_id, "error": f"Trial ask error: {{e}}", "status": "failed", "traceback": traceback.format_exc()}}))
    sys.exit(1)

try:
    print(f"Starting trial {{trial_id}}...")

    # Load ensemble dataset (validation data)
    print(f"Loading ensemble dataset...")
    ensemble_dataset = MagicDataset(
        proton_filename="{proton_file}",
        gamma_filename="{gamma_file}",
        max_samples=25000,
        debug_info=False
    )

    # Create output dir
    trial_dir = os.path.join("{run_dir}", f"trial_{{trial_id}}")
    os.makedirs(trial_dir, exist_ok=True)

    # Initialize supervisor with ensemble dataset
    print(f"Initializing training supervisor...")
    supervisor = TrainingSupervisor("custom", ensemble_dataset, trial_dir, 
                                   debug_info=False, save_model=False, save_debug_data=False)

    # Create ensemble model based on trial parameters
    print(f"Creating ensemble model...")
    ensemble_model = EnsembleModel("{cnn_path}", "{rf_path}", frozen_trial)
    supervisor.model = ensemble_model.to(supervisor.device)

    # Set hyperparameters
    supervisor.LEARNING_RATE = frozen_trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    supervisor.WEIGHT_DECAY = frozen_trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    supervisor.BATCH_SIZE = 256
    supervisor.GRAD_CLIP_NORM = frozen_trial.suggest_float('grad_clip_norm', 0.1, 5.0)

    # Train the model
    epochs = {epochs} 
    print(f"Training model for {{epochs}} epochs...")
    supervisor.train_model(epochs)

    # Calculate accuracy
    last_n_accuracies = [metrics['accuracy'] for metrics in supervisor.validation_metrics[-3:]]
    avg_accuracy = sum(last_n_accuracies) / len(last_n_accuracies)

    # Tell the study about the result
    print(f"Trial complete. Accuracy: {{avg_accuracy}}")
    study.tell(frozen_trial, avg_accuracy)

    # Print result for parent process to capture
    result = {{"trial_id": trial_id, "value": avg_accuracy, "status": "completed", "params": frozen_trial.params}}
    print(json.dumps(result))

    # Exit with success
    sys.exit(0)

except Exception as e:
    # Tell the study that the trial failed
    try:
        study.tell(frozen_trial, state=optuna.trial.TrialState.FAIL)
    except Exception as tell_error:
        print(f"Error telling study about failure: {{tell_error}}")

    # Print error for parent process to capture
    result = {{"trial_id": trial_id, "error": str(e), "status": "failed", "traceback": traceback.format_exc()}}
    print(json.dumps(result))

    # Exit with error
    sys.exit(1)

finally:
    print("Cleaning up resources...")
    if 'supervisor' in locals() and hasattr(supervisor, 'model'):
        try:
            supervisor.model = supervisor.model.cpu()
            del supervisor.model
        except Exception as cleanup_error:
            print(f"Error during model cleanup: {{cleanup_error}}")

    # Force cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Cleanup complete, exiting process.")
"""

    script_filename = f"temp_ensemble_trial_{trial_id}_{int(time.time())}.py"
    with open(script_filename, "w") as f:
        f.write(script_content)

    try:
        print(f"Executing ensemble trial {trial_id} in a separate process...")
        cmd = [sys.executable, script_filename]
        result = subprocess.run(cmd, capture_output=True, text=True)

        try:
            os.remove(script_filename)
        except:
            pass

        output = result.stdout
        stderr = result.stderr

        print(f"Output from trial {trial_id}:")
        print(output)

        if stderr:
            print(f"STDERR from trial {trial_id}:")
            print(stderr)

        json_result = None
        for line in output.strip().split("\n"):
            if line.startswith("{") and line.endswith("}"):
                try:
                    json_result = json.loads(line)
                    break
                except:
                    pass

        if json_result:
            return json_result

        return {
            "trial_id": trial_id,
            "status": "unknown",
            "stdout": output,
            "stderr": stderr,
            "exit_code": result.returncode
        }

    except Exception as e:
        print(f"Error executing subprocess for trial {trial_id}: {e}")
        traceback_str = traceback.format_exc()

        try:
            os.remove(script_filename)
        except:
            pass

        return {
            "trial_id": trial_id,
            "status": "failed",
            "error": str(e),
            "traceback": traceback_str
        }


def create_or_load_study(study_name):
    """Create a new study or load an existing one"""
    try:
        db_path = OPTUNA_DB.replace("sqlite:///", "")
        os.makedirs(os.path.dirname(os.path.abspath(db_path)) or ".", exist_ok=True)

        storage = optuna.storages.RDBStorage(
            url=OPTUNA_DB,
            engine_kwargs={"connect_args": {"timeout": 30}}
        )

        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(),
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        )
        return study
    except Exception as e:
        print(f"Error creating/loading study: {e}")
        print(traceback.format_exc())
        return None


def has_completed_trials(study):
    """Check if the study has any completed trials"""
    try:
        return any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    except:
        return False


def create_optimized_ensemble_model(best_params, cnn_path, rf_path):
    """Create the best ensemble model based on optimization results"""

    class OptimizedEnsembleModel(nn.Module):
        def __init__(self, cnn_model_path, rf_model_path, params):
            super().__init__()

            self.cnn_model = HexMagicNet()
            self.cnn_model.load_state_dict(torch.load(cnn_model_path))

            for param in self.cnn_model.parameters():
                param.requires_grad = False

            with open(rf_model_path, 'rb') as f:
                self.rf_model = pickle.load(f)

            num_layers = params['num_layers']
            input_size = 4

            layers = []
            prev_size = input_size

            for i in range(num_layers):
                size = params[f'hidden_size_{i}']
                layers.append(nn.Linear(prev_size, size))

                if params.get(f'use_batchnorm_{i}', False):
                    layers.append(nn.BatchNorm1d(size))

                activation = params.get(f'activation_{i}', 'relu')
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'leaky_relu':
                    layers.append(nn.LeakyReLU(params.get(f'leaky_relu_slope_{i}', 0.01)))
                elif activation == 'elu':
                    layers.append(nn.ELU())

                dropout_rate = params.get(f'dropout_{i}', 0.0)
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))

                prev_size = size

            layers.append(nn.Linear(prev_size, 2))

            self.ensemble_layer = nn.Sequential(*layers)

            self.cnn_weight = nn.Parameter(torch.tensor([params['initial_cnn_weight']]), requires_grad=True)

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

    return OptimizedEnsembleModel(cnn_path, rf_path, best_params)


def train_cnn(dataset):
    cnn_path = os.path.join(BASE_DIR, "cnn_base_model.pth")

    cnn_nametag = f"CNN_Base_{time.strftime('%d-%m-%Y_%H-%M-%S')}"
    cnn_output_dir = os.path.join(BASE_DIR, cnn_nametag)

    cnn_supervisor = TrainingSupervisor("hexmagicnet", dataset, cnn_output_dir,
                                        debug_info=True, save_model=True, val_split=0.1,
                                        save_debug_data=True, early_stopping=False)
    cnn_supervisor.LEARNING_RATE = 5.269632147047427e-06
    cnn_supervisor.WEIGHT_DECAY = 0.00034049323130326087
    cnn_supervisor.BATCH_SIZE = 64
    cnn_supervisor.GRAD_CLIP_NORM = 0.7168560391358462

    print(f"Training CNN model...")
    cnn_supervisor.train_model(20)
    torch.save(cnn_supervisor.model.state_dict(), cnn_path)
    print(f"CNN model saved to {cnn_path}")
    return cnn_path


def optimize_ensemble(n_trials=100, epochs=10, val_split=0.3):
    """
    Optimize the ensemble model using properly split datasets

    Args:
        n_trials: Number of Optuna trials to run
        epochs: Number of epochs to train each ensemble model
        val_split: Validation split for dataset splitting
    """
    run_timestamp = time.strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(BASE_DIR, "ensemble_optimization")
    os.makedirs(run_dir, exist_ok=True)
    study_name = f"Ensemble_Optimization_{run_timestamp}"

    print("\nSplitting datasets into train/validation sets...")
    data_dir = os.path.join(BASE_DIR, "data")
    print(f"In: {data_dir}")
    file_paths = split_parquet_files(
        PROTON_FILE,
        GAMMA_FILE,
        data_dir,
        val_split=val_split,
        random_seed=RANDOM_SEED
    )

    cnn_path = os.path.join(BASE_DIR, "cnn_model.pth")
    rf_path = os.path.join(BASE_DIR, "rf_model.pkl")

    print("\nSetting up Optuna study for ensemble optimization...")
    study = create_or_load_study(study_name)
    if study is None:
        print("Failed to create or load study")
        return None

    summary_path = os.path.join(run_dir, "optimization_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Ensemble Optimization Summary\n")
        f.write(f"{'=' * 50}\n\n")
        f.write(f"Started: {run_timestamp}\n")
        f.write(f"Number of trials: {n_trials}\n")
        f.write(f"Epochs per trial: {epochs}\n")
        f.write(f"Validation split: {val_split}\n\n")
        f.write(f"CNN base model: {cnn_path}\n")
        f.write(f"RF base model: {rf_path}\n\n")
        f.write(f"Trial results:\n")
        f.write(f"{'=' * 50}\n\n")

    print(f"\nRunning {n_trials} optimization trials...")
    for trial_id in range(n_trials):
        result = run_ensemble_trial_subprocess(
            trial_id=trial_id,
            study_name=study_name,
            proton_file=file_paths['val']['proton'],
            gamma_file=file_paths['val']['gamma'],
            cnn_path=cnn_path,
            rf_path=rf_path,
            epochs=epochs,
            run_dir=run_dir
        )

        with open(summary_path, 'a') as f:
            if result.get("status") == "completed":
                f.write(f"Trial {result.get('trial_id')} completed with value: {result.get('value')}\n")
                f.write(f"Parameters: {result.get('params')}\n\n")
                print(f"Trial {result.get('trial_id')} completed with value: {result.get('value')}")
            else:
                f.write(f"Trial {result.get('trial_id')} failed\n")
                f.write(f"Error: {result.get('error', 'Unknown error')}\n\n")
                print(f"Trial {result.get('trial_id')} failed")

        if has_completed_trials(study):
            try:
                print("\nCurrent best trial:")
                best_trial = study.best_trial
                print(f"  Value (Accuracy): {best_trial.value}")
                print(f"  Parameters: {best_trial.params}")

                with open(summary_path, 'a') as f:
                    f.write(f"Current best trial: {best_trial.number}\n")
                    f.write(f"Value (Accuracy): {best_trial.value}\n")
                    f.write(f"Parameters: {best_trial.params}\n\n")
            except Exception as e:
                print(f"Error getting best trial: {e}")

        print(f"Pausing for 2 seconds before next trial...")
        time.sleep(2)


if __name__ == "__main__":
    optimize_ensemble(n_trials=100, epochs=10, val_split=0.3)
