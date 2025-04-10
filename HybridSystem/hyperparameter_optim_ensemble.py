import optuna
import os
import time
import sys
import subprocess
import json
import traceback
import torch
import torch.nn as nn

from CNN.Architectures import HexMagicNet
import joblib
from TrainingPipeline.Datasets import MagicDataset
from TrainingPipeline.TrainingSupervisor import TrainingSupervisor

OPTUNA_DB = "sqlite:///optuna_ensemble_study.db"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HYBRID_DIR = os.path.join(BASE_DIR, "HybridSystem")
CNN_MODEL_PATH = os.path.join(HYBRID_DIR, "cnn_model.pth")
RF_MODEL_PATH = os.path.join(HYBRID_DIR, "rf_model.pkl")
ENSEMBLE_MODEL_PATH = os.path.join(HYBRID_DIR, "ensemble_model_optimized.pth")

PROTON_FILE = "../magic-protons.parquet"
GAMMA_FILE = "../magic-gammas-new.parquet"


def run_ensemble_trial_subprocess(trial_id, study_name, proton_file, gamma_file, cnn_path, rf_path, epochs):
    """
    Run a single trial as a completely separate Python process.
    This ensures complete isolation and memory cleanup between trials.
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
import joblib

# Import your modules
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
            self.rf_model = joblib.load(rf_model_path)

        num_layers = trial.suggest_int('num_layers', 1, 3)
        input_size = 4

        hidden_sizes = []
        for i in range(num_layers):
            trial.suggest_int(f'hidden_size_{{i}}', 8, 256)
            hidden_sizes.append(size)

        layers = []
        prev_size = input_size

        for i, size in enumerate(hidden_sizes):
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

            dropout_rate = trial.suggest_float(f'dropout_{{i}}', 0.0, 0.7)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            prev_size = size

        layers.append(nn.Linear(prev_size, 2))

        self.ensemble_layer = nn.Sequential(*layers)

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

    # Load dataset
    print(f"Loading dataset from {proton_file} and {gamma_file}...")
    dataset = MagicDataset("{proton_file}", "{gamma_file}", max_samples=100000, debug_info=False)

    # Create output dir
    nametag = f"{study_name}_{{time.strftime('%Y-%m-%d_%H-%M-%S')}}_trial_{{trial_id}}"
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            f"parameter_tuning/{study_name}", nametag)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize supervisor
    print(f"Initializing training supervisor...")
    supervisor = TrainingSupervisor("custom", dataset, output_dir, 
                                   debug_info=False, save_model=False, save_debug_data=False)

    # Create ensemble model based on trial parameters
    print(f"Creating ensemble model...")
    ensemble_model = EnsembleModel("{cnn_path}", "{rf_path}", frozen_trial)
    supervisor.model = ensemble_model.to(supervisor.device)

    # Set hyperparameters
    supervisor.LEARNING_RATE = frozen_trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    supervisor.WEIGHT_DECAY = frozen_trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    supervisor.BATCH_SIZE = frozen_trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    supervisor.GRAD_CLIP_NORM = frozen_trial.suggest_float('grad_clip_norm', 0.1, 5.0)

    # Train the model
    print(f"Training model for {epochs} epochs...")
    supervisor.train_model({epochs})

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
            sampler=optuna.samplers.TPESampler(seed=42),
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


def run_ensemble_optimization(proton_file, gamma_file, cnn_path, rf_path, study_name, n_trials, epochs):
    """Run the optimization for ensemble model with sequential trial execution"""
    study = create_or_load_study(study_name)
    if study is None:
        print("Failed to create or load study")
        return

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    n_completed = len(completed_trials)
    print(f"Already completed trials: {n_completed}")

    n_remaining = max(0, n_trials - n_completed)
    print(f"Remaining trials to run: {n_remaining}")

    if n_remaining <= 0:
        print("All trials have been completed")
        return study

    for trial_id in range(n_completed, n_completed + n_remaining):
        result = run_ensemble_trial_subprocess(
            trial_id=trial_id,
            study_name=study_name,
            proton_file=proton_file,
            gamma_file=gamma_file,
            cnn_path=cnn_path,
            rf_path=rf_path,
            epochs=epochs
        )

        if result.get("status") == "completed":
            print(f"Trial {result.get('trial_id')} completed with value: {result.get('value')}")
            print(f"Parameters: {result.get('params')}")
        else:
            print(f"Trial {result.get('trial_id')} failed")
            error = result.get("error", "Unknown error")
            print(f"Error: {error}")

            if "traceback" in result:
                print("Traceback:")
                print(result["traceback"])

        if has_completed_trials(study):
            try:
                print("\nCurrent best trial:")
                best_trial = study.best_trial
                print(f"  Value (Accuracy): {best_trial.value}")
                print(f"  Parameters: {best_trial.params}")
            except Exception as e:
                print(f"Error getting best trial: {e}")
        else:
            print("\nNo successful trials completed yet")

        print(f"Pausing for 3 seconds before next trial...")
        time.sleep(2)

    return study


def create_optimized_ensemble_model(best_params, cnn_path, rf_path):
    """Create the best ensemble model based on optimization results"""

    class OptimizedEnsembleModel(nn.Module):
        def __init__(self, cnn_model_path, rf_model_path, params):
            super().__init__()

            self.cnn_model = HexMagicNet()
            self.cnn_model.load_state_dict(torch.load(cnn_model_path))

            for param in self.cnn_model.parameters():
                param.requires_grad = False

            self.rf_model = joblib.load(rf_model_path)

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


def train_and_save_best_model(best_params, proton_file, gamma_file, cnn_path, rf_path, output_path, epochs=10):
    """Train and save the best model using optimal parameters"""
    print("Training optimized ensemble model...")
    dataset = MagicDataset(proton_file, gamma_file)

    ensemble_nametag = f"Optimized_Ensemble_{time.strftime('%d-%m-%Y_%H-%M-%S')}"
    ensemble_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "HybridSystem", ensemble_nametag)
    os.makedirs(ensemble_output_dir, exist_ok=True)

    model = create_optimized_ensemble_model(best_params, cnn_path, rf_path)

    supervisor = TrainingSupervisor("custom", dataset, ensemble_output_dir,
                                    debug_info=True, save_model=True,
                                    save_debug_data=True, early_stopping=False)

    supervisor.model = model.to(supervisor.device)

    supervisor.LEARNING_RATE = best_params['learning_rate']
    supervisor.WEIGHT_DECAY = best_params['weight_decay']
    supervisor.BATCH_SIZE = best_params['batch_size']
    supervisor.GRAD_CLIP_NORM = best_params['grad_clip_norm']

    # Train model
    print(f"Training optimized ensemble for {epochs} epochs...")
    supervisor.train_model(epochs)

    # Save the model
    torch.save(model.state_dict(), output_path)
    print(f"Optimized ensemble model saved to {output_path}")

    return output_path


def main():
    proton_file = "magic-protons.parquet"
    gamma_file = "magic-gammas-new.parquet"
    study_name = "Optimize_Ensemble"

    if not os.path.exists(CNN_MODEL_PATH):
        print(f"CNN model does not exist at {CNN_MODEL_PATH}. Please train it first.")
        return

    if not os.path.exists(RF_MODEL_PATH):
        print(f"Random Forest model does not exist at {RF_MODEL_PATH}. Please train it first.")
        return

    study = run_ensemble_optimization(
        proton_file=proton_file,
        gamma_file=gamma_file,
        cnn_path=CNN_MODEL_PATH,
        rf_path=RF_MODEL_PATH,
        study_name=study_name,
        n_trials=100,
        epochs=10
    )

    if study and has_completed_trials(study):
        print("\nOptimization completed!")
        print("\nBest trial:")
        print(f"  Value (Accuracy): {study.best_trial.value}")
        print("  Parameters:")
        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}")

        print("\nTraining the best model with full epochs...")
        final_model_path = train_and_save_best_model(
            best_params=study.best_trial.params,
            proton_file=proton_file,
            gamma_file=gamma_file,
            cnn_path=CNN_MODEL_PATH,
            rf_path=RF_MODEL_PATH,
            output_path=ENSEMBLE_MODEL_PATH,
            epochs=10
        )

        print("\nOptimized ensemble model training complete!")
        print(f"Model saved to: {final_model_path}")
    else:
        print("\nOptimization failed or no trials completed successfully")


if __name__ == "__main__":
    main()
