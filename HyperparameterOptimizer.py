import optuna
import os
import time
import gc
import torch
import sys
import multiprocessing as mp
from functools import partial
import subprocess
import json

OPTUNA_DB = "sqlite:///optuna_study.db"
MAX_BATCH_SIZE = 2  # Number of trials to run before restarting the process


def run_trial_subprocess(trial_id, study_name, model_name, proton_file, gamma_file, epochs):
    """
    Run a single trial as a completely separate Python process.
    This ensures complete isolation and memory cleanup between trials.
    """
    cmd = [
        sys.executable,
        "-c",
        f"""
import optuna
import os
import sys
import torch
import gc
import json

# Import your modules
from ParameterTuning.HexMagicCNN import parameterize_hex_magicnet
from TrainingPipeline.MagicDataset import MagicDataset
from TrainingPipeline.TrainingSupervisor import TrainingSupervisor
from ParameterTuning import *

# Set environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# Load study
storage = optuna.storages.RDBStorage(url="{OPTUNA_DB}")
study = optuna.load_study(study_name="{study_name}", storage=storage)

# Get a frozen trial
frozen_trial = study.ask()

try:
    # Load dataset
    dataset = MagicDataset("{proton_file}", "{gamma_file}", max_samples=100000, debug_info=False)

    # Create output dir
    nametag = f"{study_name}_{{time.strftime('%Y-%m-%d_%H-%M-%S')}}_trial_{{frozen_trial.number}}"
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            f"parameter_tuning/{study_name}", nametag)

    # Initialize supervisor
    supervisor = TrainingSupervisor("{model_name}", dataset, output_dir, 
                                   debug_info=False, save_model=False, save_debug_data=False)

    # Create model based on parameters
    if "{model_name}".lower() == "basicmagicnet":
        parameterize_func = parameterize_BasicMagicNet
    elif "{model_name}".lower() == "hexcirclenet":
        parameterize_func = parameterize_HexCircleNet
    elif "{model_name}".lower() == "hexmagicnet":
        parameterize_func = parameterize_hex_magicnet
    else:
        raise ValueError(f"Invalid Modelname for parameterization: '{model_name}'")

    # Set up the model
    supervisor.model = parameterize_func(frozen_trial).to(supervisor.device)

    # Set hyperparameters
    supervisor.LEARNING_RATE = frozen_trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True)
    supervisor.WEIGHT_DECAY = frozen_trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True)
    supervisor.GRAD_CLIP_NORM = frozen_trial.suggest_float('grad_clip_norm', 0.1, 5.0)

    # Train the model
    supervisor.train_model({epochs})

    # Calculate accuracy
    last_n_accuracies = [metrics['accuracy'] for metrics in supervisor.validation_metrics[-3:]]
    avg_accuracy = sum(last_n_accuracies) / len(last_n_accuracies)

    # Tell the study about the result
    study.tell(frozen_trial, avg_accuracy)

    # Print result for parent process to capture
    result = {{"trial_id": frozen_trial.number, "value": avg_accuracy, "status": "completed"}}
    print(json.dumps(result))

    # Exit with success
    sys.exit(0)

except Exception as e:
    # Tell the study that the trial failed
    study.tell(frozen_trial, state=optuna.trial.TrialState.FAIL)

    # Print error for parent process to capture
    result = {{"trial_id": frozen_trial.number, "error": str(e), "status": "failed"}}
    print(json.dumps(result))

    # Exit with error
    sys.exit(1)
finally:
    # Ensure everything is cleaned up
    if 'supervisor' in locals() and hasattr(supervisor, 'model'):
        supervisor.model = supervisor.model.cpu()
        del supervisor.model

    # Force cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
"""
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        try:
            output_lines = result.stdout.strip().split("\n")
            for line in output_lines:
                if line.startswith("{") and line.endswith("}"):
                    return json.loads(line)
        except json.JSONDecodeError:
            pass

        return {"trial_id": trial_id, "status": "completed", "stdout": result.stdout, "stderr": result.stderr}

    except subprocess.CalledProcessError as e:
        return {"trial_id": trial_id, "status": "failed", "error": str(e), "stdout": e.stdout, "stderr": e.stderr}


def create_or_load_study(study_name):
    """Create a new study or load an existing one"""
    storage = optuna.storages.RDBStorage(url=OPTUNA_DB)

    try:
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
        return None


def run_optimization(model_name, proton_file, gamma_file, study_name, n_trials, epochs, max_concurrent=2):
    """Run the optimization with complete process isolation for each trial"""
    study = create_or_load_study(study_name)
    if study is None:
        print("Failed to create or load study")
        return

    n_completed = len([t for t in study.trials if t.state.is_finished()])
    print(f"Already completed trials: {n_completed}")

    n_remaining = max(0, n_trials - n_completed)
    print(f"Remaining trials to run: {n_remaining}")

    if n_remaining <= 0:
        print("All trials have been completed")
        return study

    trial_ids = list(range(n_remaining))

    n_processes = min(max_concurrent, mp.cpu_count() - 1)
    print(f"Running with {n_processes} parallel processes")

    with mp.Pool(processes=n_processes) as pool:
        run_trial_fn = partial(
            run_trial_subprocess,
            study_name=study_name,
            model_name=model_name,
            proton_file=proton_file,
            gamma_file=gamma_file,
            epochs=epochs
        )

        for batch_start in range(0, n_remaining, MAX_BATCH_SIZE):
            batch_end = min(batch_start + MAX_BATCH_SIZE, n_remaining)
            batch_size = batch_end - batch_start

            print(f"\nRunning batch of {batch_size} trials ({batch_start + 1}-{batch_end} of {n_remaining})")

            batch_results = pool.map(run_trial_fn, trial_ids[batch_start:batch_end])

            for result in batch_results:
                if result.get("status") == "completed":
                    print(f"Trial {result.get('trial_id')} completed with value: {result.get('value')}")
                else:
                    print(f"Trial {result.get('trial_id')} failed: {result.get('error')}")

            print("\nCurrent best trial:")
            best_trial = study.best_trial
            print(f"  Value (Accuracy): {best_trial.value}")
            print(f"  Parameters: {best_trial.params}")

            if batch_end < n_remaining:
                print(f"Pausing for 5 seconds before next batch...")
                time.sleep(5)

    return study


def main():
    model_name = "hexmagicnet"
    proton_file = "magic-protons.parquet"
    gamma_file = "magic-gammas-new.parquet"
    study_name = f"Optimize_{model_name}"

    study = run_optimization(
        model_name=model_name,
        proton_file=proton_file,
        gamma_file=gamma_file,
        study_name=study_name,
        n_trials=300,
        epochs=10,
        max_concurrent=2
    )

    if study:
        print("\nOptimization completed!")
        print("\nBest trial:")
        print(f"  Value (Accuracy): {study.best_trial.value}")
        print("  Parameters:")
        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}")


if __name__ == "__main__":
    main()