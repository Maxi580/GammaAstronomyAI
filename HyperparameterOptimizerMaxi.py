import optuna
import os
import time
import sys
import subprocess
import json
import traceback

OPTUNA_DB = "sqlite:///optuna_study.db"


def run_trial_subprocess(trial_id, study_name, model_name, proton_file, gamma_file, epochs):
    """
    Run a single trial as a completely separate Python process.
    This ensures complete isolation and memory cleanup between trials.
    """
    print(f"\n==== Starting Trial {trial_id} ====")

    script_content = f"""
import optuna
import os
import sys
import torch
import gc
import json
import time
import traceback

# Import your modules
try:
    from ParameterTuning.HexMagicNet import parameterize_hex_magicnet
    from TrainingPipeline.MagicDataset import MagicDataset
    from TrainingPipeline.TrainingSupervisor import TrainingSupervisor
    from ParameterTuning import *
except ImportError as e:
    print(json.dumps({{"trial_id": -1, "error": f"Import error: {{e}}", "status": "failed", "traceback": traceback.format_exc()}}))
    sys.exit(1)

# Set environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# Load study
try:
    storage = optuna.storages.RDBStorage(url="{OPTUNA_DB}")
    study = optuna.load_study(study_name="{study_name}", storage=storage)
except Exception as e:
    print(json.dumps({{"trial_id": -1, "error": f"Study load error: {{e}}", "status": "failed", "traceback": traceback.format_exc()}}))
    sys.exit(1)

# Get a frozen trial
try:
    frozen_trial = study.ask()
    trial_id = frozen_trial.number
except Exception as e:
    print(json.dumps({{"trial_id": -1, "error": f"Trial ask error: {{e}}", "status": "failed", "traceback": traceback.format_exc()}}))
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
    supervisor = TrainingSupervisor("{model_name}", dataset, output_dir, 
                                   debug_info=False, save_model=False, save_debug_data=False)

    # Create model based on parameters
    print(f"Creating model...")
    if "{model_name}".lower() == "hexmagicnet":
        parameterize_func = parameterize_hex_magicnet
    else:
        raise ValueError(f"Invalid Modelname for parameterization: '{model_name}'")

    # Set up the model
    supervisor.model = parameterize_func(frozen_trial).to(supervisor.device)

    # Set hyperparameters
    supervisor.LEARNING_RATE = frozen_trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True)
    supervisor.WEIGHT_DECAY = frozen_trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True)
    supervisor.GRAD_CLIP_NORM = frozen_trial.suggest_float('grad_clip_norm', 0.1, 5.0)

    # Train the model - Note: using actual value {epochs} not the placeholder
    print(f"Training model for {epochs} epochs...")
    supervisor.train_model({epochs})

    # Calculate accuracy
    last_n_accuracies = [metrics['accuracy'] for metrics in supervisor.validation_metrics[-3:]]
    avg_accuracy = sum(last_n_accuracies) / len(last_n_accuracies)

    # Tell the study about the result
    print(f"Trial complete. Accuracy: {{avg_accuracy}}")
    study.tell(frozen_trial, avg_accuracy)

    # Print result for parent process to capture
    result = {{"trial_id": trial_id, "value": avg_accuracy, "status": "completed"}}
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
    # Ensure everything is cleaned up
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

    script_filename = f"temp_trial_{trial_id}_{int(time.time())}.py"
    with open(script_filename, "w") as f:
        f.write(script_content)

    try:
        print(f"Executing trial {trial_id} in a separate process...")
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


def run_optimization(model_name, proton_file, gamma_file, study_name, n_trials, epochs):
    """Run the optimization with sequential trial execution"""
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

    for trial_id in range(n_remaining):
        result = run_trial_subprocess(
            trial_id=trial_id,
            study_name=study_name,
            model_name=model_name,
            proton_file=proton_file,
            gamma_file=gamma_file,
            epochs=epochs
        )

        if result.get("status") == "completed":
            print(f"Trial {result.get('trial_id')} completed with value: {result.get('value')}")
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
        time.sleep(3)

    return study


def main():
    model_name = "hexmagicnet"
    proton_file = "magic-protons.parquet"
    gamma_file = "magic-gammas-new.parquet"
    study_name = f"Final_v2_Optimize_{model_name}"

    study = create_or_load_study(study_name)
    if study is None:
        print("Failed to create or load study, exiting")
        return

    study = run_optimization(
        model_name=model_name,
        proton_file=proton_file,
        gamma_file=gamma_file,
        study_name=study_name,
        n_trials=300,
        epochs=10
    )

    if study and has_completed_trials(study):
        print("\nOptimization completed!")
        print("\nBest trial:")
        print(f"  Value (Accuracy): {study.best_trial.value}")
        print("  Parameters:")
        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}")
    else:
        print("\nOptimization failed or no trials completed successfully")


if __name__ == "__main__":
    main()