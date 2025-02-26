import os
import time
import json
import matplotlib.pyplot as plt
import numpy as np
from TrainingPipeline.TrainingSupervisor import TrainingSupervisor
from TrainingPipeline.MagicDataset import MagicDataset

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'


def train_with_energy_cutoff(model_name: str, proton_file: str, gamma_file: str,
                             min_energy: float, epochs: int):
    nametag = f"{model_name}_E{min_energy}_{time.strftime('%d-%m-%Y-%H-%M-%S')}"

    proton_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), proton_file)
    gamma_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), gamma_file)
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "energy_cutoff_models", nametag)

    print(f"Starting Training with settings:")
    print(f"\t- Model = {model_name}")
    print(f"\t- Data = {proton_dir, gamma_dir}")
    print(f"\t- Energy Cutoff = {min_energy}")
    print(f"\t- Epochs = {epochs}")
    print(f"\t- Output = {output_dir}\n")

    dataset = MagicDataset(proton_file, gamma_file, debug_info=True, min_energy=min_energy)

    supervisor = TrainingSupervisor(model_name, dataset, output_dir,
                                    debug_info=True, save_model=True, save_debug_data=True)
    print(f"Model has {supervisor._count_trainable_weights()} trainable weights.")
    supervisor.train_model(epochs)

    return {
        "energy_cutoff": min_energy,
        "train_accuracy": supervisor.train_metrics[-1]["accuracy"],
        "val_accuracy": supervisor.validation_metrics[-1]["accuracy"],
        "val_precision": supervisor.validation_metrics[-1]["precision"],
        "val_recall": supervisor.validation_metrics[-1]["recall"],
        "val_f1": supervisor.validation_metrics[-1]["f1"],
        "model_dir": output_dir
    }


def run_energy_cutoff_experiment(
        model_name: str,
        proton_file: str,
        gamma_file: str,
        start_energy: float = 0.0,
        step_size: float = 10.0,
        max_energy: float = 250.0,
        epochs: int = 10
):
    results = []
    cutoffs = np.arange(start_energy, max_energy + step_size, step_size)

    experiment_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "energy_cutoff_experiments",
        f"{model_name}_{time.strftime('%d-%m-%Y-%H-%M-%S')}"
    )
    os.makedirs(experiment_dir, exist_ok=True)

    for cutoff in cutoffs:
        print(f"\n{'=' * 80}")
        print(f"Training with energy cutoff: {cutoff}")
        print(f"{'=' * 80}\n")

        try:
            result = train_with_energy_cutoff(
                model_name, proton_file, gamma_file, cutoff, epochs
            )
            results.append(result)

            save_results(results, experiment_dir)

            plot_results(results, experiment_dir)

        except Exception as e:
            print(f"Error training with cutoff {cutoff}: {str(e)}")

    return results


def save_results(results, experiment_dir):
    results_file = os.path.join(experiment_dir, "energy_cutoff_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_file}")


def plot_results(results, experiment_dir):
    """Create plots of accuracy vs energy cutoff"""
    if not results:
        return

    energy_cutoffs = [r["energy_cutoff"] for r in results]
    train_accuracies = [r["train_accuracy"] for r in results]
    val_accuracies = [r["val_accuracy"] for r in results]
    val_precisions = [r["val_precision"] for r in results]
    val_recalls = [r["val_recall"] for r in results]
    val_f1s = [r["val_f1"] for r in results]

    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance vs Energy Cutoff', fontsize=16)

    axs[0, 0].plot(energy_cutoffs, train_accuracies, 'b-o', label='Training Accuracy')
    axs[0, 0].plot(energy_cutoffs, val_accuracies, 'r-o', label='Validation Accuracy')
    axs[0, 0].set_xlabel('Energy Cutoff')
    axs[0, 0].set_ylabel('Accuracy (%)')
    axs[0, 0].set_title('Accuracy vs Energy Cutoff')
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend()

    axs[0, 1].plot(energy_cutoffs, val_precisions, 'g-o')
    axs[0, 1].set_xlabel('Energy Cutoff')
    axs[0, 1].set_ylabel('Precision (%)')
    axs[0, 1].set_title('Validation Precision vs Energy Cutoff')
    axs[0, 1].grid(True, alpha=0.3)

    axs[1, 0].plot(energy_cutoffs, val_recalls, 'm-o')
    axs[1, 0].set_xlabel('Energy Cutoff')
    axs[1, 0].set_ylabel('Recall (%)')
    axs[1, 0].set_title('Validation Recall vs Energy Cutoff')
    axs[1, 0].grid(True, alpha=0.3)

    axs[1, 1].plot(energy_cutoffs, val_f1s, 'y-o')
    axs[1, 1].set_xlabel('Energy Cutoff')
    axs[1, 1].set_ylabel('F1 Score (%)')
    axs[1, 1].set_title('Validation F1 Score vs Energy Cutoff')
    axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout(rect=(0, 0, 1, 0.96))

    plt_file = os.path.join(experiment_dir, "energy_cutoff_plots.png")
    plt.savefig(plt_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plots saved to {plt_file}")

    plt.figure(figsize=(10, 6))
    plt.plot(energy_cutoffs, train_accuracies, 'b-o', label='Training Accuracy')
    plt.plot(energy_cutoffs, val_accuracies, 'r-o', label='Validation Accuracy')
    plt.xlabel('Energy Cutoff')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Energy Cutoff')
    plt.grid(True, alpha=0.3)
    plt.legend()

    acc_plt_file = os.path.join(experiment_dir, "accuracy_vs_energy.png")
    plt.savefig(acc_plt_file, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    MODEL_NAME = "hexmagicnet"
    PROTON_FILE = "magic-protons.parquet"
    GAMMA_FILE = "magic-gammas-new.parquet"
    START_ENERGY = 0.0
    STEP_SIZE = 25.0
    MAX_ENERGY = 500.0
    EPOCHS = 15

    results = run_energy_cutoff_experiment(
        MODEL_NAME,
        PROTON_FILE,
        GAMMA_FILE,
        start_energy=START_ENERGY,
        step_size=STEP_SIZE,
        max_energy=MAX_ENERGY,
        epochs=EPOCHS
    )

    print("\nExperiment completed!")
    print("Trained models at energy cutoffs: {}".format([r['energy_cutoff'] for r in results]))