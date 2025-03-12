import os
import time
import json
import matplotlib.pyplot as plt
import numpy as np
from TrainingPipeline.MagicDataset import MagicDataset


def analyze_energy_cutoffs(proton_file: str, gamma_file: str,
                           start_energy: float = 0.0,
                           step_size: float = 50.0,
                           max_energy: float = 500.0):
    """Analyze how many events remain after different energy cutoffs using MagicDataset"""

    baseline_dataset = MagicDataset(proton_file, gamma_file, debug_info=False)

    initial_protons = baseline_dataset.n_protons
    initial_gammas = baseline_dataset.n_gammas
    initial_total = initial_protons + initial_gammas

    print(f"Initial counts:")
    print(f"Protons: {initial_protons}")
    print(f"Gammas: {initial_gammas}")
    print(f"Total: {initial_total}")

    energy_cutoffs = list(np.arange(start_energy, max_energy + step_size, step_size))

    proton_counts = []
    gamma_counts = []
    total_counts = []
    proton_percentages = []
    gamma_percentages = []
    total_percentages = []

    for cutoff in energy_cutoffs:
        print(f"\nAnalyzing energy cutoff: {cutoff} GeV")

        current_dataset = MagicDataset(
            proton_file,
            gamma_file,
            debug_info=False,
            min_true_energy=cutoff
        )

        proton_count = current_dataset.n_protons
        gamma_count = current_dataset.n_gammas
        total_count = proton_count + gamma_count

        proton_percentage = (proton_count / initial_protons) * 100
        gamma_percentage = (gamma_count / initial_gammas) * 100
        total_percentage = (total_count / initial_total) * 100

        proton_counts.append(proton_count)
        gamma_counts.append(gamma_count)
        total_counts.append(total_count)
        proton_percentages.append(proton_percentage)
        gamma_percentages.append(gamma_percentage)
        total_percentages.append(total_percentage)

        print(f"Energy cutoff {cutoff} GeV results:")
        print(f"  Protons: {proton_count} ({proton_percentage:.2f}%)")
        print(f"  Gammas: {gamma_count} ({gamma_percentage:.2f}%)")
        print(f"  Total: {total_count} ({total_percentage:.2f}%)")

    return {
        'energy_cutoffs': energy_cutoffs,
        'initial_counts': {
            'protons': initial_protons,
            'gammas': initial_gammas,
            'total': initial_total
        },
        'counts': {
            'protons': proton_counts,
            'gammas': gamma_counts,
            'total': total_counts
        },
        'percentages': {
            'protons': proton_percentages,
            'gammas': gamma_percentages,
            'total': total_percentages
        }
    }


def plot_energy_cutoff_results(results, output_dir):
    """Create plots showing percentage of events remaining after energy cutoffs"""

    os.makedirs(output_dir, exist_ok=True)

    energy_cutoffs = results['energy_cutoffs']
    initial_protons = results['initial_counts']['protons']
    initial_gammas = results['initial_counts']['gammas']
    initial_total = results['initial_counts']['total']
    proton_percentages = results['percentages']['protons']
    gamma_percentages = results['percentages']['gammas']
    total_percentages = results['percentages']['total']

    plt.figure(figsize=(10, 7))
    plt.plot(energy_cutoffs, proton_percentages, 'b-o', label='Protons')
    plt.plot(energy_cutoffs, gamma_percentages, 'r-o', label='Gammas')
    plt.plot(energy_cutoffs, total_percentages, 'g-o', label='Total')

    plt.xlabel('Energy Cutoff (GeV)')
    plt.ylabel('Percentage Remaining (%)')
    plt.title('Percentage of Particles Remaining after Energy Cutoff')

    plt.text(0.98, 0.95, f"Initial protons: {initial_protons:,}",
             transform=plt.gca().transAxes, ha='right', va='top')
    plt.text(0.98, 0.91, f"Initial gammas: {initial_gammas:,}",
             transform=plt.gca().transAxes, ha='right', va='top')
    plt.text(0.98, 0.87, f"Initial total: {initial_total:,}",
             transform=plt.gca().transAxes, ha='right', va='top')

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    min_percentage = min(min(proton_percentages), min(gamma_percentages), min(total_percentages))
    plt.ylim(bottom=max(0, min_percentage - 5), top=101)

    # Save plot
    plt_file = os.path.join(output_dir, "energy_cutoff_percentages.png")
    plt.savefig(plt_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {plt_file}")

    results_file = os.path.join(output_dir, "energy_cutoff_results.json")
    with open(results_file, 'w') as f:
        json_friendly_results = {
            'energy_cutoffs': energy_cutoffs,
            'initial_counts': {
                'protons': int(initial_protons),
                'gammas': int(initial_gammas),
                'total': int(initial_total)
            },
            'counts': {
                'protons': [int(count) for count in results['counts']['protons']],
                'gammas': [int(count) for count in results['counts']['gammas']],
                'total': [int(count) for count in results['counts']['total']]
            },
            'percentages': {
                'protons': [float(p) for p in proton_percentages],
                'gammas': [float(p) for p in gamma_percentages],
                'total': [float(p) for p in total_percentages]
            }
        }
        json.dump(json_friendly_results, f, indent=4)
    print(f"Results saved to {results_file}")


def main():
    PROTON_FILE = "magic-protons.parquet"
    GAMMA_FILE = "magic-gammas.parquet"
    START_ENERGY = 0.0
    STEP_SIZE = 50.0
    MAX_ENERGY = 500.0

    output_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "energy_cutoff_analysis",
        f"analysis_{time.strftime('%d-%m-%Y-%H-%M-%S')}"
    )

    print(f"Starting energy cutoff analysis:")
    print(f"\t- Proton File = {PROTON_FILE}")
    print(f"\t- Gamma File = {GAMMA_FILE}")
    print(f"\t- Energy Range = {START_ENERGY} to {MAX_ENERGY} GeV, step {STEP_SIZE} GeV")
    print(f"\t- Output = {output_dir}\n")

    results = analyze_energy_cutoffs(
        PROTON_FILE,
        GAMMA_FILE,
        start_energy=START_ENERGY,
        step_size=STEP_SIZE,
        max_energy=MAX_ENERGY
    )

    plot_energy_cutoff_results(results, output_dir)

    print("\nAnalysis completed!")


if __name__ == "__main__":
    main()