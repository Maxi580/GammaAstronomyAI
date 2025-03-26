import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CNN.Architectures.HexCircleCNN import HexCircleNet
from TrainingPipeline.Datasets import *


def evaluate_random_samples(model_path, proton_file, gamma_file, num_samples=10000):
    dataset = MagicDataset(proton_file, gamma_file, max_samples=100000, rescale_image=False, additional_features=['true_energy'])

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    model = HexCircleNet()
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model = model.to(device)
    model.eval()

    correct = 0
    
    protons_correct = []
    protons_incorrect = []
    gammas_correct = []
    gammas_incorrect = []

    with torch.no_grad():
        for i in range(num_samples):
            idx = np.random.randint(0, len(dataset))
            m1, m2, features, label, [energy] = dataset[idx]

            m1 = m1.unsqueeze(0).to(device)
            m2 = m2.unsqueeze(0).to(device)
            features = features.unsqueeze(0).to(device)

            output = model(m1, m2, features)
            
            probabilities = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1).item()
            gamma_prob = probabilities[0, dataset.labels[dataset.GAMMA_LABEL]].item()  # "gammacity"

            is_correct = (pred == label)
            correct += is_correct
            
            if label == dataset.labels[dataset.PROTON_LABEL]:  # Proton
                if is_correct:
                    protons_correct.append((gamma_prob, energy))
                else:
                    protons_incorrect.append((gamma_prob, energy))
            else:  # Gamma
                if is_correct:
                    gammas_correct.append((gamma_prob, energy))
                else:
                    gammas_incorrect.append((gamma_prob, energy))
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{num_samples} samples")

    accuracy = (correct / num_samples) * 100
    print(f"\nEvaluation Results:")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    
    # Helper function to safely extract x and y coordinates from a list of tuples
    def extract_coords(data):
        if data:
            # Unzip the list of tuples into two lists: x and y values
            x, y = zip(*data)
            return x, y
        else:
            return [], []

    # Extract coordinates for each category
    pc_x, pc_y = extract_coords(protons_correct)
    pi_x, pi_y = extract_coords(protons_incorrect)
    gc_x, gc_y = extract_coords(gammas_correct)
    gi_x, gi_y = extract_coords(gammas_incorrect)

    # Create the scatter plot
    plt.figure(figsize=(8, 6))

    plt.scatter(pc_x, pc_y, color='blue', label='Protons Correct', marker='o')
    plt.scatter(gc_x, gc_y, color='red', label='Gammas Correct', marker='o')
    plt.scatter(pi_x, pi_y, color='lightblue', label='Protons Incorrect', marker='o')
    plt.scatter(gi_x, gi_y, color='salmon', label='Gammas Incorrect', marker='o')

    plt.xlabel('Gammacity')
    plt.ylabel('Energy')
    plt.yscale('log')
    plt.title('Scatter Plot of Gammacity vs Energy')
    plt.legend()
    plt.tight_layout()
    plt.savefig("gammacity_vs_energy.png")
    plt.show()
    plt.clf()
    
    # Only incorrect ones
    plt.scatter(pi_x, pi_y, color='lightblue', label='Protons Incorrect', marker='o')
    plt.scatter(gi_x, gi_y, color='salmon', label='Gammas Incorrect', marker='o')

    plt.xlabel('Gammacity')
    plt.ylabel('Energy')
    plt.yscale('log')
    plt.title('Scatter Plot (Incorrect Only) of Gammacity vs Energy')
    plt.legend()
    plt.tight_layout()
    plt.savefig("gammacity_vs_energy_incorrect.png")
    plt.show()
    plt.clf()


    # --------------------------
    # Histogram of Binned Average Energy
    # --------------------------
    # For the histogram we combine correct and incorrect for each category.
    protons_data = protons_correct + protons_incorrect
    gammas_data = gammas_correct + gammas_incorrect

    # Convert list of tuples into separate numpy arrays for gammacity (x) and energy (y)
    if protons_data:
        p_x, p_y = zip(*protons_data)
        p_x = np.array(p_x)
        p_y = np.array(p_y)
    else:
        p_x, p_y = np.array([]), np.array([])

    if gammas_data:
        g_x, g_y = zip(*gammas_data)
        g_x = np.array(g_x)
        g_y = np.array(g_y)
    else:
        g_x, g_y = np.array([]), np.array([])

    # Define a function to compute the binned average energy.
    def binned_average(x, y, bins):
        counts, _ = np.histogram(x, bins=bins)
        sum_energy, _ = np.histogram(x, bins=bins, weights=y)
        # Compute average energy per bin; avoid division by zero
        avg_energy = np.divide(sum_energy, counts, out=np.zeros_like(sum_energy), where=counts>0)
        # Set bins with no data to NaN (optional)
        avg_energy[counts == 0] = np.nan
        return avg_energy, counts

    # Create 100 bins over the range of gammacity for each category.
    if p_x.size > 0:
        bins_protons = np.linspace(p_x.min(), p_x.max(), 101)
    else:
        bins_protons = np.linspace(0, 1, 101)

    if g_x.size > 0:
        bins_gammas = np.linspace(g_x.min(), g_x.max(), 101)
    else:
        bins_gammas = np.linspace(0, 1, 101)
    
    # Compute bin centers for plotting.
    p_bin_centers = (bins_protons[:-1] + bins_protons[1:]) / 2
    g_bin_centers = (bins_gammas[:-1] + bins_gammas[1:]) / 2

    p_avg_energy, p_counts = binned_average(p_x, p_y, bins_protons)
    g_avg_energy, g_counts = binned_average(g_x, g_y, bins_gammas)


    # Plot the histograms (bar plots) for protons and gammas.

    # Protons histogram: x-axis = gammacity, y-axis = average energy (log scale)
    plt.step(p_bin_centers, p_avg_energy, label='Protons')
    plt.step(g_bin_centers, g_avg_energy, label='Gammas')
    plt.xlabel('Gammacity')
    plt.ylabel('Average Energy')
    plt.yscale('log')
    plt.title('Average Energy vs Gammacity')
    plt.legend()
    plt.tight_layout()
    plt.savefig("gammacity_vs_energy_average.png")
    plt.show()
    plt.clf()
    
    # --------------------------
    # Histogram of Binned Median Energy
    # --------------------------

    # Function to compute binned median energy
    def binned_median(x, y, bins):
        # Digitize x-values into bins (indices are 1-indexed)
        bin_indices = np.digitize(x, bins)
        medians = []
        for i in range(1, len(bins)):
            # Get energy values corresponding to the i-th bin
            bin_y = y[bin_indices == i]
            if len(bin_y) > 0:
                medians.append(np.median(bin_y))
            else:
                medians.append(np.nan)
        return np.array(medians)

    p_median_energy = binned_median(p_x, p_y, bins_protons)
    g_median_energy = binned_median(g_x, g_y, bins_gammas)

    # Plot the histograms (bar plots) for protons and gammas using the median energy
    
    # Protons histogram: x-axis = gammacity, y-axis = median energy (log scale)
    plt.step(p_bin_centers, p_median_energy, label='Protons')
    plt.step(g_bin_centers, g_median_energy, label='Gammas')
    plt.xlabel('Gammacity')
    plt.ylabel('Median Energy')
    plt.yscale('log')
    plt.title('Median Energy vs Gammacity')
    plt.legend()
    plt.tight_layout()
    plt.savefig("gammacity_vs_energy_median.png")
    plt.show()
    plt.clf()
    
    # --------------------------
    # Histogram of Binned Geometric Mean Energy
    # --------------------------
    
    # Function to compute binned geometric mean energy
    def binned_geometric_mean(x, y, bins):
        # Digitize x-values into bins (indices are 1-indexed)
        bin_indices = np.digitize(x, bins)
        geo_means = []
        for i in range(1, len(bins)):
            # Get energy values corresponding to the i-th bin
            bin_y = y[bin_indices == i]
            if len(bin_y) > 0:
                # Compute the geometric mean: exp(mean(log(values)))
                gm = np.exp(np.mean(np.log(bin_y)))
                geo_means.append(gm)
            else:
                geo_means.append(np.nan)
        return np.array(geo_means)
    
    p_geo_median_energy = binned_geometric_mean(p_x, p_y, bins_protons)
    g_geo_median_energy = binned_geometric_mean(g_x, g_y, bins_gammas)
    
    # Protons histogram: x-axis = gammacity, y-axis = geometric mean energy (log scale)
    plt.step(p_bin_centers, p_geo_median_energy, label='Protons')
    plt.step(g_bin_centers, g_geo_median_energy, label='Gammas')
    plt.xlabel('Gammacity')
    plt.ylabel('Geometric Mean Energy')
    plt.yscale('log')
    plt.title('Geometric Mean Energy vs Gammacity')
    plt.legend()
    plt.tight_layout()
    plt.savefig("gammacity_vs_energy_geometricmean.png")
    plt.show()
    plt.clf()


if __name__ == "__main__":
    MODEL_PATH = "trained_model.pth"
    PROTON_FILE = "magic-protons.parquet"
    GAMMA_FILE = "magic-gammas-new-2.parquet"

    evaluate_random_samples(MODEL_PATH, PROTON_FILE, GAMMA_FILE)
