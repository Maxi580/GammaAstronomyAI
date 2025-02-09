import pandas as pd
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
from ctapipe.instrument import CameraGeometry

parquet_file_gamma = "../magic-gammas.parquet"
parquet_file_proton = "../magic-protons.parquet"

class GammaDataset(Dataset):
    """
    Reads the parquet file containing two columns:
      - "image_m1": the raw image (particle + noise)
      - "clean_image_m1": the cleaned image (particle only)
    Returns:
      x_raw: the raw image,
      x_clean: the cleaned image.
    """
    def __init__(self, parquet_file):
        self.df = pd.read_parquet(parquet_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Use the first 1039 pixels.
        x_raw = torch.tensor(row["image_m2"][:1039], dtype=torch.float32)
        x_clean = torch.tensor(row["clean_image_m2"][:1039], dtype=torch.float32)
        return x_raw, x_clean


def plot_noise_comparison(tensors_to_plot, names):
    """
    Plots 1D images over the MAGICCam geometry.
    """
    geom = CameraGeometry.from_name("MAGICCam")
    pix_x = geom.pix_x.value
    pix_y = geom.pix_y.value
    num_plots = len(tensors_to_plot)
    cols = min(num_plots, 2) if num_plots <= 4 else int(np.ceil(np.sqrt(num_plots)))
    rows = int(np.ceil(num_plots / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 7 * rows))
    if num_plots == 1:
        axes = [axes]
    elif isinstance(axes, np.ndarray):
        axes = axes.flatten()
    for i, (tensor, name) in enumerate(zip(tensors_to_plot, names)):
        t = tensor.detach().cpu().numpy()
        ax = axes[i]
        sc = ax.scatter(pix_x, pix_y, c=t, cmap='plasma', s=50)
        ax.set_title(name)
        fig.colorbar(sc, ax=ax, label="Intensity")
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()


def calculate_average_pixel_values(dataset):
    """Calculates the average value for each pixel in the dataset."""
    num_pixels = 1039
    pixel_sum = np.zeros(num_pixels)
    for i in range(len(dataset)):
        x_raw, x_clean = dataset[i]
        x_raw = x_raw - x_clean
        pixel_sum += x_raw.numpy()
    average_values = pixel_sum / len(dataset)
    return torch.tensor(average_values)

def calculate_value_counts(dataset):
    """Calculates the count of each value in the dataset, sorted by count."""
    value_counts = {}
    for i in range(len(dataset)):
        x_raw, x_clean = dataset[i]
        x_raw = x_raw - x_clean
        for value in x_raw.numpy():
            if value in value_counts:
                value_counts[value] += 1
            else:
                value_counts[value] = 1
    # Sort by count in descending order
    sorted_counts = sorted(value_counts.items(), key=lambda item: item[1], reverse=True)
    return sorted_counts

def calculate_pixel_spread(dataset):
    """Calculates the standard deviation (spread) for each pixel across the dataset."""
    num_pixels = 1039
    pixel_values = [[] for _ in range(num_pixels)] # List to store pixel values for each pixel
    for i in range(len(dataset)):
        x_raw, x_clean = dataset[i]
        x_raw = x_raw - x_clean
        for pixel_index, value in enumerate(x_raw.numpy()):
            pixel_values[pixel_index].append(value)

    pixel_spread = np.zeros(num_pixels)
    for pixel_index in range(num_pixels):
        pixel_spread[pixel_index] = np.std(pixel_values[pixel_index]) # Calculate standard deviation for each pixel

    return torch.tensor(pixel_spread)

def compare_value_counts(gamma_counts, proton_counts):
    """Compares value counts and finds values with significantly different *normalized* counts,
       accounting for potential differences in dataset lengths.
    """
    gamma_counts_dict = dict(gamma_counts)
    proton_counts_dict = dict(proton_counts)

    gamma_total_counts = sum(gamma_counts_dict.values())
    proton_total_counts = sum(proton_counts_dict.values())

    all_values = set(gamma_counts_dict.keys()) | set(proton_counts_dict.keys())
    count_differences = []

    for value in all_values:
        gamma_count = gamma_counts_dict.get(value, 0)
        proton_count = proton_counts_dict.get(value, 0)

        # Normalize counts to proportions
        gamma_proportion = gamma_count / gamma_total_counts if gamma_total_counts > 0 else 0
        proton_proportion = proton_count / proton_total_counts if proton_total_counts > 0 else 0

        difference = gamma_proportion - proton_proportion
        count_differences.append((value, difference))

    sorted_differences = sorted(count_differences, key=lambda item: abs(item[1]), reverse=True)
    return sorted_differences

def save_average_pixel_values(avg_gamma_pixels, avg_proton_pixels, filename_gamma="avg_gamma_pixels.npy", filename_proton="avg_proton_pixels.npy"):
    """Saves average pixel values to .npy files."""
    np.save(filename_gamma, avg_gamma_pixels.numpy())
    np.save(filename_proton, avg_proton_pixels.numpy())
    print(f"Average pixel values saved to: {filename_gamma}, {filename_proton}")

def save_pixel_spread(spread_gamma_pixels, spread_proton_pixels, filename_gamma="spread_gamma_pixels.npy", filename_proton="spread_proton_pixels.npy"):
    """Saves pixel spread values to .npy files."""
    np.save(filename_gamma, spread_gamma_pixels.numpy())
    np.save(filename_proton, spread_proton_pixels.numpy())
    print(f"Pixel spread values saved to: {filename_gamma}, {filename_proton}")

def save_value_counts(gamma_value_counts, proton_value_counts, filename_gamma="gamma_value_counts.txt", filename_proton="proton_value_counts.txt"):
    """Saves value counts to .txt files."""
    with open(filename_gamma, 'w') as f:
        for value, count in gamma_value_counts:
            f.write(f"{value} {count}\n")
    with open(filename_proton, 'w') as f:
        for value, count in proton_value_counts:
            f.write(f"{value} {count}\n")
    print(f"Value counts saved to: {filename_gamma}, {filename_proton}")

def save_value_count_comparison(value_count_differences, filename="value_count_comparison.txt"):
    """Saves value count comparison to a .txt file."""
    with open(filename, 'w') as f:
        f.write("Value Count Comparison (Gamma - Proton, Normalized)\n")
        f.write("Value\tNormalized Difference (Gamma - Proton)\n")
        for value, difference in value_count_differences:
            f.write(f"{value}\t{difference}\n")
    print(f"Value count comparison saved to: {filename}")

def calculate_pixel_value_counts(dataset):
    """Calculates value counts per pixel, sorted by count for each pixel."""
    num_pixels = 1039
    pixel_value_counts = [{} for _ in range(num_pixels)] # List of dictionaries, one per pixel

    for i in range(len(dataset)):
        x_raw, x_clean = dataset[i]
        x_raw = x_raw - x_clean
        for pixel_index, value in enumerate(x_raw.numpy()):
            if value in pixel_value_counts[pixel_index]:
                pixel_value_counts[pixel_index][value] += 1
            else:
                pixel_value_counts[pixel_index][value] = 1

    sorted_pixel_value_counts = []
    for pixel_idx_counts in pixel_value_counts:
        sorted_counts = sorted(pixel_idx_counts.items(), key=lambda item: item[1], reverse=True)
        sorted_pixel_value_counts.append(sorted_counts) # List of lists of (value, count) tuples, per pixel

    return sorted_pixel_value_counts

def print_pixel_value_count_outliers(sorted_pixel_value_counts, dataset_name, outlier_threshold_ratio=0.1, min_count_to_print=10):
    """Prints value count outliers per pixel to console if count > min_count_to_print."""
    print(f"\nPixel Value Count Outliers for {dataset_name} (Outlier Threshold Ratio: {outlier_threshold_ratio}, Min Count to Print: {min_count_to_print}):")
    for pixel_index, sorted_counts in enumerate(sorted_pixel_value_counts):
        if not sorted_counts: # Skip pixels with no values (shouldn't happen in this dataset but for robustness)
            continue
        most_frequent_count = sorted_counts[0][1]
        for value, count in sorted_counts[1:]: # Iterate from the second most frequent value onwards
            if count < most_frequent_count * outlier_threshold_ratio and count > min_count_to_print: # Check outlier threshold AND min count
                print(f"  Pixel {pixel_index}: Value {value:.3f} appears {count} times (most frequent: {sorted_counts[0][0]:.3f} - {most_frequent_count} times)")

def save_pixel_value_counts(sorted_pixel_value_counts, dataset_name, filename):
    """Saves pixel value counts to a file."""
    with open(filename, 'w') as f:
        f.write(f"Pixel Value Counts for {dataset_name}\n")
        for pixel_index, sorted_counts in enumerate(sorted_pixel_value_counts):
            f.write(f"Pixel {pixel_index}:\n")
            for value, count in sorted_counts:
                f.write(f"  Value: {value}, Count: {count}\n")
    print(f"Pixel value counts for {dataset_name} saved to: {filename}")

def save_pixel_value_count_outliers(sorted_pixel_value_counts, dataset_name, filename, outlier_threshold_ratio=0.1):
    """Saves pixel value count outliers to a file."""
    with open(filename, 'w') as f:
        f.write(f"Pixel Value Count Outliers for {dataset_name} (Outlier Threshold Ratio: {outlier_threshold_ratio})\n")
        for pixel_index, sorted_counts in enumerate(sorted_pixel_value_counts):
            if not sorted_counts:
                continue
            most_frequent_count = sorted_counts[0][1]
            for value, count in sorted_counts[1:]:
                if count < most_frequent_count * outlier_threshold_ratio:
                    f.write(f"  Pixel {pixel_index}: Value {value} appears {count} times (most frequent: {sorted_counts[0][0]} - {most_frequent_count} times)\n")
    print(f"Pixel value count outliers for {dataset_name} saved to: {filename}")


if __name__ == "__main__":
    gamma_dataset = GammaDataset(parquet_file_gamma)
    proton_dataset = GammaDataset(parquet_file_proton)

    # Calculate average pixel values
    avg_gamma_pixels = calculate_average_pixel_values(gamma_dataset)
    avg_proton_pixels = calculate_average_pixel_values(proton_dataset)
    diff_pixels = avg_gamma_pixels - avg_proton_pixels
    diff_pixels_inv = avg_proton_pixels - avg_gamma_pixels

    # Plot average pixel values as images
    plot_noise_comparison(
        [avg_gamma_pixels, avg_proton_pixels, diff_pixels, diff_pixels_inv],
        ["Average Gamma Pixel Values", "Average Proton Pixel Values", "Difference (Gamma - Proton), Avg", "Difference (Proton - Gamma), Avg"]
    )

    # Calculate pixel spread (standard deviation)
    spread_gamma_pixels = calculate_pixel_spread(gamma_dataset)
    spread_proton_pixels = calculate_pixel_spread(proton_dataset)

    # Plot pixel spread as images
    plot_noise_comparison(
        [spread_gamma_pixels, spread_proton_pixels],
        ["Pixel Spread (Std Dev) - Gamma", "Pixel Spread (Std Dev) - Proton"]
    )


    # Calculate and print value counts (for gamma dataset as example, can do proton as well)
    print("Value counts for Gamma dataset (sorted by count):")
    gamma_value_counts = calculate_value_counts(gamma_dataset)
    for value, count in gamma_value_counts[:20]: # Print top 20 for brevity, can print all
        print(f"{value:.3f} appears {count} times")
    if len(gamma_value_counts) > 20:
        print("...")
    print(f"Total unique values: {len(gamma_value_counts)}")


    print("\nValue counts for Proton dataset (sorted by count):")
    proton_value_counts = calculate_value_counts(proton_dataset)
    for value, count in proton_value_counts[:20]: # Print top 20 for brevity, can print all
        print(f"{value:.3f} appears {count} times")
    if len(proton_value_counts) > 20:
        print("...")
    print(f"Total unique values: {len(proton_value_counts)}")

    # Compare value counts (normalized)
    print("\nComparison of Value Counts (Sorted by Absolute Normalized Difference):")
    value_count_differences = compare_value_counts(gamma_value_counts, proton_value_counts)
    for value, difference in value_count_differences[:20]: # Print top 20 differences
        print(f"Value: {value:.3f}, Normalized Diff (Gamma - Proton): {difference:.6f}")
    if len(value_count_differences) > 20:
        print("...")
    print(f"Total unique values compared: {len(value_count_differences)}")

    # Save data to files
    save_average_pixel_values(avg_gamma_pixels, avg_proton_pixels)
    save_pixel_spread(spread_gamma_pixels, spread_proton_pixels)
    save_value_counts(gamma_value_counts, proton_value_counts)
    save_value_count_comparison(value_count_differences)

    # Calculate and print pixel value counts and outliers
    sorted_gamma_pixel_value_counts = calculate_pixel_value_counts(gamma_dataset)
    sorted_proton_pixel_value_counts = calculate_pixel_value_counts(proton_dataset)

    print_pixel_value_count_outliers(sorted_gamma_pixel_value_counts, "Gamma Dataset", min_count_to_print=50) # Print outliers to console, only if count > 10
    print_pixel_value_count_outliers(sorted_proton_pixel_value_counts, "Proton Dataset", min_count_to_print=50)

    save_pixel_value_counts(sorted_gamma_pixel_value_counts, "Gamma Dataset", "gamma_pixel_value_counts.txt") # Save all pixel value counts to file
    save_pixel_value_counts(sorted_proton_pixel_value_counts, "Proton Dataset", "proton_pixel_value_counts.txt")

    save_pixel_value_count_outliers(sorted_gamma_pixel_value_counts, "Gamma Dataset",
                                    "gamma_pixel_value_count_outliers.txt") # Save outlier info to file
    save_pixel_value_count_outliers(sorted_proton_pixel_value_counts, "Proton Dataset",
                                    "proton_pixel_value_count_outliers.txt")