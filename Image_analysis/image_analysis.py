import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from scipy import stats
from torch.utils.data import Dataset, DataLoader, random_split


# -------------------------------
# 1) Dataset definition (optional)
# -------------------------------
class MagicDataset(Dataset):
    def __init__(self, gamma_parquet, proton_parquet, transform=None):
        # Read gammas
        df_gamma = pd.read_parquet(gamma_parquet)
        df_gamma["label"] = 0  # label gammas as 0

        # Read protons
        df_proton = pd.read_parquet(proton_parquet)
        df_proton["label"] = 1  # label protons as 1

        # Combine
        self.df = pd.concat([df_gamma, df_proton], ignore_index=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # The difference images returned for training
        x_m1 = torch.tensor(row["image_m1"][:1039] - row["clean_image_m1"][:1039], dtype=torch.float32)
        x_m2 = torch.tensor(row["image_m2"][:1039] - row["clean_image_m2"][:1039], dtype=torch.float32)
        y = torch.tensor(row["label"], dtype=torch.long)
        return x_m1, x_m2, y


# -------------------------------
# 2) Load data into DataFrame
# -------------------------------
dataset = MagicDataset("../magic-gammas.parquet", "../magic-protons.parquet")
df = dataset.df

# Ensure images are always of length 1039 (truncate if needed)
df["image_m1"] = df["image_m1"].apply(lambda x: x[:1039])
df["image_m2"] = df["image_m2"].apply(lambda x: x[:1039])
df["clean_image_m1"] = df["clean_image_m1"].apply(lambda x: x[:1039])
df["clean_image_m2"] = df["clean_image_m2"].apply(lambda x: x[:1039])

# ----------------------------------------------------
# 3) Create columns for the difference images (raw-clean)
# ----------------------------------------------------
# This is the key step if you want to do statistics on raw - clean.
df["diff_image_m1"] = [
    raw - clean for raw, clean in zip(df["image_m1"], df["clean_image_m1"])
]
df["diff_image_m2"] = [
    raw - clean for raw, clean in zip(df["image_m2"], df["clean_image_m2"])
]


# -----------------------------------------
# 4) Calculate stats in a vectorized manner
# -----------------------------------------
def add_diff_stats(df, camera_column_prefix="diff_image_m1"):
    """
    For the given camera_column_prefix (e.g. 'diff_image_m1'),
    compute mean, std, skew, kurtosis, median, max, min in a vectorized way
    and add them as columns to the DataFrame.
    """
    # Convert the list of 1D arrays into a single 2D array of shape (num_samples, 1039)
    arr = np.stack(df[camera_column_prefix].values)

    # Calculate statistics along axis=1 (row-wise)
    mean_vals = arr.mean(axis=1)
    std_vals = arr.std(axis=1)
    skew_vals = stats.skew(arr, axis=1)
    kurt_vals = stats.kurtosis(arr, axis=1)
    median_vals = np.median(arr, axis=1)
    max_vals = arr.max(axis=1)
    min_vals = arr.min(axis=1)

    # Create column prefixes like "m1_mean", "m1_std", ...
    if "m1" in camera_column_prefix:
        prefix = "m1"
    else:
        prefix = "m2"

    df[f"{prefix}_mean"] = mean_vals
    df[f"{prefix}_std"] = std_vals
    df[f"{prefix}_skew"] = skew_vals
    df[f"{prefix}_kurtosis"] = kurt_vals
    df[f"{prefix}_median"] = median_vals
    df[f"{prefix}_max"] = max_vals
    df[f"{prefix}_min"] = min_vals


# Calculate stats for M1 difference images
add_diff_stats(df, "diff_image_m1")
# Calculate stats for M2 difference images
add_diff_stats(df, "diff_image_m2")

# -----------------------
# 5) Split into gamma/proton
# -----------------------
gamma_df = df[df["label"] == 0]
proton_df = df[df["label"] == 1]

# -------------------------------
# 6) Plotting settings
# -------------------------------
plt.rcParams.update({"font.size": 12})
sns.set_style("whitegrid")

# ------------------------------------------
# 7) Histogram Comparison of Key Statistics
# ------------------------------------------
stats_to_plot = ["mean", "std", "skew", "kurtosis"]
for cam in ["m1", "m2"]:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    fig.suptitle(f"Statistical Properties Comparison - {cam.upper()}", y=1.02)

    for i, stat in enumerate(stats_to_plot):
        sns.histplot(gamma_df[f"{cam}_{stat}"], ax=axes[i], color="blue", label="Gamma", kde=True)
        sns.histplot(proton_df[f"{cam}_{stat}"], ax=axes[i], color="red", label="Proton", kde=True, alpha=0.5)
        axes[i].set_title(f"{stat.upper()} Distribution")
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(f"statistical_comparison_{cam}.png", bbox_inches="tight")
    plt.close()


# --------------------------------
# 8) Average Difference Image Plots
# --------------------------------
def plot_average_images(cam_df, label, cam):
    avg_image = np.mean(np.stack(cam_df[f"diff_image_{cam}"].values), axis=0)
    plt.figure(figsize=(10, 6))
    plt.plot(avg_image)
    plt.title(f"Average {cam.upper()} (Raw - Clean) Image - {label}")
    plt.xlabel("Pixel Position")
    plt.ylabel("Intensity")
    plt.savefig(f"avg_diff_image_{label}_{cam}.png", bbox_inches="tight")
    plt.close()


for cam in ["m1", "m2"]:
    plot_average_images(gamma_df, "Gamma", cam)
    plot_average_images(proton_df, "Proton", cam)


# ----------------------------
# 9) Frequency Domain Analysis
# ----------------------------
def plot_spectral_comparison(cam):
    gamma_fft = np.abs(np.fft.fft(np.stack(gamma_df[f"diff_image_{cam}"].values), axis=1))
    proton_fft = np.abs(np.fft.fft(np.stack(proton_df[f"diff_image_{cam}"].values), axis=1))

    plt.figure(figsize=(12, 6))
    plt.plot(np.mean(gamma_fft, axis=0)[1:50], label="Gamma")
    plt.plot(np.mean(proton_fft, axis=0)[1:50], label="Proton")
    plt.title(f"Average Frequency Spectrum (First 50 bins) - {cam.upper()}")
    plt.xlabel("Frequency Bin")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.savefig(f"frequency_spectrum_{cam}.png", bbox_inches="tight")
    plt.close()


for cam in ["m1", "m2"]:
    plot_spectral_comparison(cam)

# -------------------------------
# 10) Statistical Significance Tests
# -------------------------------
results = []
for cam in ["m1", "m2"]:
    for stat in stats_to_plot:
        gamma_vals = gamma_df[f"{cam}_{stat}"]
        proton_vals = proton_df[f"{cam}_{stat}"]

        # T-test
        t_stat, t_p = stats.ttest_ind(gamma_vals, proton_vals)
        # Mann-Whitney U test
        u_stat, u_p = stats.mannwhitneyu(gamma_vals, proton_vals)

        results.append({
            "Camera": cam.upper(),
            "Statistic": stat,
            "T-test p-value": t_p,
            "Mann-Whitney p-value": u_p
        })

results_df = pd.DataFrame(results)
print("\nStatistical Test Results:")
print(results_df.to_string(float_format=lambda x: f"{x:.2e}"))

# ---------------------
# 11) Boxplot Comparison
# ---------------------
for cam in ["m1", "m2"]:
    plt.figure(figsize=(12, 8))
    for i, stat in enumerate(stats_to_plot):
        plt.subplot(2, 2, i + 1)
        sns.boxplot(x="label", y=f"{cam}_{stat}", data=df)
        plt.title(f"{stat.upper()} - {cam.upper()}")
        plt.xlabel("Class")
        plt.ylabel(stat)
        plt.xticks([0, 1], ["Gamma", "Proton"])
    plt.tight_layout()
    plt.savefig(f"boxplot_comparison_{cam}.png", bbox_inches="tight")
    plt.close()
