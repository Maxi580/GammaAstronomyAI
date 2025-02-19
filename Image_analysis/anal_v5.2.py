import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_mean_std_diff(file_path, raw_column='image_m2', clean_column='clean_image_m2', length=1039):
    df = pd.read_parquet(file_path)
    # Compute difference only where clean==0, else NaN
    diff_images = df.apply(
        lambda row: np.where(
            np.array(row[clean_column][:length]) == 0,
            np.array(row[raw_column][:length]) - np.array(row[clean_column][:length]),
            np.nan
        ),
        axis=1
    ).to_list()

    # Stack into a 2D array (rows: images, columns: pixel positions)
    images_arr = np.vstack(diff_images)

    # Pixel-wise mean and std (ignoring NaNs)
    mean = np.nanmean(images_arr, axis=0)
    std = np.nanstd(images_arr, axis=0)

    # Global average and global standard deviation across all valid pixels
    global_avg = np.nanmean(images_arr)
    global_std = np.nanstd(images_arr)

    return mean, std, global_avg, global_std


# File paths
gamma_file = '../magic-gammas.parquet'
proton_file = '../magic-protons.parquet'

# Compute statistics for gamma and proton datasets
mean_gamma, std_gamma, global_avg_gamma, global_std_gamma = load_mean_std_diff(gamma_file)
mean_proton, std_proton, global_avg_proton, global_std_proton = load_mean_std_diff(proton_file)

pixels = np.arange(1039)

plt.figure(figsize=(12, 6))

# --- Gamma Plot (Red) ---
plt.plot(pixels, mean_gamma, label='Gamma (raw - clean)', color='red')
plt.fill_between(pixels, mean_gamma - std_gamma, mean_gamma + std_gamma, color='red', alpha=0.2)
# Global average line
plt.axhline(y=global_avg_gamma, color='red', linestyle='--', label=f'Gamma Global Avg: {global_avg_gamma:.2f}')
# Global average ± global std lines
plt.axhline(y=global_avg_gamma + global_std_gamma, color='red', linestyle=':',
            label=f'Gamma Global Avg + STD: {global_avg_gamma + global_std_gamma:.2f}')
plt.axhline(y=global_avg_gamma - global_std_gamma, color='red', linestyle=':',
            label=f'Gamma Global Avg - STD: {global_avg_gamma - global_std_gamma:.2f}')

# --- Proton Plot (Blue) ---
plt.plot(pixels, mean_proton, label='Proton (raw - clean)', color='blue')
plt.fill_between(pixels, mean_proton - std_proton, mean_proton + std_proton, color='blue', alpha=0.2)
# Global average line
plt.axhline(y=global_avg_proton, color='blue', linestyle='--', label=f'Proton Global Avg: {global_avg_proton:.2f}')
# Global average ± global std lines
plt.axhline(y=global_avg_proton + global_std_proton, color='blue', linestyle=':',
            label=f'Proton Global Avg + STD: {global_avg_proton + global_std_proton:.2f}')
plt.axhline(y=global_avg_proton - global_std_proton, color='blue', linestyle=':',
            label=f'Proton Global Avg - STD: {global_avg_proton - global_std_proton:.2f}')

plt.xlabel('Pixel Index')
plt.ylabel('Average Difference')
plt.title('M2 - Pixel-wise Difference (raw - clean) with STD and Global Metrics\n(Ignoring pixels where clean != 0)')
plt.legend(loc='upper right')
plt.show()
