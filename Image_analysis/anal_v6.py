import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_mean_std_diff(file_path, raw_column='image_m1', clean_column='clean_image_m1', length=1039):
    df = pd.read_parquet(file_path)
    diff_images = df.apply(
        lambda row: np.where(
            np.array(row[clean_column][:length]) == 0,
            np.array(row[raw_column][:length]), # - np.array(row[clean_column][:length]),
            np.nan
        ),
        axis=1
    ).to_list()
    images_arr = np.vstack(diff_images)
    mean = np.nanmean(images_arr, axis=0)
    std = np.nanstd(images_arr, axis=0)
    global_avg = np.nanmean(images_arr)
    return mean, std, global_avg

gamma_file = '../magic-gammas.parquet'
proton_file = '../magic-protons.parquet'

mean_gamma, std_gamma, global_avg_gamma = load_mean_std_diff(gamma_file)
mean_proton, std_proton, global_avg_proton = load_mean_std_diff(proton_file)

pixels = np.arange(1039)

plt.figure(figsize=(12, 6))

# Gamma data (red)
plt.plot(pixels, mean_gamma, label='Gamma (raw - clean)', color='red')
plt.fill_between(pixels, mean_gamma - std_gamma, mean_gamma + std_gamma, color='red', alpha=0.2)
plt.axhline(y=global_avg_gamma, color='red', linestyle='--', label='Gamma Global Avg')

# Proton data (blue)
plt.plot(pixels, mean_proton, label='Proton (raw - clean)', color='blue')
plt.fill_between(pixels, mean_proton - std_proton, mean_proton + std_proton, color='blue', alpha=0.2)
plt.axhline(y=global_avg_proton, color='blue', linestyle='--', label='Proton Global Avg')

plt.xlabel('Pixel Index')
plt.ylabel('Average Difference')
plt.title('M1 - Pixel-wise Difference (raw - clean) with STD and Global Averages\n(Ignoring pixels where clean != 0)')
plt.legend()
plt.show()
