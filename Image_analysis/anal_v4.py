import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_and_average_diff(file_path, raw_column='image_m1', clean_column='clean_image_m1', length=1039):
    # Read the parquet file
    df = pd.read_parquet(file_path)
    # For each row, compute the difference only where the clean pixel is 0, otherwise use NaN
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
    # Compute the mean per pixel ignoring NaNs (i.e. only averaging where clean==0)
    return np.nanmean(images_arr, axis=0)


# File paths
gamma_file = '../magic-gammas.parquet'
proton_file = '../magic-protons.parquet'

# Calculate the average per pixel difference (raw - clean) for gamma and proton datasets, ignoring pixels where clean != 0
avg_gamma_diff = load_and_average_diff(gamma_file)
avg_proton_diff = load_and_average_diff(proton_file)

# Plot the averaged difference values for gamma and proton data
plt.figure(figsize=(12, 6))
plt.plot(avg_gamma_diff, label='Gamma (raw - clean)', color='red')
plt.plot(avg_proton_diff, label='Proton (raw - clean)', color='blue')
plt.xlabel('Pixel Index')
plt.ylabel('Average Difference')
plt.title(
    'Average Per Pixel Difference (raw - clean) for Gamma and Proton Data\n(ignoring pixels where clean image != 0)')
plt.legend()
plt.show()
