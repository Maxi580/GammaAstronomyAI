import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_and_average_diff(file_path, raw_column='image_m1', clean_column='clean_image_m1', length=1039):
    # Read the parquet file
    df = pd.read_parquet(file_path)
    # For each row, subtract the clean image from the raw image, slicing to the desired length
    diff_images = df.apply(lambda row: np.array(row[raw_column][:length]) - np.array(row[clean_column][:length]), axis=1).to_list()
    # Stack the difference arrays into a 2D array (rows: images, columns: pixel positions)
    images_arr = np.vstack(diff_images)
    # Compute the mean across rows (i.e. per pixel)
    return np.mean(images_arr, axis=0)

# File paths
gamma_file = '../magic-gammas.parquet'
proton_file = '../magic-protons.parquet'

# Calculate average per pixel difference for gamma and proton datasets
avg_gamma_diff = load_and_average_diff(gamma_file)
avg_proton_diff = load_and_average_diff(proton_file)

# Plotting the averaged difference images in different colors
plt.figure(figsize=(12, 6))
plt.plot(avg_gamma_diff, label='Gamma (raw - clean)', color='red')
plt.plot(avg_proton_diff, label='Proton (raw - clean)', color='blue')
plt.xlabel('Pixel Index')
plt.ylabel('Average Difference')
plt.title('Average Per Pixel Difference (raw - clean) for Gamma and Proton Data')
plt.legend()
plt.show()
