import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_and_average(file_path, column='image_m1', length=1039):
    # Read the parquet file
    df = pd.read_parquet(file_path)
    # For each row, convert the list/array to a NumPy array and slice it to the desired length
    images = df[column].apply(lambda x: np.array(x[:length])).to_list()
    # Stack all arrays into a 2D array (rows: images, columns: pixel positions)
    images_arr = np.vstack(images)
    # Compute the mean across rows (i.e. per pixel)
    return np.mean(images_arr, axis=0)

# File paths
gamma_file = '../magic-gammas.parquet'
proton_file = '../magic-protons.parquet'

# Choose the column to use. If you prefer 'clean_image_m1', change the column name below.
column_to_use = 'clean_image_m1'

# Calculate average per pixel for gamma and proton datasets
avg_gamma = load_and_average(gamma_file, column=column_to_use, length=1039)
avg_proton = load_and_average(proton_file, column=column_to_use, length=1039)

# Plotting the averaged images in different colors
plt.figure(figsize=(12, 6))
plt.plot(avg_gamma, label='Gamma', color='red')
plt.plot(avg_proton, label='Proton', color='blue')
plt.xlabel('Pixel Index')
plt.ylabel('Average Intensity')
plt.title('Average Per Pixel for Gamma and Proton Data')
plt.legend()
plt.show()
