import pandas as pd
import matplotlib.pyplot as plt
from ctapipe.instrument import CameraGeometry

# Load MAGIC camera geometry
geom = CameraGeometry.from_name("MAGICCam")
pix_x = geom.pix_x.value  # Pixel x coordinates
pix_y = geom.pix_y.value  # Pixel y coordinates

def plot_images(row):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))  # Larger figure

    # Plot M1 image
    img_m1 = row["image_m1"][:1039] - row["clean_image_m1"][:1039]
    #img_m1[:len(img_m1)-50] = 0
    sc1 = ax1.scatter(pix_x, pix_y, c=img_m1, cmap='plasma', s=50)  # Larger marker & distinct colormap
    ax1.set_title("M1 Camera")
    fig.colorbar(sc1, ax=ax1, label="Intensity")

    # Plot M2 image
    img_m2 = row["image_m2"][:1039] - row["clean_image_m2"][:1039]
    #img_m2[:len(img_m2)-50] = 0
    sc2 = ax2.scatter(pix_x, pix_y, c=img_m2, cmap='plasma', s=50)
    ax2.set_title("M2 Camera")
    fig.colorbar(sc2, ax=ax2, label="Intensity")

    plt.tight_layout()
    plt.show()

# Load data (example with gamma file)


comb_df = pd.read_parquet("../magic-protons.parquet")

for i in range(10):
    plot_images(comb_df.iloc[i])  # Plot first gamma event

for i in range(-10, 0):
    plot_images(comb_df.iloc[i])  # Plot first gamma event

#gamma_df = pd.read_parquet("../magic-gammas.parquet")

#for i in range(10):
#    plot_images(gamma_df.iloc[i])  # Plot first gamma event

# For protons:
#proton_df = pd.read_parquet("../magic-protons.parquet")

#for i in range(10):
#    plot_images(proton_df.iloc[i])  # Plot first gamma event