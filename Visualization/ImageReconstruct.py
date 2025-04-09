import os
import numpy as np
import matplotlib.pyplot as plt
from importlib.resources import files
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay
import torch

from CNN.MagicConv.NeighborLogic import unpool_array


POOLING_KERNEL_SIZE = 2


def reconstruct_image(array_1039, save_path, title=None, pooling_count=0):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    f = str(files("ctapipe_io_magic").joinpath("resources/MAGICCam.camgeom.fits.gz"))
    geom = CameraGeometry.from_table(f)

    if torch.is_tensor(array_1039):
        array_1039 = array_1039.cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")
    ax.axis("off")  # Hide axes for a cleaner look
    disp = CameraDisplay(geom, ax=ax)

    if pooling_count > 0:
        array_1039, valid_indices = unpool_array(
            array_1039,
            pooling_kernel_size=2,
            num_pooling_layers=POOLING_KERNEL_SIZE
        )

        mask = np.ones_like(array_1039, dtype=bool)
        mask[valid_indices] = False
        masked_data = np.ma.masked_array(array_1039, mask=mask)
        disp.image = masked_data
        disp.cmap.set_bad('gray', alpha=0.2)
    else:
        disp.image = array_1039

    if title:
        plt.title(title)

    plt.colorbar(disp.pixels, ax=ax, label='Intensity')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
