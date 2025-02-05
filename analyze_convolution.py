import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from importlib.resources import files
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay

from CombinedNet.CombinedNet import CombinedNet
from CNN.HexLayers.ConvHex import ConvHex
from CNN.HexLayers.neighbor import unpool_array
from CombinedNet.magicDataset import MagicDataset

POOLING_KERNEL_SIZE = 2


def reconstruct_image(array_1039, save_path, title=None, pooling_count=0):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    f = str(files("ctapipe_io_magic").joinpath("resources/MAGICCam.camgeom.fits.gz"))
    geom = CameraGeometry.from_table(f)

    if torch.is_tensor(array_1039):
        array_1039 = array_1039.cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 10))
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


def extract_conv_arrays(tensor, cnn_idx, pooling_count, output_dir):
    data = tensor.squeeze().cpu().detach().numpy()
    for channel_idx in range(data.shape[0]):
        reconstruct_image(
            data[channel_idx],
            output_dir + f"cnn{cnn_idx}/{channel_idx}.png",
            pooling_count=pooling_count
        )


def simulate_forward_pass(image, model, output_dir):
    pooling_cnt = 0

    for idx, layer in enumerate(model.m1_cnn.cnn):
        if isinstance(layer, ConvHex):
            image = layer(image)
            extract_conv_arrays(image, idx, pooling_cnt, output_dir)
        elif isinstance(layer, nn.MaxPool1d):
            image = layer(image)
            pooling_cnt += 1
        else:
            image = layer(image)


def debug_forward_pass(model_path, prefix, image_m1, image_m2):
    output_dir = f"convolution_analysis/image{prefix}/reconstructed_cnn/"
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CombinedNet()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    image_m1 = (image_m1.to(device))[:1039]
    image_m2 = (image_m2.to(device))[:1039]
    reconstruct_image(image_m1, output_dir + "m1_original.png")
    reconstruct_image(image_m2, output_dir + "m2_original.png")

    if image_m1.dim() == 1:
        image_m1 = image_m1.unsqueeze(0).unsqueeze(0)
    elif image_m1.dim() == 2:
        image_m1 = image_m1.unsqueeze(0)

    if image_m2.dim() == 1:
        image_m2 = image_m2.unsqueeze(0).unsqueeze(0)
    elif image_m2.dim() == 2:
        image_m2 = image_m2.unsqueeze(0)

    simulate_forward_pass(image_m1, model, output_dir + "/m1/")
    simulate_forward_pass(image_m2, model, output_dir + "/m2/")


if __name__ == "__main__":
    model_path = "masked_model.pth"

    proton_file = "magic-protons.parquet"
    gamma_file = "magic-gammas.parquet"
    dataset = MagicDataset(proton_file, gamma_file, mask_rings=10, debug_info=False)
    num_samples = 3

    total_samples = len(dataset)
    random_indices = np.random.choice(total_samples, num_samples, replace=False)
    label_names = {v: k for k, v in dataset.labels.items()}

    for i, idx in enumerate(random_indices):
        m1_image, m2_image, features, label = dataset[idx]

        prefix = f"{i}_{label_names[label]}"
        debug_forward_pass(model_path, prefix, m1_image, m2_image)

        print(f"Processed sample {i + 1}/{num_samples}: {prefix}")
