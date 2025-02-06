import os

import numpy as np
import torch
import torch.nn as nn

from CNN.Architectures.BasicMagicCNN import BasicMagicNet
from CNN.MagicConv.MagicConv import MagicConv
from TrainingPipeline.MagicDataset import MagicDataset
from MagicTelescope.ImageReconstruct import reconstruct_image

POOLING_KERNEL_SIZE = 2


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
        if isinstance(layer, MagicConv):
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

    model = BasicMagicNet()
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
    dataset = MagicDataset(proton_file, gamma_file, mask_rings=13, debug_info=False)
    num_samples = 3

    total_samples = len(dataset)
    random_indices = np.random.choice(total_samples, num_samples, replace=False)
    label_names = {v: k for k, v in dataset.labels.items()}

    for i, idx in enumerate(random_indices):
        m1_image, m2_image, features, label = dataset[idx]

        prefix = f"{i}_{label_names[label]}"
        debug_forward_pass(model_path, prefix, m1_image, m2_image)

        print(f"Processed sample {i + 1}/{num_samples}: {prefix}")
