#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Magic Telescope Model with Visualizations:
- Gradient-based Saliency Maps
- Integrated Gradients
- Attention Heatmaps using a custom TransformerEncoderLayer

Model parameters updated: emb_dim=64, n_heads=2, ff_dim=128, n_layers=2, n_classes=2.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from astropy.wcs.wcsapi.high_level_api import high_level_objects_to_values
from ctapipe.instrument import CameraGeometry
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt


# -----------------------------------------------------
# 1) DATASET - Return (x_m1, x_m2), label
# -----------------------------------------------------
class MagicDataset(Dataset):
    def __init__(self, gamma_parquet, proton_parquet, transform=None):
        df_gamma = pd.read_parquet(gamma_parquet)
        df_gamma["label"] = 0  # label gammas as 0
        df_proton = pd.read_parquet(proton_parquet)
        df_proton["label"] = 1  # label protons as 1
        self.df = pd.concat([df_gamma, df_proton], ignore_index=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x_m1 = torch.tensor(row["image_m1"][:1039] - row["clean_image_m1"][:1039], dtype=torch.float32)
        x_m2 = torch.tensor(row["image_m2"][:1039] - row["clean_image_m2"][:1039], dtype=torch.float32)
        y = torch.tensor(row["label"], dtype=torch.long)
        return x_m1, x_m2, y


# -----------------------------------------------------
# 2) PATCH + POSITIONAL ENCODING
# -----------------------------------------------------
class PatchEmbedding(nn.Module):
    def __init__(self, in_dim=1039, patch_size=1, emb_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size, emb_dim)

    def forward(self, x):
        length = x.shape[1]
        mod = length % self.patch_size
        if mod != 0:
            pad_length = self.patch_size - mod
            x = nn.functional.pad(x, (0, pad_length), mode='constant', value=0)
        x = x.view(x.shape[0], -1, self.patch_size)
        return self.proj(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :]


# -----------------------------------------------------
# 2.5) CUSTOM TRANSFORMER ENCODER LAYER THAT RETURNS ATTENTION WEIGHTS
# Updated forward() to accept an extra 'is_causal' parameter.
# -----------------------------------------------------
class TransformerEncoderLayerWithAttn(nn.TransformerEncoderLayer):
    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        # Force need_weights=True (and average_attn_weights=False for raw per-head scores)
        attn_output, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        self.last_attn_weights = attn_weights
        return src


# -----------------------------------------------------
# 3) SHAPE TRANSFORMER (single-branch) using the custom encoder layer
# -----------------------------------------------------
class ShapeTransformer(nn.Module):
    def __init__(self, emb_dim=64, n_heads=2, ff_dim=128, n_layers=2, max_len=2000):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_dim=1039, patch_size=1, emb_dim=emb_dim)
        self.pos_encoder = PositionalEncoding(emb_dim, max_len)
        encoder_layer = TransformerEncoderLayerWithAttn(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.final_linear = nn.Identity()

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)  # TransformerEncoder may pass is_causal keyword.
        x = x.mean(dim=1)
        return self.final_linear(x)


# -----------------------------------------------------
# 4) COMBINED TRANSFORMER - merges two branches
# -----------------------------------------------------
class CombinedTransformer(nn.Module):
    def __init__(self, emb_dim=64, n_heads=2, ff_dim=128, n_layers=2, n_classes=2):
        super().__init__()
        self.transformer_m1 = ShapeTransformer(
            emb_dim=emb_dim,
            n_heads=n_heads,
            ff_dim=ff_dim,
            n_layers=n_layers
        )
        self.transformer_m2 = ShapeTransformer(
            emb_dim=emb_dim,
            n_heads=n_heads,
            ff_dim=ff_dim,
            n_layers=n_layers
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(emb_dim, n_classes)
        )

    def forward(self, x_m1, x_m2):
        out_m1 = self.transformer_m1(x_m1)
        out_m2 = self.transformer_m2(x_m2)
        combined = torch.cat([out_m1, out_m2], dim=1)
        return self.classifier(combined)


# -----------------------------------------------------
# 5) VISUALIZATION FUNCTIONS
# -----------------------------------------------------
def visualize_saliency(model, x_m1, x_m2, target_class, device):
    model.eval()
    x_m1 = x_m1.clone().detach().to(device)
    x_m2 = x_m2.clone().detach().to(device)
    x_m1.requires_grad_()
    x_m2.requires_grad_()

    outputs = model(x_m1, x_m2)
    score = outputs[0, target_class]
    model.zero_grad()
    score.backward()

    saliency_m1 = x_m1.grad.abs().detach().cpu().numpy().squeeze()
    saliency_m2 = x_m2.grad.abs().detach().cpu().numpy().squeeze()

    m1_max_ind = saliency_m1.argmax()
    m2_max_ind = saliency_m2.argmax()

    print("Max saliency m1: ", m1_max_ind, "  -  ", saliency_m1[m1_max_ind])
    print("Max saliency m2: ", m2_max_ind, "  -  ", saliency_m2[m2_max_ind])

    m1_min_ind = saliency_m1.argmin()
    m2_min_ind = saliency_m2.argmin()

    print("Min saliency m1: ", m1_min_ind, "  -  ", saliency_m1[m1_min_ind])
    print("Min saliency m2: ", m2_min_ind, "  -  ", saliency_m2[m2_min_ind])

    min_max_values_m1 = [0] * 1039
    min_max_values_m2 = [0] * 1039

    min_max_values_m1[m1_max_ind] = 1
    min_max_values_m1[m1_min_ind] = -1

    min_max_values_m2[m2_max_ind] = 1
    min_max_values_m2[m2_min_ind] = -1

    min_max_values_tensor_m1 = torch.tensor(min_max_values_m1, dtype=torch.float32)
    min_max_values_tensor_m2 = torch.tensor(min_max_values_m2, dtype=torch.float32)

    plot_noise_comparison([min_max_values_tensor_m1, min_max_values_tensor_m2], ["M1", "M2"])

    plt.figure(figsize=(24, 10))
    plt.subplot(1, 2, 1)
    plt.title("Gradient Saliency - m1")
    plt.plot(saliency_m1, color='blue')
    plt.xlabel("Input Index")
    plt.ylabel("Gradient Magnitude")

    plt.subplot(1, 2, 2)
    plt.title("Gradient Saliency - m2")
    plt.plot(saliency_m2, color='green')
    plt.xlabel("Input Index")
    plt.ylabel("Gradient Magnitude")
    plt.tight_layout()
    plt.show()


def integrated_gradients(model, x_m1, x_m2, target_class, baseline=None, steps=50, device='cpu'):
    if baseline is None:
        baseline_m1 = torch.zeros_like(x_m1)
        baseline_m2 = torch.zeros_like(x_m2)
    else:
        baseline_m1, baseline_m2 = baseline

    diff_m1 = x_m1 - baseline_m1
    diff_m2 = x_m2 - baseline_m2
    total_gradients_m1 = torch.zeros_like(x_m1)
    total_gradients_m2 = torch.zeros_like(x_m2)

    for alpha in torch.linspace(0, 1, steps).to(device):
        scaled_x_m1 = baseline_m1 + alpha * diff_m1
        scaled_x_m2 = baseline_m2 + alpha * diff_m2
        scaled_x_m1.requires_grad_()
        scaled_x_m2.requires_grad_()
        outputs = model(scaled_x_m1, scaled_x_m2)
        score = outputs[0, target_class]
        model.zero_grad()
        score.backward(retain_graph=True)
        total_gradients_m1 += scaled_x_m1.grad
        total_gradients_m2 += scaled_x_m2.grad

    avg_gradients_m1 = total_gradients_m1 / steps
    avg_gradients_m2 = total_gradients_m2 / steps

    integrated_grads_m1 = diff_m1 * avg_gradients_m1
    integrated_grads_m2 = diff_m2 * avg_gradients_m2

    m1_max_ind = integrated_grads_m1[0].argmax()
    m2_max_ind = integrated_grads_m2[0].argmax()

    print("Max integrated_grads m1: ", m1_max_ind, "  -  ", integrated_grads_m1[0][m1_max_ind])
    print("Max integrated_grads m2: ", m2_max_ind, "  -  ", integrated_grads_m2[0][m2_max_ind])

    m1_min_ind = integrated_grads_m1[0].argmin()
    m2_min_ind = integrated_grads_m2[0].argmin()

    print("Min integrated_grads m1: ", m1_min_ind, "  -  ", integrated_grads_m1[0][m1_min_ind])
    print("Min integrated_grads m2: ", m2_min_ind, "  -  ", integrated_grads_m2[0][m2_min_ind])

    min_max_values_m1 = [0] * 1039
    min_max_values_m2 = [0] * 1039

    min_max_values_m1[m1_max_ind] = 1
    min_max_values_m1[m1_min_ind] = -1

    min_max_values_m2[m2_max_ind] = 1
    min_max_values_m2[m2_min_ind] = -1

    min_max_values_tensor_m1 = torch.tensor(min_max_values_m1, dtype=torch.float32)
    min_max_values_tensor_m2 = torch.tensor(min_max_values_m2, dtype=torch.float32)

    plot_noise_comparison([min_max_values_tensor_m1, min_max_values_tensor_m2], ["M1", "M2"])

    return integrated_grads_m1, integrated_grads_m2


def visualize_integrated_gradients(model, x_m1, x_m2, target_class, device):
    ig_m1, ig_m2 = integrated_gradients(model, x_m1, x_m2, target_class, device=device)
    ig_m1 = ig_m1.detach().cpu().numpy().squeeze()
    ig_m2 = ig_m2.detach().cpu().numpy().squeeze()

    plt.figure(figsize=(24, 10))
    plt.subplot(1, 2, 1)
    plt.title("Integrated Gradients - m1")
    plt.plot(ig_m1, color='purple')
    plt.xlabel("Input Index")
    plt.ylabel("Attribution")

    plt.subplot(1, 2, 2)
    plt.title("Integrated Gradients - m2")
    plt.plot(ig_m2, color='orange')
    plt.xlabel("Input Index")
    plt.ylabel("Attribution")
    plt.tight_layout()
    plt.show()


def visualize_attention_heatmaps(model, x_m1, x_m2, device):
    model.eval()
    attention_weights = {}

    def get_attention_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                attn = output[1]
                attention_weights[name] = attn.detach().cpu()
            else:
                print(f"Warning: No attention weights returned in {name}.")
                attention_weights[name] = None

        return hook

    hooks = []
    for i, layer in enumerate(model.transformer_m1.transformer_encoder.layers):
        hook = layer.self_attn.register_forward_hook(get_attention_hook(f"m1_layer_{i}"))
        hooks.append(hook)
    for i, layer in enumerate(model.transformer_m2.transformer_encoder.layers):
        hook = layer.self_attn.register_forward_hook(get_attention_hook(f"m2_layer_{i}"))
        hooks.append(hook)

    _ = model(x_m1, x_m2)

    for key, attn in attention_weights.items():
        if attn is None:
            continue
        attn_sample = attn[0]  # take first sample
        num_heads = attn_sample.shape[0]
        fig, axes = plt.subplots(1, num_heads, figsize=(num_heads * 24, 24))
        fig.suptitle(f"Attention Weights: {key}", fontsize=12)
        for h in range(num_heads):
            ax = axes[h] if num_heads > 1 else axes
            im = ax.imshow(attn_sample[h].numpy(), cmap='viridis')
            ax.set_title(f"Head {h}")
            ax.set_xlabel("Key Position")
            ax.set_ylabel("Query Position")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    for hook in hooks:
        hook.remove()


def plot_noise_comparison(tensors_to_plot, names):
    """
    Plots 1D images over the MAGICCam geometry.
    """
    geom = CameraGeometry.from_name("MAGICCam")
    pix_x = geom.pix_x.value
    pix_y = geom.pix_y.value
    num_plots = len(tensors_to_plot)
    cols = min(num_plots, 2) if num_plots <= 4 else int(np.ceil(np.sqrt(num_plots)))
    rows = int(np.ceil(num_plots / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 7 * rows))
    if num_plots == 1:
        axes = [axes]
    elif isinstance(axes, np.ndarray):
        axes = axes.flatten()
    for i, (tensor, name) in enumerate(zip(tensors_to_plot, names)):
        t = tensor.detach().cpu().numpy()
        ax = axes[i]
        sc = ax.scatter(pix_x, pix_y, c=t, cmap='plasma', s=50)
        ax.set_title(name)
        fig.colorbar(sc, ax=ax, label="Intensity")
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------
# 6) MAIN: DEMONSTRATE VISUALIZATIONS
# -----------------------------------------------------
if __name__ == "__main__":
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )
    print("Using device:", device)

    gamma_file = "../magic-gammas_small_part1.parquet"
    proton_file = "../magic-protons_small_part1.parquet"

    dataset = MagicDataset(gamma_parquet=gamma_file, proton_parquet=proton_file)
    #train_size = int(0.7 * len(dataset))
    #val_size = len(dataset) - train_size
    #_, val_dataset = random_split(dataset, [train_size, val_size])


    # 5000 ist interessant
    # -2 -3,0.6ausschlag
    x_m1, x_m2, y = dataset[5000]
    print("The sample is labeled as:", "Gamma" if y.item() == 0 else "Proton", "  -  ", y)

    x_m1 = x_m1.unsqueeze(0).to(device)
    x_m2 = x_m2.unsqueeze(0).to(device)

    # Add a random small value to the image to simulate noise
    #x_m1 += torch.randn_like(x_m1) * 0.1
    #x_m2 += torch.randn_like(x_m2) * 0.1

    target_class = y.item()

    image_m1_lt5 = x_m1.clone()
    image_m1_lt5[image_m1_lt5 > 7] = 7

    image_m2_lt5 = x_m2.clone()
    image_m2_lt5[image_m2_lt5 > 7] = 7

    plot_noise_comparison([image_m1_lt5.squeeze(), image_m2_lt5.squeeze()], ["M1", "M2"])

    #model = CombinedTransformer(
    #    emb_dim=64,
    #    n_heads=1,
    #    ff_dim=64,
    #    n_layers=1,
    #    n_classes=2
    #).to(device)

    model = CombinedTransformer(
        emb_dim=32,
        n_heads=1,
        ff_dim=1,  # 128
        n_layers=1,
        n_classes=2
    ).to(device)

    # Predict the sample
    # Predict the sample
    model.eval()
    with torch.no_grad():
        output = model(x_m1, x_m2)
        predicted_class = torch.argmax(output, dim=1).item()
        print(f"Predicted class: {'Gamma' if predicted_class == 0 else 'Proton'}")
        print(f"The prediction is {'CORRECT' if predicted_class == target_class else 'INCORRECT'}")


    model_path = "best_model_dual_fin.pt"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded model weights from", model_path)
    else:
        print("Trained model weights not found. Visualizations will use the untrained model.")

    print("Generating Gradient-based Saliency Maps...")
    visualize_saliency(model, x_m1, x_m2, target_class, device)

    print("Generating Integrated Gradients...")
    visualize_integrated_gradients(model, x_m1, x_m2, target_class, device)

    print("Generating Attention Heatmaps...")
    visualize_attention_heatmaps(model, x_m1, x_m2, device)
