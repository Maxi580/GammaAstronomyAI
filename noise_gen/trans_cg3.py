#!/usr/bin/env python3
"""
Adversarial Noise Generation for MAGIC Telescope Images using Transformers.
The code loads images from a parquet file, computes the noise (raw – cleaned),
and trains a Generator (Transformer) to generate noise in regions where the
cleaned image is nonzero (i.e. where particle signals are present). A
Discriminator (Transformer) is trained to distinguish “real” noise (from background
pixels) from the generated noise in particle regions.
"""

import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from ctapipe.instrument import CameraGeometry

# ------------------ Positional Encoding ------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# ------------------ Generator (Model1) ------------------
class Generator(nn.Module):
    def __init__(self, input_dim=2, d_model=64, nhead=8, num_layers=2, seq_len=1039):
        """
        Generator takes a 2-channel input per pixel:
            - Channel 0: True noise (raw - cleaned)
            - Channel 1: Binary mask (1 if cleaned pixel nonzero i.e. particle)
        and outputs a 1D noise prediction for each pixel.
        """
        super().__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=0.1, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_linear = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (batch, seq_len, 2)
        x = self.input_linear(x)             # (batch, seq_len, d_model)
        x = self.pos_enc(x)                  # add positional encoding
        x = x.transpose(0, 1)                # (seq_len, batch, d_model)
        x = self.transformer(x)              # (seq_len, batch, d_model)
        x = x.transpose(0, 1)                # (batch, seq_len, d_model)
        out = self.output_linear(x)          # (batch, seq_len, 1)
        return out.squeeze(-1)               # (batch, seq_len)

# ------------------ Discriminator (Model2) ------------------
class Discriminator(nn.Module):
    def __init__(self, input_dim=1, d_model=256, nhead=8, num_layers=4, seq_len=1039):
        """
        Discriminator takes a single-channel noise input per pixel (either real or generated)
        and outputs a logit (before sigmoid) per pixel indicating the probability of being real.
        """
        super().__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=0.1, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_linear = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (batch, seq_len) or (batch, seq_len, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        x = self.input_linear(x)            # (batch, seq_len, d_model)
        x = self.pos_enc(x)
        x = x.transpose(0, 1)               # (seq_len, batch, d_model)
        x = self.transformer(x)
        x = x.transpose(0, 1)               # (batch, seq_len, d_model)
        out = self.output_linear(x)         # (batch, seq_len, 1)
        return out.squeeze(-1)              # (batch, seq_len)

# ------------------ Dataset ------------------
class GammaDataset(Dataset):
    def __init__(self, parquet_file):
        self.df = pd.read_parquet(parquet_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load a single row and extract the first 1039 values for each image.
        row = self.df.iloc[idx]
        x_m1 = torch.tensor(row["image_m1"][:1039], dtype=torch.float32)
        x_c_m1 = torch.tensor(row["clean_image_m1"][:1039], dtype=torch.float32)
        return x_m1, x_c_m1

def plot_noise_comparison(tensors_to_plot, names):
    """Plot noise comparisons in a grid.

    Args:
        tensors_to_plot: List of tensors to plot.
        names: List of names/titles for each plot.
    """
    geom = CameraGeometry.from_name("MAGICCam")
    pix_x = geom.pix_x.value
    pix_y = geom.pix_y.value

    num_plots = len(tensors_to_plot)

    if num_plots == 0:
        print("No tensors provided to plot.")
        return

    # Determine grid dimensions (approximate square for >4 plots, otherwise fit to a reasonable rectangle)
    if num_plots <= 4:
        cols = min(num_plots, 2)
        rows = int(np.ceil(num_plots / cols))
    else:
        cols = int(np.ceil(np.sqrt(num_plots)))
        rows = int(np.ceil(num_plots / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 7 * rows))

    # Handle the case where there's only one plot
    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()  # Flatten the axes array for easy iteration

    for i, (tensor, name) in enumerate(zip(tensors_to_plot, names)):
        tensor_lt5 = tensor
        tensor_lt5[tensor_lt5 > 5] = 5
        ax = axes[i]
        sc = ax.scatter(pix_x, pix_y, c=tensor_lt5.cpu().detach().numpy(),
                        cmap='plasma', s=50)
        ax.set_title(name)
        fig.colorbar(sc, ax=ax, label="Intensity")

    # Hide any unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

# ------------------ Training Loop ------------------
def train():
    # Settings & Hyperparameters
    gamma_file = "../magic-gammas_part1.parquet"
    batch_size = 32
    num_epochs = 10
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dataset and dataloader
    dataset = GammaDataset(gamma_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Instantiate models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # Optimizers and loss functions
    gen_optimizer = optim.Adam(generator.parameters(), lr=lr)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=lr)
    bce_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    for epoch in range(num_epochs):
        for batch_idx, (x_m1, x_c_m1) in enumerate(dataloader):
            # Move data to device
            x_m1 = x_m1.to(device)           # (B, 1039)
            x_c_m1 = x_c_m1.to(device)         # (B, 1039)

            # Compute "true" noise = raw - cleaned.
            true_noise = x_m1 - x_c_m1         # (B, 1039)

            # Create binary mask: 1 where cleaned image is nonzero (particle region),
            # 0 where cleaned image is zero (background).
            mask = (x_c_m1 != 0).float()       # (B, 1039)

            # ------------------ Generator Forward Pass ------------------
            # Generator input: concatenate true_noise and mask as two channels.
            gen_input = torch.stack([true_noise, mask], dim=-1)  # (B, 1039, 2)
            gen_noise = generator(gen_input)                    # (B, 1039)

            # ------------------ Discriminator Training ------------------
            # For discriminator, use:
            # - Real noise: from background pixels (mask==0) --> true_noise.
            # - Fake noise: from particle pixels (mask==1) --> gen_noise.
            # Use gen_noise.detach() so that gradients don't flow back to the generator here.
            disc_input = torch.where(mask == 0, true_noise, gen_noise.detach())
            disc_target = torch.where(mask == 0,
                                      torch.ones_like(true_noise),
                                      torch.zeros_like(true_noise))
            disc_optimizer.zero_grad()
            disc_pred = discriminator(disc_input)  # (B, 1039)
            loss_disc = bce_loss(disc_pred, disc_target)
            loss_disc.backward()
            disc_optimizer.step()

            # ------------------ Generator Training ------------------
            gen_optimizer.zero_grad()
            # Adversarial loss: force discriminator to classify generated noise (in particle regions)
            # as real (target=1). Only apply adversarial loss on pixels with mask==1.
            disc_pred_gen = discriminator(gen_noise)  # (B, 1039)
            if mask.sum() > 0:
                adv_loss = bce_loss(disc_pred_gen[mask == 1],
                                    torch.ones_like(disc_pred_gen[mask == 1]))
            else:
                adv_loss = torch.tensor(0.0, device=device)

            # Reconstruction loss on background pixels (mask==0): ensure generated noise equals true noise.
            if (1 - mask).sum() > 0:
                recon_loss = l1_loss(gen_noise * (1 - mask), true_noise * (1 - mask))
            else:
                recon_loss = torch.tensor(0.0, device=device)

            loss_gen = adv_loss + recon_loss
            loss_gen.backward()
            gen_optimizer.step()

            # ------------------ Logging & Plotting ------------------
            if batch_idx % 20 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}] | "
                      f"D Loss: {loss_disc.item():.4f} | G Loss: {loss_gen.item():.4f}")
                # Plot the first sample in the batch.
                sample_true = true_noise[0]
                sample_gen = gen_noise[0]
                plot_noise_comparison(
                    [sample_true, sample_gen, mask[0], disc_pred[0], disc_input[0]],
                        ["actual", "generated", "mask", "disc_pred", "disc_input"],)


if __name__ == "__main__":
    train()
