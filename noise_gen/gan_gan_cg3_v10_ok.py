#!/usr/bin/env python3
"""
Adversarial Noise Generation for MAGIC Telescope Images using:
 - A Transformer-based Generator with spiking output (“SpikeTransformerGenerator”)
   that produces sparse, spiky noise via a dual head (spike probability and amplitude).
 - A Graph-based Discriminator (with sigmoid output) that processes the hexagonal camera geometry.

Loss per sample:
 • Discriminator: maximize separation between its average output on mask vs. non‐mask pixels.
 • Generator: minimize that separation on mask pixels + reconstruct background (non‐mask)
   to match true noise, with an additional regularization term that encourages binary (spiky)
   spike probabilities.
"""

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import spectral_norm
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
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# ------------------ Spike Transformer Generator ------------------
class SpikeTransformerGenerator(nn.Module):
    def __init__(self, input_dim=2, d_model=128, nhead=4, num_layers=2, seq_len=1039, temperature=0.05):
        """
        Expects per-pixel input with 2 channels:
          - Channel 0: true noise (raw - cleaned)
          - Channel 1: binary mask (1 if particle, 0 if background)
        Outputs two values per pixel:
          • A spike probability (between 0 and 1) – low temperature encourages near-binary output.
          • A spike amplitude (nonnegative).
        The final noise is given by: noise = spike_probability * spike_amplitude.
        """
        super().__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=0.1, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Two heads:
        self.spike_prob_head = nn.Linear(d_model, 1)
        self.spike_amp_head = nn.Linear(d_model, 1)
        self.temperature = temperature

    def forward(self, x):
        # x: (B, seq_len, input_dim)
        x = self.input_linear(x)              # (B, seq_len, d_model)
        x = self.pos_enc(x)
        x = x.transpose(0, 1)                 # (seq_len, B, d_model)
        x = self.transformer(x)
        x = x.transpose(0, 1)                 # (B, seq_len, d_model)
        # Spike probability with low temperature for sharper transitions.
        spike_prob = torch.sigmoid(self.spike_prob_head(x) / self.temperature)  # (B, seq_len, 1)
        # Spike amplitude: ensure nonnegative.
        spike_amp = torch.relu(self.spike_amp_head(x))                          # (B, seq_len, 1)
        noise = spike_prob * spike_amp
        return noise.squeeze(-1), spike_prob.squeeze(-1)  # Both (B, seq_len)

# ------------------ Graph Convolution Layer with Spectral Normalization ------------------
class GraphConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = spectral_norm(nn.Linear(in_channels, out_channels))

    def forward(self, x, edge_index):
        """
        x: (num_nodes, in_channels)
        edge_index: (2, E) with rows [target, source]
        Aggregates neighbor features by averaging.
        """
        num_nodes, _ = x.size()
        agg = torch.zeros_like(x)
        row, col = edge_index
        agg = agg.index_add(0, row, x[col])
        degree = torch.zeros(num_nodes, device=x.device)
        ones = torch.ones(row.size(0), device=x.device)
        degree = degree.index_add(0, row, ones).unsqueeze(-1).clamp(min=1)
        agg = agg / degree
        return self.linear(agg)

# ------------------ Graph-Based Discriminator ------------------
class GraphDiscriminator(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=64, num_layers=3, seq_len=1039, k=6):
        """
        Processes the noise signal arranged on the hexagonal grid.
        Outputs a per-pixel value squashed to [0,1] via sigmoid.
        """
        super().__init__()
        self.num_layers = num_layers
        self.gconvs = nn.ModuleList()
        self.gconvs.append(GraphConvLayer(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.gconvs.append(GraphConvLayer(hidden_channels, hidden_channels))
        self.gconvs.append(GraphConvLayer(hidden_channels, 1))
        # Build graph connectivity from MAGICCam geometry.
        geom = CameraGeometry.from_name("MAGICCam")
        pix_x = geom.pix_x.value
        pix_y = geom.pix_y.value
        coords = np.stack([pix_x, pix_y], axis=1)  # (num_nodes, 2)
        self.register_buffer('edge_index', self.build_graph_edges(coords, k))

    def build_graph_edges(self, coords, k=6):
        num_nodes = coords.shape[0]
        edge_list = []
        dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        for i in range(num_nodes):
            sorted_indices = np.argsort(dist[i])
            # Exclude self and take k nearest neighbors.
            neighbors = sorted_indices[1:k+1]
            for j in neighbors:
                edge_list.append((i, j))
                edge_list.append((j, i))
        edge_index = np.array(edge_list).T  # (2, num_edges)
        return torch.tensor(edge_index, dtype=torch.long)

    def forward(self, x):
        # x: (B, seq_len) or (B, seq_len, 1)
        if x.dim() == 3:
            x = x.squeeze(-1)  # (B, seq_len)
        batch_size, num_nodes = x.size()
        outputs = []
        for i in range(batch_size):
            node_features = x[i].unsqueeze(-1)  # (num_nodes, 1)
            for layer in self.gconvs[:-1]:
                node_features = torch.relu(layer(node_features, self.edge_index))
            node_features = self.gconvs[-1](node_features, self.edge_index)
            outputs.append(node_features.squeeze(-1))
        return torch.sigmoid(torch.stack(outputs, dim=0))  # (B, num_nodes)

# ------------------ Dataset ------------------
class GammaDataset(Dataset):
    def __init__(self, parquet_file):
        self.df = pd.read_parquet(parquet_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x_m1 = torch.tensor(row["image_m1"][:1039], dtype=torch.float32)
        x_c_m1 = torch.tensor(row["clean_image_m1"][:1039], dtype=torch.float32)
        return x_m1, x_c_m1

# ------------------ Plotting ------------------
def plot_noise_comparison(tensors_to_plot, names):
    geom = CameraGeometry.from_name("MAGICCam")
    pix_x = geom.pix_x.value
    pix_y = geom.pix_y.value
    num_plots = len(tensors_to_plot)
    if num_plots <= 4:
        cols = min(num_plots, 2)
        rows = int(np.ceil(num_plots / cols))
    else:
        cols = int(np.ceil(np.sqrt(num_plots)))
        rows = int(np.ceil(num_plots / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 7 * rows))
    axes = axes.flatten() if num_plots > 1 else [axes]
    for i, (tensor, name) in enumerate(zip(tensors_to_plot, names)):
        tensor_lt5 = tensor.clone()
        tensor_lt5[tensor_lt5 > 5] = 5
        if tensor_lt5.size(0) < 1039:
            tensor_lt5 = torch.cat((tensor_lt5, torch.zeros(1039 - tensor_lt5.size(0),
                                                               dtype=tensor_lt5.dtype,
                                                               device=tensor_lt5.device)))
        ax = axes[i]
        sc = ax.scatter(pix_x, pix_y, c=tensor_lt5.cpu().detach().numpy(),
                        cmap='plasma', s=50)
        ax.set_title(name)
        fig.colorbar(sc, ax=ax, label="Intensity")
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()

# ------------------ Training Loop ------------------
def train():
    gamma_file = "../magic-gammas_part1.parquet"
    batch_size = 32
    num_epochs = 10
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = GammaDataset(gamma_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Use the transformer-based generator with spiking output.
    generator = SpikeTransformerGenerator().to(device)
    discriminator = GraphDiscriminator().to(device)

    gen_optimizer = optim.Adam(generator.parameters(), lr=lr)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=lr)
    l1_loss = nn.L1Loss()
    lambda_reg = 0.1  # Regularization strength for spike probability binarization

    for epoch in range(num_epochs):
        for batch_idx, (x_m1, x_c_m1) in enumerate(dataloader):
            x_m1 = x_m1.to(device)          # (B, 1039)
            x_c_m1 = x_c_m1.to(device)        # (B, 1039)
            true_noise = x_m1 - x_c_m1        # (B, 1039)
            mask = (x_c_m1 != 0).float()      # (B, 1039)

            # ---------- Generator Forward Pass ----------
            gen_input = torch.stack([true_noise, mask], dim=-1)  # (B, 1039, 2)
            gen_noise, spike_prob = generator(gen_input)          # Both (B, 1039)

            # ---------- Discriminator Training ----------
            # For background (mask==0), use true_noise; for particle pixels (mask==1), use gen_noise (detach)
            disc_input = torch.where(mask == 0, true_noise, gen_noise.detach())
            disc_optimizer.zero_grad()
            disc_pred = discriminator(disc_input)  # (B, 1039), in [0,1]

            # Compute per-sample averages on masked and non-masked regions.
            mu_mask = torch.sum(disc_pred * mask, dim=1) / (mask.sum(dim=1) + 1e-8)
            mu_nonmask = torch.sum(disc_pred * (1 - mask), dim=1) / (((1 - mask).sum(dim=1)) + 1e-8)
            loss_disc = -torch.mean(torch.abs(mu_mask - mu_nonmask))
            loss_disc.backward()
            disc_optimizer.step()

            # ---------- Generator Training ----------
            gen_optimizer.zero_grad()
            disc_pred_gen = discriminator(gen_noise)  # (B, 1039)
            mu_mask_gen = torch.sum(disc_pred_gen * mask, dim=1) / (mask.sum(dim=1) + 1e-8)
            mu_nonmask_gen = torch.sum(disc_pred_gen * (1 - mask), dim=1) / (((1 - mask).sum(dim=1)) + 1e-8)
            adv_loss = torch.mean(torch.abs(mu_mask_gen - mu_nonmask_gen))
            recon_loss = l1_loss(gen_noise * (1 - mask), true_noise * (1 - mask))
            # Regularization to encourage spike_prob to be near 0 or 1 (i.e. binary)
            spike_reg_loss = lambda_reg * torch.mean(spike_prob * (1 - spike_prob))
            loss_gen = recon_loss + adv_loss + spike_reg_loss
            loss_gen.backward()
            gen_optimizer.step()

            if batch_idx % 40 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}] | D Loss: {loss_disc.item():.4f} | G Loss: {loss_gen.item():.4f}")
                # Plot sample outputs.
                sample_true = true_noise[0]
                sample_gen = gen_noise[0]
                sample_mask = mask[0]
                sample_disc_pred = disc_pred[0]
                sample_disc_input = disc_input[0]
                only_generated_noise = torch.where(mask == 0, 0, gen_noise.detach())[0]

                plot_noise_comparison(
                    [sample_true, sample_gen, sample_mask, sample_disc_pred, sample_disc_input, only_generated_noise],
                    ["True Noise", "Generated Noise", "Mask", "Discriminator Prediction", "Discriminator Input", "Generated Noise"]
                )

if __name__ == "__main__":
    train()
