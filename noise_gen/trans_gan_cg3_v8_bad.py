#!/usr/bin/env python3
"""
Adversarial Noise Generation for MAGIC Telescope Images using a Transformer-based Generator
and a graph-based Discriminator (with label smoothing, instance noise, and adjusted lr) that leverages
the hexagonal camera geometry.
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

# ------------------ Positional Encoding (for Generator) ------------------
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
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# ------------------ Generator (Model1) ------------------
class Generator(nn.Module):
    def __init__(self, input_dim=2, d_model=128, nhead=4, num_layers=2, seq_len=1039):
        """
        Generator takes a 2-channel input per pixel:
          - Channel 0: True noise (raw - cleaned)
          - Channel 1: Binary mask (1 if cleaned pixel nonzero)
        and outputs a 1D noise prediction for each pixel.
        """
        super().__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=0.1, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_linear = nn.Linear(d_model, 1)
    def forward(self, x):
        x = self.input_linear(x)
        x = self.pos_enc(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)
        out = self.output_linear(x)
        return out.squeeze(-1)

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

# ------------------ Graph-Based Discriminator (Model2) ------------------
class GraphDiscriminator(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=64, num_layers=3, seq_len=1039, k=6):
        """
        Uses a simple GCN to process the noise signal arranged on the hexagonal grid.
        The network outputs raw scores (no sigmoid).
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
        coords = np.stack([pix_x, pix_y], axis=1)
        self.register_buffer('edge_index', self.build_graph_edges(coords, k))
    def build_graph_edges(self, coords, k=6):
        num_nodes = coords.shape[0]
        edge_list = []
        dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        for i in range(num_nodes):
            sorted_indices = np.argsort(dist[i])
            neighbors = sorted_indices[1:k+1]
            for j in neighbors:
                edge_list.append((i, j))
                edge_list.append((j, i))
        edge_index = np.array(edge_list).T
        return torch.tensor(edge_index, dtype=torch.long)
    def forward(self, x):
        # x: (batch, seq_len) or (batch, seq_len, 1)
        if x.dim() == 3:
            x = x.squeeze(-1)
        batch_size, num_nodes = x.size()
        outputs = []
        for i in range(batch_size):
            node_features = x[i].unsqueeze(-1)  # (num_nodes, 1)
            for layer in self.gconvs[:-1]:
                node_features = torch.relu(layer(node_features, self.edge_index))
            node_features = self.gconvs[-1](node_features, self.edge_index)
            outputs.append(node_features.squeeze(-1))
        return torch.stack(outputs, dim=0)  # (batch, num_nodes)

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
        tensor_lt5 = torch.cat((tensor_lt5,
                                torch.zeros(1039 - len(tensor_lt5), dtype=tensor_lt5.dtype,
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
    gen_lr = 1e-4
    disc_lr = 1e-5  # Lower lr for the discriminator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = GammaDataset(gamma_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    generator = Generator().to(device)
    discriminator = GraphDiscriminator().to(device)
    gen_optimizer = optim.Adam(generator.parameters(), lr=gen_lr)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=disc_lr)
    mse_loss = nn.MSELoss(reduction='none')
    l1_loss = nn.L1Loss()
    # Weight factors for pixel losses
    pos_weight = 5.0
    neg_weight = 1.0

    for epoch in range(num_epochs):
        for batch_idx, (x_m1, x_c_m1) in enumerate(dataloader):
            x_m1 = x_m1.to(device)      # (B, 1039)
            x_c_m1 = x_c_m1.to(device)    # (B, 1039)
            true_noise = x_m1 - x_c_m1    # (B, 1039)
            mask = (x_c_m1 != 0).float()  # (B, 1039)

            # ---------- Generator Forward Pass ----------
            gen_input = torch.stack([true_noise, mask], dim=-1)  # (B, 1039, 2)
            gen_noise = generator(gen_input)                    # (B, 1039)

            # ---------- Discriminator Training ----------
            # For discriminator:
            # - Background (mask==0): true_noise → real (target=0.9 after smoothing)
            # - Particle regions (mask==1): gen_noise (detached) → fake (target=0)
            disc_input = torch.where(mask == 0, true_noise, gen_noise.detach())
            # Add instance noise
            disc_input = disc_input + 0.01 * torch.randn_like(disc_input)
            real_target = 0.9 * torch.ones_like(true_noise)  # one-sided label smoothing
            fake_target = torch.zeros_like(true_noise)
            disc_target = torch.where(mask == 0, real_target, fake_target)
            disc_optimizer.zero_grad()
            disc_pred = discriminator(disc_input)  # (B, 1039); raw scores
            loss_disc_all = mse_loss(disc_pred, disc_target)
            weights = torch.where(mask == 1, pos_weight * torch.ones_like(mask),
                                  neg_weight * torch.ones_like(mask))
            loss_disc = torch.mean(loss_disc_all * weights)
            loss_disc.backward()
            disc_optimizer.step()

            # ---------- Generator Training ----------
            gen_optimizer.zero_grad()
            disc_pred_gen = discriminator(gen_noise)
            # For generator: in particle regions push disc_pred_gen toward 0.9 (real)
            adv_loss = torch.mean(mse_loss(disc_pred_gen[mask == 1],
                                           0.9 * torch.ones_like(disc_pred_gen[mask == 1])))
            # Reconstruction loss on background regions (mask==0)
            if (1 - mask).sum() > 0:
                recon_loss = l1_loss(gen_noise * (1 - mask), true_noise * (1 - mask))
            else:
                recon_loss = torch.tensor(0.0, device=device)
            loss_gen = adv_loss + recon_loss
            loss_gen.backward()
            gen_optimizer.step()

            if batch_idx % 20 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}] | "
                      f"D Loss: {loss_disc.item():.4f} | G Loss: {loss_gen.item():.4f}")
                sample_true = true_noise[0]
                sample_gen = gen_noise[0]
                plot_noise_comparison([sample_true, sample_gen, mask[0], disc_pred[0], disc_input[0]],
                                      ["actual", "generated", "mask", "disc_pred", "disc_input"])

if __name__ == "__main__":
    train()
