#!/usr/bin/env python3
"""
Adversarial Noise Generation for MAGIC Telescope Images using:
 - A Transformer-based Generator whose output is forced to be nonnegative.
 - A graph-based Discriminator (with sigmoid output) that processes the hexagonal camera geometry.

Loss function (per sample):
  • Discriminator: maximize the separation between its average output on mask vs. non-mask pixels.
  • Generator: minimize that separation on mask pixels + reconstruct the background (non-mask) to match true noise.
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
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ------------------ Generator (Model1) ------------------
class Generator(nn.Module):
    def __init__(self, input_dim=2, d_model=128, nhead=4, num_layers=2, seq_len=1039):
        """
        Expects per-pixel input with 2 channels:
          - Channel 0: true noise (raw - cleaned)
          - Channel 1: binary mask (1 if particle, 0 if background)
        Outputs a 1D noise prediction per pixel.
        The final activation ensures nonnegative outputs.
        """
        super().__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=0.1, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_linear = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_linear(x)  # (B, seq_len, d_model)
        x = self.pos_enc(x)
        x = x.transpose(0, 1)  # (seq_len, B, d_model)
        x = self.transformer(x)
        x = x.transpose(0, 1)  # (B, seq_len, d_model)
        out = self.output_linear(x)  # (B, seq_len, 1)
        # Enforce nonnegative outputs (true noise values are > 0)
        return torch.relu(out.squeeze(-1))  # (B, seq_len)


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
        Processes the noise signal arranged on the hexagonal grid.
        Outputs a value per pixel that is squashed into [0,1] (via sigmoid).
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
        # Compute pairwise distances.
        dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        for i in range(num_nodes):
            sorted_indices = np.argsort(dist[i])
            # Exclude self (first element) and take k nearest neighbors.
            neighbors = sorted_indices[1:k + 1]
            for j in neighbors:
                edge_list.append((i, j))
                edge_list.append((j, i))
        edge_index = np.array(edge_list).T  # shape: (2, num_edges)
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
        # Stack and squash outputs to [0,1]
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
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = GammaDataset(gamma_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    generator = Generator().to(device)
    discriminator = GraphDiscriminator().to(device)

    gen_optimizer = optim.Adam(generator.parameters(), lr=lr)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

    l1_loss = nn.L1Loss()

    # For each sample:
    #   - Discriminator sees true_noise on background (mask==0) and gen_noise on particles (mask==1).
    #   - It is trained to maximize the separation (difference) between its average outputs on these two regions.
    #   - The generator is trained with two terms:
    #         a) Reconstruction loss on background (non-mask) so that gen_noise ≈ true_noise there.
    #         b) An adversarial loss that minimizes the difference between the average discriminator outputs on mask and non-mask pixels.

    for epoch in range(num_epochs):
        for batch_idx, (x_m1, x_c_m1) in enumerate(dataloader):
            x_m1 = x_m1.to(device)  # (B, 1039)
            x_c_m1 = x_c_m1.to(device)  # (B, 1039)
            true_noise = x_m1 - x_c_m1  # (B, 1039)
            mask = (x_c_m1 != 0).float()  # (B, 1039)

            # ---------- Generator Forward Pass ----------
            gen_input = torch.stack([true_noise, mask], dim=-1)  # (B, 1039, 2)
            gen_noise = generator(gen_input)  # (B, 1039), nonnegative by ReLU

            # ---------- Discriminator Training ----------
            # For background (mask==0), use true_noise; for particles (mask==1), use gen_noise (detach)
            disc_input = torch.where(mask == 0, true_noise, gen_noise.detach())
            disc_optimizer.zero_grad()
            disc_pred = discriminator(disc_input)  # (B, 1039), in [0,1]

            # Compute per-sample averages
            mu_mask = torch.sum(disc_pred * mask, dim=1) / (mask.sum(dim=1) + 1e-8)
            mu_nonmask = torch.sum(disc_pred * (1 - mask), dim=1) / (((1 - mask).sum(dim=1)) + 1e-8)
            # Discriminator maximizes separation -> loss = -|mu_mask - mu_nonmask|
            loss_disc = -torch.mean(torch.abs(mu_mask - mu_nonmask))
            loss_disc.backward()
            disc_optimizer.step()

            # ---------- Generator Training ----------
            gen_optimizer.zero_grad()
            disc_pred_gen = discriminator(gen_noise)  # (B, 1039)
            mu_mask_gen = torch.sum(disc_pred_gen * mask, dim=1) / (mask.sum(dim=1) + 1e-8)
            mu_nonmask_gen = torch.sum(disc_pred_gen * (1 - mask), dim=1) / (((1 - mask).sum(dim=1)) + 1e-8)
            # Generator tries to minimize the separation (adversarial loss)
            adv_loss = torch.mean(torch.abs(mu_mask_gen - mu_nonmask_gen))
            # Plus, add a reconstruction loss on background (non-mask) pixels so that they remain unchanged.
            recon_loss = l1_loss(gen_noise * (1 - mask), true_noise * (1 - mask))
            loss_gen = recon_loss + adv_loss
            loss_gen.backward()
            gen_optimizer.step()

            if batch_idx % 20 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}] Batch [{batch_idx}] | "
                      f"D Loss: {loss_disc.item():.4f} | G Loss: {loss_gen.item():.4f}")
                sample_true = true_noise[0]
                sample_gen = gen_noise[0]
                plot_noise_comparison(
                    [sample_true, sample_gen, mask[0], disc_pred[0], disc_input[0]],
                    ["true_noise", "gen_noise", "mask", "disc_pred", "disc_input"],
                )


if __name__ == "__main__":
    train()
