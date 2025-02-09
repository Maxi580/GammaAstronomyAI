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

class Generator(nn.Module):
    def __init__(self, input_dim=2, d_model=256, nhead=4, num_layers=4, seq_len=1039):
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
    def __init__(self, in_channels=1, hidden_channels=128, num_layers=3, seq_len=1039, k=6):
        super().__init__()
        self.num_layers = num_layers
        self.gconvs = nn.ModuleList()
        self.gconvs.append(GraphConvLayer(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.gconvs.append(GraphConvLayer(hidden_channels, hidden_channels))
        self.gconvs.append(GraphConvLayer(hidden_channels, 1))
        # Build graph connectivity from MAGICCam geometry.
        geom = CameraGeometry.from_name("MAGICCam")
        pix_x = geom.pix_x.value  # (num_pixels,)
        pix_y = geom.pix_y.value
        coords = np.stack([pix_x, pix_y], axis=1)  # shape: (num_pixels, 2)
        self.register_buffer('edge_index', self.build_graph_edges(coords, k))
        # Precompute pairwise distances (Euclidean) on the hexagonal grid.
        # This matrix is (num_pixels x num_pixels) and never changes.
        dist_mat = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        self.register_buffer('distance_matrix', torch.tensor(dist_mat, dtype=torch.float32))
        self.d_max = float(self.distance_matrix.max())

    def build_graph_edges(self, coords, k=6):
        num_nodes = coords.shape[0]
        edge_list = []
        dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        for i in range(num_nodes):
            sorted_indices = np.argsort(dist[i])
            neighbors = sorted_indices[1:k + 1]
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


# ------------------ Helper: Compute Distance-Based Weights ------------------
def compute_distance_weights(mask, distance_matrix, d_max, pos_weight, neg_weight):
    """
    For each image in the batch, compute a per-pixel weight based on the distance to
    the nearest particle pixel (mask==1). Here, mask is a tensor of shape (B, 1039)
    with values 0 or 1.
    """
    B, num_pixels = mask.shape
    weights = []
    # Loop over the batch (1039 is small so a Python loop is acceptable)
    for i in range(B):
        m = mask[i]  # (1039,)
        # Find indices where the particle is present.
        particle_idx = (m > 0).nonzero(as_tuple=True)[0]
        if len(particle_idx) == 0:
            # No particle pixels → treat all pixels as far from any particle.
            d = torch.full((num_pixels,), d_max, device=mask.device)
        else:
            # distance_matrix: (num_pixels, num_pixels)
            # For each pixel, take the minimum distance among the particle pixels.
            d_all = distance_matrix[:, particle_idx]  # shape: (num_pixels, num_particle_pixels)
            d, _ = torch.min(d_all, dim=1)  # (num_pixels,)
        # Normalize distance and compute weight.
        # Pixels right on the particle (d==0) get neg_weight; pixels far away get pos_weight.
        w = neg_weight + (pos_weight - neg_weight) * (d / d_max)
        weights.append(w)
    return torch.stack(weights, dim=0)  # (B, num_pixels)


# ------------------ Plotting ------------------
def plot_noise_comparison(tensors_to_plot, names):
    geom = CameraGeometry.from_name("MAGICCam")
    pix_x = geom.pix_x.value
    pix_y = geom.pix_y.value
    num_plots = len(tensors_to_plot)
    cols = min(num_plots, 2) if num_plots <= 4 else int(np.ceil(np.sqrt(num_plots)))
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



# ------------------ Training Loop (Modified) ------------------
def train():
    gamma_file = "../magic-protons_part1.parquet"
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
    mse_loss = nn.MSELoss(reduction='none')
    l1_loss = nn.L1Loss()

    # Set base weights (you can adjust these as needed)
    pos_weight = 5.0  # Intended for “pure” background (far from particle)
    neg_weight = 1.0  # Intended for pixels right on the particle region

    for epoch in range(num_epochs):
        for batch_idx, (x_m1, x_c_m1) in enumerate(dataloader):
            x_m1 = x_m1.to(device)  # (B, 1039)
            x_c_m1 = x_c_m1.to(device)  # (B, 1039)
            true_noise = x_m1 - x_c_m1  # (B, 1039)
            # mask: 1 where the cleaned image is nonzero (particle region), else 0.
            mask = (x_c_m1 != 0).float()  # (B, 1039)

            # ---------- Generator Forward Pass ----------
            gen_input = torch.stack([true_noise, mask], dim=-1)  # (B, 1039, 2)
            gen_noise = generator(gen_input)  # (B, 1039)

            # ---------- Discriminator Training ----------
            # Use true_noise for background pixels and gen_noise for particle regions.
            disc_input = torch.where(mask == 0, true_noise, gen_noise.detach())
            # Targets: background pixels → 1 (real noise), particle pixels → 0 (fake noise)
            real_target = torch.ones_like(true_noise)
            fake_target = torch.zeros_like(true_noise)
            disc_target = torch.where(mask == 0, real_target, fake_target)

            # Compute per-pixel weights based on distance to the particle.
            # Note: discriminator.distance_matrix is on the same device as discriminator.
            weights = compute_distance_weights(mask, discriminator.distance_matrix, discriminator.d_max,
                                               pos_weight, neg_weight)
            disc_optimizer.zero_grad()
            disc_pred = discriminator(disc_input)  # (B, 1039); raw scores
            loss_disc_all = mse_loss(disc_pred, disc_target)
            loss_disc = torch.mean(loss_disc_all * weights)
            loss_disc.backward()
            disc_optimizer.step()

            # ---------- Generator Training ----------
            gen_optimizer.zero_grad()
            disc_pred_gen = discriminator(gen_noise)
            # For the generator, we care about the particle regions (mask==1)
            adv_loss = torch.mean(mse_loss(disc_pred_gen[mask == 1],
                                           torch.ones_like(disc_pred_gen[mask == 1])))
            # Reconstruction loss on background regions (mask==0)
            if (1 - mask).sum() > 0:
                recon_loss = l1_loss(gen_noise * (1 - mask), true_noise * (1 - mask))
            else:
                recon_loss = torch.tensor(0.0, device=device)
            loss_gen = adv_loss + recon_loss
            loss_gen.backward()
            gen_optimizer.step()

            if batch_idx % 20 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}] Batch [{batch_idx}] | "
                      f"D Loss: {loss_disc.item():.4f} | G Loss: {loss_gen.item():.4f}")
                sample_true = true_noise[0]
                sample_gen = gen_noise[0]
                masked_noise = torch.where(mask == 0, 0, gen_noise.detach())[0]
                plot_noise_comparison([sample_true, sample_gen, mask[0], disc_pred[0], disc_input[0], masked_noise],
                                      ["actual", "generated", "mask", "disc_pred", "disc_input", "masked_noise"])


if __name__ == "__main__":
    train()
