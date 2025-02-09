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
        """
        Implements standard positional encoding.
        """
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
        """
        Args:
            x: Tensor of shape (B, seq_len, d_model)
        Returns:
            Tensor with positional encodings added.
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# ------------------ Conditional Noise Generator ------------------
class ConditionalNoiseGenerator(nn.Module):
    """
    The generator takes a two-channel input per pixel:
      - Channel 0: The cleaned image value.
      - Channel 1: A per-pixel random noise seed.
    It outputs a prediction for the noise (raw minus clean).
    """
    def __init__(self, input_dim=2, d_model=128, nhead=4, num_layers=2, seq_len=1039):
        super().__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=0.1, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_linear = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (B, seq_len, 2)
        x = self.input_linear(x)             # (B, seq_len, d_model)
        x = self.pos_enc(x)
        x = x.transpose(0, 1)                # (seq_len, B, d_model) for transformer
        x = self.transformer(x)
        x = x.transpose(0, 1)                # (B, seq_len, d_model)
        out = self.output_linear(x)          # (B, seq_len, 1)
        return out.squeeze(-1)               # (B, seq_len)

# ------------------ Conditional Discriminator ------------------
class ConditionalDiscriminator(nn.Module):
    """
    The discriminator is conditioned on the cleaned image.
    It takes a two-channel input per pixel:
      - Channel 0: The cleaned image.
      - Channel 1: A noise image (either real or generated).
    It outputs a per-pixel logit (before sigmoid) which should be high
    (close to 1) for real noise and low (close to 0) for fake noise.
    """
    def __init__(self, input_dim=2, d_model=128, nhead=4, num_layers=2, seq_len=1039):
        super().__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=0.1, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_linear = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (B, seq_len, 2)
        x = self.input_linear(x)             # (B, seq_len, d_model)
        x = self.pos_enc(x)
        x = x.transpose(0, 1)                # (seq_len, B, d_model)
        x = self.transformer(x)
        x = x.transpose(0, 1)                # (B, seq_len, d_model)
        out = self.output_linear(x)          # (B, seq_len, 1)
        return out.squeeze(-1)               # (B, seq_len)

# ------------------ Dataset ------------------
class GammaDataset(Dataset):
    """
    Reads the parquet file containing two columns:
      - "image_m1": the raw image (particle + noise)
      - "clean_image_m1": the cleaned image (particle only)
    Returns:
      x_raw: the raw image,
      x_clean: the cleaned image.
    """
    def __init__(self, parquet_file):
        self.df = pd.read_parquet(parquet_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Use the first 1039 pixels.
        x_raw = torch.tensor(row["image_m1"][:1039], dtype=torch.float32)
        x_clean = torch.tensor(row["clean_image_m1"][:1039], dtype=torch.float32)
        return x_raw, x_clean

# ------------------ Plotting Function ------------------
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

# ------------------ Training Loop (Conditional GAN with Weighted Reconstruction Loss) ------------------
def train():
    # Path to the MAGIC telescope data.
    parquet_file = "../magic-gammas.parquet"
    batch_size = 32
    num_epochs = 10
    lr = 1e-4
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    dataset = GammaDataset(parquet_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize generator and discriminator.
    generator = ConditionalNoiseGenerator().to(device)
    discriminator = ConditionalDiscriminator().to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

    # Using BCEWithLogitsLoss to work directly with logits.
    bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    l1_loss = nn.L1Loss(reduction='none')

    # Loss weighting coefficients.
    lambda_adv = 1.0    # adversarial loss weight
    lambda_rec = 10.0   # reconstruction loss weight (you may want to tune this)

    # Additional factor for weighting reconstruction error by the magnitude of the noise.
    rec_weight_factor = 1.0

    generator.train()
    discriminator.train()
    for epoch in range(num_epochs):
        for batch_idx, (x_raw, x_clean) in enumerate(dataloader):
            x_raw = x_raw.to(device)      # (B, 1039)
            x_clean = x_clean.to(device)  # (B, 1039)

            # Compute the "true" noise (raw - clean).
            real_noise = x_raw - x_clean   # (B, 1039)

            # Create a valid-mask for background pixels only (where cleaned image is nearly zero).
            valid_mask = (x_clean.abs() < 1e-6).float()  # (B, 1039)

            # ----------------------
            # Train Discriminator
            # ----------------------
            optimizer_D.zero_grad()

            # For the generator input, sample per-pixel random noise.
            rand_seed = torch.randn_like(x_clean)  # (B, 1039)
            # Generator input: [cleaned image, random seed] as two channels.
            gen_input = torch.stack([x_clean, rand_seed], dim=-1)  # (B, 1039, 2)
            # Generate fake noise.
            fake_noise = generator(gen_input)  # (B, 1039)

            # Prepare discriminator inputs (concatenate condition and noise):
            # For real noise:
            real_disc_input = torch.stack([x_clean, real_noise], dim=-1)  # (B, 1039, 2)
            # For fake noise:
            fake_disc_input = torch.stack([x_clean, fake_noise.detach()], dim=-1)  # (B, 1039, 2)

            # Discriminator predictions.
            pred_real = discriminator(real_disc_input)  # (B, 1039)
            pred_fake = discriminator(fake_disc_input)   # (B, 1039)

            # Target labels.
            target_real = torch.ones_like(pred_real)
            target_fake = torch.zeros_like(pred_fake)

            # Compute BCE loss on valid (background) pixels only.
            loss_real = bce_loss(pred_real, target_real)
            loss_fake = bce_loss(pred_fake, target_fake)
            loss_real = (loss_real * valid_mask).sum() / (valid_mask.sum() + 1e-8)
            loss_fake = (loss_fake * valid_mask).sum() / (valid_mask.sum() + 1e-8)
            loss_D = (loss_real + loss_fake) / 2.0

            loss_D.backward()
            optimizer_D.step()

            # ----------------------
            # Train Generator
            # ----------------------
            optimizer_G.zero_grad()
            # We reuse fake_noise from above.
            fake_disc_input = torch.stack([x_clean, fake_noise], dim=-1)
            pred_fake_for_G = discriminator(fake_disc_input)  # (B, 1039)

            # Generator tries to fool the discriminator (labels = 1).
            adv_loss = bce_loss(pred_fake_for_G, target_real)
            adv_loss = (adv_loss * valid_mask).sum() / (valid_mask.sum() + 1e-8)

            # Weighted reconstruction loss: errors on high–noise pixels are amplified.
            # Compute per-pixel weights based on the magnitude of the true noise.
            rec_weights = 1.0 + rec_weight_factor * real_noise.abs()  # (B, 1039)
            # Apply L1 loss between fake and real noise.
            rec_loss_all = l1_loss(fake_noise, real_noise)
            rec_loss = (rec_loss_all * rec_weights * valid_mask).sum() / ((rec_weights * valid_mask).sum() + 1e-8)

            loss_G = lambda_adv * adv_loss + lambda_rec * rec_loss
            loss_G.backward()
            optimizer_G.step()

            if batch_idx % 20 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}] | D Loss: {loss_D.item():.4f} | G Loss: {loss_G.item():.4f}")
                # For visualization, pick the first sample in the batch.
                sample_idx = 0
                sample_clean = x_clean[sample_idx]
                sample_real_noise = real_noise[sample_idx]
                sample_fake_noise = fake_noise[sample_idx]
                plot_noise_comparison(
                    [sample_clean, sample_real_noise, sample_fake_noise, real_disc_input[1][:, 1],
                     fake_disc_input[1][:, 1]],
                    ["Clean Image", "Real Noise (raw - clean)", "Generated Noise", "Real Discriminator Input",
                     "Fake Discriminator Input"]
                )

if __name__ == "__main__":
    train()
