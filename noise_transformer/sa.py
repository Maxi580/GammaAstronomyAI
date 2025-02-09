import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from ctapipe.instrument import CameraGeometry


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


# ------------------ CNN-based Conditional Noise Generator ------------------
class ConditionalNoiseGeneratorCNN(nn.Module):
    """
    A convolutional generator that takes a two-channel input per pixel:
      - Channel 0: The cleaned image.
      - Channel 1: A per-pixel random noise seed.
    It outputs a prediction for the noise (raw minus clean).
    """

    def __init__(self, seq_len=1039):
        super().__init__()
        # Input: (B, 2, seq_len)
        self.conv1 = nn.Conv1d(2, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 1, kernel_size=3, padding=1)
        # You might experiment with a final activation.
        # For example, if you normalize to [0,1] then use a sigmoid (and later rescale).
        # Here, we leave it linear.

    def forward(self, x):
        # x: (B, 2, seq_len)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)  # (B, 1, seq_len)
        return x.squeeze(1)  # (B, seq_len)


# ------------------ CNN-based Conditional Discriminator ------------------
class ConditionalDiscriminatorCNN(nn.Module):
    """
    A convolutional discriminator conditioned on the cleaned image.
    Input is two channels:
      - Channel 0: Cleaned image.
      - Channel 1: Noise image (either real or generated).
    Outputs per-pixel logits.
    """

    def __init__(self, seq_len=1039):
        super().__init__()
        # Input: (B, 2, seq_len)
        self.conv1 = nn.Conv1d(2, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # x: (B, 2, seq_len)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)  # (B, 1, seq_len)
        return x.squeeze(1)  # (B, seq_len)


# ------------------ Training Loop (CNN-based Conditional GAN) ------------------
def train():
    parquet_file = "../magic-gammas.parquet"
    batch_size = 32
    num_epochs = 10
    lr = 1e-4

    # Device selection.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    dataset = GammaDataset(parquet_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize CNN-based generator and discriminator.
    generator = ConditionalNoiseGeneratorCNN().to(device)
    discriminator = ConditionalDiscriminatorCNN().to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

    # Loss functions.
    bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    l1_loss = nn.L1Loss(reduction='none')

    # Loss weighting coefficients.
    lambda_adv = 1.0  # adversarial loss weight
    lambda_rec = 10.0  # reconstruction loss weight (adjust as needed)
    rec_weight_factor = 1.0

    generator.train()
    discriminator.train()
    for epoch in range(num_epochs):
        for batch_idx, (x_raw, x_clean) in enumerate(dataloader):
            # Optionally, normalize your data here if needed.
            # For example, if your values are in [0, 10] range, you might do:
            # x_raw = x_raw / 10.0; x_clean = x_clean / 10.0
            x_raw = x_raw.to(device)  # (B, seq_len)
            x_clean = x_clean.to(device)  # (B, seq_len)
            real_noise = x_raw - x_clean  # (B, seq_len)

            # Create a valid-mask for background pixels only (where cleaned image is nearly zero).
            valid_mask = (x_clean.abs() < 1e-6).float()  # (B, seq_len)

            # ----------------------
            # Train Discriminator
            # ----------------------
            optimizer_D.zero_grad()
            rand_seed = torch.randn_like(x_clean)  # (B, seq_len)
            # Generator input: [cleaned image, random seed] as channels.
            gen_input = torch.stack([x_clean, rand_seed], dim=1)  # (B, 2, seq_len)
            fake_noise = generator(gen_input)  # (B, seq_len)

            # Prepare discriminator inputs.
            real_disc_input = torch.stack([x_clean, real_noise], dim=1)  # (B, 2, seq_len)
            fake_disc_input = torch.stack([x_clean, fake_noise.detach()], dim=1)  # (B, 2, seq_len)

            pred_real = discriminator(real_disc_input)  # (B, seq_len)
            pred_fake = discriminator(fake_disc_input)  # (B, seq_len)

            target_real = torch.ones_like(pred_real)
            target_fake = torch.zeros_like(pred_fake)

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
            fake_disc_input = torch.stack([x_clean, fake_noise], dim=1)  # (B, 2, seq_len)
            pred_fake_for_G = discriminator(fake_disc_input)

            adv_loss = bce_loss(pred_fake_for_G, target_real)
            adv_loss = (adv_loss * valid_mask).sum() / (valid_mask.sum() + 1e-8)

            rec_weights = 1.0 + rec_weight_factor * real_noise.abs()  # amplify errors on high peaks
            rec_loss_all = l1_loss(fake_noise, real_noise)
            rec_loss = (rec_loss_all * rec_weights * valid_mask).sum() / ((rec_weights * valid_mask).sum() + 1e-8)

            loss_G = lambda_adv * adv_loss + lambda_rec * rec_loss
            loss_G.backward()
            optimizer_G.step()

            if batch_idx % 20 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}] Batch [{batch_idx}] | D Loss: {loss_D.item():.4f} | G Loss: {loss_G.item():.4f}")
                sample_idx = 0
                sample_clean = x_clean[sample_idx]
                sample_real_noise = real_noise[sample_idx]
                sample_fake_noise = fake_noise[sample_idx]
                plot_noise_comparison(
                    [sample_clean, sample_real_noise, sample_fake_noise],
                    ["Clean Image", "Real Noise (raw - clean)", "Generated Noise"]
                )


if __name__ == "__main__":
    train()
