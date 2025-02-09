#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd


# --- Dataset Definition ---
class MagicGammaDataset(Dataset):
    def __init__(self, parquet_file):
        self.df = pd.read_parquet(parquet_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Use only the first 1039 values from each image
        x_m1 = torch.tensor(row["image_m1"][:1039], dtype=torch.float32)
        x_c_m1 = torch.tensor(row["clean_image_m1"][:1039], dtype=torch.float32)
        return x_m1, x_c_m1


# --- Generator: 1D U-Net ---
class UNet1D(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, features=64):
        super(UNet1D, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv1d(in_channels, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(features, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.encoder2 = nn.Sequential(
            nn.Conv1d(features, features * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(features * 2, features * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.bottleneck = nn.Sequential(
            nn.Conv1d(features * 2, features * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(features * 4, features * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.ConvTranspose1d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv1d(features * 4, features * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(features * 2, features * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up1 = nn.ConvTranspose1d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv1d(features * 2, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(features, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.out_conv = nn.Conv1d(features, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)  # (B, features, L)
        p1 = self.pool1(e1)  # (B, features, L/2)
        e2 = self.encoder2(p1)  # (B, features*2, L/2)
        p2 = self.pool2(e2)  # (B, features*2, L/4)
        b = self.bottleneck(p2)  # (B, features*4, L/4)
        up2 = self.up2(b)  # (B, features*2, L/2)
        cat2 = torch.cat([up2, e2], dim=1)
        d2 = self.decoder2(cat2)  # (B, features*2, L/2)
        up1 = self.up1(d2)  # (B, features, L)
        cat1 = torch.cat([up1, e1], dim=1)
        d1 = self.decoder1(cat1)  # (B, features, L)
        out = self.out_conv(d1)  # (B, out_channels, L)
        return out


# --- Discriminator: 1D CNN ---
class Discriminator1D(nn.Module):
    def __init__(self, in_channels=1, features=64):
        super(Discriminator1D, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(features, features * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(features * 2, features * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(features * 4, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # Pixel-wise probability output in [0,1]
        )

    def forward(self, x):
        return self.net(x)


# --- Main Training Loop ---
def main():
    # Hyperparameters
    parquet_file = "../magic-gammas.parquet"
    num_epochs = 10
    batch_size = 16
    learning_rate = 1e-4
    lambda_adv = 1.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data preparation
    dataset = MagicGammaDataset(parquet_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize models
    generator = UNet1D(in_channels=2, out_channels=1).to(device)
    discriminator = Discriminator1D(in_channels=1).to(device)

    # Loss functions
    l1_loss = nn.L1Loss()
    bce_loss = nn.BCELoss()

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for batch_idx, (x_m1, x_c_m1) in enumerate(dataloader):
            print(f"Epoch [{epoch + 1}/{num_epochs}] Batch [{batch_idx}/{len(dataloader)}]")

            # Move data to device
            x_m1 = x_m1.to(device)  # Raw image: (B, 1039)
            x_c_m1 = x_c_m1.to(device)  # Clean image (particle): (B, 1039)

            # Compute ground-truth noise (raw minus cleaned)
            gt_noise = x_m1 - x_c_m1  # (B, 1039)
            # Create mask: 1 where particle is present (i.e. cleaned image non-zero), 0 otherwise.
            mask = (x_c_m1 != 0).float()  # (B, 1039)

            # Reshape for channel dimension
            gt_noise = gt_noise.unsqueeze(1)  # (B, 1, 1039)
            mask = mask.unsqueeze(1)  # (B, 1, 1039)

            # Generator input: [raw_noise, mask]
            gen_input = torch.cat([gt_noise, mask], dim=1)  # (B, 2, 1039)

            # --- Update Discriminator ---
            optimizer_D.zero_grad()
            # Generator produces noise prediction
            noise_pred = generator(gen_input)  # (B, 1, 1039)
            # Discriminator must classify each pixel as "generated" (1) or "real" (0).
            # Ground-truth for discriminator is the mask (1 for missing noise / particle region, 0 for real noise)
            disc_pred = discriminator(noise_pred.detach())
            loss_D = bce_loss(disc_pred, mask)
            loss_D.backward()
            optimizer_D.step()

            # --- Update Generator ---
            optimizer_G.zero_grad()
            noise_pred = generator(gen_input)
            disc_pred = discriminator(noise_pred)
            # Reconstruction loss on background (where mask==0, i.e. real noise exists)
            recon_loss = l1_loss(noise_pred * (1 - mask), gt_noise * (1 - mask))
            # Adversarial loss on particle regions (mask==1): Generator tries to fool Discriminator
            # (i.e. force disc_pred to 0 in missing noise regions)
            adv_loss = bce_loss(disc_pred * mask, torch.zeros_like(disc_pred * mask))
            loss_G = recon_loss + lambda_adv * adv_loss
            loss_G.backward()
            optimizer_G.step()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}] Batch [{batch_idx}/{len(dataloader)}] "
                      f"Loss_D: {loss_D.item():.4f}, Loss_G: {loss_G.item():.4f}, "
                      f"Recon: {recon_loss.item():.4f}, Adv: {adv_loss.item():.4f}")


if __name__ == "__main__":
    main()
