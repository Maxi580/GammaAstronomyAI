import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch.nn.functional as F


##############################################################################
# 1) Dataset
#    We assume disc_target = (x_c_m1 == 0).float(), i.e. 1 => "noise area",
#    0 => "particle area". The user wants to IGNORE the 0-area for D's loss.
##############################################################################
class MAGICNoiseDataset(Dataset):
    def __init__(self, gamma_file):
        self.df = pd.read_parquet(gamma_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        x_m1 = torch.tensor(row["image_m1"][:1039], dtype=torch.float32)
        x_c_m1 = torch.tensor(row["clean_image_m1"][:1039], dtype=torch.float32)

        # Noise = (raw - cleaned)
        noise = x_m1 - x_c_m1

        # disc_target => 1 if x_c_m1==0 => "pure noise area", 0 => "particle area".
        # The user wants to skip (ignore) the 0-area in the D's loss,
        # so disc_target=1 => area to be used (real noise region),
        # disc_target=0 => ignore in the D's loss.
        disc_target = (x_c_m1 == 0).float()

        return noise, disc_target, x_m1, x_c_m1


##############################################################################
# 2) Positional Encoding for 1D Transformers
##############################################################################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # shape (1, max_len, d_model)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


##############################################################################
# 3) Transformer Generator - unbounded output
##############################################################################
class TransformerGenerator(nn.Module):
    def __init__(self, d_model=64, nhead=8, num_layers=2):
        super().__init__()
        self.d_model = d_model

        self.input_proj = nn.Linear(1, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                         num_layers=num_layers)
        self.output_fc = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: [B, 1039]
        x = x.unsqueeze(-1)  # => [B, 1039, 1]
        x = self.input_proj(x)  # => [B, 1039, d_model]
        x = self.pos_encoding(x)  # => [B, 1039, d_model]
        x = self.transformer_encoder(x)  # => [B, 1039, d_model]
        x = self.output_fc(x).squeeze(-1)  # => [B, 1039]
        return x  # unbounded


##############################################################################
# 4) Transformer Discriminator - output in [0,1]
#    We add a final Sigmoid so the result is strictly between 0 and 1.
##############################################################################
class TransformerDiscriminator(nn.Module):
    def __init__(self, d_model=64, nhead=8, num_layers=2):
        super().__init__()
        self.d_model = d_model

        self.input_proj = nn.Linear(1, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                         num_layers=num_layers)
        self.output_fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, 1039]
        x = x.unsqueeze(-1)  # => [B, 1039, 1]
        x = self.input_proj(x)  # => [B, 1039, d_model]
        x = self.pos_encoding(x)  # => [B, 1039, d_model]
        x = self.transformer_encoder(x)  # => [B, 1039, d_model]
        x = self.output_fc(x).squeeze(-1)  # => [B, 1039]
        x = self.sigmoid(x)  # => [B, 1039], in [0,1]
        return x


##############################################################################
# 5) A helper for masked BCE.
#    We only compute BCE in areas mask=1, ignoring mask=0.
##############################################################################
def masked_bce(pred, target, mask):
    """
    pred, target, mask: all [B, 1039]
    We'll do elementwise BCE, then multiply by mask,
    then sum over all, then divide by the sum of mask.
    This effectively ignores those pixels with mask=0.
    """
    # 'pred' is in [0,1] (since we used final sigmoid).
    # 'target' is 0 or 1.
    # 'mask' is 0 or 1.
    # We'll do:
    bce_eltwise = F.binary_cross_entropy(pred, target, reduction='none')
    masked_loss = (bce_eltwise * mask).sum() / (mask.sum() + 1e-8)
    return masked_loss


##############################################################################
# 6) Training Loop
#    Typical "GAN style" but ignoring disc_target=0 region in both real & fake passes.
#    disc_target=1 => "noise region => real data", so we treat it as label=1 for real pass.
#    For the generator's fake pass, we also use that region => label=0 to train D,
#    but G tries to push it to 1 => fool D.
#    However, user specifically wants ignoring disc_target=0, so we skip it in losses.
##############################################################################
def train_adversarial(
        generator,
        discriminator,
        dataloader,
        g_optimizer,
        d_optimizer,
        device='cpu',
        alpha=0.0  # weight for optional difference (MSE) loss
):
    generator.train()
    discriminator.train()

    for noise, disc_target, x_m1, x_c_m1 in dataloader:
        noise = noise.to(device)
        disc_target = disc_target.to(device)  # 1 => "noise region" to use, 0 => ignore
        x_m1 = x_m1.to(device)
        x_c_m1 = x_c_m1.to(device)

        # We'll define:
        # mask_for_loss = disc_target (==1 in noise region, 0 in particle region)
        # Because user wants to ignore the 0 region in D's loss calculations.
        mask_for_loss = disc_target

        #######################################################################
        # 1) Train Discriminator
        #######################################################################
        d_optimizer.zero_grad()

        # a) Real pass
        # We feed the real noise array "noise".
        # We want the D to output 1 => "real" in the region disc_target=1,
        # ignoring disc_target=0.
        pred_real = discriminator(noise)
        # label=1 in that region
        real_labels = torch.ones_like(pred_real)
        loss_d_real = masked_bce(pred_real, real_labels, mask_for_loss)

        # b) Fake pass
        # We generate noise for the region disc_target=1,
        # combine with original "noise" or do direct pass. For standard "GAN style",
        # let's do the same shape pass (the region disc_target=1 is supposed to be "real noise",
        # so the generator tries to produce something D can't distinguish).
        fake_noise = generator(noise)
        # We'll just pass the "fake_noise" directly or combine with "noise."
        # But user is focusing on the zero area => let's do direct "fake_noise" to keep it simpler.
        # If you want inpainting style, combine them. But let's follow typical approach:
        pred_fake = discriminator(fake_noise)
        # D should output 0 => "fake" in the region disc_target=1 => so label=0
        fake_labels = torch.zeros_like(pred_fake)
        loss_d_fake = masked_bce(pred_fake, fake_labels, mask_for_loss)

        loss_d = 0.5 * (loss_d_real + loss_d_fake)
        loss_d.backward()
        d_optimizer.step()

        #######################################################################
        # 2) Train Generator
        #######################################################################
        g_optimizer.zero_grad()

        fake_noise_g = generator(noise)
        pred_fake_g = discriminator(fake_noise_g)

        # Now G wants to fool D => produce output=1 in that region disc_target=1.
        # ignoring the region disc_target=0.
        target_for_g = torch.ones_like(pred_fake_g)
        loss_g_adv = masked_bce(pred_fake_g, target_for_g, mask_for_loss)

        # Optional difference loss to keep generated noise close to original
        if alpha > 0.0:
            mse = nn.MSELoss()
            diff_loss = mse(fake_noise_g * mask_for_loss, noise * mask_for_loss)
            loss_g = loss_g_adv + alpha * diff_loss
        else:
            loss_g = loss_g_adv

        loss_g.backward()
        g_optimizer.step()


##############################################################################
# 7) Example usage
##############################################################################
if __name__ == "__main__":
    gamma_file = "../magic-gammas.parquet"
    dataset = MAGICNoiseDataset(gamma_file)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    generator = TransformerGenerator(d_model=64, nhead=8, num_layers=2).to(device)
    discriminator = TransformerDiscriminator(d_model=64, nhead=8, num_layers=2).to(device)

    g_optimizer = optim.Adam(generator.parameters(), lr=1e-4)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4)

    # Train for a few epochs
    for epoch in range(5):
        train_adversarial(
            generator,
            discriminator,
            dataloader,
            g_optimizer,
            d_optimizer,
            device=device,
            alpha=0.1  # e.g. weighting for difference loss
        )
        print(f"Finished epoch {epoch + 1}")
