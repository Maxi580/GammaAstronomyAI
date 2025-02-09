import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ctapipe.instrument import CameraGeometry

from noise_gen.trans_cg import TransformerGenerator, TransformerDiscriminator, MAGICNoiseDataset, masked_bce


def plot_noise_comparison(tensors_to_plot, names):
    """Plot noise comparisons in a 2x2 grid

    Args:
        tensors_to_plot: List of tensors to plot
        names: List of names/titles for each plot
        pix_x: x coordinates for scatter plot
        pix_y: y coordinates for scatter plot
        batch_idx: Batch index for the suptitle
    """

    geom = CameraGeometry.from_name("MAGICCam")
    pix_x = geom.pix_x.value
    pix_y = geom.pix_y.value

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 14))
    axes = [ax1, ax2, ax3, ax4]

    for tensor, name, ax in zip(tensors_to_plot, names, axes):
        sc = ax.scatter(pix_x, pix_y, c=tensor.cpu().detach().numpy(),
                        cmap='plasma', s=50)
        ax.set_title(name)
        fig.colorbar(sc, ax=ax, label="Intensity")

    plt.tight_layout()
    plt.show()


def train_adversarial_with_vis(
        generator, discriminator,
        dataloader,
        g_optimizer, d_optimizer,
        device='cpu',
        alpha=0.1
):
    # Get MAGIC camera geometry for visualization
    geom = CameraGeometry.from_name("MAGICCam")
    pix_x = geom.pix_x.value
    pix_y = geom.pix_y.value

    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    generator.train()
    discriminator.train()


    for batch_idx, (noise, disc_target, x_m1, x_c_m1) in enumerate(dataloader):
        print(f"Batch {batch_idx} / {len(dataloader)}")


        noise = noise.to(device)
        disc_target = disc_target.to(device)
        # disc_target = 1 - disc_target
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


        print(f"Loss D: {loss_d.item()} - Loss G: {loss_g.item()}")

        # Visualize every 20 batches
        if batch_idx % 10 == 0:
            # Get first sample from batch for visualization
            noise_lt5 = noise[0]
            noise_lt5[noise_lt5 > 5] = 5

            fake_noise_lt5 = fake_noise[0]
            fake_noise_lt5[fake_noise_lt5 > 5] = 5

            disc_target_lt5 = disc_target[0]
            disc_target_lt5[disc_target_lt5 > 5] = 5

            logits_real_tg_lt5 = pred_fake_g[0]
            logits_real_tg_lt5[logits_real_tg_lt5 > 5] = 5

            plot_noise_comparison(
                [noise_lt5,logits_real_tg_lt5,disc_target_lt5,fake_noise_lt5,],
                ["Noise", "Logits Real", "Disc Target", "Fake Noise"],
                pix_x,
                pix_y,
                batch_idx
            )

    return


# Modified main section
if __name__ == "__main__":
    gamma_file = "../magic-protons_part1.parquet"
    dataset = MAGICNoiseDataset(gamma_file)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    generator = TransformerGenerator(d_model=256, nhead=8, num_layers=2).to(device)
    discriminator = TransformerDiscriminator(d_model=128, nhead=8, num_layers=4).to(device)

    g_optimizer = optim.Adam(generator.parameters(), lr=1e-4)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4)

    # Train with visualization
    for epoch in range(5):
        print(f"Epoch {epoch + 1}")
        train_adversarial_with_vis(
            generator, discriminator,
            dataloader,
            g_optimizer, d_optimizer,
            device=device
        )
        print(f"Finished epoch {epoch + 1}")