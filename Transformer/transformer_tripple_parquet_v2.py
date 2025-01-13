# Potential reasons why adding numeric features might degrade performance:
# 1) Many NaNs / incorrect default fill values.
# 2) Very different value ranges -> Transformers or feed-forward might struggle if not normalized.
# 3) Possibly some numeric features are noisy or uncorrelated.

# Common fixes:
# - Fill or drop NaNs carefully.
# - Normalize/scale numeric features.
# - Use simpler or fewer numeric features. Too many can increase noise.

# Example: Add normalization. (We do a quick mean/std over entire dataset, then standardize.)

import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

from Transformer.MaxiDataset import MaxiDataset

NUMERIC_COLS = [
    #"event_number",
    #"run_number",
    "true_energy",
    "true_theta",
    "true_phi",
    "true_telescope_theta",
    "true_telescope_phi",
    "true_first_interaction_height",
    "hillas_length_m1",
    "hillas_width_m1",
    "hillas_delta_m1",
    "hillas_size_m1",
    "hillas_cog_x_m1",
    "hillas_cog_y_m1",
    "hillas_sin_delta_m1",
    "hillas_cos_delta_m1",
    "hillas_length_m2",
    "hillas_width_m2",
    "hillas_delta_m2",
    "hillas_size_m2",
    "hillas_cog_x_m2",
    "hillas_cog_y_m2",
    "hillas_sin_delta_m2",
    "hillas_cos_delta_m2",
    "stereo_direction_x",
    "stereo_direction_y",
    "stereo_zenith",
    "stereo_azimuth",
    "stereo_dec",
    "stereo_ra",
    "stereo_theta2",
    "stereo_core_x",
    "stereo_core_y",
    "stereo_impact_m1",
    "stereo_impact_m2",
    "stereo_impact_azimuth_m1",
    "stereo_impact_azimuth_m2",
    "stereo_shower_max_height",
    "stereo_xmax",
    "stereo_cherenkov_radius",
    "stereo_cherenkov_density",
    "stereo_baseline_phi_m1",
    "stereo_baseline_phi_m2",
    "stereo_image_angle",
    "stereo_cos_between_shower",
    "pointing_zenith",
    "pointing_azimuth",
    "time_gradient_m1",
    "time_gradient_m2",
    "true_impact_m1",
    "true_impact_m2",
    "source_alpha_m1",
    "source_dist_m1",
    "source_cos_delta_alpha_m1",
    "source_dca_m1",
    "source_dca_delta_m1",
    "source_alpha_m2",
    "source_dist_m2",
    "source_cos_delta_alpha_m2",
    "source_dca_m2",
    "source_dca_delta_m2",
]

# -----------------------------------------------
# 1) Compute mean/stdev over the entire dataset.
#    We'll use a simple helper function.
# -----------------------------------------------
def compute_numeric_stats(df):
    """Compute mean/stdev for numeric columns. Return two dicts: mean_dict, std_dict."""
    mean_dict = {}
    std_dict = {}
    for col in NUMERIC_COLS:
        col_vals = pd.to_numeric(df[col], errors='coerce')
        mean_dict[col] = col_vals.mean(skipna=True)
        std_dict[col] = col_vals.std(skipna=True)
    return mean_dict, std_dict

class MagicDataset(Dataset):
    def __init__(self, gamma_parquet, proton_parquet, mean_dict=None, std_dict=None):
        df_gamma = pd.read_parquet(gamma_parquet)
        df_gamma["label"] = 0
        df_proton = pd.read_parquet(proton_parquet)
        df_proton["label"] = 1
        self.df = pd.concat([df_gamma, df_proton], ignore_index=True)

        # If no stats provided, compute from entire dataset
        if mean_dict is None or std_dict is None:
            self.mean_dict, self.std_dict = compute_numeric_stats(self.df)
        else:
            self.mean_dict, self.std_dict = mean_dict, std_dict

        # Clean up NaNs
        self.df = self.df.fillna(0.0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Image data
        x_m1 = torch.tensor(row["image_m1"][:1039], dtype=torch.float32)
        x_m2 = torch.tensor(row["image_m2"][:1039], dtype=torch.float32)

        # Numeric data -> standardize
        numeric_values = []
        for col in NUMERIC_COLS:
            val = row[col]
            mu = self.mean_dict[col] if self.std_dict[col] == self.std_dict[col] else 0.0
            sigma = self.std_dict[col] if self.std_dict[col] != 0.0 else 1.0
            # Replace NaN with 0
            if pd.isna(val):
                val = 0.0
            numeric_values.append((val - mu) / sigma)
        x_num = torch.tensor(numeric_values, dtype=torch.float32)

        # Label
        y = torch.tensor(row["label"], dtype=torch.long)

        return (x_m1, x_m2, x_num), y

# ----------------------------------------
# Transformers for image branches
# ----------------------------------------
class PatchEmbedding(nn.Module):
    def __init__(self, in_dim=1039, patch_size=8, emb_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size, emb_dim)

    def forward(self, x):
        length = x.shape[1]
        mod = length % self.patch_size
        if mod != 0:
            pad_length = self.patch_size - mod
            x = nn.functional.pad(x, (0, pad_length), mode='constant', value=0)
        x = x.view(x.shape[0], -1, self.patch_size)
        return self.proj(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :]

class ShapeTransformer(nn.Module):
    def __init__(self, emb_dim=128, n_heads=8, ff_dim=256, n_layers=4):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_dim=1039, patch_size=8, emb_dim=emb_dim)
        self.pos_encoder = PositionalEncoding(emb_dim, max_len=2000)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.final_linear = nn.Identity()

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.final_linear(x)

# ----------------------------------------
# Small MLP for numeric data
# ----------------------------------------
class NumericBranch(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

# ----------------------------------------
# Combine 2 transformers + numeric MLP
# ----------------------------------------
class CombinedModel(nn.Module):
    def __init__(self,
                 emb_dim=512,
                 n_heads=8,
                 ff_dim=1024,
                 n_layers=4,
                 num_features=len(NUMERIC_COLS),
                 hidden_num_branch=256,
                 n_classes=2):
        super().__init__()
        self.transformer_m1 = ShapeTransformer(emb_dim=emb_dim,
                                               n_heads=n_heads,
                                               ff_dim=ff_dim,
                                               n_layers=n_layers)
        self.transformer_m2 = ShapeTransformer(emb_dim=emb_dim,
                                               n_heads=n_heads,
                                               ff_dim=ff_dim,
                                               n_layers=n_layers)
        self.numeric_branch = NumericBranch(input_dim=num_features,
                                            hidden_dim=hidden_num_branch,
                                            out_dim=emb_dim)
        self.classifier = nn.Sequential(
            nn.Linear(3 * emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(emb_dim, n_classes)
        )

    def forward(self, x_m1, x_m2, x_num):
        out_m1 = self.transformer_m1(x_m1)
        out_m2 = self.transformer_m2(x_m2)
        out_num = self.numeric_branch(x_num)
        merged = torch.cat([out_m1, out_m2, out_num], dim=1)
        return self.classifier(merged)

# ----------------------------------------
# Train/Eval
# ----------------------------------------
def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, counter, correct, total = 0, 0, 0, 0
    for (x_m1, x_m2, x_num), y in dataloader:
        x_m1, x_m2, x_num, y = x_m1.to(device), x_m2.to(device), x_num.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x_m1, x_m2, x_num)
        loss = criterion(outputs, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

        if counter % 100 == 0:
            print(f"Batch {counter} - Loss: {loss.item():.4f} - Accuracy: {correct / total:.4f}")
            correct, total = 0, 0
        counter += 1

    return running_loss / len(dataloader)

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for (x_m1, x_m2, x_num), y in dataloader:
            x_m1, x_m2, x_num, y = x_m1.to(device), x_m2.to(device), x_num.to(device), y.to(device)

            outputs = model(x_m1, x_m2, x_num)
            loss = criterion(outputs, y)
            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return running_loss / len(dataloader), correct / total

def report_misclassified(model, dataset, device):
    model.eval()
    count = 0
    with torch.no_grad():
        for i in range(len(dataset)):
            if count >= 300:
                break
            (x_m1, x_m2, x_num), y = dataset[i]
            x_m1, x_m2, x_num = (x_m1.unsqueeze(0).to(device),
                                 x_m2.unsqueeze(0).to(device),
                                 x_num.unsqueeze(0).to(device))
            pred = model(x_m1, x_m2, x_num).argmax(dim=1).item()
            if pred != y.item():
                print(f"Sample {i} - Actual: {y.item()}, Predicted: {pred}")
            count += 1

# ----------------------------------------
# Main
# ----------------------------------------
if __name__ == "__main__":
    BATCH_SIZE = 128
    LR = 1e-5
    EPOCHS = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)
    gamma_file = "../magic-gammas_part1.parquet"
    proton_file = "../magic-protons_part1.parquet"

    # If you want to pre-compute stats over entire data:
    tmp_df_g = pd.read_parquet(gamma_file)
    tmp_df_p = pd.read_parquet(proton_file)
    full_df = pd.concat([tmp_df_g, tmp_df_p], ignore_index=True)
    mean_dict, std_dict = compute_numeric_stats(full_df)

    #dataset = MagicDataset(gamma_file, proton_file, mean_dict, std_dict)
    dataset = MaxiDataset(gamma_filename=gamma_file, proton_filename=proton_file, max_samples=10000)

    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = CombinedModel(
        emb_dim=512,
        n_heads=8,
        ff_dim=1024,
        n_layers=4,
        num_features=len(NUMERIC_COLS),
        hidden_num_branch=256,
        n_classes=2
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}")
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1} - "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model_numeric.pt")

    torch.save(model.state_dict(), "trained_combined_numeric.pt")
    print("Model saved to trained_combined_numeric.pt")

    report_misclassified(model, val_dataset, device)
