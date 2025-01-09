import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader

NUMERIC_COLS = [
    "true_energy", "true_theta", "true_phi", "true_telescope_theta",
    "true_telescope_phi", "true_first_interaction_height", "hillas_length_m1",
    "hillas_width_m1", "hillas_delta_m1", "hillas_size_m1", "hillas_cog_x_m1",
    "hillas_cog_y_m1", "hillas_sin_delta_m1", "hillas_cos_delta_m1",
    "hillas_length_m2", "hillas_width_m2", "hillas_delta_m2", "hillas_size_m2",
    "hillas_cog_x_m2", "hillas_cog_y_m2", "hillas_sin_delta_m2",
    "hillas_cos_delta_m2", "stereo_direction_x", "stereo_direction_y",
    "stereo_zenith", "stereo_azimuth", "stereo_dec", "stereo_ra",
    "stereo_theta2", "stereo_core_x", "stereo_core_y", "stereo_impact_m1",
    "stereo_impact_m2", "stereo_impact_azimuth_m1", "stereo_impact_azimuth_m2",
    "stereo_shower_max_height", "stereo_xmax", "stereo_cherenkov_radius",
    "stereo_cherenkov_density", "stereo_baseline_phi_m1",
    "stereo_baseline_phi_m2", "stereo_image_angle", "stereo_cos_between_shower",
    "pointing_zenith", "pointing_azimuth", "time_gradient_m1",
    "time_gradient_m2", "true_impact_m1", "true_impact_m2", "source_alpha_m1",
    "source_dist_m1", "source_cos_delta_alpha_m1", "source_dca_m1",
    "source_dca_delta_m1", "source_alpha_m2", "source_dist_m2",
    "source_cos_delta_alpha_m2", "source_dca_m2", "source_dca_delta_m2"
]


def compute_numeric_stats(df):
    mean_dict, std_dict = {}, {}
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

        if mean_dict is None or std_dict is None:
            self.mean_dict, self.std_dict = compute_numeric_stats(self.df)
        else:
            self.mean_dict, self.std_dict = mean_dict, std_dict

        self.df = self.df.fillna(0.0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x_m1 = torch.tensor(row["image_m1"][:1039], dtype=torch.float32)
        x_m2 = torch.tensor(row["image_m2"][:1039], dtype=torch.float32)

        numeric_values = []
        for col in NUMERIC_COLS:
            val = row[col]
            mu = self.mean_dict[col] if self.std_dict[col] == self.std_dict[col] else 0.0
            sigma = self.std_dict[col] if self.std_dict[col] != 0.0 else 1.0
            if pd.isna(val):
                val = 0.0
            numeric_values.append((val - mu) / sigma)
        x_num = torch.tensor(numeric_values, dtype=torch.float32)

        y = torch.tensor(row["label"], dtype=torch.long)
        return (x_m1, x_m2, x_num), y


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


class CombinedModel(nn.Module):
    def __init__(self, emb_dim=512, n_heads=8, ff_dim=1024, n_layers=4,
                 num_features=len(NUMERIC_COLS), hidden_num_branch=256, n_classes=2):
        super().__init__()
        self.transformer_m1 = ShapeTransformer(emb_dim=emb_dim, n_heads=n_heads,
                                               ff_dim=ff_dim, n_layers=n_layers)
        self.transformer_m2 = ShapeTransformer(emb_dim=emb_dim, n_heads=n_heads,
                                               ff_dim=ff_dim, n_layers=n_layers)
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


def evaluate_model(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for (x_m1, x_m2, x_num), y in dataloader:
            x_m1 = x_m1.to(device)
            x_m2 = x_m2.to(device)
            x_num = x_num.to(device)
            y = y.to(device)

            outputs = model(x_m1, x_m2, x_num)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            if total % 1000 == 0:
                print(f"Processed {total} samples - Current accuracy: {correct / total:.4f}")

    return correct / total


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Update these paths
    gamma_file = "../magic-gammas_part2.parquet"
    proton_file = "../magic-protons_part2.parquet"
    model_path = "best_model_numeric.pt"

    # Load the test dataset
    print("Loading dataset...")
    dataset = MagicDataset(gamma_file, proton_file)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

    # Initialize and load the model
    print("Loading model...")
    model = CombinedModel(
        emb_dim=512,
        n_heads=8,
        ff_dim=1024,
        n_layers=4,
        num_features=len(NUMERIC_COLS),
        hidden_num_branch=256,
        n_classes=2
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))

    # Evaluate
    print("Starting evaluation...")
    accuracy = evaluate_model(model, dataloader, device)
    print(f"\nFinal accuracy on test set: {accuracy:.4f}")