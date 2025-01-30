import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# -----------------------------------------------------
# 1) DATASET - Return (x_m1, x_m2), label
# -----------------------------------------------------
class MagicDataset(Dataset):
    def __init__(self, gamma_parquet, proton_parquet, transform=None):
        df_gamma = pd.read_parquet(gamma_parquet)
        df_gamma["label"] = 0

        df_proton = pd.read_parquet(proton_parquet)
        df_proton["label"] = 1

        self.df = pd.concat([df_gamma, df_proton], ignore_index=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x_m1 = torch.tensor(row["image_m1"][:1039], dtype=torch.float32)
        x_m2 = torch.tensor(row["image_m2"][:1039], dtype=torch.float32)
        y = torch.tensor(row["label"], dtype=torch.long)
        return x_m1, x_m2, y

# -----------------------------------------------------
# 2) PATCH + POSITIONAL ENCODING (unchanged)
# -----------------------------------------------------
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

# -----------------------------------------------------
# 3) SHAPE TRANSFORMER (single-branch)
# -----------------------------------------------------
class ShapeTransformer(nn.Module):
    def __init__(self, emb_dim=128, n_heads=8, ff_dim=256, n_layers=4, n_classes=2, max_len=2000):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_dim=1039, patch_size=8, emb_dim=emb_dim)
        self.pos_encoder = PositionalEncoding(emb_dim, max_len)
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

# -----------------------------------------------------
# 4) COMBINED TRANSFORMER - merges two branches
# -----------------------------------------------------
class CombinedTransformer(nn.Module):
    def __init__(self, emb_dim=512, n_heads=8, ff_dim=1024, n_layers=4, n_classes=2):
        super().__init__()
        self.transformer_m1 = ShapeTransformer(
            emb_dim=emb_dim,
            n_heads=n_heads,
            ff_dim=ff_dim,
            n_layers=n_layers,
            n_classes=n_classes
        )
        self.transformer_m2 = ShapeTransformer(
            emb_dim=emb_dim,
            n_heads=n_heads,
            ff_dim=ff_dim,
            n_layers=n_layers,
            n_classes=n_classes
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(emb_dim, n_classes)
        )

    def forward(self, x_m1, x_m2):
        out_m1 = self.transformer_m1(x_m1)
        out_m2 = self.transformer_m2(x_m2)
        combined = torch.cat([out_m1, out_m2], dim=1)
        return self.classifier(combined)

# -----------------------------------------------------
# 5) EVALUATION UTILS
# -----------------------------------------------------
def evaluate(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    counter = 0
    with torch.no_grad():
        for x_m1, x_m2, y in dataloader:
            x_m1, x_m2, y = x_m1.to(device), x_m2.to(device), y.to(device)
            out = model(x_m1, x_m2)
            preds = torch.argmax(out, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            counter += 1
            if counter % 100 == 0:
                print(f"correct: {correct}, total: {total}")
    return correct / total

# -----------------------------------------------------
# 6) MAIN
# -----------------------------------------------------
if __name__ == "__main__":
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )

    # Update with your paths:
    gamma_file = "../magic-gammas_part2.parquet"
    proton_file = "../magic-protons_part2.parquet"
    model_path = "best_model_dual.pt" # or "best_model_dual_fin.pt"

    # Load dataset and model
    full_dataset = MagicDataset(gamma_file, proton_file)
    full_loader = DataLoader(full_dataset, batch_size=256, shuffle=True)

    model = CombinedTransformer().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Evaluate
    accuracy = evaluate(model, full_loader, device)
    print(f"Accuracy on entire dataset: {accuracy:.4f}")