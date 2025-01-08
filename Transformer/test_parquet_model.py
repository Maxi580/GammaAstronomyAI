import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class MagicDataset(Dataset):
    def __init__(self, gamma_parquet, proton_parquet):
        df_gamma = pd.read_parquet(gamma_parquet)
        df_gamma["label"] = 0
        df_proton = pd.read_parquet(proton_parquet)
        df_proton["label"] = 1
        self.df = pd.concat([df_gamma, df_proton], ignore_index=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = torch.tensor(row["image_m1"], dtype=torch.float32)
        y = torch.tensor(row["label"], dtype=torch.long)
        return x, y

class PatchEmbedding(nn.Module):
    def __init__(self, in_dim=1039, patch_size=8, emb_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size, emb_dim)

    def forward(self, x):
        length = x.shape[1]
        mod = length % self.patch_size
        if mod != 0:
            x = nn.functional.pad(x, (0, self.patch_size - mod), value=0)
        x = x.view(x.size(0), -1, self.patch_size)
        return self.proj(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :]

class ShapeTransformer(nn.Module):
    def __init__(self, emb_dim=256, n_heads=8, ff_dim=512, n_layers=4, n_classes=2, max_len=2000):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_dim=1039, patch_size=8, emb_dim=emb_dim)
        self.pos_encoder = PositionalEncoding(emb_dim, max_len)
        layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=n_heads, dim_feedforward=ff_dim, batch_first=True)
        # Use the same name as in training:
        self.transformer_encoder = nn.TransformerEncoder(layer, n_layers)
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(emb_dim, n_classes)
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return self.classifier(x.mean(dim=1))


def evaluate(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    counter = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = torch.argmax(out, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            counter += 1
            if counter % 100 == 0:
                print(f"correct: {correct}, total: {total}")
    return correct / total

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Update with your paths:
    gamma_file = "../magic-gammas.parquet"
    proton_file = "../magic-protons.parquet"
    model_path = "best_model_real_data_1.pt"

    # Load dataset and model
    full_dataset = MagicDataset(gamma_file, proton_file)
    full_loader = DataLoader(full_dataset, batch_size=256, shuffle=False)

    model = ShapeTransformer().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Evaluate
    accuracy = evaluate(model, full_loader, device)
    print(f"Accuracy on entire dataset: {accuracy:.4f}")
