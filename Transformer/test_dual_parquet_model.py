import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# -----------------------------------------------------
# 1) DATASET - Return (x_m1, x_m2), label
# -----------------------------------------------------
class MagicDataset(Dataset):
    def __init__(self, combined_parquet, transform=None):
        # Read combined file
        self.df = pd.read_parquet(combined_parquet)
        # No labels needed for this task

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x_m1 = torch.tensor(row["image_m1"][:1039], dtype=torch.float32)
        x_m2 = torch.tensor(row["image_m2"][:1039], dtype=torch.float32)

        return x_m1, x_m2

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
        # Final MLP for this single branch. We only output an embedding here,
        # so let's keep it minimal: e.g. the final feature we want is x.mean(dim=1).
        self.final_linear = nn.Identity()  # We'll extract the mean and pass it on.

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # final embedding
        return self.final_linear(x)

# -----------------------------------------------------
# 4) COMBINED TRANSFORMER - merges two branches
# -----------------------------------------------------
class CombinedTransformer(nn.Module):
    def __init__(self, emb_dim=512, n_heads=8, ff_dim=1024, n_layers=4, n_classes=2):
        super().__init__()
        # Two separate shape transformers
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
        # Merge embeddings from both branches -> final classification
        self.classifier = nn.Sequential(
            nn.Linear(2 * emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(emb_dim, n_classes)
        )

    def forward(self, x_m1, x_m2):
        out_m1 = self.transformer_m1(x_m1)  # shape [batch, emb_dim]
        out_m2 = self.transformer_m2(x_m2)  # shape [batch, emb_dim]
        combined = torch.cat([out_m1, out_m2], dim=1)
        return self.classifier(combined)

# -----------------------------------------------------
# 5) INFERENCE
# -----------------------------------------------------
def perform_inference(model, dataloader, device, output_file):
    model.eval()
    with torch.no_grad(), open(output_file, "w") as f:
        for x_m1, x_m2 in dataloader:
            x_m1, x_m2 = x_m1.to(device), x_m2.to(device)
            outputs = model(x_m1, x_m2)
            preds = torch.argmax(outputs, dim=1)
            for pred in preds:
                f.write(f"{pred.item()}\n")

# -----------------------------------------------------
# 6) MAIN
# -----------------------------------------------------
if __name__ == "__main__":
    BATCH_SIZE = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using: ", device)

    # Files
    combined_file = "../combined_file_reverse.parquet" # Replace with combined file
    output_file = "output.txt"

    # Dataset
    dataset = MagicDataset(combined_parquet=combined_file)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    model = CombinedTransformer(
        emb_dim=512,
        n_heads=8,
        ff_dim=1024,
        n_layers=4,
        n_classes=2
    ).to(device)

    # Load state dict
    model.load_state_dict(torch.load("best_model_dual.pt"))

    # Inference
    perform_inference(model, dataloader, device, output_file)
    print(f"Predictions written to {output_file}")