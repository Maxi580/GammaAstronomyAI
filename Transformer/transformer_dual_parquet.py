import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split

from Transformer.MaxiDataset import MaxiDataset


# -----------------------------------------------------
# 1) DATASET - Return (x_m1, x_m2), label
# -----------------------------------------------------
class MagicDataset(Dataset):
    def __init__(self, gamma_parquet, proton_parquet, transform=None):
        # Read gammas
        df_gamma = pd.read_parquet(gamma_parquet)
        df_gamma["label"] = 0  # label gammas as 0

        # Read protons
        df_proton = pd.read_parquet(proton_parquet)
        df_proton["label"] = 1  # label protons as 1

        # Combine
        self.df = pd.concat([df_gamma, df_proton], ignore_index=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x_m1 = torch.tensor(row["image_m1"][:1039] - row["clean_image_m1"][:1039], dtype=torch.float32)
        x_m2 = torch.tensor(row["image_m2"][:1039] - row["clean_image_m2"][:1039], dtype=torch.float32)
        y = torch.tensor(row["label"], dtype=torch.long)
        #x_m1[0] = y
        #x_m2[0] = y
        #if self.transform:
        #    x_m1 = self.transform(x_m1)
        #    x_m2 = self.transform(x_m2)
        return x_m1, x_m2, y


# -----------------------------------------------------
# 2) PATCH + POSITIONAL ENCODING (unchanged)
# -----------------------------------------------------
class PatchEmbedding(nn.Module):
    def __init__(self, in_dim=1039, patch_size=1, emb_dim=128):
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
        self.patch_embedding = PatchEmbedding(in_dim=1039, patch_size=1, emb_dim=emb_dim)
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
# 5) TRAIN/EVAL UTILS (mostly unchanged)
# -----------------------------------------------------
def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, counter, correct, total = 0, 0, 0, 0
    for x_m1, x_m2, y in dataloader:
        x_m1, x_m2, y = x_m1.to(device), x_m2.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x_m1, x_m2)
        loss = criterion(outputs, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

        if counter % 100 == 0:
            print(f"Batch {counter} - Loss: {loss.item()} - Accuracy: {correct / total:.4f}")
            correct, total = 0, 0
        counter += 1
    return running_loss / len(dataloader)

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for x_m1, x_m2, y in dataloader:
            x_m1, x_m2, y = x_m1.to(device), x_m2.to(device), y.to(device)
            outputs = model(x_m1, x_m2)
            loss = criterion(outputs, y)
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return running_loss / len(dataloader), correct / total

def report_misclassified(model, dataset, device):
    model.eval()
    with torch.no_grad():
        count = 0
        for i in range(len(dataset)):
            if count > 300:
                break
            x_m1, x_m2, y = dataset[i]
            x_m1, x_m2 = x_m1.unsqueeze(0).to(device), x_m2.unsqueeze(0).to(device)
            pred = model(x_m1, x_m2).argmax(dim=1).item()
            if pred != y.item():
                print(f"Sample {i} - Actual: {y.item()}, Predicted: {pred}")
            count += 1


# -----------------------------------------------------
# 6) MAIN
# -----------------------------------------------------
if __name__ == "__main__":
    BATCH_SIZE = 64
    LR = 1e-4
    EPOCHS = 2
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )
    print("Using device:", device)
    # Files
    gamma_file = "../magic-gammas_small_part1.parquet"
    proton_file = "../magic-protons_small_part1.parquet"

    # Dataset + split
    #dataset = MaxiDataset(gamma_filename=gamma_file, proton_filename=proton_file)
    dataset = MagicDataset(gamma_parquet=gamma_file, proton_parquet=proton_file)
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Two-branch model
    model = CombinedTransformer(
        emb_dim=32,
        n_heads=1,
        ff_dim=1, #128
        n_layers=1,
        n_classes=2
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        print("Epoch", epoch + 1)
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model_dual.pt")

    torch.save(model.state_dict(), "best_model_dual_fin.pt")
    print("Model saved to trained_shape_transformer_50k.pt")

    report_misclassified(model, val_dataset, device)
