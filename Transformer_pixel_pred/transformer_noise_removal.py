import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F

from Transformer_pixel_pred.reconstruct_from_array import main_reconstruct


# --------- Dataset that loads X=combined, Y=pixel_array (0/1 for each of 1039 pixels) ---------
class ShapePixelDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.json')])
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data_file = self.data_files[idx]
        with open(os.path.join(self.data_dir, data_file), 'r') as f:
            data_json = json.load(f)
        x = torch.tensor(data_json["combined"], dtype=torch.float32)
        y = torch.tensor(data_json["pixel_array"], dtype=torch.float32)  # shape mask
        # y = (y != 0).float()

        if self.transform:
            x = self.transform(x)
        return x, y

# --------- Patch Embedding (same idea, but final output will be 1039 dims) ---------
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=8, emb_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size, emb_dim)

    def forward(self, x):
        # x: [batch_size, seq_len]
        seq_len = x.shape[1]
        pad_len = self.patch_size - (seq_len % self.patch_size) if seq_len % self.patch_size != 0 else 0
        x = F.pad(x, (0, pad_len), mode='constant', value=0)
        x = x.view(x.shape[0], -1, self.patch_size)
        return self.proj(x)  # [batch_size, new_seq_len, emb_dim]

# --------- Positional Encoding ---------
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
        return x + self.pe[: x.size(1)]

# --------- Transformer predicting each pixel -> 1039 outputs ---------
class PixelTransformer(nn.Module):
    def __init__(self, patch_size=8, emb_dim=128,
                 n_heads=8, ff_dim=256, n_layers=4):
        super().__init__()
        self.patch_embedding = PatchEmbedding(patch_size=patch_size, emb_dim=emb_dim)
        self.pos_encoder = PositionalEncoding(emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=0.1, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(emb_dim, patch_size)

    def forward(self, x):
        # x: [batch_size, 1039]
        x = self.patch_embedding(x)  # -> [B, new_seq_len, emb_dim]
        x = self.pos_encoder(x)  # -> [B, new_seq_len, emb_dim]
        x = self.transformer_encoder(x)  # -> [B, new_seq_len, emb_dim]
        x = self.fc_out(x)  # -> [B, new_seq_len, patch_size]
        x = x.view(x.size(0), -1)  # -> [B, new_seq_len*patch_size]

        # slice or pad to exactly 1039
        if x.size(1) > 1039:
            x = x[:, :1039]
        elif x.size(1) < 1039:
            diff = 1039 - x.size(1)
            x = F.pad(x, (0, diff), 'constant', 0)
        return x


# --------- Train & Evaluate ---------
def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        out = model(X)  # [batch_size, 1039]
        loss = loss_fn(out, Y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            out = model(X)
            loss = loss_fn(out, Y)
            total_loss += loss.item()
    return total_loss / len(loader)

# ----------------- Main Script -------------------
def train(data_dir, batch_size=16):
    dataset = ShapePixelDataset(data_dir=data_dir)

    # Example split
    train_size = int(len(dataset)*0.7)
    val_size   = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PixelTransformer(
        # in_dim=1039,
        patch_size=8,
        emb_dim=128,
        n_heads=8,
        ff_dim=256,
        n_layers=4
    ).to(device)

    # Each pixel is either in shape or not => BCEWithLogitsLoss
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    best_val_loss = float('inf')
    EPOCHS = 10

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss   = eval_epoch(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_pixel_model.pt")

    print("Training complete. Final model saved as best_pixel_model.pt")


def test(model_path, data_dir, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_class = PixelTransformer
    dataset_class = ShapePixelDataset

    model = model_class().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2) Load dataset + dataloader
    test_dataset = dataset_class(data_dir=data_dir)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_thresh_labels = []
    all_raw_labels = []
    all_filenames = []  # New list to store filenames

    with torch.no_grad():
        for batch_idx, (X, Y) in enumerate(test_loader):
            X, Y = X.to(device), Y.to(device)
            logits = model(X)  # [batch_size, 1039]
            preds = (torch.sigmoid(logits) > 0.5).float()
            labels = (Y > 0.5).float()  # threshold the actual labels

            # Get filenames for this batch
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(test_dataset))
            batch_filenames = test_dataset.data_files[start_idx:end_idx]

            all_preds.append(preds.cpu())
            all_thresh_labels.append(labels.cpu())
            all_raw_labels.append(Y.cpu())
            all_filenames.extend(batch_filenames)  # Add filenames for this batch

    # Concatenate all results
    all_preds = torch.cat(all_preds, dim=0)  # [N, 1039]
    all_thresh_labels = torch.cat(all_thresh_labels, dim=0)  # [N, 1039]
    all_raw_labels = torch.cat(all_raw_labels, dim=0)  # [N, 1039]

    # --- 3) Pixel-level Accuracy (thresholded) ---
    correct_pixels = (all_preds == all_thresh_labels).sum().item()
    total_pixels = all_thresh_labels.numel()
    accuracy = correct_pixels / total_pixels
    print(f"Pixel-level Accuracy (thresholded): {accuracy:.4f}\n")

    # --- 4) Identify first 30 samples that have at least one pixel mismatch ---
    mismatches = (all_preds != all_thresh_labels).any(dim=1)
    wrong_idx = mismatches.nonzero(as_tuple=True)[0]

    for i in wrong_idx[:30]:
        filename = all_filenames[i]
        diff_pixels = (all_preds[i] != all_thresh_labels[i]).nonzero(as_tuple=True)[0]
        print(f"Sample index: {i.item()} (File: {filename})")
        print(f"Mismatched pixel indices: {diff_pixels.tolist()}")
        print("Predicted vs Thresholded Label (and raw label) only for mismatched pixels:")
        for pix_id in diff_pixels:
            pred_val = all_preds[i][pix_id].item()
            thresh_val = all_thresh_labels[i][pix_id].item()
            raw_val = all_raw_labels[i][pix_id].item()
            print(f"  Pixel {pix_id.item()} -> pred: {pred_val}, label(thr): {thresh_val}, label(raw): {raw_val}")
        print("----\n")

def create(model_path, data_dir, output_dir, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PixelTransformer().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    output_arrays_dir = os.path.join(output_dir, "arrays")
    os.makedirs(output_arrays_dir, exist_ok=True)

    dataset = ShapePixelDataset(data_dir=data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 4) Generate and save predictions
    with torch.no_grad():
        for batch_idx, (X, Y) in enumerate(dataloader):
            X = X.to(device)
            logits = model(X)
            preds = torch.sigmoid(logits)

            # Get filenames for this batch
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(dataset))
            batch_filenames = dataset.data_files[start_idx:end_idx]

            # 5) Save each prediction as a JSON file
            for i in range(preds.shape[0]):
                filename = os.path.splitext(batch_filenames[i])[0] + ".json"  # Change extension to .json
                output_path = os.path.join(output_arrays_dir, filename)

                # Load the original data to get noise and combined
                original_data_path = os.path.join(data_dir, batch_filenames[i])
                with open(original_data_path, 'r') as f:
                  original_data = json.load(f)

                # Create a dictionary in the desired format
                output_data = {
                    "pixel_array": preds[i].cpu().tolist(),  # Predicted pixel array (probabilities)
                    "noise": original_data["noise"],  # Same as original
                    "combined": original_data["combined"]  # Same as original
                }

                # Save the predicted data as JSON
                with open(output_path, "w") as outfile:
                    json.dump(output_data, outfile, indent=4)

    print(f"Predicted arrays saved to: {output_arrays_dir}")
    main_reconstruct()


if __name__ == "__main__":
    #train(data_dir="../simulated_data_15k_gn/arrays", batch_size=16)
    #test(model_path="best_pixel_model.pt", data_dir="../simulated_data_validation_sn/arrays", batch_size=16)
    create(model_path="best_pixel_model.pt",data_dir="../simulated_data_4k_gn/arrays",output_dir="pixel_results",batch_size=16)