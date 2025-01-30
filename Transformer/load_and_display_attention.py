import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from ctapipe.instrument import CameraGeometry

# ----------------------------
# 1) DATASET
# ----------------------------
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
        x_m1 = torch.tensor(row["image_m1"][:1039] - row["clean_image_m1"][:1039], dtype=torch.float32)
        x_m2 = torch.tensor(row["image_m2"][:1039] - row["clean_image_m2"][:1039], dtype=torch.float32)
        y = torch.tensor(row["label"], dtype=torch.long)
        return x_m1, x_m2, y

# ----------------------------
# 2) PATCH + POSITIONAL ENCODING
# ----------------------------
class PatchEmbedding(nn.Module):
    def __init__(self, in_dim=1039, patch_size=8, emb_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size, emb_dim)

    def forward(self, x):
        # x: [batch, 1039]
        length = x.shape[1]
        mod = length % self.patch_size
        if mod != 0:
            pad_length = self.patch_size - mod
            x = nn.functional.pad(x, (0, pad_length), mode='constant', value=0)
        x = x.view(x.shape[0], -1, self.patch_size)  # [batch, #patches, patch_size]
        return self.proj(x)                          # [batch, #patches, emb_dim]

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

# ----------------------------
# 3) CUSTOM ENCODER LAYER (to extract attention)
# ----------------------------
class TransformerEncoderLayerWithAttn(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # IMPORTANT: 'average_attn_weights=False' to keep per-head
        src2, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )
        # Standard post-attn steps
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # FFN
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights  # shape: [batch, n_heads, seq_len, seq_len]

# ----------------------------
# 4) SHAPE TRANSFORMER (single-branch), returning optional attention
# ----------------------------
class ShapeTransformer(nn.Module):
    def __init__(self, emb_dim=128, n_heads=8, ff_dim=256, n_layers=4, max_len=2000):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_dim=1039, patch_size=64, emb_dim=emb_dim)
        self.pos_encoder = PositionalEncoding(emb_dim, max_len)
        # Build a stack of custom layers so we can retrieve attn
        layers = []
        for _ in range(n_layers):
            layer = TransformerEncoderLayerWithAttn(
                d_model=emb_dim,
                nhead=n_heads,
                dim_feedforward=ff_dim,
                dropout=0.1,
                batch_first=True
            )
            layers.append(layer)
        self.layers = nn.ModuleList(layers)
        # Final MLP or identity
        self.final_linear = nn.Identity()

    def forward(self, x, return_attention=False):
        x = self.patch_embedding(x)
        x = self.pos_encoder(x)

        all_attn = []
        for layer in self.layers:
            x, attn = layer(x)
            if return_attention:
                all_attn.append(attn)  # shape [batch_size, n_heads, seq_len, seq_len]
        x = x.mean(dim=1)
        if return_attention:
            return self.final_linear(x), all_attn
        else:
            return self.final_linear(x)

# ----------------------------
# 5) COMBINED TRANSFORMER (two-branch)
# ----------------------------
class CombinedTransformer(nn.Module):
    def __init__(self, emb_dim=512, n_heads=8, ff_dim=1024, n_layers=4, n_classes=2):
        super().__init__()
        self.transformer_m1 = ShapeTransformer(
            emb_dim=emb_dim, n_heads=n_heads, ff_dim=ff_dim, n_layers=n_layers
        )
        self.transformer_m2 = ShapeTransformer(
            emb_dim=emb_dim, n_heads=n_heads, ff_dim=ff_dim, n_layers=n_layers
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(emb_dim, n_classes)
        )

    def forward(self, x_m1, x_m2, return_attention=False):
        if return_attention:
            out_m1, attn_m1 = self.transformer_m1(x_m1, return_attention=True)
            out_m2, attn_m2 = self.transformer_m2(x_m2, return_attention=True)
        else:
            out_m1 = self.transformer_m1(x_m1)
            out_m2 = self.transformer_m2(x_m2)
            attn_m1 = attn_m2 = None

        combined = torch.cat([out_m1, out_m2], dim=1)
        logits = self.classifier(combined)
        if return_attention:
            return logits, attn_m1, attn_m2
        return logits

# ----------------------------
# 6) TRAIN & EVAL UTILS
# ----------------------------
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
            print(f"Batch {counter} - Loss: {loss.item():.4f} - Acc: {correct/total:.4f}")
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
            if count > 100:
                break
            x_m1, x_m2, y = dataset[i]
            x_m1, x_m2 = x_m1.unsqueeze(0).to(device), x_m2.unsqueeze(0).to(device)
            pred = model(x_m1, x_m2).argmax(dim=1).item()
            if pred != y.item():
                print(f"Sample {i} - Actual: {y.item()}, Predicted: {pred}")
            count += 1

# ----------------------------
# 7A) PLOT SINGLE BATCH ATTENTION
# ----------------------------
def plot_attention_heads(attn_tensor, layer_idx=0, batch_idx=0):
    """
    attn_tensor: list of [batch_size, n_heads, seq_len, seq_len]
    layer_idx: which of the transformer's layers to look at
    batch_idx: which sample in the batch to visualize
    """
    attn_layer = attn_tensor[layer_idx][batch_idx]  # shape [n_heads, seq_len, seq_len]
    n_heads = attn_layer.size(0)
    fig, axes = plt.subplots(1, n_heads, figsize=(3*n_heads, 3))
    if n_heads == 1:
        axes = [axes]
    for head in range(n_heads):
        axes[head].imshow(attn_layer[head].cpu(), cmap='viridis')
        axes[head].set_title(f"Head {head}")
        axes[head].axis('off')
    plt.show()

# ----------------------------
# 7B) AVERAGE ATTENTION
# ----------------------------
def compute_average_attention(model, val_loader, device):
    """
    Returns two lists: attn_means_m1 and attn_means_m2,
    each a list of [n_heads, seq_len, seq_len] for each layer,
    containing the average attention over the entire val set.
    """
    model.eval()
    num_layers = len(model.transformer_m1.layers)  # number of layers in each shape transformer
    attn_sums_m1 = [0] * num_layers
    attn_sums_m2 = [0] * num_layers
    count = 0

    with torch.no_grad():
        for x_m1, x_m2, y in val_loader:
            x_m1, x_m2 = x_m1.to(device), x_m2.to(device)
            _, attn_m1, attn_m2 = model(x_m1, x_m2, return_attention=True)
            batch_size = x_m1.size(0)
            for i in range(num_layers):
                # attn_m1[i] -> [batch_size, n_heads, seq_len, seq_len]
                attn_sums_m1[i] += attn_m1[i].sum(dim=0)  # sum over batch => [n_heads, seq_len, seq_len]
                attn_sums_m2[i] += attn_m2[i].sum(dim=0)
            count += batch_size

    # divide sums by total # samples => average
    attn_means_m1 = [s / count for s in attn_sums_m1]  # each => [n_heads, seq_len, seq_len]
    attn_means_m2 = [s / count for s in attn_sums_m2]
    return attn_means_m1, attn_means_m2

def plot_avg_attention(attn_map):
    """
    attn_map: [n_heads, seq_len, seq_len] for a single layer
    Plots each head side by side
    """
    n_heads = attn_map.size(0)
    fig, axes = plt.subplots(1, n_heads, figsize=(3*n_heads, 3))
    if n_heads == 1:
        axes = [axes]
    for head in range(n_heads):
        axes[head].imshow(attn_map[head].cpu(), cmap='viridis')
        axes[head].set_title(f"Head {head}")
        axes[head].axis('off')
    plt.show()

# ----------------------------
# 8) IMAGE PLOTTING EXAMPLE
# ----------------------------
geom = CameraGeometry.from_name("MAGICCam")
pix_x = geom.pix_x.value
pix_y = geom.pix_y.value

def plot_camera_images(row):
    """
    row: a DataFrame row with columns ["image_m1","clean_image_m1","image_m2","clean_image_m2"]
         each containing array-like of length >=1039
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    img_m1 = row["image_m1"][:1039] - row["clean_image_m1"][:1039]
    sc1 = ax1.scatter(pix_x, pix_y, c=img_m1, cmap='plasma', s=50)
    ax1.set_title("M1 Camera")
    fig.colorbar(sc1, ax=ax1, label="Intensity")

    img_m2 = row["image_m2"][:1039] - row["clean_image_m2"][:1039]
    sc2 = ax2.scatter(pix_x, pix_y, c=img_m2, cmap='plasma', s=50)
    ax2.set_title("M2 Camera")
    fig.colorbar(sc2, ax=ax2, label="Intensity")

    plt.tight_layout()
    plt.show()

# ----------------------------
# 9) MAIN
# ----------------------------
if __name__ == "__main__":
    BATCH_SIZE = 128
    LR = 1e-5
    EPOCHS = 1
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    print("Using device:", device)

    gamma_file = "../magic-gammas.parquet"
    proton_file = "../magic-protons.parquet"

    # Build dataset
    dataset = MagicDataset(gamma_parquet=gamma_file, proton_parquet=proton_file)
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Build model
    model = CombinedTransformer(
        emb_dim=512,
        n_heads=1,  # put 1 head if you want to see a single attention map clearly
        ff_dim=1024,
        n_layers=2, # smaller number for quick demo
        n_classes=2
    ).to(device)

    # Optionally train if no checkpoint found
    if not os.path.exists("best_model_dual.pt"):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=LR)
        best_val_loss = float('inf')
        for epoch in range(EPOCHS):
            print(f"Epoch {epoch+1}")
            train_loss = train_model(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
            print(f"Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Val Acc={val_acc:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "best_model_dual.pt")
        print("Training complete. Model saved to best_model_dual.pt")
    else:
        print("Loading from checkpoint best_model_dual.pt")
        model.load_state_dict(torch.load("best_model_dual.pt", map_location=device))

    # --- 1) Compute average attention across entire val set ---
    attn_means_m1, attn_means_m2 = compute_average_attention(model, val_loader, device)
    # Plot average attention of layer 0 for M1
    print("Plotting average attention (layer 0, M1):")
    plot_avg_attention(attn_means_m1[0])  # shape [n_heads, seq_len, seq_len]

    # --- 2) Single example: show its camera images & single attention map
    # pick some index in val_dataset
    single_idx = 0
    x_m1, x_m2, y = val_dataset[single_idx]
    # move to device, add batch dimension
    x_m1_b = x_m1.unsqueeze(0).to(device)
    x_m2_b = x_m2.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits, attn_m1, attn_m2 = model(x_m1_b, x_m2_b, return_attention=True)
        predicted = torch.argmax(logits, dim=1).item()

    # Print label vs predicted
    print(f"Example idx={single_idx}: true label={y}, predicted={predicted}")

    # Plot camera images for that single sample
    # val_dataset.indices[single_idx] => original row index in the full dataset
    real_idx = val_dataset.indices[single_idx]
    row = dataset.df.iloc[real_idx]
    plot_camera_images(row)

    # Plot attention for layer 0, batch_idx=0, M1
    print("Plotting single-sample attention (M1, layer 0, head 0):")
    plot_attention_heads(attn_m1, layer_idx=0, batch_idx=0)
