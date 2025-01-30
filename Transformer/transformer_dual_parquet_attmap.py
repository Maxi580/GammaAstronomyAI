import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split

# For geometry-based image plotting:
from ctapipe.instrument import CameraGeometry


# ----------------------------
# CONFIG
# ----------------------------
class Config:
    # Data Config
    GAMMA_FILE = "../magic-gammas.parquet"
    PROTON_FILE = "../magic-protons.parquet"
    BATCH_SIZE = 128
    TRAIN_SPLIT = 0.7

    # Model Architecture
    PATCH_SIZE = 8
    EMB_DIM = 16
    N_HEADS = 1
    FF_DIM = 1
    N_LAYERS = 1
    N_CLASSES = 2

    # Training
    LEARNING_RATE = 1e-4
    EPOCHS = 1
    GRAD_CLIP = 1.0

    # Misc
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    def __str__(self):
        return f"p{self.PATCH_SIZE}_e{self.EMB_DIM}_h{self.N_HEADS}_l{self.N_LAYERS}"

    @property
    def model_info(self):
        return {
            "patch_size": self.PATCH_SIZE,
            "emb_dim": self.EMB_DIM,
            "n_heads": self.N_HEADS,
            "ff_dim": self.FF_DIM,
            "n_layers": self.N_LAYERS,
        }


# Create global config
CFG = Config()


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

        mod = 1039 - 1039 % CFG.PATCH_SIZE
        x_m1 = torch.tensor(row["image_m1"][:mod] - row["clean_image_m1"][:mod], dtype=torch.float32)
        x_m2 = torch.tensor(row["image_m2"][:mod] - row["clean_image_m2"][:mod], dtype=torch.float32)
        y = torch.tensor(row["label"], dtype=torch.long)
        return x_m1, x_m2, y


# ----------------------------
# 2) PATCH + POSITIONAL ENCODING
# ----------------------------
class PatchEmbedding(nn.Module):
    def __init__(self, in_dim=1039, patch_size=None, emb_dim=None):
        super().__init__()
        self.patch_size = patch_size or CFG.PATCH_SIZE
        emb_dim = emb_dim or CFG.EMB_DIM
        self.proj = nn.Linear(self.patch_size, emb_dim)

    def forward(self, x):
        # x: [batch, 1039]
        length = x.shape[1]
        mod = length % self.patch_size
        if mod != 0:
            pad_length = self.patch_size - mod
            x = nn.functional.pad(x, (0, pad_length), mode='constant', value=0)
        x = x.view(x.shape[0], -1, self.patch_size)  # [batch, #patches, patch_size]
        return self.proj(x)  # [batch, #patches, emb_dim]


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
        src2, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights


# ----------------------------
# 4) SHAPE TRANSFORMER (single-branch)
# ----------------------------
class ShapeTransformer(nn.Module):
    def __init__(self, emb_dim=None, n_heads=None, ff_dim=None, n_layers=None, max_len=2000):
        super().__init__()
        emb_dim = emb_dim or CFG.EMB_DIM
        n_heads = n_heads or CFG.N_HEADS
        ff_dim = ff_dim or CFG.FF_DIM
        n_layers = n_layers or CFG.N_LAYERS

        self.patch_embedding = PatchEmbedding()
        self.pos_encoder = PositionalEncoding(emb_dim, max_len)

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
        self.final_linear = nn.Identity()

    def forward(self, x, return_attention=False):
        x = self.patch_embedding(x)
        x = self.pos_encoder(x)

        all_attn = []
        for layer in self.layers:
            x, attn = layer(x)
            if return_attention:
                all_attn.append(attn)
        x = x.mean(dim=1)
        if return_attention:
            return self.final_linear(x), all_attn
        else:
            return self.final_linear(x)


# ----------------------------
# 5) COMBINED TRANSFORMER (two-branch)
# ----------------------------
class CombinedTransformer(nn.Module):
    def __init__(self, emb_dim=None, n_heads=None, ff_dim=None, n_layers=None, n_classes=None):
        super().__init__()
        emb_dim = emb_dim or CFG.EMB_DIM
        n_classes = n_classes or CFG.N_CLASSES

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
        nn.utils.clip_grad_norm_(model.parameters(), CFG.GRAD_CLIP)
        optimizer.step()

        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

        if counter % 100 == 0:
            print(f"Batch {counter} - Loss: {loss.item():.4f} - Acc: {correct / total:.4f}")
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


def report_misclassified(model, dataset, device, max_samples=100):
    model.eval()
    with torch.no_grad():
        count = 0
        for i in range(len(dataset)):
            if count >= max_samples:
                break
            x_m1, x_m2, y = dataset[i]
            x_m1, x_m2 = x_m1.unsqueeze(0).to(device), x_m2.unsqueeze(0).to(device)
            pred = model(x_m1, x_m2).argmax(dim=1).item()
            if pred != y.item():
                print(f"Sample {i} - Actual: {y.item()}, Predicted: {pred}")
            count += 1


# ----------------------------
# 7) ATTENTION DISPLAY UTILS
# ----------------------------
def plot_attention_heads(attn_tensor, layer_idx=0, batch_idx=0, title="Default", val_acc=None, n_samples=None):
    attn_layer = attn_tensor[layer_idx][batch_idx]
    n_heads = attn_layer.size(0)
    fig, axes = plt.subplots(1, n_heads, figsize=(30 * n_heads, 30))
    if n_heads == 1:
        axes = [axes]
    for head in range(n_heads):
        print("attn_layer.shape:", attn_layer.shape)
        print("attn_layer[head].shape:", attn_layer[head].shape)

        attention_matrix = attn_layer[head].cpu()

        # Make the matrix symmetric by averaging with its transpose
        attention_matrix = (attention_matrix + attention_matrix.T) / 2
        # Flip vertically for visualization
        attention_matrix = attention_matrix.flip(0)

        axes[head].imshow(attention_matrix, cmap='viridis')

        head_title = f"Head {head}, {title}\n"
        if val_acc is not None:
            head_title += f"Val Acc: {val_acc:.3f}"
        if n_samples is not None:
            head_title += f" (n={n_samples})"
        axes[head].set_title(head_title)
        axes[head].axis('off')

    plt.tight_layout()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs('./plots', exist_ok=True)

    config_str = str(CFG)
    filename = f'./plots/attn_l{layer_idx}_{title}_{config_str}_{timestamp}.png'
    plt.savefig(filename, bbox_inches='tight')
    plt.show()


# ----------------------------
# 8) IMAGE PLOTTING UTILS
# ----------------------------
# Load MAGIC camera geometry
geom = CameraGeometry.from_name("MAGICCam")
pix_x = geom.pix_x.value
pix_y = geom.pix_y.value


def plot_camera_images(row):
    """Plot raw camera images from a DataFrame row"""
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
    print(f"Using device: {CFG.DEVICE}")
    print(f"Model config: {str(CFG)}")

    # Dataset + split
    dataset = MagicDataset(gamma_parquet=CFG.GAMMA_FILE, proton_parquet=CFG.PROTON_FILE)
    train_size = int(CFG.TRAIN_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True)

    # Two-branch model
    model = CombinedTransformer().to(CFG.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CFG.LEARNING_RATE)

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(CFG.EPOCHS):
        print(f"Epoch {epoch + 1}")
        train_loss = train_model(model, train_loader, criterion, optimizer, CFG.DEVICE)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, CFG.DEVICE)
        print(f"Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Val Acc={val_acc:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"best_model_{str(CFG)}.pt")

    # Compute average attention over samples
    model.eval()
    average_over = 800000
    gamma_count = proton_count = 0
    gamma_attn_m1 = gamma_attn_m2 = None
    proton_attn_m1 = proton_attn_m2 = None

    with torch.no_grad():
        for x_m1, x_m2, y in val_loader:
            x_m1, x_m2 = x_m1.to(CFG.DEVICE), x_m2.to(CFG.DEVICE)
            logits, attn_m1, attn_m2 = model(x_m1, x_m2, return_attention=True)

            batch_size = y.size(0)
            for i in range(batch_size):
                label = y[i].item()
                if label == 0 and gamma_count < average_over:
                    if gamma_attn_m1 is None:
                        num_layers = len(attn_m1)
                        n_heads = attn_m1[0].shape[1]
                        seq_len = attn_m1[0].shape[2]
                        gamma_attn_m1 = [torch.zeros(n_heads, seq_len, seq_len, device=CFG.DEVICE)
                                         for _ in range(num_layers)]
                        gamma_attn_m2 = [torch.zeros(n_heads, seq_len, seq_len, device=CFG.DEVICE)
                                         for _ in range(num_layers)]
                    for l in range(len(attn_m1)):
                        gamma_attn_m1[l] += attn_m1[l][i]
                        gamma_attn_m2[l] += attn_m2[l][i]
                    gamma_count += 1
                elif label == 1 and proton_count < average_over:
                    if proton_attn_m1 is None:
                        num_layers = len(attn_m1)
                        n_heads = attn_m1[0].shape[1]
                        seq_len = attn_m1[0].shape[2]
                        proton_attn_m1 = [torch.zeros(n_heads, seq_len, seq_len, device=CFG.DEVICE)
                                          for _ in range(num_layers)]
                        proton_attn_m2 = [torch.zeros(n_heads, seq_len, seq_len, device=CFG.DEVICE)
                                          for _ in range(num_layers)]
                    for l in range(len(attn_m1)):
                        proton_attn_m1[l] += attn_m1[l][i]
                        proton_attn_m2[l] += attn_m2[l][i]
                    proton_count += 1

                if gamma_count >= average_over and proton_count >= average_over:
                    break
            if gamma_count >= average_over and proton_count >= average_over:
                break

        # Compute average attention
        if gamma_count > 0:
            gamma_attn_m1 = [attn / gamma_count for attn in gamma_attn_m1]
            gamma_attn_m2 = [attn / gamma_count for attn in gamma_attn_m2]
        if proton_count > 0:
            proton_attn_m1 = [attn / proton_count for attn in proton_attn_m1]
            proton_attn_m2 = [attn / proton_count for attn in proton_attn_m2]

        # Plot average attention for M1, layer 0
        if gamma_attn_m1 is not None:
            print(f"Average attention for {gamma_count} Gammas (M1, Layer 0):")
            gamma_attn_plot = [attn.unsqueeze(0) for attn in gamma_attn_m1]
            plot_attention_heads(gamma_attn_plot, layer_idx=0, batch_idx=0,
                                 title="Gamma", val_acc=val_acc, n_samples=gamma_count)
        if proton_attn_m1 is not None:
            print(f"Average attention for {proton_count} Protons (M1, Layer 0):")
            proton_attn_plot = [attn.unsqueeze(0) for attn in proton_attn_m1]
            plot_attention_heads(proton_attn_plot, layer_idx=0, batch_idx=0,
                                 title="Proton", val_acc=val_acc, n_samples=proton_count)

    # Show misclassified examples
    report_misclassified(model, val_dataset, CFG.DEVICE)

    # Example of plotting raw camera images
    # df_example = pd.read_parquet(CFG.PROTON_FILE)
    # for i in range(3):
    #    plot_camera_images(df_example.iloc[i])