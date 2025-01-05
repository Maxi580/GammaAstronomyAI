import os
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader, random_split


class ShapeDataset(Dataset):
    def __init__(self, data_dir, ann_dir, transform=None):
        self.data_files = sorted([
            f for f in os.listdir(data_dir) if f.endswith('.json')
        ])
        self.data_dir = data_dir
        self.ann_dir = ann_dir
        self.transform = transform
        self.label_map = {"square": 0, "ellipse": 1}

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data_file = self.data_files[idx]
        with open(os.path.join(self.data_dir, data_file), 'r') as f:
            data_json = json.load(f)
        x = torch.tensor(data_json["combined"], dtype=torch.float32)

        #x = np.array(data_json["combined"])
        #scaler = MinMaxScaler()
        #x = scaler.fit_transform(x.reshape(-1, 1)).flatten()
        #x = torch.tensor(x, dtype=torch.float32)

        ann_file = data_file.replace('.json', '.txt')
        with open(os.path.join(self.ann_dir, ann_file), 'r') as f:
            label_str = f.read().strip()
        y = torch.tensor(self.label_map[label_str], dtype=torch.long)

        if self.transform:
            x = self.transform(x)

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
    def __init__(self, emb_dim=128, n_heads=8, ff_dim=256, n_layers=4, n_classes=2, max_len=2000):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_dim=1039, patch_size=8, emb_dim=emb_dim)
        self.pos_encoder = PositionalEncoding(emb_dim, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_heads, dim_feedforward=ff_dim, dropout=0.1, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(emb_dim, n_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        # x = nn.LogSoftmax(dim=1)(x)
        return x




def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y).sum().item()
    return running_loss / len(dataloader), correct / len(dataloader.dataset)


# -------- Function to report misclassified samples --------
def report_misclassified(model, dataset, device):
    model.eval()
    count = 0
    with torch.no_grad():
        for i in range(len(dataset)):
            count += 1
            if count > 300:
                break
            x, y = dataset[i]
            x = x.unsqueeze(0).to(device)
            pred = model(x).argmax(dim=1).item()
            if pred != y.item():
                print(f"Sample {i} - Actual: {y.item()}, Predicted: {pred}")


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, pred, target):
        loss = 0
        for i in range(len(target)):
            if target[i] == 0:
                loss += (1 - pred[i][0]) ** 2 + (pred[i][1]) ** 2
                if pred[i][0] < pred[i][1]:
                    loss += 2
            else:
                loss += (pred[i][0]) ** 2 + (1 - pred[i][1]) ** 2
                if pred[i][1] < pred[i][0]:
                    loss += 2
        return loss / len(target)


if __name__ == "__main__":
    BATCH_SIZE = 32
    LR = 1e-4
    EPOCHS = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    direc = "../simulated_data_10k_nn/"

    dataset = ShapeDataset(
        data_dir= direc + "arrays",
        ann_dir= direc + "annotations"
    )

    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = ShapeTransformer(
        emb_dim=256,
        n_heads=16,
        ff_dim=512,
        n_layers=6,
        n_classes=2
    ).to(device)

    #criterion = nn.CrossEntropyLoss()
    #criterion = nn.NLLLoss()
    criterion = CustomLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-7)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)  # More aggressive LR scheduling


    best_val_loss = float('inf')

    print("Starting training...")
    for epoch in range(EPOCHS):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")


    # -------- Model Saving (Paste these lines at the end of main) --------
    torch.save(model.state_dict(), "trained_shape_transformer.pt")
    print("Model saved to trained_shape_transformer.pt")

    # -------- Example usage of reporting misclassified samples --------
    report_misclassified(model, val_dataset, device)
