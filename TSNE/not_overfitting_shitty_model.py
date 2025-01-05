# improved_tsne_classifier.py

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

class ShapeDataset(Dataset):
    def __init__(self, data_dir, ann_dir):
        self.data_files = sorted([
            f for f in os.listdir(data_dir) if f.endswith('.json')
        ])
        self.data_dir = data_dir
        self.ann_dir = ann_dir
        self.label_map = {"square": 0, "ellipse": 1}

        # Preload
        self.X = []
        self.Y = []
        for data_file in self.data_files:
            with open(os.path.join(self.data_dir, data_file), 'r') as f:
                data_json = json.load(f)
            combined_array = data_json["combined"]
            self.X.append(combined_array)

            ann_file = data_file.replace('.json', '.txt')
            with open(os.path.join(self.ann_dir, ann_file), 'r') as f:
                label_str = f.read().strip()
            self.Y.append(self.label_map[label_str])

        self.X = np.array(self.X, dtype=np.float32)
        self.Y = np.array(self.Y, dtype=np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        return x, y


# Deeper MLP with dropout & batch norm
class MLPClassifier(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y).sum().item()
    return total_loss / len(dataloader), correct / len(dataloader.dataset)


if __name__ == "__main__":
    data_dir = "../simulated_data_10k_nn/arrays"
    ann_dir  = "../simulated_data_10k_nn/annotations"
    dataset  = ShapeDataset(data_dir, ann_dir)

    print("Scaling data...")
    scaler = StandardScaler()
    dataset.X = scaler.fit_transform(dataset.X)

    print("Applying PCA (1039 → 50 dims)...")
    pca_dim = 50
    pca = PCA(n_components=pca_dim, random_state=123)
    dataset.X = pca.fit_transform(dataset.X)

    print("Running t-SNE on the PCA output...")
    tsne = TSNE(
        n_components=2,     # Final 2D embedding
        init='pca',
        random_state=123,
        perplexity=40,      # Tweakable
        n_iter=1500,        # Tweakable
        learning_rate='auto'
    )
    dataset.X = tsne.fit_transform(dataset.X)

    # Train/Val split
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    BATCH_SIZE = 32
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPClassifier(input_dim=2, hidden_dim=512, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    EPOCHS = 300
    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}: "
              f"Train Loss={train_loss:.4f}, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_tsne_model.pt")

    print("Training finished. Saved to best_tsne_model.pt.")

    # Check some misclassifications
    model.eval()
    check_count = 0
    for i in range(len(val_dataset)):
        if check_count > 300:
            break
        x, y = val_dataset[i]
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
        pred = model(x_tensor).argmax(dim=1).item()
        if pred != y:
            print(f"Misclassified Val Sample {i}: Actual={y}, Predicted={pred}")
        check_count += 1

