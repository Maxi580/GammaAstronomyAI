import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


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
            nn.Sigmoid()  # Added to match training architecture
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)


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


def report_misclassified(model, dataset, device):
    model.eval()
    count = 0
    with torch.no_grad():
        for i in range(len(dataset)):
            count += 1
            if count > 300:
                break

            x, y = dataset[i]
            data_file = dataset.data_files[i]
            x = x.unsqueeze(0).to(device)
            outputs = model(x)
            pred = (outputs[:, 1] > outputs[:, 0]).long().item()  # Modified prediction logic
            if pred != y.item():
                print(f"Index: {i}, File: {data_file} - Actual: {y.item()}, Predicted: {pred}")
                print(f"Confidence scores - Class 0: {outputs[0][0]:.4f}, Class 1: {outputs[0][1]:.4f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    direc = "../simulated_data_5k_nn/"

    dataset = ShapeDataset(
        data_dir=direc + "arrays",
        ann_dir=direc + "annotations"
    )

    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = ShapeTransformer(
        emb_dim=256,
        n_heads=16,
        ff_dim=512,
        n_layers=6,
        n_classes=2
    ).to(device)

    model.load_state_dict(torch.load("./best_model.pt", map_location=device))
    model.eval()
    print("Model loaded successfully")

    # Evaluate using the custom loss
    criterion = CustomLoss()
    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            running_loss += loss.item()
            preds = (outputs[:, 1] > outputs[:, 0]).long()  # Modified prediction logic
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = running_loss / len(loader)
    accuracy = correct / total
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Overall Accuracy on dataset: {accuracy:.4f}")

    # Print misclassified examples
    print("\nMisclassified examples:")
    report_misclassified(model, dataset, device)