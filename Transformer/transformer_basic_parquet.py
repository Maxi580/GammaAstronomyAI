import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split

class MagicDataset(Dataset):
    def __init__(self, gamma_parquet, proton_parquet, transform=None):
        # Read gammas
        df_gamma = pd.read_parquet(gamma_parquet)
        print("Length gamma", len(df_gamma))
        #df_gamma = df_gamma.iloc[:int(0.8 * len(df_gamma))]  # Use first 80%
        print("Length gamma", len(df_gamma))
        df_gamma["label"] = 0  # label gammas as 0
        #df_gamma["label"] = torch.randint(0, 2, (len(df_gamma),))

        # Read protons
        df_proton = pd.read_parquet(proton_parquet)
        print("Length proton", len(df_proton))
        #df_proton = df_proton.iloc[:int(0.8 * len(df_proton))]  # Use first 80%
        print("Length proton", len(df_proton))
        df_proton["label"] = 1  # label protons as 1
        #df_proton["label"] = torch.randint(0, 2, (len(df_proton),))

        # Combine into one dataframe
        self.df = pd.concat([df_gamma, df_proton], ignore_index=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Example: convert the stored list of pixel intensities to a tensor
        # Adjust column name(s) to match what you have:
        # e.g. row["image_m1.list"] might be a Python list already
        pixel_list = row["image_m2"][:1039]

        # Convert to float32 tensor.
        # You might want to combine image_m1.list and image_m2.list, etc.
        x = torch.tensor(pixel_list, dtype=torch.float32)
        #print("x: ", x, " shape: ", x.shape, " first element: ", x[0], " last element: ", x[-1])
        y = torch.tensor(row["label"], dtype=torch.long)
        #(y)
        if self.transform:
            x = self.transform(x)

        return x, y


# -------------- Example usage below (rest is mostly unchanged) --------------
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
            nn.Linear(emb_dim, n_classes)
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)


def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    counter = 0
    correct = 0
    total = 0
    #number_y_is_one = 0
    #number_y_is_zero = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        #number_y_is_one += torch.sum(y).item()
        #number_y_is_zero += y.size(0) - torch.sum(y).item()
        optimizer.zero_grad()
        outputs = model(x)
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
            #print(f"Number of 1s: {number_y_is_one}, Number of 0s: {number_y_is_zero}")
            correct = 0
            total = 0
        counter += 1
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

if __name__ == "__main__":
    BATCH_SIZE = 128
    LR = 1e-5
    EPOCHS = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using: ", device)

    # Point to your actual parquet files
    gamma_file = "../magic-gammas_part1.parquet"
    proton_file = "../magic-protons_part1.parquet"

    dataset = MagicDataset(
        gamma_parquet=gamma_file,
        proton_parquet=proton_file
    )

    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = ShapeTransformer(
        emb_dim=512,
        n_heads=8,
        ff_dim=1024,
        n_layers=4,
        n_classes=2
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        print("Epoch", epoch + 1)
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        print("Train Loss:", train_loss)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model_single.pt")

    torch.save(model.state_dict(), "trained_shape_transformer_single.pt")
    print("Model saved to trained_shape_transformer_single.pt")

    report_misclassified(model, val_dataset, device)
