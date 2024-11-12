import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'simulated_data')
ARRAY_DIR = os.path.join(DATA_DIR, 'arrays')
ANNOTATION_DIR = os.path.join(DATA_DIR, 'annotations')
IMAGE_SIZE = (128, 128)


class ShapeDataset(Dataset):
    def __init__(self, array_dir, annotation_dir):
        self.array_dir = array_dir
        self.annotation_dir = annotation_dir
        self.arrays = sorted(os.listdir(array_dir))
        self.labels = {'square': 0, 'ellipse': 1}

    def __len__(self):
        return len(self.arrays)

    def __getitem__(self, idx):
        array_path = os.path.join(self.array_dir, self.arrays[idx])
        label_path = os.path.join(self.annotation_dir, self.arrays[idx].replace('.json', '.txt'))

        with open(array_path, 'r') as f:
            array = json.loads(f.read().strip())

        with open(label_path, 'r') as f:
            label = f.read().strip()

        return torch.tensor([array]), self.labels[label]


class SimpleShapeCNN(nn.Module):
    def __init__(self):
        super(SimpleShapeCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout1d(0.1),

            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout1d(0.1),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout1d(0.1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 129, 128), # 64*129 because of pooling
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_model(model, train_loader, val_loader, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    best_val_acc = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            train_loss += loss.item()

        train_acc = 100. * train_correct / train_total

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                val_loss += loss.item()

        val_acc = 100. * val_correct / val_total

        scheduler.step(val_loss)

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Train Loss: {train_loss / len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss / len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        print('--------------------')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model_arrays.pth')


def main():
    print("Loading Test Data into Datasets...")
    
    full_dataset = ShapeDataset(
        ARRAY_DIR,
        ANNOTATION_DIR
    )

    # Use 80% of data for training and 20% for validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    print("Loading Data Loaders...")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    
    print("Starting Training...")

    model = SimpleShapeCNN()

    train_model(model, train_loader, val_loader, num_epochs=30)


if __name__ == '__main__':
    main()
