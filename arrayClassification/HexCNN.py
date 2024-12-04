import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import os
import json

from arrayClassification.HexConv.HexConv import ConvHex

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'simulated_data')
ARRAY_DIR = os.path.join(DATA_DIR, 'arrays')
ANNOTATION_DIR = os.path.join(DATA_DIR, 'annotations')


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

    def get_distribution(self):
        all_labels = []
        for array_file in self.arrays:
            label_path = os.path.join(self.annotation_dir, array_file.replace('.json', '.txt'))
            with open(label_path, 'r') as f:
                label = f.read().strip()
            all_labels.append(self.labels[label])

        total_samples = len(all_labels)
        label_counts = {}
        for label_name, label_idx in self.labels.items():
            count = all_labels.count(label_idx)
            percentage = (count / total_samples) * 100
            label_counts[label_name] = {
                'count': count,
                'percentage': percentage
            }

        return {
            'total_samples': total_samples,
            'distribution': label_counts
        }


class HexCNN(nn.Module):
    def __init__(self):
        super(HexCNN, self).__init__()

        self.features = nn.Sequential(
            # Capture hexagonal structure !!!
            ConvHex(in_channels=1, out_channels=16),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout1d(0.1),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout1d(0.15),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16576, 32),  # 64*129 (1039 / 2 / 2 = 129)
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 2)
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

        train_preds = []
        train_labels = []
        train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            train_loss += loss.item()

        train_acc = 100. * np.mean(np.array(train_labels) == np.array(train_preds))
        train_precision = 100. * precision_score(train_labels, train_preds, average='weighted', zero_division=0)
        train_recall = 100. * recall_score(train_labels, train_preds, average='weighted')
        train_f1 = 100. * f1_score(train_labels, train_preds, average='weighted')

        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, predicted = outputs.max(1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_loss += loss.item()
        scheduler.step(val_loss)

        val_acc = 100. * np.mean(np.array(val_labels) == np.array(val_preds))
        val_precision = 100. * precision_score(val_labels, val_preds, average='weighted', zero_division=0)
        val_recall = 100. * recall_score(val_labels, val_preds, average='weighted')
        val_f1 = 100. * f1_score(val_labels, val_preds, average='weighted')

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print('Training Metrics:')
        print(f'Loss: {train_loss / len(train_loader):.4f}')
        print(f'Accuracy: {train_acc:.2f}%')
        print(f'Precision: {train_precision:.2f}%')
        print(f'Recall: {train_recall:.2f}%')
        print(f'F1-Score: {train_f1:.2f}%')

        print('\nValidation Metrics:')
        print(f'Loss: {val_loss / len(val_loader):.4f}')
        print(f'Accuracy: {val_acc:.2f}%')
        print(f'Precision: {val_precision:.2f}%')
        print(f'Recall: {val_recall:.2f}%')
        print(f'F1-Score: {val_f1:.2f}%')
        print('-' * 50)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model_arrays.pth')


def main():
    print("Loading Test Data into Datasets...")
    
    full_dataset = ShapeDataset(
        ARRAY_DIR,
        ANNOTATION_DIR
    )

    dist_info = full_dataset.get_distribution()
    print("\nDataset Overview:")
    print(f"Total number of samples: {dist_info['total_samples']}")
    print("\nClass Distribution:")
    for label, info in dist_info['distribution'].items():
        print(f"{label}: {info['count']} samples ({info['percentage']:.2f}%)")

    # Use 70% of data for training and 30% for validation
    train_size = int(0.7 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    print("Loading Data Loaders...")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    
    print("Starting Training...")

    model = HexCNN()

    train_model(model, train_loader, val_loader, num_epochs=10)


if __name__ == '__main__':
    main()
