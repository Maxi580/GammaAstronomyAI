import json
import os
import time
from typing import List, TypedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from arrayClassification.CNN.HexCNN import HexCNN
from arrayClassification.CNN.shapeDataset import ShapeDataset
from arrayClassification.CNN.SimpleShapeCNN import SimpleShapeCNN

MetricsDict = TypedDict(
    "MetricsDict",
    {
        "loss": float,
        "accuracy": float,
        "precision": float,
        "recall": float,
        "f1": float,
    },
)


class TrainingSupervisor:
    model: nn.Module
    device: torch.device
    training_data_loader: DataLoader
    validation_data_loader: DataLoader

    training_metrics: List[MetricsDict] = []
    validation_metrics: List[MetricsDict] = []

    def __init__(self, modelname: str, output_dir: str) -> None:
        self.modelname = modelname

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_training_data(self, training_dir):
        print("Loading Training Data into Datasets...\n")

        self.full_dataset = ShapeDataset(training_dir)

        # Print info about the training data
        self.dist_info = self.full_dataset.get_distribution()
        print("Full Dataset Overview:")
        print(f"Total number of samples: {self.dist_info["total_samples"]}\n")
        print("Class Distribution:")
        for label, info in self.dist_info["distribution"].items():
            print(f"{label}: {info["count"]} samples ({info["percentage"]:.2f}%)")
        print("\n")

        # Use 70% of data for training and 30% for validation
        train_size = int(0.7 * len(self.full_dataset))
        val_size = len(self.full_dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.full_dataset, [train_size, val_size]
        )

        print("Loading Data Loaders...\n")

        self.training_data_loader = DataLoader(
            self.train_dataset, batch_size=32, shuffle=True
        )
        self.validation_data_loader = DataLoader(
            self.val_dataset, batch_size=32, shuffle=True
        )

    def start_training(self, epochs: int, info_prints: bool = False):

        print("Starting Training...")

        match self.modelname.lower():
            case "hexcnn":
                self.model = HexCNN()
            case "simpleshapecnn":
                self.model = SimpleShapeCNN()
            case _:
                raise ValueError(f"Invalid Modelname: '{self.modelname}'")

        self.model: nn.Module = self.model.to(self.device)

        self.training_metrics = []
        self.validation_metrics = []

        self._train_model(epochs, info_prints)

    def _train_model(self, num_epochs: int, info_prints: bool = False):

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3)

        best_validation_accuracy = 0

        for epoch in range(num_epochs):
            print(f"Training Epoch {epoch + 1}/{num_epochs}...")
            self.model.train()

            self._calc_train_metrics(optimizer, criterion)

            self.model.eval()

            validation_accuracy = self._calc_validation_metrics(scheduler, criterion)

            if info_prints:
                self._print_metrics_last_epoch()

            print("-" * 50)

            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.output_dir, "trained_model.pth"),
                )

    def _calc_train_metrics(self, optimizer: optim.Optimizer, criterion):
        # Test accuracy on training data for current epoch
        train_preds = []
        train_labels = []
        train_loss = 0
        for inputs, labels in self.training_data_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            train_loss += loss.item()

        # Calculate metrics on training data for current epoch
        accuracy = 100.0 * np.mean(np.array(train_labels) == np.array(train_preds))
        precision = 100.0 * precision_score(
            train_labels, train_preds, average="weighted", zero_division=0
        )
        recall = 100.0 * recall_score(train_labels, train_preds, average="weighted")
        f1 = 100.0 * f1_score(train_labels, train_preds, average="weighted")

        self.training_metrics.append(
            {
                "loss": train_loss / len(self.training_data_loader),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    def _calc_validation_metrics(self, scheduler, criterion) -> float:
        # Test accuracy on validation data for current epoch
        val_loss = 0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for inputs, labels in self.validation_data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                _, predicted = outputs.max(1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_loss += loss.item()
        scheduler.step(val_loss)

        # Calculate metrics on validation data for current epoch
        accuracy = 100.0 * np.mean(np.array(val_labels) == np.array(val_preds))
        precision = 100.0 * precision_score(
            val_labels, val_preds, average="weighted", zero_division=0
        )
        recall = 100.0 * recall_score(val_labels, val_preds, average="weighted")
        f1 = 100.0 * f1_score(val_labels, val_preds, average="weighted")

        self.validation_metrics.append(
            {
                "loss": val_loss / len(self.validation_data_loader),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

        return accuracy

    def _print_metrics_last_epoch(self):
        train_metrics = self.training_metrics[-1]
        val_metrics = self.validation_metrics[-1]

        print("\nTraining Metrics:")
        print(f"Loss: {train_metrics["loss"]:.4f}")
        print(f"Accuracy: {train_metrics["accuracy"]:.2f}%")
        print(f"Precision: {train_metrics["precision"]:.2f}%")
        print(f"Recall: {train_metrics["recall"]:.2f}%")
        print(f"F1-Score: {train_metrics["f1"]:.2f}%")

        print("\nValidation Metrics:")
        print(f"Loss: {val_metrics["loss"]:.4f}")
        print(f"Accuracy: {val_metrics["accuracy"]:.2f}%")
        print(f"Precision: {val_metrics["precision"]:.2f}%")
        print(f"Recall: {val_metrics["recall"]:.2f}%")
        print(f"F1-Score: {val_metrics["f1"]:.2f}%")
