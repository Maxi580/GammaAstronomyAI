import json
import os
from typing import List, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

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
        "tn": float,
        "fp": float,
        "fn": float,
        "tp": float,
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
        print(f"Training on Device: {self.device}")

    def load_training_data(self, training_dir):
        print("Loading Training Data into Datasets...\n")
        self.training_dir = training_dir

        self.full_dataset = ShapeDataset(training_dir)

        # Print info about the training data
        self.data_distribution = self.full_dataset.get_distribution()
        print("Full Dataset Overview:")
        print(f"Total number of samples: {self.data_distribution["total_samples"]}\n")
        print("Class Distribution:")
        for label, info in self.data_distribution["distribution"].items():
            print(f"{label}: {info["count"]} samples ({info["percentage"]:.2f}%)")
        print("\n")

        # Stratified splitting using sklearn
        # Use 70% of data for training and 30% for validation
        print("Splitting Dataset equally...\n")
        labels = torch.tensor(
            [self.full_dataset[i][1] for i in range(len(self.full_dataset))]
        )
        train_indices, val_indices = train_test_split(
            np.arange(len(self.full_dataset)),
            test_size=0.3,
            stratify=labels.numpy(),  # Use the labels for stratification
            random_state=42,
        )

        # Create subsets for training and testing
        self.train_dataset = Subset(self.full_dataset, train_indices)
        self.val_dataset = Subset(self.full_dataset, val_indices)

        print("Loading Data Loaders...\n")

        self.training_data_loader = DataLoader(
            self.train_dataset, batch_size=32, shuffle=True
        )
        self.validation_data_loader = DataLoader(
            self.val_dataset, batch_size=32, shuffle=True
        )

    def start_training(self, epochs: int, info_prints: bool = False):
        print("Starting Training...")
        self.epochs = epochs

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

        self.training_confusion_matrices = []
        self.validation_confusion_matrices = []

        self._train_model(epochs, info_prints)

    def _train_model(self, num_epochs: int, info_prints: bool = False):

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.01)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3)

        best_validation_accuracy = 0

        for epoch in range(num_epochs):
            print(f"Training Epoch {epoch + 1}/{num_epochs}...")
            self.model.train()

            self._training_step(optimizer, criterion)

            self.model.eval()

            validation_accuracy = self._validation_step(scheduler, criterion)

            if info_prints:
                self._print_metrics_last_epoch()

            print("-" * 50)

            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.output_dir, "trained_model.pth"),
                )

    def _training_step(self, optimizer: optim.Optimizer, criterion):
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
        self._calc_metrics(
            self.training_metrics,
            train_labels,
            train_preds,
            train_loss / len(self.training_data_loader),
            self.training_confusion_matrices,
        )

    def _validation_step(self, scheduler, criterion) -> float:
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
        accuracy = self._calc_metrics(
            self.validation_metrics,
            val_labels,
            val_preds,
            val_loss / len(self.validation_data_loader),
            self.validation_confusion_matrices,
        )

        return accuracy

    def _calc_metrics(self, metrics: List[MetricsDict], y_pred, y_true, loss, cm_list):
        # Calculate metrics on training data for current epoch
        accuracy = 100.0 * accuracy_score(y_true, y_pred)
        precision = 100.0 * precision_score(y_true, y_pred, zero_division=0)
        recall = 100.0 * recall_score(y_true, y_pred)
        f1 = 100.0 * f1_score(y_true, y_pred)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        cm_list.append(cm)

        metrics.append(
            {
                "loss": loss,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tn": float(tn),
                "fp": float(fp),
                "fn": float(fn),
                "tp": float(tp),
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

    def write_results(self):
        print("Writing Training Results...")

        data = {
            "dataset": {
                "directory": self.training_dir,
                "distribution": self.data_distribution,
            },
            "epochs": self.epochs,
            "model": {
                "name": self.modelname,
                "structure": self._get_model_structure(),
                "trainable_weights": self._count_trainable_weights(),
            },
            "training_metrics": self.training_metrics,
            "validation_metrics": self.validation_metrics,
        }

        with open(os.path.join(self.output_dir, "info.json"), "w") as file:
            file.write(json.dumps(data, indent=4))

        epochs = np.arange(1, self.epochs + 1)

        # Create a diagram with loss values
        graphs = [
            (epochs, [m["loss"] for m in self.training_metrics], "Training"),
            (epochs, [m["loss"] for m in self.validation_metrics], "Validation"),
        ]
        self._create_diagram(
            filename="loss_diagram.png",
            title="Loss Diagram",
            x_axis=((1, self.epochs), None, "Epochs"),
            y_axis=((0, 1), np.arange(0, 1.1, 0.1), "Loss Value"),
            graphs=graphs,
        )

        # Create Diagram for training metrics
        graphs = [
            (epochs, [m["accuracy"] for m in self.training_metrics], "Accuracy"),
            (epochs, [m["precision"] for m in self.training_metrics], "Precision"),
            (epochs, [m["recall"] for m in self.training_metrics], "Recall"),
            (epochs, [m["f1"] for m in self.training_metrics], "F1"),
        ]
        self._create_diagram(
            filename="training_metrics.png",
            title="Training Metrics",
            x_axis=((1, self.epochs), None, "Epochs"),
            y_axis=((0, 100), np.arange(0, 101, 10), "Percentage"),
            graphs=graphs,
        )

        # Create Diagram for validation metrics
        graphs = [
            (epochs, [m["accuracy"] for m in self.validation_metrics], "Accuracy"),
            (epochs, [m["precision"] for m in self.validation_metrics], "Precision"),
            (epochs, [m["recall"] for m in self.validation_metrics], "Recall"),
            (epochs, [m["f1"] for m in self.validation_metrics], "F1"),
        ]
        self._create_diagram(
            filename="validation_metrics.png",
            title="Validation Metrics",
            x_axis=((1, self.epochs), None, "Epochs"),
            y_axis=((0, 100), np.arange(0, 101, 10), "Percentage"),
            graphs=graphs,
        )

        # Create Diagram for Accuracy Comparison
        graphs = [
            (
                epochs,
                [m["accuracy"] for m in self.training_metrics],
                "Accuracy Training",
            ),
            (
                epochs,
                [m["accuracy"] for m in self.validation_metrics],
                "Accuracy Validation",
            ),
        ]
        self._create_diagram(
            filename="accuracy_comparison.png",
            title="Accuracy Comparison",
            x_axis=((1, self.epochs), None, "Epochs"),
            y_axis=((0, 100), np.arange(0, 101, 10), "Percentage"),
            graphs=graphs,
        )

        # Plot Confusion matrices
        # TODO: add correct labels
        training_cm = ConfusionMatrixDisplay(self.training_confusion_matrices[-1])
        training_cm.plot().figure_.savefig(
            os.path.join(self.output_dir, "training_confusion_matrix.png")
        )

        validation_cm = ConfusionMatrixDisplay(self.validation_confusion_matrices[-1])
        validation_cm.plot().figure_.savefig(
            os.path.join(self.output_dir, "validation_confusion_matrix.png")
        )

    def _get_model_structure(self):
        model_structure = {}
        for name, module in self.model.named_modules():
            if name:
                model_structure[name] = str(module)
        return model_structure

    def _count_trainable_weights(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _create_diagram(self, filename: str, title: str, x_axis, y_axis, graphs):
        ax = plt.subplot()

        for x, y, label in graphs:
            ax.plot(x, y, label=label)

        xlim, xticks, xlabel = x_axis
        ylim, yticks, ylabel = y_axis

        ax.set(
            xlim=xlim,
            ylim=ylim,
        )

        if xticks is not None:
            ax.set_xticks(xticks)
        if yticks is not None:
            ax.set_yticks(yticks)

        ax.set_title(title)
        ax.legend()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.figure.savefig(os.path.join(self.output_dir, filename))
        ax.clear()
