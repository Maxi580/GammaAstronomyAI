import os
from typing import TypedDict, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from cnn.Architectures.HexCNN import HexCNN
from cnn.shapeDataset import ShapeDataset
from cnn.Architectures.BasicCNN import BasicCNN
from cnn.resultsWriter import ResultsWriter

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
        "label_0": str,
        "label_1": str
    },
)


def print_metrics(metrics: MetricsDict):
    print(f"\nModel Performance Metrics:")
    print(f"{'Accuracy:':<12} {metrics['accuracy']:>6.2f}%")
    print(f"{'Precision:':<12} {metrics['precision']:>6.2f}%")
    print(f"{'Recall:':<12} {metrics['recall']:>6.2f}%")
    print(f"{'F1-Score:':<12} {metrics['f1']:>6.2f}%")
    print(f"{'Loss:':<12} {metrics['loss']:>6.4f}")

    print("\nConfusion Matrix:")
    print("─" * 45)
    print(f"│              │      Predicted      │")
    print(f"│              │   0-{metrics['label_0']}  1-{metrics['label_1']}   │")
    print(f"├──────────────┼───────────────────┤")
    print(f"│ Actual    0  │    {metrics['tn']:4.0f}      {metrics['fp']:4.0f}    │")
    print(f"│ {metrics['label_0']:<10}  │                   │")
    print(f"│ Actual    1  │    {metrics['fn']:4.0f}      {metrics['tp']:4.0f}    │")
    print(f"│ {metrics['label_1']:<10}  │                   │")
    print("─" * 45)


class TrainingSupervisor:
    DATA_TEST_SPLIT: float = 0.3
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 1e-3
    ADAM_BETA_1: float = 0.9
    ADAM_BETA_2: float = 0.999
    ADAM_EPSILON: float = 1e-8
    WEIGHT_DECAY: float = 1e-2
    SCHEDULER_MIN_LR: float = 1e-4
    SCHEDULER_MAX_LR: float = 1e-3
    SCHEDULER_MODE: Literal["triangular", "triangular2", "exp_range"] = "triangular2"
    SCHEDULER_CYCLE_MOMENTUM: bool = False
    GRAD_CLIP_NORM: float = 1

    def __init__(self, model_name: str, input_dir: str, output_dir: str, debug_info: bool = True,
                 save_model: bool = True) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.debug_info = debug_info
        self.save_model = save_model

        if debug_info or save_model:
            os.makedirs(output_dir, exist_ok=True)
        if debug_info:
            print(f"Training on Device: {self.device}")

        self.model_name = model_name
        self.model = self.load_model()
        self.validation_metrics = []
        self.train_metrics = []

        self.input_dir = input_dir
        self.dataset, self.train_dataset, self.val_dataset, self.training_data_loader, self.val_data_loader \
            = self.load_training_data()

        self.output_dir = output_dir

    def load_training_data(self) -> tuple[ShapeDataset, Subset, Subset, DataLoader, DataLoader]:
        if self.debug_info:
            print("Loading Training Data...\n")

        dataset = ShapeDataset(self.input_dir)

        if self.debug_info:
            data_distribution = dataset.get_distribution()
            print("Full Dataset Overview:")
            print(f"Total number of samples: {data_distribution["total_samples"]}\n")
            print("Class Distribution:")
            for label, info in data_distribution["distribution"].items():
                print(f"{label}: {info["count"]} samples ({info["percentage"]:.2f}%)")
            print("\n")

        # Stratified splitting using sklearn
        # Use 70% of data for training and 30% for validation
        labels = torch.tensor(
            [dataset[i][1] for i in range(len(dataset))]
        )

        train_indices, val_indices = train_test_split(
            np.arange(len(dataset)),
            test_size=self.DATA_TEST_SPLIT,
            stratify=labels.numpy(),
            random_state=42,
        )

        if self.debug_info:
            print("\nAfter split label distribution:")
            print("Training set:")
            train_labels = labels[train_indices]
            unique, counts = np.unique(train_labels.numpy(), return_counts=True)
            for label, count in zip(unique, counts):
                idx_to_label = {v: k for k, v in dataset.labels.items()}
                label_name = idx_to_label[label]
                print(f"{label_name} (label {label}): {count} samples ({count / len(train_labels) * 100:.2f}%)")

            print("\nValidation set:")
            val_labels = labels[val_indices]
            unique, counts = np.unique(val_labels.numpy(), return_counts=True)
            for label, count in zip(unique, counts):
                idx_to_label = {v: k for k, v in dataset.labels.items()}
                label_name = idx_to_label[label]
                print(f"{label_name} (label {label}): {count} samples ({count / len(val_labels) * 100:.2f}%)")
            print("\n")

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)

        training_data_loader = DataLoader(
            train_dataset, batch_size=self.BATCH_SIZE, shuffle=True
        )
        val_data_loader = DataLoader(
            val_dataset, batch_size=self.BATCH_SIZE, shuffle=False
        )

        if self.debug_info:
            print("Dataset loaded.")
            print("\n")

        return dataset, train_dataset, val_dataset, training_data_loader, val_data_loader

    def load_model(self):
        match self.model_name.lower():
            case "hexcnn":
                model = HexCNN()
            case "basiccnn":
                model = BasicCNN()
            case _:
                raise ValueError(f"Invalid Modelname: '{self.model_name}'")

        if self.debug_info:
            print("Model loaded.")

        return model.to(self.device)

    def train_model(self, epochs: int):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.LEARNING_RATE,
            betas=(self.ADAM_BETA_1, self.ADAM_BETA_2),
            eps=self.ADAM_EPSILON,
            weight_decay=self.WEIGHT_DECAY
        )

        steps_per_epoch = len(self.training_data_loader)
        scheduler = optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=self.SCHEDULER_MIN_LR,
            max_lr=self.SCHEDULER_MAX_LR,
            step_size_up=2 * steps_per_epoch,
            mode=self.SCHEDULER_MODE,  # Learning rate policy
            cycle_momentum=self.SCHEDULER_CYCLE_MOMENTUM  # Don't cycle momentum for Adam
        )

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

        best_validation_accuracy = 0
        for epoch in range(epochs):
            if self.debug_info:
                print(f"Training Epoch {epoch + 1}/{epochs}...")

            self.model.train()

            train_metrics = self._training_step(optimizer, criterion)
            if self.debug_info:
                print(f"Training Metrics of epoch: {epoch}: \n")
                print_metrics(train_metrics)

            self.model.eval()

            val_metrics = self._validation_step(scheduler, criterion)
            if self.debug_info:
                print(f"Validation Metrics of epoch: {epoch}: \n")
                print_metrics(val_metrics)
                print("-" * 50)

            if val_metrics['accuracy'] > best_validation_accuracy:
                best_validation_accuracy = val_metrics['accuracy']
                if self.save_model:
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(self.output_dir, "trained_model.pth"),
                    )

        if self.debug_info:
            self.write_results(epochs)

    def _training_step(self, optimizer: optim.Optimizer, criterion) -> dict[str, float]:
        # Test accuracy on training data for current epoch
        train_preds = []
        train_labels = []
        train_loss = 0
        for inputs, labels in self.training_data_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.GRAD_CLIP_NORM)
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            train_loss += loss.item()

        metrics = self.calc_metrics(
            train_labels,
            train_preds,
            train_loss / len(self.training_data_loader),
        )

        self.train_metrics.append(metrics)
        return metrics

    def _validation_step(self, scheduler, criterion) -> dict[str, float]:
        # Test accuracy on validation data for current epoch
        val_loss = 0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for inputs, labels in self.val_data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                _, predicted = outputs.max(1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_loss += loss.item()
        scheduler.step()

        # Calculate metrics on validation data for current epoch
        metrics = self.calc_metrics(
            val_labels,
            val_preds,
            val_loss / len(self.val_data_loader),
        )

        self.validation_metrics.append(metrics)
        return metrics

    def calc_metrics(self, y_pred, y_true, loss):
        print(f"Unique values in y_true: {np.unique(y_true, return_counts=True)}")
        print(f"Unique values in y_pred: {np.unique(y_pred, return_counts=True)}")
        accuracy = 100.0 * accuracy_score(y_true, y_pred)
        precision = 100.0 * precision_score(y_true, y_pred, zero_division=0)
        recall = 100.0 * recall_score(y_true, y_pred)
        f1 = 100.0 * f1_score(y_true, y_pred)

        # Get label mapping (which actual label is 0 and which is 1)
        label_to_idx = self.dataset.labels
        idx_to_label = {v: k for k, v in label_to_idx.items()}
        print(f"Label mapping: {label_to_idx}")

        cm = confusion_matrix(y_pred, y_true, labels=[0, 1])
        print(f"Raw confusion matrix:\n{cm}")

        tn, fp, fn, tp = cm.ravel()

        metrics = {
            "loss": loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tn": float(tn),
            "fp": float(fp),
            "fn": float(fn),
            "tp": float(tp),
            "label_0": idx_to_label[0],
            "label_1": idx_to_label[1]
        }

        return metrics

    def _get_model_structure(self):
        model_structure = {}
        for name, module in self.model.named_modules():
            if name:
                model_structure[name] = str(module)
        return model_structure

    def _count_trainable_weights(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def write_results(self, epochs):
        training_data = {
            "dataset": {
                "directory": self.input_dir,
                "distribution": self.dataset.get_distribution(),
            },
            "epochs": epochs,
            "model": {
                "name": self.model_name,
                "structure": self._get_model_structure(),
                "trainable_weights": self._count_trainable_weights(),
            },
            "training_metrics": self.train_metrics,
            "validation_metrics": self.validation_metrics,
        }

        writer = ResultsWriter(self.output_dir)
        writer.save_training_results(training_data)
