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
from CombinedNet.CombinedNet import CombinedNet
from CombinedNet.magicDataset import MagicDataset
from CombinedNet.resultsWriter import ResultsWriter

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


def calc_metrics(y_true, y_pred, loss):
    accuracy = 100.0 * accuracy_score(y_true, y_pred)
    precision = 100.0 * precision_score(y_true, y_pred, zero_division=0)
    recall = 100.0 * recall_score(y_true, y_pred, zero_division=0)
    f1 = 100.0 * f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

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
    }

    return metrics


def print_metrics(labels, metrics: MetricsDict):
    print(f"Model Performance Metrics:")
    print(f"{'Accuracy:':<12} {metrics['accuracy']:>6.2f}%")
    print(f"{'Precision:':<12} {metrics['precision']:>6.2f}%")
    print(f"{'Recall:':<12} {metrics['recall']:>6.2f}%")
    print(f"{'F1-Score:':<12} {metrics['f1']:>6.2f}%")
    print(f"{'Loss:':<12} {metrics['loss']:>6.4f}")

    print("\nConfusion Matrix:")
    print("─" * 45)
    print(f"│              │      Predicted    │")
    print(f"│              │          0      1 │")
    print(f"├──────────────┼───────────────────┤")
    print(f"│ Actual    0  │    {metrics['tn']:4.0f}      {metrics['fp']:4.0f}│")
    print(f"│              │                   │")
    print(f"│ Actual    1  │    {metrics['fn']:4.0f}      {metrics['tp']:4.0f}│")
    print(f"│              │")
    print("─" * 45)
    # Get label mapping (which actual label is 0 and which is 1)
    print(f"Label mapping: {labels}")


def inference(data_loader, labels, model_path):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = CombinedNet()

    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    print(f"Evaluating model on device: {device}")
    print(f"Total batches to process: {len(data_loader)}")

    with torch.no_grad():
        for batch_idx, (m1_images, m2_images, features, labels) in enumerate(data_loader):
            m1_images = m1_images.to(device)
            m2_images = m2_images.to(device)
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(m1_images, m2_images, features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)

    # Calculate metrics using the average loss and all predictions
    metrics = calc_metrics(all_labels, all_preds, avg_loss)
    print_metrics(labels, metrics)

    return metrics


class TrainingSupervisor:
    TEMP_DATA_SPLIT: float = 0.3
    TEST_DATA_SPLIT: float = 0
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.00001
    WEIGHT_DECAY: float = 0.01
    SCHEDULER_MODE: Literal["triangular", "triangular2", "exp_range"] = "triangular2"
    SCHEDULER_CYCLE_MOMENTUM: bool = False
    GRAD_CLIP_NORM: float = 5.0

    def __init__(self, model_name: str, dataset: MagicDataset, output_dir: str, debug_info: bool = True) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available()
                                   else "cpu")
        self.debug_info = debug_info
        os.makedirs(output_dir, exist_ok=True)

        if debug_info:
            print(f"Training on Device: {self.device}")

        self.model_name = model_name
        self.model = self.load_model()
        self.validation_metrics = []
        self.train_metrics = []
        self.inference_metrics = []

        self.dataset = dataset
        (self.train_dataset, self.val_dataset, self.test_dataset, self.training_data_loader, self.val_data_loader,
         self.test_data_loader) = self.load_training_data()

        self.output_dir = output_dir
        self.model_path = os.path.join(self.output_dir, "trained_model.pth")

    def load_training_data(self) -> tuple[Subset, Subset, Subset, DataLoader, DataLoader, DataLoader]:
        if self.debug_info:
            print("Loading Training Data...\n")

        if self.debug_info:
            data_distribution = self.dataset.get_distribution()
            print("Full Dataset Overview:")
            total_samples = data_distribution["total_samples"]
            print(f"Total number of samples: {total_samples}\n")
            print("Class Distribution:")
            for label, info in data_distribution["distribution"].items():
                count = info["count"]
                percentage = info["percentage"]
                print(f"{label}: {count} samples ({percentage}%)")
            print("\n")

        # Stratified splitting using sklearn
        # Use 70% of data for training and 30% for validation
        # Create labels array directly from metadata
        n_protons = self.dataset.n_protons
        n_gammas = self.dataset.n_gammas

        # First n_protons items are proton labels (Logic is mirrored magicDataset)
        labels = np.full(n_protons + n_gammas, self.dataset.labels[self.dataset.PROTON_LABEL])
        # Last n_gammas items are gamma labels
        labels[n_protons:] = self.dataset.labels[self.dataset.GAMMA_LABEL]

        # Stratified splitting using sklearn (shuffles indices)
        train_indices, temp_indices = train_test_split(
            np.arange(len(self.dataset)),
            test_size=self.TEMP_DATA_SPLIT,
            stratify=labels,
            shuffle=True,
            random_state=42,
        )

        val_indices, test_indices = train_test_split(
            temp_indices,
            test_size=self.TEST_DATA_SPLIT,
            stratify=labels[temp_indices],
            shuffle=True,
            random_state=42
        )

        if self.debug_info:
            print("\nAfter split label distribution:")
            print("Training set:")
            train_labels = labels[train_indices]
            unique, counts = np.unique(train_labels, return_counts=True)
            for label, count in zip(unique, counts):
                idx_to_label = {v: k for k, v in self.dataset.labels.items()}
                label_name = idx_to_label[label]
                print(f"{label_name} (label {label}): {count} samples ({count / len(train_labels) * 100:.2f}%)")

            print("\nValidation set:")
            val_labels = labels[val_indices]
            unique, counts = np.unique(val_labels, return_counts=True)
            for label, count in zip(unique, counts):
                idx_to_label = {v: k for k, v in self.dataset.labels.items()}
                label_name = idx_to_label[label]
                print(f"{label_name} (label {label}): {count} samples ({count / len(val_labels) * 100:.2f}%)")
            print("\n")

            print("\nTest set:")
            test_labels = labels[test_indices]
            unique, counts = np.unique(test_labels, return_counts=True)
            for label, count in zip(unique, counts):
                idx_to_label = {v: k for k, v in self.dataset.labels.items()}
                label_name = idx_to_label[label]
                print(f"{label_name} (label {label}): {count} samples ({count / len(test_labels) * 100:.2f}%)")
            print("\n")

        train_dataset = Subset(self.dataset, train_indices)
        val_dataset = Subset(self.dataset, val_indices)
        test_dataset = Subset(self.dataset, test_indices)

        training_data_loader = DataLoader(
            train_dataset, batch_size=self.BATCH_SIZE, shuffle=True
        )
        val_data_loader = DataLoader(
            val_dataset, batch_size=self.BATCH_SIZE, shuffle=False
        )
        test_data_loader = DataLoader(
            test_dataset, batch_size=self.BATCH_SIZE, shuffle=False
        )

        if self.debug_info:
            print("Dataset loaded.")
            print("\n")

        return train_dataset, val_dataset, test_dataset, training_data_loader, val_data_loader, test_data_loader

    def load_model(self):
        match self.model_name.lower():
            case "combinednet":
                model = CombinedNet()
            case _:
                raise ValueError(f"Invalid Modelname: '{self.model_name}'")

        if self.debug_info:
            print("Model loaded.")

        return model.to(self.device)

    def calculate_weight_distribution(self):
        total_samples = len(self.dataset)
        n_protons = self.dataset.n_protons
        n_gammas = self.dataset.n_gammas
        weight_proton = total_samples / (2 * n_protons)
        weight_gamma = total_samples / (2 * n_gammas)
        weights_sum = weight_proton + weight_gamma
        weight_proton = weight_proton / weights_sum
        weight_gamma = weight_gamma / weights_sum
        return weight_proton, weight_gamma

    def train_model(self, epochs: int):
        weight_proton, weight_gamma = self.calculate_weight_distribution()
        class_weights = torch.tensor([weight_proton, weight_gamma]).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.LEARNING_RATE,
            weight_decay=self.WEIGHT_DECAY
        )

        steps_per_epoch = len(self.training_data_loader)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.LEARNING_RATE * 2,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
            div_factor=5,
        )

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

        best_validation_accuracy = 0
        for epoch in range(epochs):
            if self.debug_info:
                print(f"Training Epoch {epoch + 1}/{epochs}...")

            train_metrics = self._training_step(optimizer, criterion)
            if self.debug_info:
                print(f"\nTraining Metrics of epoch: {epoch + 1}: \n")
                print_metrics(self.dataset.labels, train_metrics)

            val_metrics = self._validation_step(criterion)
            scheduler.step()

            if self.debug_info:
                print(f"\nValidation Metrics of epoch: {epoch + 1}: \n")
                print_metrics(self.dataset.labels, val_metrics)
                print("-" * 50)

            if val_metrics['accuracy'] > best_validation_accuracy:
                best_validation_accuracy = val_metrics['accuracy']
                torch.save(
                    self.model.state_dict(),
                    self.model_path,
                )

        if self.debug_info:
            self.write_results(epochs)

        """if self.debug_info:
            print("Running Inference on Test Sample")
        metrics = inference(self.test_data_loader, self.dataset.labels, self.model_path)
        self.inference_metrics.append(metrics)"""

    def _extract_batch(self, batch):
        m1_images, m2_images, features, labels = batch

        m1_images = m1_images.to(self.device)
        m2_images = m2_images.to(self.device)
        features = features.to(self.device)
        labels = labels.to(self.device)

        return m1_images, m2_images, features, labels

    def _training_step(self, optimizer: optim.Optimizer, criterion) -> dict[str, float]:
        train_preds = []
        train_labels = []
        train_loss = 0

        self.model.train()
        batch_cntr = 1
        total_batches = len(self.training_data_loader)
        for batch in self.training_data_loader:
            m1_images, m2_images, features, labels = self._extract_batch(batch)

            optimizer.zero_grad()
            outputs = self.model(m1_images, m2_images, features)

            loss = criterion(outputs, labels)

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.GRAD_CLIP_NORM)

            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            train_loss += loss.item()

            if self.debug_info and batch_cntr % 1000 == 0:
                current_accuracy = 100.0 * accuracy_score(train_labels, train_preds)
                print(f"       Accuracy for batch {batch_cntr} of {total_batches}: {current_accuracy} ")
            batch_cntr += 1

        metrics = calc_metrics(
            train_labels,
            train_preds,
            train_loss / len(self.training_data_loader),
        )

        self.train_metrics.append(metrics)
        return metrics

    def _validation_step(self, criterion) -> dict[str, float]:
        val_loss = 0
        val_preds = []
        val_labels = []

        self.model.eval()
        with torch.no_grad():
            for batch in self.val_data_loader:
                m1_images, m2_images, features, labels = self._extract_batch(batch)

                outputs = self.model(m1_images, m2_images, features)

                loss = criterion(outputs, labels)
                _, predicted = outputs.max(1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_loss += loss.item()

        metrics = calc_metrics(
            val_labels,
            val_preds,
            val_loss / len(self.val_data_loader),
        )

        self.validation_metrics.append(metrics)
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
            "inference_metrics": self.inference_metrics,
        }

        writer = ResultsWriter(self.output_dir)
        writer.save_training_results(training_data)
