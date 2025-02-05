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
from CNN.Architectures.BasicMagicCNN import BasicMagicNet
from TrainingPipeline.ResultsWriter import ResultsWriter
from ParquetJsonTransformation.jsonDataset import jsonDataset

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


def calc_metrics(y_pred, y_true, loss):
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


def _print_split_info(dataset, labels, train_indices, val_indices):
    train_labels = labels[train_indices]
    val_labels = labels[val_indices]

    print("\nSplit Distribution:")
    for split_name, split_labels in [("Training", train_labels), ("Validation", val_labels)]:
        unique, counts = np.unique(split_labels, return_counts=True)
        print(f"\n{split_name} set:")
        for label, count in zip(unique, counts):
            idx_to_label = {v: k for k, v in dataset.labels.items()}
            label_name = idx_to_label[label]
            percent = count / len(split_labels) * 100
            print(f"{label_name} (label {label}): {count} samples ({percent:.2f}%)")


class jsonTrainingSupervisor:
    DATA_TEST_SPLIT: float = 0.3
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.004362359948002615
    ADAM_BETA_1: float = 0.8666112052644459
    ADAM_BETA_2: float = 0.9559910148600463
    ADAM_EPSILON: float = 1e-8
    WEIGHT_DECAY: float = 0.0020743152186914865
    SCHEDULER_MIN_LR: float = 8.603675064364842e-05
    SCHEDULER_MAX_LR: float = 0.00010192925124725547
    SCHEDULER_MODE: Literal["triangular", "triangular2", "exp_range"] = "triangular2"
    SCHEDULER_CYCLE_MOMENTUM: bool = False
    GRAD_CLIP_NORM: float = 4.260615936053168

    def __init__(self, model_name: str, data_dir: str, output_dir: str, debug_info: bool = True,
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

        self.data_dir = data_dir
        self.dataset, self.train_dataset, self.val_dataset, self.training_data_loader, self.val_data_loader \
            = self.load_training_data()

        self.output_dir = output_dir

    def _split_dataset(self, dataset):
        indices = np.arange(len(dataset))

        # Create labels array for stratification
        labels = []
        for idx in indices:
            _, _, _, label = dataset[idx]
            labels.append(label)
        labels = np.array(labels)

        train_indices, val_indices = train_test_split(
            indices,
            test_size=self.DATA_TEST_SPLIT,
            stratify=labels,
            random_state=42
        )

        if self.debug_info:
            _print_split_info(dataset, labels, train_indices, val_indices)

        return Subset(dataset, train_indices), Subset(dataset, val_indices)

    def load_training_data(self) -> tuple[jsonDataset, Subset, Subset, DataLoader, DataLoader]:
        if self.debug_info:
            print("Loading Training Data...\n")

        dataset = jsonDataset(self.data_dir)

        if self.debug_info:
            data_distribution = dataset.get_distribution()
            print("Full Dataset Overview:")
            print(f"Total number of samples: {data_distribution["total_samples"]}\n")
            print("Class Distribution:")
            for label, info in data_distribution["distribution"].items():
                print(f"{label}: {info["count"]} samples ({info["percentage"]:.2f}%)")
            print("\n")

        train_dataset, val_dataset = self._split_dataset(dataset)

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
            case "combinednet":
                model = BasicMagicNet()
            case _:
                raise ValueError(f"Invalid Modelname: '{self.model_name}'")

        if self.debug_info:
            print("Model loaded.")

        return model.to(self.device)

    def train_model(self, epochs: int):
        total_samples = len(self.dataset)
        n_protons = self.dataset.n_protons
        n_gammas = self.dataset.n_gammas
        weight_proton = total_samples / (2 * n_protons)
        weight_gamma = total_samples / (2 * n_gammas)
        class_weights = torch.tensor([weight_proton, weight_gamma]).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

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
            mode=self.SCHEDULER_MODE,
            cycle_momentum=self.SCHEDULER_CYCLE_MOMENTUM
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
                print(f"\nTraining Metrics of epoch: {epoch}: \n")
                self.print_metrics(train_metrics)

            val_metrics = self._validation_step(criterion)
            scheduler.step()

            if self.debug_info:
                print(f"\nValidation Metrics of epoch: {epoch}: \n")
                self.print_metrics(val_metrics)
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

    def _extract_batch(self, batch):
        m1_images, m2_images, features, labels = batch

        m1_images = m1_images.to(self.device)
        m2_images = m2_images.to(self.device)
        features = features.to(self.device)
        labels = labels.to(self.device)

        if (torch.isnan(m1_images).any() or torch.isinf(m1_images).any() or
                torch.isnan(m2_images).any() or torch.isinf(m2_images).any() or
                torch.isnan(features).any() or torch.isinf(features).any()):

            m1_images = torch.nan_to_num(m1_images, 0.0)
            m2_images = torch.nan_to_num(m2_images, 0.0)
            features = torch.nan_to_num(features, 0.0)

        return m1_images, m2_images, features, labels

    def _training_step(self, optimizer: optim.Optimizer, criterion) -> dict[str, float]:
        train_preds = []
        train_labels = []
        train_loss = 0

        self.model.train()
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

    def print_metrics(self, metrics: MetricsDict):
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
        print(f"Label mapping: {self.dataset.labels}")

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
        }

        writer = ResultsWriter(self.output_dir)
        writer.save_training_results(training_data)
