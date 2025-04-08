import os
import datetime
from typing import Literal, TypedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from TrainingPipeline.Datasets import MagicDataset
from TrainingPipeline.ResultsWriter import ResultsWriter

from CNN.Architectures import *

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    if "proton_loss" in metrics and "gamma_loss" in metrics:
        print(f"{'Proton Loss:':<12} {metrics['proton_loss']:>6.4f}")
        print(f"{'Gamma Loss:':<12} {metrics['gamma_loss']:>6.4f}")

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


def worker_init_fn(worker_id):
    np.random.seed(42 + worker_id)


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class TrainingSupervisor:
    VAL_SPLIT: float = 0.3
    
    # Params from Maxis Branch
    LEARNING_RATE = 5.269632147047427e-06
    WEIGHT_DECAY = 0.00034049323130326087
    BATCH_SIZE = 64
    GRAD_CLIP_NORM = 0.7168560391358462
    
    # LEARNING_RATE = 1e-4
    # WEIGHT_DECAY = 1e-4
    # BATCH_SIZE = 64
    # GRAD_CLIP_NORM = 1.0
    SCHEDULER_MODE: Literal["triangular", "triangular2", "exp_range"] = "triangular2"
    SCHEDULER_CYCLE_MOMENTUM: bool = False
    SCHEDULER_STEP_SIZE = 4
    SCHEDULER_BASE_LR = 1e-4
    SCHEDULER_MAX_LR = 1e-2

    def __init__(self, model_name: str, dataset: MagicDataset, output_dir: str, debug_info: bool = True,
                 save_model: bool = False, save_debug_data: bool = True, early_stopping: bool = True) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available()
                                   else "cpu")
        self.debug_info = debug_info
        self.save_model = save_model
        self.save_debug_data = save_debug_data
        self.early_stopping = early_stopping

        if self.save_debug_data or self.save_model:
            os.makedirs(output_dir, exist_ok=True)

        if debug_info:
            print(f"Training on Device: {self.device}")

        self.model_name = model_name
        self.model = self.load_model()
        self.validation_metrics = []
        self.train_metrics = []
        self.inference_metrics = []

        self.dataset = dataset
        (self.train_dataset, self.val_dataset, self.training_data_loader, self.val_data_loader) = (
            self.load_training_data())

        self.output_dir = output_dir
        self.model_path = os.path.join(self.output_dir, "trained_model.pth")

    def load_training_data(self) -> tuple[Subset, Subset, DataLoader, DataLoader]:
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
        # Get labels from dataset
        labels = self.dataset.get_all_labels()

        # Stratified splitting using sklearn (shuffles indices)
        train_indices, val_indices = train_test_split(
            np.arange(len(self.dataset)),
            test_size=self.VAL_SPLIT,
            stratify=labels,
            shuffle=True,
            random_state=42,
        )

        """val_indices, test_indices = train_test_split(
            temp_indices,
            test_size=self.TEST_DATA_SPLIT,
            stratify=labels[temp_indices],
            shuffle=True,
            random_state=42
        )"""

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

            """print("\nTest set:")
            test_labels = labels[test_indices]
            unique, counts = np.unique(test_labels, return_counts=True)
            for label, count in zip(unique, counts):
                idx_to_label = {v: k for k, v in self.dataset.labels.items()}
                label_name = idx_to_label[label]
                print(f"{label_name} (label {label}): {count} samples ({count / len(test_labels) * 100:.2f}%)")
            print("\n")"""

        train_dataset = Subset(self.dataset, train_indices)
        val_dataset = Subset(self.dataset, val_indices)
        # test_dataset = Subset(self.dataset, test_indices)

        generator = torch.Generator()
        generator.manual_seed(42)

        training_data_loader = DataLoader(
            train_dataset, batch_size=self.BATCH_SIZE, shuffle=True, generator=generator, worker_init_fn=worker_init_fn
        )
        val_data_loader = DataLoader(
            val_dataset, batch_size=self.BATCH_SIZE, shuffle=False, generator=generator, worker_init_fn=worker_init_fn
        )

        if self.debug_info:
            print("Dataset loaded.")
            print("\n")

        return train_dataset, val_dataset, training_data_loader, val_data_loader

    def load_model(self):
        match self.model_name.lower():
            case "basicmagicnet":
                model = BasicMagicNet()
            case "mlp":
                model = MLP()
            case "statsmagicnet":
                model = StatsMagicNet()
            case "hexcirclenet":
                model = HexCircleNet()
            case "hexmagicnet":
                model = HexMagicNet()
            case "hexagdlynet":
                model = HexagdlyNet()
            case "simple1dnet":
                model = Simple1dNet()
            case "custom":
                model = nn.Module()
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
        self.training_start_time = datetime.datetime.now()
        weight_proton, weight_gamma = self.calculate_weight_distribution()
        class_weights = torch.tensor([weight_proton, weight_gamma]).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.LEARNING_RATE,
            weight_decay=self.WEIGHT_DECAY
        )

        scheduler = optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=self.SCHEDULER_BASE_LR,
            max_lr=self.SCHEDULER_MAX_LR,
            step_size_up=self.SCHEDULER_STEP_SIZE,
            mode=self.SCHEDULER_MODE,
            cycle_momentum=self.SCHEDULER_CYCLE_MOMENTUM
        )

        early_stopping = EarlyStopping(patience=3, min_delta=0.0001)

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
                if self.save_model:
                    torch.save(
                        self.model.state_dict(),
                        self.model_path,
                    )

            if self.early_stopping:
                early_stopping(val_metrics['loss'])
                if early_stopping.early_stop:
                    if self.debug_info:
                        print(f"Early stopping triggered at epoch {epoch + 1}")
                    break
                
        self.training_duration = datetime.datetime.now() - self.training_start_time
        if self.debug_info:
            print(f"Total Training duration: {str(self.training_duration)}")

        if self.save_debug_data:
            self.write_results(epoch + 1)

    def _extract_batch(self, batch):
        *data, labels, _ = batch

        return [d.to(self.device) for d in data], labels.to(self.device)

    def _training_step(self, optimizer: optim.Optimizer, criterion) -> dict[str, float]:
        train_preds = []
        train_labels = []
        train_loss = 0

        # New accumulators for per-class loss
        proton_loss_total = 0.0
        gamma_loss_total = 0.0
        count_proton = 0
        count_gamma = 0

        # Create a criterion with reduction='none' to obtain per-sample losses
        criterion_none = nn.CrossEntropyLoss(
            label_smoothing=criterion.label_smoothing, 
            reduction='none'
        )

        self.model.train()
        batch_cntr = 1
        total_batches = len(self.training_data_loader)
        for batch in self.training_data_loader:
            data, labels = self._extract_batch(batch)

            optimizer.zero_grad()
            outputs = self.model(*data)

            loss = criterion(outputs, labels)
            # Compute per-sample losses
            per_sample_losses = criterion_none(outputs, labels)
            # Accumulate losses per class
            for loss_val, label in zip(per_sample_losses, labels):
                if label.item() == self.dataset.labels[self.dataset.PROTON_LABEL]:
                    proton_loss_total += loss_val.item()
                    count_proton += 1
                else:
                    gamma_loss_total += loss_val.item()
                    count_gamma += 1

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
        # Add per-class losses to metrics
        metrics["proton_loss"] = proton_loss_total / count_proton if count_proton > 0 else 0.0
        metrics["gamma_loss"] = gamma_loss_total / count_gamma if count_gamma > 0 else 0.0

        self.train_metrics.append(metrics)
        return metrics

    def _validation_step(self, criterion) -> dict[str, float]:
        val_loss = 0
        val_preds = []
        val_labels = []

        # New accumulators for per-class loss in validation
        proton_loss_total = 0.0
        gamma_loss_total = 0.0
        count_proton = 0
        count_gamma = 0

        # Create a criterion with reduction='none' for per-sample loss calculation
        criterion_none = nn.CrossEntropyLoss(
            label_smoothing=criterion.label_smoothing, 
            reduction='none'
        )

        self.model.eval()
        with torch.no_grad():
            for batch in self.val_data_loader:
                data, labels = self._extract_batch(batch)

                outputs = self.model(*data)

                loss = criterion(outputs, labels)
                per_sample_losses = criterion_none(outputs, labels)
                
                for loss_val, label in zip(per_sample_losses, labels):
                    if label.item() == self.dataset.labels[self.dataset.PROTON_LABEL]:
                        proton_loss_total += loss_val.item()
                        count_proton += 1
                    else:
                        gamma_loss_total += loss_val.item()
                        count_gamma += 1

                _, predicted = outputs.max(1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_loss += loss.item()

        metrics = calc_metrics(
            val_labels,
            val_preds,
            val_loss / len(self.val_data_loader),
        )
        # Add per-class losses to validation metrics
        metrics["proton_loss"] = proton_loss_total / count_proton if count_proton > 0 else 0.0
        metrics["gamma_loss"] = gamma_loss_total / count_gamma if count_gamma > 0 else 0.0

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
            "time": {
                "total": str(self.training_duration),
                "avg_per_epoch": str(self.training_duration / epochs),
            },
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