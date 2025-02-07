import torch
from torch.utils.data import DataLoader, Subset
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from typing import Tuple
from TrainingPipeline.MagicDataset import MagicDataset


def get_batch_stats(img_batch):
    return torch.stack([
        img_batch.mean(dim=1),
        img_batch.std(dim=1),
        (img_batch < 0).float().mean(dim=1),
        img_batch.min(dim=1).values,
        img_batch.max(dim=1).values,
        (img_batch ** 2).mean(dim=1),
        torch.quantile(img_batch, 0.25, dim=1),
        torch.quantile(img_batch, 0.5, dim=1),
        torch.quantile(img_batch, 0.75, dim=1)
    ], dim=1)


def prepare_data(loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    all_features = []
    all_labels = []

    for m1, m2, _, labels in loader:
        m1_stats = get_batch_stats(m1)
        m2_stats = get_batch_stats(m2)

        combined_stats = torch.cat([m1_stats, m2_stats], dim=1)

        all_features.append(combined_stats)
        all_labels.append(labels)

    features = torch.cat(all_features, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()

    return features, labels


class StatsSVM:
    def __init__(self, kernel='rbf'):
        self.svm = SVC(kernel=kernel, probability=True)

    def fit(self, train_loader: DataLoader) -> None:
        print("Preparing training data...")
        X_train, y_train = prepare_data(train_loader)

        print("Training SVM...")
        self.svm.fit(X_train, y_train)

    def evaluate(self, val_loader: DataLoader) -> dict:
        print("Preparing validation data...")
        X_val, y_val = prepare_data(val_loader)

        print("Making predictions...")
        predictions = self.svm.predict(X_val)
        probabilities = self.svm.predict_proba(X_val)

        accuracy = accuracy_score(y_val, predictions)
        report = classification_report(y_val, predictions, output_dict=True)

        print("\nResults:")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print("\nDetailed Classification Report:")
        print(classification_report(y_val, predictions))

        return {
            'accuracy': accuracy,
            'report': report,
            'predictions': predictions,
            'probabilities': probabilities,
            'true_labels': y_val
        }


def main():
    print("Loading dataset...")
    dataset = MagicDataset("magic-protons.parquet", "magic-gammas.parquet")

    print("Creating stratified split...")

    n_protons = dataset.n_protons
    n_gammas = dataset.n_gammas
    labels = np.full(n_protons + n_gammas, dataset.labels[dataset.PROTON_LABEL])
    labels[n_protons:] = dataset.labels[dataset.GAMMA_LABEL]

    # Perform stratified split
    indices = np.arange(len(dataset))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # Print distribution information
    train_labels = labels[train_indices]
    val_labels = labels[val_indices]

    print("\nData Distribution:")
    print("Training set:")
    print(f"Protons: {np.sum(train_labels == 0)} ({np.mean(train_labels == 0) * 100:.1f}%)")
    print(f"Gammas: {np.sum(train_labels == 1)} ({np.mean(train_labels == 1) * 100:.1f}%)")
    print("\nValidation set:")
    print(f"Protons: {np.sum(val_labels == 0)} ({np.mean(val_labels == 0) * 100:.1f}%)")
    print(f"Gammas: {np.sum(val_labels == 1)} ({np.mean(val_labels == 1) * 100:.1f}%)")

    # Create and train SVM
    print("\nInitializing SVM classifier...")
    svm_classifier = StatsSVM(kernel='rbf')

    print("\nTraining SVM...")
    svm_classifier.fit(train_loader)

    print("\nEvaluating SVM...")
    results = svm_classifier.evaluate(val_loader)


if __name__ == "__main__":
    main()
