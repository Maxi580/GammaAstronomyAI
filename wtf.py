from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from TrainingPipeline.MagicDataset import MagicDataset
from CNN.Architectures.StatsModel import get_batch_stats


def threshold_classifier(m2_square_mean: float, m2_threshold: float) -> int:
    """
        0 for proton, 1 for gamma
    """
    if m2_square_mean > m2_threshold:
        return 1
    else:
        return 0


def evaluate_threshold_classifier(dataloader: DataLoader, m2_threshold: float,
                                  num_samples: int = 10000) -> Dict:
    samples_processed = 0
    all_predictions = []
    all_labels = []

    for _, m2, _, labels in dataloader:
        if samples_processed >= num_samples:
            break

        m2_stats = get_batch_stats(m2)

        m2_square_mean = m2_stats[:, 5].numpy()

        predictions = [
            threshold_classifier(m2_sq, m2_threshold)
            for m2_sq in m2_square_mean
        ]

        all_predictions.extend(predictions)
        all_labels.extend(labels.numpy())
        samples_processed += len(labels)

    predictions = np.array(all_predictions)
    labels = np.array(all_labels)

    proton_mask = labels == 0
    gamma_mask = labels == 1

    proton_accuracy = np.mean(predictions[proton_mask] == labels[proton_mask]) * 100
    gamma_accuracy = np.mean(predictions[gamma_mask] == labels[gamma_mask]) * 100
    balanced_accuracy = (proton_accuracy + gamma_accuracy) / 2

    return {
        "proton_accuracy": proton_accuracy,
        "gamma_accuracy": gamma_accuracy,
        "balanced_accuracy": balanced_accuracy
    }


def main():
    dataset = MagicDataset("magic-protons.parquet", "magic-gammas.parquet")
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    m2_threshold = 691

    results = evaluate_threshold_classifier(loader, m2_threshold, num_samples=10000)

    print("=== Threshold-Based Classifier Results ===")
    print(f"Proton Accuracy: {results['proton_accuracy']:.2f}%")
    print(f"Gamma Accuracy: {results['gamma_accuracy']:.2f}%")
    print(f"Balanced Accuracy: {results['balanced_accuracy']:.2f}%")


if __name__ == "__main__":
    main()
