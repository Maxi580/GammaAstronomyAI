from TrainingPipeline.MagicDataset import MagicDataset


def evaluate_simple_classifier(dataset, n_samples=1000):
    threshold = 0.008
    correct = 0

    for idx in range(n_samples):
        m1, _, _, true_label = dataset[idx]
        neg_ratio = (m1 < 0).float().mean().item()
        pred_label = dataset.labels['proton'] if neg_ratio < threshold else dataset.labels['gamma']
        correct += (pred_label == true_label)

    print(f"Accuracy on {n_samples} samples: {(correct / n_samples) * 100:.2f}%")


if __name__ == '__main__':
    dataset = MagicDataset("magic-protons.parquet", "magic-gammas.parquet", debug_info=False)
    evaluate_simple_classifier(dataset)