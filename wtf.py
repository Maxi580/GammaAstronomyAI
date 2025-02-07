from TrainingPipeline.MagicDataset import MagicDataset


def evaluate_simple_classifier(dataset):
    threshold = 0.007
    correct = 0

    for idx in range(len(dataset)):
        m1, _, _, true_label = dataset[idx]
        neg_ratio = (m1 < 0).float().mean().item()
        pred_label = dataset.labels['proton'] if neg_ratio < threshold else dataset.labels['gamma']
        correct += (pred_label == true_label)

    print(f"Accuracy on {len(dataset)} samples: {(correct / len(dataset)) * 100:.2f}%")


if __name__ == '__main__':
    dataset = MagicDataset("magic-protons.parquet", "magic-gammas.parquet", debug_info=False)
    evaluate_simple_classifier(dataset)
