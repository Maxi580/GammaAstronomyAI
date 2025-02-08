import os
import sys
from torch.utils.data import DataLoader
from TrainingPipeline.MagicDataset import MagicDataset
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CNN.Architectures.StatsModel import get_batch_stats
from wtf.minmaxclassifier import rule_based_minmax_classifier


def evaluate_classifier_with_certainty(dataset):
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    true_labels_certain = []
    predicted_labels_certain = []

    certain = 0
    uncertain = 0
    total_samples = 0

    for m1, m2, _, labels in loader:
        m1_stats = get_batch_stats(m1)
        m2_stats = get_batch_stats(m2)

        for i in range(len(labels)):
            total_samples += 1
            pred = rule_based_minmax_classifier(m1_stats[i], m2_stats[i])

            if pred != -1:
                certain += 1
                true_labels_certain.append(labels[i].item())
                predicted_labels_certain.append(pred)
            else:
                uncertain += 1

        if total_samples % 10000 == 0:
            print(f"Processed {total_samples} samples")

    print(f"\nTotal samples: {total_samples}")
    print(f"Certain: {certain} ({certain / total_samples * 100:.2f}%)")
    print(f"Uncertain: {uncertain} ({uncertain / total_samples * 100:.2f}%)")

    if true_labels_certain:
        accuracy = accuracy_score(true_labels_certain, predicted_labels_certain)
        conf_matrix = confusion_matrix(true_labels_certain, predicted_labels_certain)

        print(f"\nAccuracy (certain predictions only): {accuracy:.4f}")
        print("\nConfusion Matrix (certain predictions only):")
        print("            Predicted")
        print("             P    G")
        print(f"Actual P  {conf_matrix[0][0]:4d} {conf_matrix[0][1]:4d}")
        print(f"       G  {conf_matrix[1][0]:4d} {conf_matrix[1][1]:4d}")


def main():
    dataset = MagicDataset("magic-protons.parquet", "magic-gammas.parquet", debug_info=False)
    print("Evaluating rule-based classifier...")
    evaluate_classifier_with_certainty(dataset)


if __name__ == "__main__":
    main()
