import torch
from torch.utils.data import DataLoader
from TrainingPipeline.MagicDataset import MagicDataset
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def get_batch_stats(img_batch):
    return torch.stack([
        img_batch.mean(dim=1),
        img_batch.std(dim=1),
        (img_batch < 0).float().mean(dim=1),
        img_batch.min(dim=1).values,
        img_batch.max(dim=1).values,
        (img_batch ** 2).mean(dim=1),
        (img_batch != 0).float().mean(dim=1),
        torch.quantile(img_batch, 0.25, dim=1),
        torch.quantile(img_batch, 0.5, dim=1),
        torch.quantile(img_batch, 0.75, dim=1)
    ], dim=1)


def rule_based_classifier(stats_m1, stats_m2):
    neg_ratio_m1 = stats_m1[2]
    neg_ratio_m2 = stats_m2[2]

    std_m1 = stats_m1[1]
    std_m2 = stats_m2[1]

    max_m1 = stats_m1[4]
    max_m2 = stats_m2[4]

    # Rule 1: Negative ratio threshold (protons have higher neg_ratio)
    neg_ratio_threshold = 0.002  # Based on the statistics we saw
    rule1 = (neg_ratio_m1 < neg_ratio_threshold) and (neg_ratio_m2 < neg_ratio_threshold)

    # Rule 2: Standard deviation threshold (gammas have higher std)
    std_threshold = 9.0  # Based on the statistics we saw
    rule2 = (std_m1 > std_threshold) or (std_m2 > std_threshold)

    # Rule 3: Maximum value threshold (gammas have higher max values)
    max_threshold = 100.0  # Based on the statistics we saw
    rule3 = (max_m1 > max_threshold) or (max_m2 > max_threshold)

    # Combine rules - if any two rules suggest gamma, classify as gamma
    rules_satisfied = sum([rule1, rule2, rule3])
    return 1 if rules_satisfied >= 2 else 0


def evaluate_classifier(dataset):
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    true_labels = []
    predicted_labels = []
    sample_count = 0

    for m1, m2, _, labels in loader:

        # Get statistics for both telescopes
        m1_stats = get_batch_stats(m1)
        m2_stats = get_batch_stats(m2)

        # Make predictions for each sample in batch
        for i in range(len(labels)):
            pred = rule_based_classifier(m1_stats[i], m2_stats[i])

            true_labels.append(labels[i].item())
            predicted_labels.append(pred)

        sample_count += len(labels)
        if sample_count % 1000 == 0:
            print(f"Processed {sample_count} samples")

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    report = classification_report(true_labels, predicted_labels,
                                   target_names=['Proton', 'Gamma'])

    print("\nClassification Report:")
    print(report)

    print("\nConfusion Matrix:")
    print("            Predicted")
    print("             P    G")
    print(f"Actual P  {conf_matrix[0][0]:4d} {conf_matrix[0][1]:4d}")
    print(f"       G  {conf_matrix[1][0]:4d} {conf_matrix[1][1]:4d}")

    # Calculate additional metrics
    tn, fp, fn, tp = conf_matrix.ravel()
    sensitivity = tp / (tp + fn)  # True Positive Rate / Recall for Gammas
    specificity = tn / (tn + fp)  # True Negative Rate for Protons

    print(f"\nOverall Accuracy: {accuracy:.4f}")
    print(f"Sensitivity (Gamma Detection Rate): {sensitivity:.4f}")
    print(f"Specificity (Proton Detection Rate): {specificity:.4f}")

    return accuracy, sensitivity, specificity


def main():
    dataset = MagicDataset("magic-protons.parquet", "magic-gammas.parquet", debug_info=False)

    print("Evaluating rule-based classifier...")
    evaluate_classifier(dataset)


if __name__ == "__main__":
    main()
