import os
import sys
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader
from TrainingPipeline.MagicDataset import MagicDataset
from sklearn.metrics import accuracy_score, confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CNN.Architectures.StatsModel import get_batch_stats
from TrainingPipeline.MagicDataset import collect_statistics


def combine_results(m1_result, m2_result):
    if m1_result == m2_result:
        return m1_result

    if m1_result != -1 and m2_result == -1:
        return m1_result
    if m2_result != -1 and m1_result == -1:
        return m2_result

    return -1


class OptimizedClassifier:
    def __init__(self, telescope_stats):
        self.m1_proton_stats = telescope_stats["m1_proton"]
        self.m1_gamma_stats = telescope_stats["m1_gamma"]
        self.m2_proton_stats = telescope_stats["m2_proton"]
        self.m2_gamma_stats = telescope_stats["m2_gamma"]

        self.m1_proton_min = self.m1_proton_stats.min(axis=0)
        self.m1_proton_max = self.m1_proton_stats.max(axis=0)
        self.m1_gamma_min = self.m1_gamma_stats.min(axis=0)
        self.m1_gamma_max = self.m1_gamma_stats.max(axis=0)

        self.m2_proton_min = self.m2_proton_stats.min(axis=0)
        self.m2_proton_max = self.m2_proton_stats.max(axis=0)
        self.m2_gamma_min = self.m2_gamma_stats.min(axis=0)
        self.m2_gamma_max = self.m2_gamma_stats.max(axis=0)

        self.m1_proton_means = self.m1_proton_stats.mean(axis=0)
        self.m1_proton_stds = self.m1_proton_stats.std(axis=0)
        self.m1_gamma_means = self.m1_gamma_stats.mean(axis=0)
        self.m1_gamma_stds = self.m1_gamma_stats.std(axis=0)

        self.m2_proton_means = self.m2_proton_stats.mean(axis=0)
        self.m2_proton_stds = self.m2_proton_stats.std(axis=0)
        self.m2_gamma_means = self.m2_gamma_stats.mean(axis=0)
        self.m2_gamma_stds = self.m2_gamma_stats.std(axis=0)

        self.m1_proton_std_lower = self.m1_proton_means - self.m1_proton_stds
        self.m1_proton_std_upper = self.m1_proton_means + self.m1_proton_stds
        self.m1_gamma_std_lower = self.m1_gamma_means - self.m1_gamma_stds
        self.m1_gamma_std_upper = self.m1_gamma_means + self.m1_gamma_stds

        self.m2_proton_std_lower = self.m2_proton_means - self.m2_proton_stds
        self.m2_proton_std_upper = self.m2_proton_means + self.m2_proton_stds
        self.m2_gamma_std_lower = self.m2_gamma_means - self.m2_gamma_stds
        self.m2_gamma_std_upper = self.m2_gamma_means + self.m2_gamma_stds

        self.decision_stats = {
            'minmax_m1': defaultdict(int),
            'minmax_m2': defaultdict(int),
            'std_m1': defaultdict(int),
            'std_m2': defaultdict(int),
            'metric_decisions': defaultdict(int)
        }

        self.metric_names = [
            'mean', 'std', 'neg_ratio', 'min_val', 'max_val',
            'squared_mean', 'unused', 'q25', 'q50', 'q75'
        ]

    def classify_m1_minmax(self, stats):
        stats = np.array(stats)

        gamma_violations = (stats < self.m1_gamma_min) | (stats > self.m1_gamma_max)
        proton_violations = (stats < self.m1_proton_min) | (stats > self.m1_proton_max)

        if np.any(gamma_violations):
            for idx, is_violation in enumerate(gamma_violations):
                if is_violation:
                    self.decision_stats['metric_decisions'][f'M1_minmax_proton_{self.metric_names[idx]}'] += 1
            self.decision_stats['minmax_m1']['proton'] += 1
            return 0

        if np.any(proton_violations):
            for idx, is_violation in enumerate(proton_violations):
                if is_violation:
                    self.decision_stats['metric_decisions'][f'M1_minmax_gamma_{self.metric_names[idx]}'] += 1
            self.decision_stats['minmax_m1']['gamma'] += 1
            return 1

        self.decision_stats['minmax_m1']['uncertain'] += 1
        return -1

    def classify_m2_minmax(self, stats):
        stats = np.array(stats)

        gamma_violations = (stats < self.m2_gamma_min) | (stats > self.m2_gamma_max)
        proton_violations = (stats < self.m2_proton_min) | (stats > self.m2_proton_max)

        if np.any(gamma_violations):
            for idx, is_violation in enumerate(gamma_violations):
                if is_violation:
                    self.decision_stats['metric_decisions'][f'M2_minmax_proton_{self.metric_names[idx]}'] += 1
            self.decision_stats['minmax_m2']['proton'] += 1
            return 0

        if np.any(proton_violations):
            for idx, is_violation in enumerate(proton_violations):
                if is_violation:
                    self.decision_stats['metric_decisions'][f'M2_minmax_gamma_{self.metric_names[idx]}'] += 1
            self.decision_stats['minmax_m2']['gamma'] += 1
            return 1

        self.decision_stats['minmax_m2']['uncertain'] += 1
        return -1

    def classify_m1_std(self, stats):
        stats = np.array(stats)

        proton_matches = (stats >= self.m1_proton_std_lower) & (stats <= self.m1_proton_std_upper)
        gamma_matches = (stats >= self.m1_gamma_std_lower) & (stats <= self.m1_gamma_std_upper)

        proton_count = np.sum(proton_matches)
        gamma_count = np.sum(gamma_matches)

        if proton_count > gamma_count:
            for idx, is_match in enumerate(proton_matches):
                if is_match:
                    self.decision_stats['metric_decisions'][f'M1_std_proton_{self.metric_names[idx]}'] += 1
            self.decision_stats['std_m1']['proton'] += 1
            return 0
        elif gamma_count > proton_count:
            for idx, is_match in enumerate(gamma_matches):
                if is_match:
                    self.decision_stats['metric_decisions'][f'M1_std_gamma_{self.metric_names[idx]}'] += 1
            self.decision_stats['std_m1']['gamma'] += 1
            return 1

        self.decision_stats['std_m1']['uncertain'] += 1
        return -1

    def classify_m2_std(self, stats):
        stats = np.array(stats)

        proton_matches = (stats >= self.m2_proton_std_lower) & (stats <= self.m2_proton_std_upper)
        gamma_matches = (stats >= self.m2_gamma_std_lower) & (stats <= self.m2_gamma_std_upper)

        proton_count = np.sum(proton_matches)
        gamma_count = np.sum(gamma_matches)

        if proton_count > gamma_count:
            for idx, is_match in enumerate(proton_matches):
                if is_match:
                    self.decision_stats['metric_decisions'][f'M2_std_proton_{self.metric_names[idx]}'] += 1
            self.decision_stats['std_m2']['proton'] += 1
            return 0
        elif gamma_count > proton_count:
            for idx, is_match in enumerate(gamma_matches):
                if is_match:
                    self.decision_stats['metric_decisions'][f'M2_std_gamma_{self.metric_names[idx]}'] += 1
            self.decision_stats['std_m2']['gamma'] += 1
            return 1

        self.decision_stats['std_m2']['uncertain'] += 1
        return -1

    def classify_minmax(self, stats_m1, stats_m2):
        m1_result = self.classify_m1_minmax(stats_m1)
        m2_result = self.classify_m2_minmax(stats_m2)
        return combine_results(m1_result, m2_result)

    def classify_std(self, stats_m1, stats_m2):
        m1_result = self.classify_m1_std(stats_m1)
        m2_result = self.classify_m2_std(stats_m2)
        return combine_results(m1_result, m2_result)

    def print_decision_statistics(self):
        print("\nClassification Decision Statistics:")
        print("\nMin-Max Classification Decisions:")
        for telescope in ['minmax_m1', 'minmax_m2']:
            print(f"\n{telescope.upper()}:")
            total = sum(self.decision_stats[telescope].values())
            for decision, count in self.decision_stats[telescope].items():
                print(f"  {decision}: {count} ({count / total * 100:.2f}%)")

        print("\nStandard Deviation Classification Decisions:")
        for telescope in ['std_m1', 'std_m2']:
            print(f"\n{telescope.upper()}:")
            total = sum(self.decision_stats[telescope].values())
            for decision, count in self.decision_stats[telescope].items():
                print(f"  {decision}: {count} ({count / total * 100:.2f}%)")

        print("\nMost Influential Metrics in Decisions:")
        metrics_sorted = sorted(self.decision_stats['metric_decisions'].items(),
                                key=lambda x: x[1], reverse=True)
        total_metric_decisions = sum(count for _, count in metrics_sorted)

        print("\nTop 20 most influential metrics:")
        for metric, count in metrics_sorted[:20]:
            print(f"  {metric}: {count} ({count / total_metric_decisions * 100:.2f}%)")


def evaluate_classifier_with_certainty(dataset, dataset_stats):
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    classifier = OptimizedClassifier(dataset_stats)

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
            pred = classifier.classify_minmax(m1_stats[i], m2_stats[i])
            if pred == -1:
                pred = classifier.classify_std(m1_stats[i], m2_stats[i])

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
    classifier.print_decision_statistics()

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
    print(f"Gathering Statistics")
    dataset_stats = collect_statistics(dataset)
    print("Evaluating rule-based classifier...")
    evaluate_classifier_with_certainty(dataset, dataset_stats)


if __name__ == "__main__":
    main()
