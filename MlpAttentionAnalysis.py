import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader
from typing import Dict

from CNN.Architectures.StatsModel import get_batch_stats, StatsMagicNet
from TrainingPipeline.Datasets.MagicDataset import MagicDataset


class StatsMagicNetAnalyzer:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = StatsMagicNet()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()
        self.model = self.model.to(self.device)

        self.activations = defaultdict(list)
        self.stats = defaultdict(list)

        self.feature_names = [
            'M1 Mean', 'M1 Std', 'M1 Neg Ratio', 'M1 Min', 'M1 Max',
            'M1 Square Mean', 'M1 NonZero Ratio', 'M1 25%', 'M1 50%', 'M1 75%',
            'M2 Mean', 'M2 Std', 'M2 Neg Ratio', 'M2 Min', 'M2 Max',
            'M2 Square Mean', 'M2 NonZero Ratio', 'M2 25%', 'M2 50%', 'M2 75%'
        ]

        self._register_hooks()

    def _activation_hook(self, name: str):
        def hook(module, input, output):
            self.activations[name].append(output.detach().cpu())

        return hook

    def _register_hooks(self):
        for name, module in self.model.classifier.named_children():
            if isinstance(module, (nn.Linear, nn.ReLU)):
                module.register_forward_hook(self._activation_hook(f"{name}"))

    def analyze_batch(self, dataloader: DataLoader) -> Dict:
        samples_processed = 0
        all_predictions = []
        all_labels = []
        all_stats = []

        with torch.no_grad():
            for m1, m2, _, labels in dataloader:
                m1 = m1.to(self.device)
                m2 = m2.to(self.device)

                m1_stats = get_batch_stats(m1)
                m2_stats = get_batch_stats(m2)
                combined_stats = torch.cat([m1_stats, m2_stats], dim=1)
                all_stats.append(combined_stats.cpu())

                outputs = self.model(m1, m2, None)
                preds = outputs.argmax(dim=1)

                all_predictions.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

                samples_processed += len(labels)

        predictions = np.array(all_predictions)
        labels = np.array(all_labels)
        stats = torch.cat(all_stats, dim=0).numpy()

        return self._analyze_results(predictions, labels, stats)

    def _analyze_results(self, predictions: np.ndarray, labels: np.ndarray,
                         inputs: np.ndarray) -> Dict:
        # Split data by true class
        proton_mask = labels == 0
        gamma_mask = labels == 1

        proton_inputs = inputs[proton_mask]
        gamma_inputs = inputs[gamma_mask]
        proton_preds = predictions[proton_mask]
        gamma_preds = predictions[gamma_mask]

        # Calculate per-class accuracy
        proton_accuracy = np.mean(proton_preds == labels[proton_mask]) * 100
        gamma_accuracy = np.mean(gamma_preds == labels[gamma_mask]) * 100

        results = {
            "class_accuracy": {
                "proton": proton_accuracy,
                "gamma": gamma_accuracy,
                "balanced": (proton_accuracy + gamma_accuracy) / 2
            },
            "feature_importance": {},
            "layer_activations": {},
            "per_class_stats": {
                "proton": {},
                "gamma": {}
            }
        }

        # Analyze features
        for i, feature_name in enumerate(self.feature_names):
            proton_correct = proton_inputs[proton_preds == 0][:, i]
            proton_incorrect = proton_inputs[proton_preds == 1][:, i]
            gamma_correct = gamma_inputs[gamma_preds == 1][:, i]
            gamma_incorrect = gamma_inputs[gamma_preds == 0][:, i]

            proton_importance = abs(np.mean(proton_correct) - np.mean(proton_incorrect))
            gamma_importance = abs(np.mean(gamma_correct) - np.mean(gamma_incorrect))

            results["per_class_stats"]["proton"][feature_name] = {
                "correct_mean": np.mean(proton_correct),
                "incorrect_mean": np.mean(proton_incorrect),
                "importance": proton_importance
            }

            results["per_class_stats"]["gamma"][feature_name] = {
                "correct_mean": np.mean(gamma_correct),
                "incorrect_mean": np.mean(gamma_incorrect),
                "importance": gamma_importance
            }

            results["feature_importance"][feature_name] = {
                "balanced_importance": (proton_importance + gamma_importance) / 2,
                "proton_importance": proton_importance,
                "gamma_importance": gamma_importance
            }

        # Analyze layer activations
        for layer_name, acts in self.activations.items():
            layer_acts = torch.cat(acts, dim=0).numpy()

            proton_acts = layer_acts[proton_mask]
            gamma_acts = layer_acts[gamma_mask]

            results["layer_activations"][layer_name] = {
                "proton": {
                    "mean": np.mean(proton_acts),
                    "std": np.std(proton_acts)
                },
                "gamma": {
                    "mean": np.mean(gamma_acts),
                    "std": np.std(gamma_acts)
                }
            }

        self.activations.clear()
        return results


def print_analysis(results: Dict):
    """Print the analysis results with class-balanced metrics"""
    print("\n=== Model Accuracy ===")
    print(f"Proton Accuracy: {results['class_accuracy']['proton']:.2f}%")
    print(f"Gamma Accuracy: {results['class_accuracy']['gamma']:.2f}%")
    print(f"Balanced Accuracy: {results['class_accuracy']['balanced']:.2f}%")

    print("\n=== Feature Importance (Class-Balanced) ===")
    sorted_features = sorted(
        results["feature_importance"].items(),
        key=lambda x: x[1]["balanced_importance"],
        reverse=True
    )

    for feature_name, importance in sorted_features:
        print(f"\n{feature_name}:")
        print(f"  Balanced Importance: {importance['balanced_importance']:.4f}")
        print(f"  Proton Importance: {importance['proton_importance']:.4f}")
        print(f"  Gamma Importance: {importance['gamma_importance']:.4f}")

        proton_stats = results["per_class_stats"]["proton"][feature_name]
        gamma_stats = results["per_class_stats"]["gamma"][feature_name]

        print(f"  Proton - Correct Mean: {proton_stats['correct_mean']:.4f}")
        print(f"  Proton - Incorrect Mean: {proton_stats['incorrect_mean']:.4f}")
        print(f"  Gamma - Correct Mean: {gamma_stats['correct_mean']:.4f}")
        print(f"  Gamma - Incorrect Mean: {gamma_stats['incorrect_mean']:.4f}")

    print("\n=== Layer Activation Analysis (Per Class) ===")
    for layer_name, stats in results["layer_activations"].items():
        print(f"\n{layer_name}:")
        print(f"  Proton - Mean: {stats['proton']['mean']:.4f}, Std: {stats['proton']['std']:.4f}")
        print(f"  Gamma - Mean: {stats['gamma']['mean']:.4f}, Std: {stats['gamma']['std']:.4f}")


def main():
    dataset = MagicDataset("magic-protons.parquet", "magic-gammas.parquet")
    analyzer = StatsMagicNetAnalyzer("trained_model.pth")

    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    results = analyzer.analyze_batch(loader)

    print_analysis(results)


if __name__ == "__main__":
    main()