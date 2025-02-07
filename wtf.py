import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader
from typing import Dict

from CNN.Architectures.StatsModel import get_batch_stats, StatsMagicNet
from TrainingPipeline.MagicDataset import MagicDataset


class MLPActivationAnalyzer:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = StatsMagicNet()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.model = self.model.to(self.device)

        # Store activations and gradients
        self.activations = defaultdict(list)
        self.feature_importance = None
        self.input_features = []

        # Feature names for interpretation
        self.feature_names = [
            'M1 Mean', 'M1 Std', 'M1 Neg Ratio', 'M1 Min', 'M1 Max',
            'M1 Square Mean', 'M1 25%', 'M1 50%', 'M1 75%',
            'M2 Mean', 'M2 Std', 'M2 Neg Ratio', 'M2 Min', 'M2 Max',
            'M2 Square Mean', 'M2 25%', 'M2 50%', 'M2 75%'
        ]

        # Register hooks for all layers
        self._register_hooks()

    def _activation_hook(self, name: str):
        def hook(module, input, output):
            self.activations[name].append(output.detach().cpu())

        return hook

    def _register_hooks(self):
        # Register hooks for each layer in the classifier
        for name, module in self.model.classifier.named_children():
            if isinstance(module, (nn.Linear, nn.ReLU)):
                module.register_forward_hook(self._activation_hook(f"{name}"))

    def analyze_batch(self, dataloader: DataLoader, num_samples: int = 1000) -> Dict:
        """Analyze model behavior on a batch of data"""
        samples_processed = 0
        all_predictions = []
        all_labels = []
        all_inputs = []

        with torch.no_grad():
            for m1, m2, _, labels in dataloader:
                if samples_processed >= num_samples:
                    break

                m1 = m1.to(self.device)
                m2 = m2.to(self.device)

                # Get statistical features
                m1_stats = get_batch_stats(m1)
                m2_stats = get_batch_stats(m2)
                combined_stats = torch.cat([m1_stats, m2_stats], dim=1)

                # Store inputs for analysis
                all_inputs.append(combined_stats.cpu())

                # Get predictions
                outputs = self.model.classifier(combined_stats)
                preds = outputs.argmax(dim=1)

                all_predictions.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

                samples_processed += len(labels)

        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        labels = np.array(all_labels)
        inputs = torch.cat(all_inputs, dim=0).numpy()

        return self._analyze_results(predictions, labels, inputs)

    def _analyze_results(self, predictions: np.ndarray, labels: np.ndarray,
                         inputs: np.ndarray) -> Dict:
        """Analyze the results and compute feature importance"""
        correct_mask = predictions == labels

        results = {
            "accuracy": np.mean(correct_mask) * 100,
            "feature_importance": {},
            "layer_activations": {},
            "feature_correlations": {}
        }

        # Analyze input features
        for i, feature_name in enumerate(self.feature_names):
            # Compute feature importance based on prediction correctness
            correct_mean = inputs[correct_mask, i].mean()
            incorrect_mean = inputs[~correct_mask, i].mean()
            importance = abs(correct_mean - incorrect_mean)

            results["feature_importance"][feature_name] = {
                "correct_mean": correct_mean,
                "incorrect_mean": incorrect_mean,
                "importance": importance
            }

        # Analyze layer activations
        for layer_name, acts in self.activations.items():
            if not acts:  # Skip if no activations stored
                continue

            # Concatenate all batches
            layer_acts = torch.cat(acts, dim=0).numpy()

            # Compute statistics for correct vs incorrect predictions
            correct_acts = layer_acts[correct_mask]
            incorrect_acts = layer_acts[~correct_mask]

            results["layer_activations"][layer_name] = {
                "mean_diff": np.abs(np.mean(correct_acts, axis=0) - np.mean(incorrect_acts, axis=0)).mean(),
                "correct_mean": np.mean(correct_acts),
                "incorrect_mean": np.mean(incorrect_acts),
                "correct_std": np.std(correct_acts),
                "incorrect_std": np.std(incorrect_acts),
                "activation_std": np.std(layer_acts)
            }

        # Compute feature correlations with correctness
        correlations = []
        for i in range(inputs.shape[1]):
            correlation = np.corrcoef(inputs[:, i], correct_mask)[0, 1]
            correlations.append((self.feature_names[i], correlation))

        results["feature_correlations"] = dict(sorted(
            correlations,
            key=lambda x: abs(x[1]),
            reverse=True
        ))

        # Clear stored activations
        self.activations.clear()

        return results


def print_analysis(results: Dict):
    """Print the analysis results in a readable format"""
    print(f"\nModel Accuracy: {results['accuracy']:.2f}%\n")

    print("=== Feature Importance ===")
    sorted_features = sorted(
        results["feature_importance"].items(),
        key=lambda x: x[1]["importance"],
        reverse=True
    )

    for feature_name, stats in sorted_features:
        print(f"\n{feature_name}:")
        print(f"  Importance Score: {stats['importance']:.4f}")
        print(f"  Mean (correct predictions): {stats['correct_mean']:.4f}")
        print(f"  Mean (incorrect predictions): {stats['incorrect_mean']:.4f}")

    print("\n=== Layer Activation Analysis ===")
    for layer_name, stats in results["layer_activations"].items():
        print(f"\n{layer_name}:")
        print(f"  Mean activation difference: {stats['mean_diff']:.4f}")
        print(f"  Activation std: {stats['activation_std']:.4f}")

    print("\n=== Top Feature Correlations with Correctness ===")
    for feature, correlation in list(results["feature_correlations"].items())[:5]:
        print(f"{feature}: {correlation:.4f}")


def main():
    # Initialize dataset and analyzer
    dataset = MagicDataset("magic-protons.parquet", "magic-gammas.parquet")
    analyzer = MLPActivationAnalyzer("trained_model.pth")

    # Create dataloader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Run analysis
    results = analyzer.analyze_batch(loader, num_samples=1000)

    # Print results
    print_analysis(results)


if __name__ == "__main__":
    main()