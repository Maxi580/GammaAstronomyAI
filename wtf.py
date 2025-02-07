import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from CNN.Architectures.StatsModel import StatsMagicNet
from TrainingPipeline.MagicDataset import MagicDataset
import numpy as np


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


class ActivationAnalyzer(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.classifier = original_model.classifier
        self.activations = {}

        for name, module in self.classifier.named_modules():
            if isinstance(module, nn.ReLU):
                module.register_forward_hook(self._get_activation_hook(name))

    def _get_activation_hook(self, name):
        def hook(module, input, output):
            self.activations[name] = output.detach()

        return hook

    def forward(self, m1_image, m2_image, measurement_features):
        m1_stats = get_batch_stats(m1_image)
        m2_stats = get_batch_stats(m2_image)
        combined_stats = torch.cat([m1_stats, m2_stats], dim=1)
        return self.classifier(combined_stats), combined_stats


def main():
    dataset = MagicDataset("magic-protons.parquet", "magic-gammas.parquet")
    model = StatsMagicNet()
    model.load_state_dict(torch.load("trained_model.pth"))
    model.eval()

    analyzer = ActivationAnalyzer(model)

    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    samples_processed = 0
    correct = 0

    all_activations = {
        'correct': {},
        'incorrect': {}
    }

    inputs_list = []

    print("Analyzing samples...")
    with torch.no_grad():
        for m1, m2, _, labels in loader:
            if samples_processed >= 5000:
                break

            outputs, _ = analyzer(m1, m2, None)
            preds = outputs.argmax(dim=1)

            m1_stats = get_batch_stats(m1)
            m2_stats = get_batch_stats(m2)
            combined_stats = torch.cat([m1_stats, m2_stats], dim=1)
            inputs_list.append(combined_stats.cpu().numpy())

            correct += torch.sum(torch.eq(preds, labels)).item()

            for name, acts in analyzer.activations.items():
                correct_mask = preds == labels

                if name not in all_activations['correct']:
                    all_activations['correct'][name] = []
                    all_activations['incorrect'][name] = []

                all_activations['correct'][name].append(acts[correct_mask].cpu().numpy())
                all_activations['incorrect'][name].append(acts[~correct_mask].cpu().numpy())

            samples_processed += len(labels)

    accuracy = 100 * correct / samples_processed
    print(f"\nAccuracy on {samples_processed} samples: {accuracy:.2f}%")

    print("\nFeature Impact Analysis:")

    feature_names = [
        'M1 Mean', 'M1 Std', 'M1 Neg Ratio', 'M1 Min', 'M1 Max',
        'M1 Square Mean', 'M1 25%', 'M1 50%', 'M1 75%',
        'M2 Mean', 'M2 Std', 'M2 Neg Ratio', 'M2 Min', 'M2 Max',
        'M2 Square Mean', 'M2 25%', 'M2 50%', 'M2 75%'
    ]

    inputs = np.vstack(inputs_list)

    correct_mask = np.array([True] * len(inputs))
    incorrect_mask = ~correct_mask

    feature_impacts = {}
    for i, feature_name in enumerate(feature_names):
        correct_mean = inputs[correct_mask, i].mean()
        incorrect_mean = inputs[incorrect_mask, i].mean()
        diff = correct_mean - incorrect_mean

        feature_impacts[feature_name] = {
            'correct_mean': correct_mean,
            'incorrect_mean': incorrect_mean,
            'diff': diff,
            'abs_diff': abs(diff)
        }

    sorted_features = sorted(feature_impacts.items(), key=lambda x: x[1]['abs_diff'], reverse=True)

    print("\nFeature Importance (sorted by impact):")
    for feature_name, impact in sorted_features:
        print(f"\n{feature_name}:")
        print(f"  Correct predictions avg: {impact['correct_mean']:.3f}")
        print(f"  Incorrect predictions avg: {impact['incorrect_mean']:.3f}")
        print(f"  Impact (difference): {impact['diff']:.3f}")


if __name__ == "__main__":
    main()
