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

    print("Analyzing samples...")

    with torch.no_grad():
        for m1, m2, _, labels in loader:
            if samples_processed >= 50000:
                break

            outputs, _ = analyzer(m1, m2, None)
            preds = outputs.argmax(dim=1)

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

    print("\nNeuron Analysis:")
    for layer_name in all_activations['correct'].keys():
        correct_acts = np.vstack(all_activations['correct'][layer_name])
        incorrect_acts = np.vstack(all_activations['incorrect'][layer_name])

        correct_mean = correct_acts.mean(axis=0)
        incorrect_mean = incorrect_acts.mean(axis=0)

        neuron_diffs = correct_mean - incorrect_mean
        most_discriminative = np.argsort(np.abs(neuron_diffs))[-5:]

        print(f"\n{layer_name}:")
        for neuron_idx in most_discriminative[::-1]:
            diff = neuron_diffs[neuron_idx]
            print(f"Neuron {neuron_idx}: activation diff = {diff:.3f} "
                  f"(correct: {correct_mean[neuron_idx]:.3f}, "
                  f"incorrect: {incorrect_mean[neuron_idx]:.3f})")


if __name__ == "__main__":
    main()
