import torch
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CNN.Architectures.BasicMagicCNN import BasicMagicNet
from TrainingPipeline.MagicDataset import MagicDataset


def evaluate_multiple_pairs(model_path, proton_file, gamma_file, num_pairs=1000):
    dataset = MagicDataset(proton_file, gamma_file)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = BasicMagicNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    proton_correct = 0
    gamma_correct = 0
    proton_probs = []
    gamma_probs = []

    with torch.no_grad():
        for i in range(num_pairs):
            idx = np.random.randint(0, len(dataset) - 1)
            m1, m2, features, label = dataset[idx]

            m1 = m1.unsqueeze(0).to(device)
            m2 = m2.unsqueeze(0).to(device)
            features = features.unsqueeze(0).to(device)

            output = model(m1, m2, features)
            _, pred = output.max(1)
            proton_correct += (output.item() == label)

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{num_pairs} pairs")

    proton_acc = proton_correct / num_pairs * 100
    gamma_acc = gamma_correct / num_pairs * 100

    print("\nEvaluation Results:")
    print(f"Proton Accuracy: {proton_acc:.2f}%")
    print(f"Gamma Accuracy: {gamma_acc:.2f}%")
    print(f"Overall Accuracy: {((proton_correct + gamma_correct) / (2 * num_pairs)) * 100:.2f}%")

    proton_probs = np.array(proton_probs)
    gamma_probs = np.array(gamma_probs)
    print("\nAverage Prediction Probabilities:")
    print(f"Proton Images - Proton: {proton_probs[:, 0].mean():.3f}, Gamma: {proton_probs[:, 1].mean():.3f}")
    print(f"Gamma Images - Proton: {gamma_probs[:, 0].mean():.3f}, Gamma: {gamma_probs[:, 1].mean():.3f}")


if __name__ == "__main__":
    MODEL_PATH = "trained_model.pth"
    PROTON_FILE = "magic-protons.parquet"
    GAMMA_FILE = "magic-gammas.parquet"

    evaluate_multiple_pairs(MODEL_PATH, PROTON_FILE, GAMMA_FILE)
