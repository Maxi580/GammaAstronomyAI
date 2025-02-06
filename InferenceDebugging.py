import torch
from magicDataset import MagicDataset
from BasicMagicNet import BasicMagicNet
import numpy as np


def evaluate_random_samples(model_path, proton_file, gamma_file, num_samples=1000):
    dataset = MagicDataset(proton_file, gamma_file)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    model = BasicMagicNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    correct = 0
    predicted_labels = []
    true_labels = []

    with torch.no_grad():
        for i in range(num_samples):
            idx = np.random.randint(0, len(dataset))
            noisy_m1, noisy_m2, clean_m1, clean_m2, features, label = dataset[idx]

            noisy_m1 = noisy_m1.unsqueeze(0).to(device)
            noisy_m2 = noisy_m2.unsqueeze(0).to(device)
            features = features.unsqueeze(0).to(device)

            output = model(noisy_m1, noisy_m2, features)
            pred = output.argmax(dim=1).item()

            correct += (pred == label)
            predicted_labels.append(pred)
            true_labels.append(label)

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{num_samples} samples")

    accuracy = (correct / num_samples) * 100
    print(f"\nEvaluation Results:")
    print(f"Overall Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    MODEL_PATH = "trained_model.pth"
    PROTON_FILE = "magic-protons.parquet"
    GAMMA_FILE = "magic-gammas.parquet"

    evaluate_random_samples(MODEL_PATH, PROTON_FILE, GAMMA_FILE)