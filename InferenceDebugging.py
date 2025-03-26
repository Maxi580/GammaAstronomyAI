import os
import sys
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CNN.Architectures import *
from TrainingPipeline.Datasets import *

def evaluate_random_samples(model_path, proton_file, gamma_file, num_samples=10000):
    dataset = MagicDataset(proton_file, gamma_file, max_samples=100000, rescale_image=False)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    model = HexCircleNet()
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model = model.to(device)
    model.eval()

    correct = 0

    with torch.no_grad():
        for i in range(num_samples):
            idx = np.random.randint(0, len(dataset))
            m1, m2, features, label, *_ = dataset[idx]

            m1 = m1.unsqueeze(0).to(device)
            m2 = m2.unsqueeze(0).to(device)
            features = features.unsqueeze(0).to(device)

            output = model(m1, m2, features)
            pred = output.argmax(dim=1).item()

            correct += (pred == label)
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{num_samples} samples")

    accuracy = (correct / num_samples) * 100
    print(f"\nEvaluation Results:")
    print(f"Overall Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    MODEL_PATH = "trained_model.pth"
    PROTON_FILE = "magic-protons.parquet"
    GAMMA_FILE = "magic-gammas-new-2.parquet"

    evaluate_random_samples(MODEL_PATH, PROTON_FILE, GAMMA_FILE)
