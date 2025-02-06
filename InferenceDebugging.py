import torch
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CNN.Architectures.BasicMagicCNN import BasicMagicNet
from TrainingPipeline.MagicDataset import MagicDataset


def evaluate_multiple_pairs(model_path, proton_file, gamma_file, num_pairs=100):
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
            proton_idx = np.random.randint(0, dataset.n_protons)
            gamma_idx = np.random.randint(dataset.n_protons, len(dataset))

            m1_p, m2_p, features_p, label_p = dataset[proton_idx]
            m1_g, m2_g, features_g, label_g = dataset[gamma_idx]

            m1_p = m1_p.unsqueeze(0).to(device)
            m2_p = m2_p.unsqueeze(0).to(device)
            features_p = features_p.unsqueeze(0).to(device)
            output_p = model(m1_p, m2_p, features_p)
            probs_p = torch.softmax(output_p, dim=1)[0]
            _, pred_p = output_p.max(1)
            proton_correct += (pred_p.item() == label_p)
            proton_probs.append(probs_p.cpu().numpy())

            m1_g = m1_g.unsqueeze(0).to(device)
            m2_g = m2_g.unsqueeze(0).to(device)
            features_g = features_g.unsqueeze(0).to(device)
            output_g = model(m1_g, m2_g, features_g)
            probs_g = torch.softmax(output_g, dim=1)[0]
            _, pred_g = output_g.max(1)
            gamma_correct += (pred_g.item() == label_g)
            gamma_probs.append(probs_g.cpu().numpy())

            if (i + 1) % 10 == 0:
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
