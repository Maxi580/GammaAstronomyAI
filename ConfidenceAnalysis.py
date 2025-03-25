import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CNN.Architectures.HexCircleCNN import HexCircleNet
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
    
    all_confidences = []
    all_predicted_labels = []
    all_correct = []
    
    proton_correct_conf = []
    proton_incorrect_conf = []
    gamma_correct_conf = []
    gamma_incorrect_conf = []
    
    proton_gammacity = []
    gamma_gammacity = []

    with torch.no_grad():
        for i in range(num_samples):
            idx = np.random.randint(0, len(dataset))
            m1, m2, features, label, *_ = dataset[idx]

            m1 = m1.unsqueeze(0).to(device)
            m2 = m2.unsqueeze(0).to(device)
            features = features.unsqueeze(0).to(device)

            output = model(m1, m2, features)
            
            probabilities = torch.softmax(output, dim=1)
            confidence = probabilities.max().item()
            pred = output.argmax(dim=1).item()
            gamma_prob = probabilities[0, 1].item() # "gammacity"

            is_correct = (pred == label)
            correct += is_correct

            all_confidences.append(confidence)
            all_predicted_labels.append(pred)
            all_correct.append(is_correct)
            
            if label == 0:  # Proton
                if is_correct:
                    proton_correct_conf.append(confidence)
                else:
                    proton_incorrect_conf.append(confidence)
                    
                proton_gammacity.append(gamma_prob)
            else:  # label == 1 -> Gamma
                if is_correct:
                    gamma_correct_conf.append(confidence)
                else:
                    gamma_incorrect_conf.append(confidence)
                    
                gamma_gammacity.append(gamma_prob)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{num_samples} samples")

    accuracy = (correct / num_samples) * 100
    print(f"\nEvaluation Results:")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    
    # Plotting the results
    # x-axis: predicted label, y-axis: confidence
    # Color: green if correct, red if wrong
    x_jittered_correct = []
    y_conf_correct = []
    x_jittered_incorrect = []
    y_conf_incorrect = []

    for i, pred_label in enumerate(all_predicted_labels):
        # Small random shift in [-0.2, 0.2] around 0 or 1
        jitter = (np.random.rand() - 0.5) * 0.4
        if all_correct[i]:
            x_jittered_correct.append(pred_label + jitter)
            y_conf_correct.append(all_confidences[i])
        else:
            x_jittered_incorrect.append(pred_label + jitter)
            y_conf_incorrect.append(all_confidences[i])

    # Jittered Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x_jittered_correct, y_conf_correct, c='green', alpha=0.6, s=10, label='Correct')
    plt.scatter(x_jittered_incorrect, y_conf_incorrect, c='red', alpha=0.6, s=10, label='Incorrect')

    # Format the plot
    plt.xlabel("Predicted Label (with jitter)")
    plt.ylabel("Confidence")
    plt.title("Model Confidence vs Predicted Label\n(Green: Correct, Red: Incorrect)")
    plt.xticks([0, 1], labels=['0', '1'])  # Keep the x-ticks at 0 and 1
    plt.grid(True)
    plt.legend()
    plt.savefig("confidence-analysis-1.png")
    plt.clf()
    
    # Histograms of Gammacity
    plt.figure(figsize=(10, 6))

    plt.hist(proton_gammacity, bins=100, histtype='step', label='Protons')
    plt.hist(gamma_gammacity, bins=100, histtype='step', label='Gammas')

    plt.xlabel("Gammacity (Predicted Gamma Probability)")
    plt.ylabel("Counts")
    plt.title("Distribution of Gammacity for Protons and Gammas")
    plt.legend()
    plt.grid(True)
    plt.savefig("confidence-analysis-2-1.png")
    plt.clf()
    
    # Histograms of Gammacity (Relative)
    plt.figure(figsize=(10, 6))

    plt.hist(proton_gammacity, bins=100, histtype='step', label='Protons',  density=True)
    plt.hist(gamma_gammacity, bins=100, histtype='step', label='Gammas',  density=True)

    plt.xlabel("Gammacity (Predicted Gamma Probability)")
    plt.ylabel("Relative Counts")
    plt.title("Distribution of Gammacity for Protons and Gammas (Relative)")
    plt.legend()
    plt.grid(True)
    plt.savefig("confidence-analysis-2-2.png")
    plt.clf()
    
    # --- Histograms of Confidence Distribution ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Proton - Correct
    axes[0, 0].hist(proton_correct_conf, bins=50, color='green', alpha=0.7)
    axes[0, 0].set_title("Proton (label=0) - Correct")
    axes[0, 0].set_xlim([0, 1])

    # Proton - Incorrect
    axes[0, 1].hist(proton_incorrect_conf, bins=50, color='red', alpha=0.7)
    axes[0, 1].set_title("Proton (label=0) - Incorrect")
    axes[0, 1].set_xlim([0, 1])

    # Gamma - Correct
    axes[1, 0].hist(gamma_correct_conf, bins=50, color='green', alpha=0.7)
    axes[1, 0].set_title("Gamma (label=1) - Correct")
    axes[1, 0].set_xlim([0, 1])

    # Gamma - Incorrect
    axes[1, 1].hist(gamma_incorrect_conf, bins=50, color='red', alpha=0.7)
    axes[1, 1].set_title("Gamma (label=1) - Incorrect")
    axes[1, 1].set_xlim([0, 1])

    # Common x/y labels
    for ax in axes.flat:
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Count")

    plt.tight_layout()
    plt.savefig("confidence-analysis-3.png")
    


if __name__ == "__main__":
    MODEL_PATH = "trained_model.pth"
    PROTON_FILE = "magic-protons.parquet"
    GAMMA_FILE = "magic-gammas-new-2.parquet"

    evaluate_random_samples(MODEL_PATH, PROTON_FILE, GAMMA_FILE)
