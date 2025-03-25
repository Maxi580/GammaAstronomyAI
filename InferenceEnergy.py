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
    
    protons_correct = []
    protons_incorrect = []
    gammas_correct = []
    gammas_incorrect = []

    with torch.no_grad():
        for i in range(num_samples):
            idx = np.random.randint(0, len(dataset))
            m1, m2, features, label, event_num, run_num, energy = dataset[idx]

            m1 = m1.unsqueeze(0).to(device)
            m2 = m2.unsqueeze(0).to(device)
            features = features.unsqueeze(0).to(device)

            output = model(m1, m2, features)
            
            probabilities = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1).item()
            gamma_prob = probabilities[0, 1].item() # "gammacity"

            is_correct = (pred == label)
            correct += is_correct
            
            if label == 0:  # Proton
                if is_correct:
                    protons_correct.append((gamma_prob, energy))
                else:
                    protons_incorrect.append((gamma_prob, energy))

            else:  # label == 1 -> Gamma
                if is_correct:
                    gammas_correct.append((gamma_prob, energy))
                else:
                    gammas_incorrect.append((gamma_prob, energy))
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{num_samples} samples")

    accuracy = (correct / num_samples) * 100
    print(f"\nEvaluation Results:")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    
    # Helper function to safely extract x and y coordinates from a list of tuples
    def extract_coords(data):
        if data:
            # Unzip the list of tuples into two lists: x and y values
            x, y = zip(*data)
            return x, y
        else:
            return [], []

    # Extract coordinates for each category
    pc_x, pc_y = extract_coords(protons_correct)
    pi_x, pi_y = extract_coords(protons_incorrect)
    gc_x, gc_y = extract_coords(gammas_correct)
    gi_x, gi_y = extract_coords(gammas_incorrect)

    # Create the scatter plot
    plt.figure(figsize=(8, 6))

    plt.scatter(pc_x, pc_y, color='blue', label='Protons Correct', marker='o')
    plt.scatter(gc_x, gc_y, color='red', label='Gammas Correct', marker='o')
    plt.scatter(pi_x, pi_y, color='lightblue', label='Protons Incorrect', marker='o')
    plt.scatter(gi_x, gi_y, color='salmon', label='Gammas Incorrect', marker='o')

    plt.xlabel('Gammacity')
    plt.ylabel('Energy')
    plt.yscale('log')
    plt.title('Scatter Plot of Gammacity vs Energy')
    plt.legend()
    plt.tight_layout()
    plt.savefig("gammacity_vs_energy.png")
    plt.show()
    plt.clf()
    
    # Only incorrect ones
    plt.scatter(pi_x, pi_y, color='lightblue', label='Protons Incorrect', marker='o')
    plt.scatter(gi_x, gi_y, color='salmon', label='Gammas Incorrect', marker='o')

    plt.xlabel('Gammacity')
    plt.ylabel('Energy')
    plt.yscale('log')
    plt.title('Scatter Plot of Gammacity vs Energy')
    plt.legend()
    plt.tight_layout()
    plt.savefig("gammacity_vs_energy2.png")
    plt.show()
    


if __name__ == "__main__":
    MODEL_PATH = "trained_model.pth"
    PROTON_FILE = "magic-protons.parquet"
    GAMMA_FILE = "magic-gammas-new-2.parquet"

    evaluate_random_samples(MODEL_PATH, PROTON_FILE, GAMMA_FILE)
