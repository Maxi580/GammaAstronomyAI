import torch
import torch.nn as nn
from CombinedNet.CombinedNet import CombinedNet
from CombinedNet.magicDataset import MagicDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def evaluate_saved_model(model_path, proton_file, gamma_file, batch_size=32):
    # Load the dataset
    dataset = MagicDataset(proton_file, gamma_file)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = CombinedNet()

    # Load saved weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Evaluation variables
    all_preds = []
    all_labels = []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    print(f"Evaluating model on device: {device}")
    print(f"Total batches to process: {len(data_loader)}")

    # Evaluation loop
    with torch.no_grad():
        for batch_idx, (m1_images, m2_images, labels) in enumerate(data_loader):
            if batch_idx % 100 == 0:
                print(f"Processing batch {batch_idx}/{len(data_loader)}")

            m1_images = m1_images.to(device)
            m2_images = m2_images.to(device)
            labels = labels.to(device)

            outputs = model(m1_images, m2_images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    avg_loss = total_loss / len(data_loader)

    # Print results
    print("\nEvaluation Results:")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-Score: {f1 * 100:.2f}%")
    print("\nConfusion Matrix:")
    print("─" * 45)
    print(f"│              │      Predicted    │")
    print(f"│              │          0      1 │")
    print(f"├──────────────┼───────────────────┤")
    print(f"│ Actual    0  │    {cm[0, 0]:4.0f}    {cm[0, 1]:4.0f}│")
    print(f"│              │                   │")
    print(f"│ Actual    1  │    {cm[1, 0]:4.0f}    {cm[1, 1]:4.0f}│")
    print("─" * 45)

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }


if __name__ == "__main__":
    MODEL_PATH = "trained_models/CombinedNet__2025-01-08_02-42-32/trained_model.pth"
    PROTON_FILE = "magic-protons.parquet"
    GAMMA_FILE = "magic-gammas.parquet"

    results = evaluate_saved_model(MODEL_PATH, PROTON_FILE, GAMMA_FILE)