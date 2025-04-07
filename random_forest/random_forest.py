import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from TrainingPipeline.Datasets.MagicDataset import MagicDataset

def train_random_forest_classifier(proton_file, gamma_file, test_size=0.3):
    print("Loading the MAGIC dataset...")
    dataset = MagicDataset(
        proton_filename=proton_file,
        gamma_filename=gamma_file,
        debug_info=True,
    )

    print("\nExtracting features and labels for training...")
    X = []
    y = []

    batch_size = 128
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for batch in tqdm(loader, desc="Processing batches"):
        m1_batch, m2_batch, features_batch, labels_batch = batch

        # Add the features to our dataset (these are already the numerical features)
        for i in range(len(labels_batch)):
            X.append(features_batch[i].numpy())
            y.append(labels_batch[i].item())

    X = np.array(X)
    y = np.array(y)

    print(f"\nExtracted features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Class distribution: {np.bincount(y)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")

    print("\nTraining Random Forest classifier...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nRandom Forest Test Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Proton', 'Gamma']))

    # Calculate ROC curve
    y_prob = rf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    results = {
        'model': rf,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr,
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }

    return results


def plot_results(results):
    """Plot the results from the random forest classifier"""

    # Plot feature importances
    plt.figure(figsize=(12, 8))

    # Get top 20 features
    top_features = results['feature_importances'][:20]
    features = [f[0] for f in top_features]
    importances = [f[1] for f in top_features]

    sns.barplot(x=importances, y=features)
    plt.title('Top 20 Feature Importances')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importances.png')

    # Plot ROC curve
    plt.figure(figsize=(8, 8))
    plt.plot(results['fpr'], results['tpr'], color='darkorange', lw=2,
             label=f'ROC curve (area = {results["roc_auc"]:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('roc_curve.png')

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Proton', 'Gamma'],
                yticklabels=['Proton', 'Gamma'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')

    print("Plots saved to current directory")


if __name__ == "__main__":
    proton_file = "magic-protons.parquet"
    gamma_file ="magic-gammas-new.parquet"
    test_size = 0.3

    results = train_random_forest_classifier(
        proton_file=proton_file,
        gamma_file=gamma_file,
        test_size=test_size,
    )

    plot_results(results)

    print("\nTraining and evaluation complete!")