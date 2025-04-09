import numpy as np
import sys
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader
import cudf
import cuml
from cuml.ensemble import RandomForestClassifier as cuRF
import joblib
from joblib import parallel_backend

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from TrainingPipeline.Datasets.MagicDataset import MagicDataset


def optimize_random_forest(X_train, y_train, cv=3, sample_size=20000):
    print("Optimizing Random Forest hyperparameters on a subset of the data...")

    if len(X_train) > sample_size:
        X_sample, _, y_sample, _ = train_test_split(
            X_train, y_train,
            train_size=sample_size,
            stratify=y_train,
            random_state=42
        )
        print(f"Using {sample_size} samples for optimization ({sample_size / len(X_train):.1%} of training data)")
    else:
        X_sample = X_train
        y_sample = y_train
        print(f"Using all {len(X_train)} samples for optimization (dataset smaller than requested sample)")

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }

    rf = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=cv,
        n_jobs=-1,
        verbose=2,
        scoring='accuracy'
    )

    grid_search.fit(X_sample, y_sample)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    best_params = grid_search.best_params_
    print("Training final model with best parameters on full training set...")
    best_model = RandomForestClassifier(random_state=42, **best_params)
    best_model.fit(X_train, y_train)

    return best_model


def train_random_forest_classifier(proton_file, gamma_file, path, test_size=0.3, optimize=False):
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
        m1_batch, m2_batch, features_batch, labels_batch, _ = batch

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

    X_train_gpu = cudf.DataFrame(X_train)
    y_train_gpu = cudf.Series(y_train)
    X_test_gpu = cudf.DataFrame(X_test)
    y_test_gpu = cudf.Series(y_test)

    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    print("\nTraining GPU-accelerated Random Forest classifier...")

    rf = cuRF(
        n_estimators=200,
        max_depth=30,
        n_bins=128,
        random_state=42,
        verbose=True
    )

    rf.fit(X_train_gpu, y_train_gpu)

    y_pred = rf.predict(X_test_gpu).to_numpy()
    y_prob = rf.predict_proba(X_test_gpu)[:, 1].to_numpy()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(rf, path)
    print(f"GPU model saved to {path}")

    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nRandom Forest Test Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Proton', 'Gamma']))

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