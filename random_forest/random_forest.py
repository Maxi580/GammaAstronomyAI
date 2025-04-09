import time

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
from cuml.model_selection import train_test_split as cu_train_test_split
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

    print("\nProcessing data directly to GPU...")
    start_time = time.time()

    batch_size = 1024
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    feature_cols = [f'feature_{i}' for i in range(51)]  # Based on extract_features returning 51 features

    first_batch = next(iter(loader))
    _, _, features_batch, labels_batch, _ = first_batch

    X_gpu = cudf.DataFrame(features_batch.numpy().astype(np.float32), columns=feature_cols)
    y_gpu = cudf.Series(labels_batch.numpy().astype(np.int32))

    for i, batch in enumerate(tqdm(loader, desc="Processing batches to GPU")):
        if i == 0:
            continue

        _, _, features_batch, labels_batch, _ = batch

        features_cudf = cudf.DataFrame(features_batch.numpy().astype(np.float32), columns=feature_cols)
        labels_cudf = cudf.Series(labels_batch.numpy().astype(np.int32))

        X_gpu = cudf.concat([X_gpu, features_cudf], ignore_index=True)
        y_gpu = cudf.concat([y_gpu, labels_cudf], ignore_index=True)

    data_processing_time = time.time() - start_time
    print(f"GPU data processing time: {data_processing_time:.2f} seconds")

    print(f"\nTotal samples on GPU: {len(X_gpu)}")
    print(f"Class distribution: {y_gpu.value_counts().to_pandas().to_dict()}")

    print("\nSplitting data on GPU...")
    X_train_gpu, X_test_gpu, y_train_gpu, y_test_gpu = cu_train_test_split(
        X_gpu, y_gpu, test_size=test_size, random_state=42, stratify=y_gpu
    )

    print(f"Training set size: {len(X_train_gpu)}")
    print(f"Testing set size: {len(X_test_gpu)}")

    print("\nTraining GPU-accelerated Random Forest classifier...")
    start_time = time.time()

    rf = cuRF(
        random_state=42,
        verbose=True,
        n_estimators=200,
        max_depth=30,
        n_bins=128
    )

    rf.fit(X_train_gpu, y_train_gpu)
    fit_time = time.time() - start_time
    print(f"Training time: {fit_time:.2f} seconds")

    start_time = time.time()
    y_pred = rf.predict(X_test_gpu)
    y_prob = rf.predict_proba(X_test_gpu)[:, 1]
    pred_time = time.time() - start_time
    print(f"Prediction time: {pred_time:.2f} seconds")

    y_test_np = y_test_gpu.to_numpy()
    y_pred_np = y_pred.to_numpy()
    y_prob_np = y_prob.to_numpy()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(rf, path)
    print(f"GPU model saved to {path}")

    accuracy = accuracy_score(y_test_np, y_pred_np)
    print(f"\nRandom Forest Test Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test_np, y_pred_np, target_names=['Proton', 'Gamma']))

    fpr, tpr, thresholds = roc_curve(y_test_np, y_prob_np)
    roc_auc = auc(fpr, tpr)

    results = {
        'model': rf,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr,
        'confusion_matrix': confusion_matrix(y_test_np, y_pred_np)
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
