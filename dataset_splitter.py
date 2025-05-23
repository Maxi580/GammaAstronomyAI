import pyarrow.parquet as pq
import pandas as pd
import os
import numpy as np


def split_parquet_files(proton_path, gamma_path, output_dir, val_split=0.3, test_split=0.0, random_seed=42):
    """
    Split proton and gamma parquet files into train/validation/test sets.

    Args:
        proton_path: Path to proton parquet file
        gamma_path: Path to gamma parquet file
        output_dir: Directory to save split files
        val_split: Fraction for validation set
        test_split: Fraction for test set (optional)
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with paths to generated files
    """
    train_proton_path = os.path.join(output_dir, "train_proton.parquet")
    train_gamma_path = os.path.join(output_dir, "train_gamma.parquet")
    val_proton_path = os.path.join(output_dir, "val_proton.parquet")
    val_gamma_path = os.path.join(output_dir, "val_gamma.parquet")
    required_files = [train_proton_path, train_gamma_path, val_proton_path, val_gamma_path]

    files_exist = all(os.path.exists(f) for f in required_files)

    if files_exist:
        print("Dataset split files already exist, skipping split operation.")
        result = {
            'train': {
                'proton': train_proton_path,
                'gamma': train_gamma_path
            },
            'val': {
                'proton': val_proton_path,
                'gamma': val_gamma_path
            }
        }

        return result

    os.makedirs(output_dir, exist_ok=True)

    np.random.seed(random_seed)

    print(f"Reading proton data from {proton_path}")
    proton_df = pd.read_parquet(proton_path)
    proton_indices = np.random.permutation(len(proton_df))

    print(f"Reading gamma data from {gamma_path}")
    gamma_df = pd.read_parquet(gamma_path)
    gamma_indices = np.random.permutation(len(gamma_df))

    proton_val_idx = int(len(proton_df) * (1 - val_split - test_split))
    proton_test_idx = int(len(proton_df) * (1 - test_split))

    gamma_val_idx = int(len(gamma_df) * (1 - val_split - test_split))
    gamma_test_idx = int(len(gamma_df) * (1 - test_split))

    proton_train = proton_df.iloc[proton_indices[:proton_val_idx]]
    proton_val = proton_df.iloc[proton_indices[proton_val_idx:proton_test_idx]]
    proton_test = proton_df.iloc[proton_indices[proton_test_idx:]] if test_split > 0 else None

    gamma_train = gamma_df.iloc[gamma_indices[:gamma_val_idx]]
    gamma_val = gamma_df.iloc[gamma_indices[gamma_val_idx:gamma_test_idx]]
    gamma_test = gamma_df.iloc[gamma_indices[gamma_test_idx:]] if test_split > 0 else None

    print(f"Saving train files (proton: {len(proton_train)}, gamma: {len(gamma_train)})")
    proton_train.to_parquet(train_proton_path)
    gamma_train.to_parquet(train_gamma_path)

    print(f"Saving validation files (proton: {len(proton_val)}, gamma: {len(gamma_val)})")
    proton_val.to_parquet(val_proton_path)
    gamma_val.to_parquet(val_gamma_path)

    result = {
        'train': {
            'proton': train_proton_path,
            'gamma': train_gamma_path
        },
        'val': {
            'proton': val_proton_path,
            'gamma': val_gamma_path
        }
    }

    if test_split > 0:
        test_proton_path = os.path.join(output_dir, "test_proton.parquet")
        test_gamma_path = os.path.join(output_dir, "test_gamma.parquet")

        print(f"Saving test files (proton: {len(proton_test)}, gamma: {len(gamma_test)})")
        proton_test.to_parquet(test_proton_path)
        gamma_test.to_parquet(test_gamma_path)

        result['test'] = {
            'proton': test_proton_path,
            'gamma': test_gamma_path
        }

    return result
