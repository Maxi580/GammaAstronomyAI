import cudf
import cuml
from cuml.ensemble import RandomForestClassifier
from cuml.model_selection import train_test_split
from cuml.metrics import accuracy_score
import joblib
import pandas as pd
import numpy as np
import time
from TrainingPipeline.Datasets.MagicDataset import MagicDataset, read_parquet_limit, extract_features


def train_random_forest_classifier_gpu(
        proton_file,
        gamma_file,
        path=None,
        test_size=0.3,
        n_estimators=100,
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=1
):
    print("Loading data...")
    start_time = time.time()

    proton_metadata = cudf.io.read_parquet_metadata(proton_file)
    gamma_metadata = cudf.io.read_parquet_metadata(gamma_file)

    n_protons = proton_metadata.num_rows
    n_gammas = gamma_metadata.num_rows

    print(f"Loading {n_protons} proton samples and {n_gammas} gamma samples")

    proton_data = cudf.read_parquet(proton_file)
    gamma_data = cudf.read_parquet(gamma_file)

    print(f"Data loaded in {time.time() - start_time:.2f} seconds")

    print("Extracting features...")
    feature_start = time.time()

    proton_features = extract_features_gpu(proton_data)
    gamma_features = extract_features_gpu(gamma_data)

    proton_labels = cudf.Series(np.zeros(len(proton_data)), dtype='int32')
    gamma_labels = cudf.Series(np.ones(len(gamma_data)), dtype='int32')

    X = cudf.concat([proton_features, gamma_features])
    y = cudf.concat([proton_labels, gamma_labels])

    print(f"Features extracted in {time.time() - feature_start:.2f} seconds")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    print("Training Random Forest model on GPU...")
    training_start = time.time()

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        n_streams=1,
        n_bins=128,
        max_features=None,
        bootstrap=True,
        random_state=42
    )

    model.fit(X_train, y_train)

    training_time = time.time() - training_start
    print(f"Model trained in {training_time:.2f} seconds")

    print("Evaluating model...")


    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.4f}")

    # Save the model if path is provided
    if path:
        joblib.dump(model, path)
        print(f"Model saved to {path}")

    return {
        "model": model,
        "accuracy": accuracy,
        "training_time": training_time,
        "n_samples": len(X)
    }


def extract_features_gpu(df):
    """
    Adapt the extract_features function to work with cuDF DataFrames
    This returns all features as a cuDF DataFrame
    """
    # The same feature names as in your original extract_features function
    feature_columns = [
        # Hillas M1 features
        'hillas_length_m1', 'hillas_width_m1', 'hillas_delta_m1',
        'hillas_size_m1', 'hillas_cog_x_m1', 'hillas_cog_y_m1',
        'hillas_sin_delta_m1', 'hillas_cos_delta_m1',

        # Hillas M2 features
        'hillas_length_m2', 'hillas_width_m2', 'hillas_delta_m2',
        'hillas_size_m2', 'hillas_cog_x_m2', 'hillas_cog_y_m2',
        'hillas_sin_delta_m2', 'hillas_cos_delta_m2',

        # Stereo features
        'stereo_direction_x', 'stereo_direction_y', 'stereo_zenith',
        'stereo_azimuth', 'stereo_dec', 'stereo_ra', 'stereo_theta2',
        'stereo_core_x', 'stereo_core_y', 'stereo_impact_m1',
        'stereo_impact_m2', 'stereo_impact_azimuth_m1',
        'stereo_impact_azimuth_m2', 'stereo_shower_max_height',
        'stereo_xmax', 'stereo_cherenkov_radius',
        'stereo_cherenkov_density', 'stereo_baseline_phi_m1',
        'stereo_baseline_phi_m2', 'stereo_image_angle',
        'stereo_cos_between_shower',

        # Pointing features
        'pointing_zenith', 'pointing_azimuth',

        # Time gradient features
        'time_gradient_m1', 'time_gradient_m2',

        # Source M1 features
        'source_alpha_m1', 'source_dist_m1',
        'source_cos_delta_alpha_m1', 'source_dca_m1',
        'source_dca_delta_m1',

        # Source M2 features
        'source_alpha_m2', 'source_dist_m2',
        'source_cos_delta_alpha_m2', 'source_dca_m2',
        'source_dca_delta_m2'
    ]

    # Create a new DataFrame with only the features we need
    features_df = df[feature_columns].copy()

    # Replace NaN values with 0
    features_df = features_df.fillna(0.0)

    return features_df