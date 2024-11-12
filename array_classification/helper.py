import os
import glob
import numpy as np
import json


def load_data_from_json(data_dir, annotation_dir):
    arrays = []
    labels = []

    json_files = sorted(glob.glob(os.path.join(data_dir, '*.json')))

    for json_file in json_files:
        # Get the base filename to find corresponding annotation
        base_name = os.path.basename(json_file)
        annotation_file = os.path.join(annotation_dir,
                                       base_name.replace('.json', '.txt'))

        # Check if both files exist
        if not os.path.exists(annotation_file):
            print(f"Warning: No annotation found for {base_name}, skipping...")
            continue

        try:
            # Load the array from JSON
            with open(json_file, 'r') as f:
                array_data = json.load(f)

            # Convert to numpy array
            array = np.array(array_data)

            with open(annotation_file, 'r') as f:
                label = f.read().strip()

            arrays.append(array)
            labels.append(label)

        except Exception as e:
            print(f"Error processing {base_name}: {str(e)}")
            continue

    return arrays, labels


def load_and_preprocess_data(array_list, labels):
    features = []

    for arr in array_list:
        # Extract features from the 1D array
        features.append(extract_features(arr))

    return np.array(features), np.array(labels)


def extract_features(array):
    features = [
        np.mean(array),  # Mean value
        np.std(array),  # Standard deviation
        np.median(array),  # Median value
        np.max(array),  # Maximum value
        np.min(array),  # Minimum value
        np.sum(array),  # Sum of values
        np.percentile(array, 25),  # First quartile
        np.percentile(array, 75),  # Third quartile
        np.count_nonzero(array),  # Number of non-zero elements
    ]

    non_zero_indices = np.nonzero(array)[0]
    if len(non_zero_indices) > 0:
        features.extend([
            non_zero_indices[0],  # First non-zero position
            non_zero_indices[-1],  # Last non-zero position
            np.mean(non_zero_indices),  # Mean position of non-zero elements
            np.std(non_zero_indices)  # Spread of non-zero elements
        ])
    else:
        features.extend([0, 0, 0, 0])

    return features
