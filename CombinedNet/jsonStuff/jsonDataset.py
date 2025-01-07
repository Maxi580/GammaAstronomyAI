import torch
from torch.utils.data import Dataset
import json
import os


def extract_features(data):
    features = []

    m1_raw = torch.tensor(data["images"]["raw"]["m1"], dtype=torch.float32)
    m2_raw = torch.tensor(data["images"]["raw"]["m2"], dtype=torch.float32)

    for feature_group in ["true_parameters", "hillas_m1", "hillas_m2", "stereo",
                          "pointing", "time_gradient", "source_m1", "source_m2"]:
        group_data = data["features"][feature_group]
        features.extend(float(val) for val in group_data.values())

    return m1_raw, m2_raw, torch.tensor(features, dtype=torch.float32)


class jsonDataset(Dataset):
    PROTON_LABEL = 'proton'
    GAMMA_LABEL = 'gamma'

    def __init__(self, data_dir: str, debug_info: bool = True):
        self.debug_info = debug_info
        if debug_info:
            print(f"Loading data from: {data_dir}")

        self.array_dir = os.path.join(data_dir, "arrays")
        self.annotation_dir = os.path.join(data_dir, "annotations")

        if not os.path.exists(self.array_dir) or not os.path.exists(self.annotation_dir):
            raise ValueError(f"Invalid data directory structure in {data_dir}")

        self.arrays = sorted(os.listdir(self.array_dir))
        self.labels = {self.PROTON_LABEL: 0, self.GAMMA_LABEL: 1}

        # Count samples
        self.n_protons = 0
        self.n_gammas = 0

        for array_file in self.arrays:
            label_path = os.path.join(self.annotation_dir, array_file.replace(".json", ".txt"))
            with open(label_path, "r") as f:
                label = f.read().strip()
                if label == self.PROTON_LABEL:
                    self.n_protons += 1
                else:
                    self.n_gammas += 1

        if debug_info:
            print(f"Found {self.n_protons} proton and {self.n_gammas} gamma samples")

    def __len__(self):
        return len(self.arrays)

    def __getitem__(self, idx):
        array_file = self.arrays[idx]
        array_path = os.path.join(self.array_dir, array_file)
        label_path = os.path.join(self.annotation_dir, array_file.replace(".json", ".txt"))

        with open(array_path, "r") as f:
            data = json.load(f)
        with open(label_path, "r") as f:
            label = f.read().strip()

        m1_raw, m2_raw, features = extract_features(data)
        return m1_raw, m2_raw, features, self.labels[label]

    def get_distribution(self):
        total_samples = len(self.arrays)
        distribution = {
            self.PROTON_LABEL: {
                'count': self.n_protons,
                'percentage': (self.n_protons / total_samples) * 100
            },
            self.GAMMA_LABEL: {
                'count': self.n_gammas,
                'percentage': (self.n_gammas / total_samples) * 100
            }
        }
        return {'total_samples': total_samples, 'distribution': distribution}