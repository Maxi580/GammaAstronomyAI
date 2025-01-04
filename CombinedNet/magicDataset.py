import json
import os
from typing import Dict, Tuple, Any

import torch
from torch.utils.data import Dataset


def _extract_features(features_data: Dict[str, Dict[str, Any]]) -> torch.Tensor:
    features = []

    true_params = features_data["true_parameters"]
    features.extend([
        true_params["energy"],
        true_params["theta"],
        true_params["phi"],
        true_params["telescope_theta"],
        true_params["telescope_phi"],
        true_params["first_interaction_height"],
        true_params["impact_m1"],
        true_params["impact_m2"]
    ])

    hillas_m1 = features_data["hillas_m1"]
    features.extend([
        hillas_m1["length"],
        hillas_m1["width"],
        hillas_m1["delta"],
        hillas_m1["size"],
        hillas_m1["cog_x"],
        hillas_m1["cog_y"],
        hillas_m1["sin_delta"],
        hillas_m1["cos_delta"]
    ])

    hillas_m2 = features_data["hillas_m2"]
    features.extend([
        hillas_m2["length"],
        hillas_m2["width"],
        hillas_m2["delta"],
        hillas_m2["size"],
        hillas_m2["cog_x"],
        hillas_m2["cog_y"],
        hillas_m2["sin_delta"],
        hillas_m2["cos_delta"]
    ])

    stereo = features_data["stereo"]
    features.extend([
        stereo["direction_x"],
        stereo["direction_y"],
        stereo["zenith"],
        stereo["azimuth"],
        stereo["dec"],
        stereo["ra"],
        stereo["theta2"],
        stereo["core_x"],
        stereo["core_y"],
        stereo["impact_m1"],
        stereo["impact_m2"],
        stereo["impact_azimuth_m1"],
        stereo["impact_azimuth_m2"],
        stereo["shower_max_height"],
        stereo["xmax"],
        stereo["cherenkov_radius"],
        stereo["cherenkov_density"],
        stereo["baseline_phi_m1"],
        stereo["baseline_phi_m2"],
        stereo["image_angle"],
        stereo["cos_between_shower"]
    ])

    pointing = features_data["pointing"]
    features.extend([
        pointing["zenith"],
        pointing["azimuth"]
    ])

    time_gradient = features_data["time_gradient"]
    features.extend([
        time_gradient["m1"],
        time_gradient["m2"]
    ])

    source_m1 = features_data["source_m1"]
    features.extend([
        source_m1["alpha"],
        source_m1["dist"],
        source_m1["cos_delta_alpha"],
        source_m1["dca"],
        source_m1["dca_delta"]
    ])

    source_m2 = features_data["source_m2"]
    features.extend([
        source_m2["alpha"],
        source_m2["dist"],
        source_m2["cos_delta_alpha"],
        source_m2["dca"],
        source_m2["dca_delta"]
    ])

    return torch.tensor(features, dtype=torch.float32)


class MagicDataset(Dataset):
    def __init__(self, testdata_dir: str):
        if not os.path.isdir(testdata_dir):
            raise Exception(f"Directory not found: '{testdata_dir}'")

        self.array_dir = os.path.join(testdata_dir, "arrays")
        self.annotation_dir = os.path.join(testdata_dir, "annotations")

        if not os.path.isdir(self.array_dir):
            raise Exception(f"Directory not found: '{self.array_dir}'")

        if not os.path.isdir(self.annotation_dir):
            raise Exception(f"Directory not found: '{self.annotation_dir}'")

        self.arrays = sorted(os.listdir(self.array_dir))
        self.labels = self.detect_labels()

    def __len__(self):
        return len(self.arrays)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        data_path = os.path.join(self.array_dir, self.arrays[idx])
        label_path = os.path.join(
            self.annotation_dir, self.arrays[idx].replace(".json", ".txt")
        )

        with open(label_path, "r") as f:
            label = f.read().strip()

        with open(data_path, "r") as f:
            data = json.load(f)

        try:
            m1_raw = torch.tensor(data["images"]["raw"]["m1"], dtype=torch.float32)
            m2_raw = torch.tensor(data["images"]["raw"]["m2"], dtype=torch.float32)
            features = _extract_features(data["features"])
        except Exception as e:
            print(f"Source: {data_path}")
            raise e

        return m1_raw, m2_raw, features, self.labels[label]

    def detect_labels(self) -> Dict[str, int]:
        unique_labels = set()
        for array_file in self.arrays:
            label_path = os.path.join(
                self.annotation_dir, array_file.replace(".json", ".txt")
            )
            with open(label_path, "r") as f:
                label = f.read().strip()
            unique_labels.add(label)

        sorted_labels = sorted(list(unique_labels))

        return {label: idx for idx, label in enumerate(sorted_labels)}

    def get_distribution(self) -> Dict[str, Any]:
        all_labels = []
        for array_file in self.arrays:
            label_path = os.path.join(
                self.annotation_dir, array_file.replace(".json", ".txt")
            )
            with open(label_path, "r") as f:
                label = f.read().strip()
            all_labels.append(self.labels[label])

        total_samples = len(all_labels)
        label_counts = {}
        for label_name, label_idx in self.labels.items():
            count = all_labels.count(label_idx)
            percentage = (count / total_samples) * 100
            label_counts[label_name] = {"count": count, "percentage": percentage}

        return {"total_samples": total_samples, "distribution": label_counts}


"""if __name__ == '__main__':
    mg = MagicDataset("magic_protons_full")
    print(f"Distribution: {mg.get_distribution()}")
    print(f"Labels: {mg.detect_labels()}")
    print(f"Sample: {mg[0]}")"""
