import json
import os

import torch
from torch.utils.data import Dataset

from arrayClassification.CNN.constants import LABELS, LABELS_SQUARE


class ShapeDataset(Dataset):
    def __init__(self, testdata_dir):
        if not os.path.isdir(testdata_dir):
            raise Exception(f"Directory not found: '{testdata_dir}'")

        self.array_dir = os.path.join(testdata_dir, "arrays")
        self.annotation_dir = os.path.join(testdata_dir, "annotations")

        if not os.path.isdir(self.array_dir):
            raise Exception(f"Directory not found: '{self.array_dir}'")

        if not os.path.isdir(self.annotation_dir):
            raise Exception(f"Directory not found: '{self.annotation_dir}'")

        self.arrays = sorted(os.listdir(self.array_dir))

        # TODO: make this selection automatic
        self.labels = LABELS_SQUARE  # Or LABELS for centered and normal ellipses

        self.data_set_distribution = self._get_distribution()
    def __len__(self):
        return len(self.arrays)

    def __getitem__(self, idx):
        array_path = os.path.join(self.array_dir, self.arrays[idx])
        label_path = os.path.join(
            self.annotation_dir, self.arrays[idx].replace(".json", ".txt")
        )

        with open(array_path, "r") as f:
            array = json.loads(f.read().strip())["pixel_array"]

        with open(label_path, "r") as f:
            label = f.read().strip()

        return torch.tensor([array]), self.labels[label]

    def _get_distribution(self):
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
        for label_name, label_idx in LABELS.items():
            count = all_labels.count(label_idx)
            percentage = (count / total_samples) * 100
            label_counts[label_name] = {"count": count, "percentage": percentage}

        return {"total_samples": total_samples, "distribution": label_counts}
