import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Dict, Tuple, Any, Optional
import pyarrow.parquet as pq


class MagicDataset(Dataset):
    GAMMA_LABEL: str = 'gamma'
    PROTON_LABEL: str = 'proton'

    def __init__(self, proton_filename: str, gamma_filename: str, max_samples: Optional[int] = None,
                 debug_info: bool = True):
        self.debug_info = debug_info

        if self.debug_info:
            print(f"Initializing dataset from:")
            print(f"Proton file: {proton_filename}")
            print(f"Gamma file: {gamma_filename}")

        self.proton_metadata = pq.read_metadata(proton_filename)
        self.gamma_metadata = pq.read_metadata(gamma_filename)

        total_samples = self.proton_metadata.num_rows + self.gamma_metadata.num_rows
        original_proton_ratio = self.proton_metadata.num_rows / total_samples

        if max_samples is not None:
            self.n_protons = min(int(max_samples * original_proton_ratio), self.proton_metadata.num_rows)
            self.n_gammas = min(max_samples - self.n_protons, self.gamma_metadata.num_rows)
        else:
            self.n_protons = self.proton_metadata.num_rows
            self.n_gammas = self.gamma_metadata.num_rows

        self.proton_data = pd.read_parquet(proton_filename).iloc[:self.n_protons]
        self.gamma_data = pd.read_parquet(gamma_filename).iloc[:self.n_gammas]
        self.length = self.n_protons + self.n_gammas
        self.labels = {self.PROTON_LABEL: 0, self.GAMMA_LABEL: 1}

        if self.debug_info:
            print(f"\nDataset initialized with {self.length} total samples:")
            print(f"Protons: {self.n_protons}")
            print(f"Gammas: {self.n_gammas}")
            print(f"Label mapping: {self.labels}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if idx < self.n_protons:
            row = self.proton_data.iloc[idx]
            label = self.PROTON_LABEL
        else:
            row = self.gamma_data.iloc[idx - self.n_protons]
            label = self.GAMMA_LABEL

        m1_raw = torch.tensor(row['image_m1'], dtype=torch.float32)
        m2_raw = torch.tensor(row['image_m2'], dtype=torch.float32)

        return m1_raw, m2_raw, self.labels[label]

    def get_distribution(self) -> Dict[str, Any]:
        total_samples = self.length

        distribution = {
            'proton': {
                'count': self.n_protons,
                'percentage': (self.n_protons / total_samples) * 100
            },
            'gamma': {
                'count': self.n_gammas,
                'percentage': (self.n_gammas / total_samples) * 100
            }
        }

        return {'total_samples': total_samples, 'distribution': distribution}
