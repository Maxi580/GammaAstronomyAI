import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Dict, Tuple, Any, Optional
import pyarrow.parquet as pq


def extract_features(row: pd.Series) -> torch.Tensor:
    features = []

    features.extend([
        float(row['true_energy']),
        float(row['true_theta']),
        float(row['true_phi']),
        float(row['true_telescope_theta']),
        float(row['true_telescope_phi']),
        float(row['true_first_interaction_height']),
        float(row['true_impact_m1']),
        float(row['true_impact_m2'])
    ])

    features.extend([
        float(row['hillas_length_m1']),
        float(row['hillas_width_m1']),
        float(row['hillas_delta_m1']),
        float(row['hillas_size_m1']),
        float(row['hillas_cog_x_m1']),
        float(row['hillas_cog_y_m1']),
        float(row['hillas_sin_delta_m1']),
        float(row['hillas_cos_delta_m1'])
    ])

    features.extend([
        float(row['hillas_length_m2']),
        float(row['hillas_width_m2']),
        float(row['hillas_delta_m2']),
        float(row['hillas_size_m2']),
        float(row['hillas_cog_x_m2']),
        float(row['hillas_cog_y_m2']),
        float(row['hillas_sin_delta_m2']),
        float(row['hillas_cos_delta_m2'])
    ])

    stereo_features = [
        'stereo_direction_x', 'stereo_direction_y', 'stereo_zenith',
        'stereo_azimuth', 'stereo_dec', 'stereo_ra', 'stereo_theta2',
        'stereo_core_x', 'stereo_core_y', 'stereo_impact_m1',
        'stereo_impact_m2', 'stereo_impact_azimuth_m1',
        'stereo_impact_azimuth_m2', 'stereo_shower_max_height',
        'stereo_xmax', 'stereo_cherenkov_radius',
        'stereo_cherenkov_density', 'stereo_baseline_phi_m1',
        'stereo_baseline_phi_m2', 'stereo_image_angle',
        'stereo_cos_between_shower'
    ]
    features.extend([float(row[col]) for col in stereo_features])

    features.extend([
        float(row['pointing_zenith']),
        float(row['pointing_azimuth'])
    ])

    features.extend([
        float(row['time_gradient_m1']),
        float(row['time_gradient_m2'])
    ])

    source_m1_features = [
        'source_alpha_m1', 'source_dist_m1',
        'source_cos_delta_alpha_m1', 'source_dca_m1',
        'source_dca_delta_m1'
    ]
    features.extend([float(row[col]) for col in source_m1_features])

    source_m2_features = [
        'source_alpha_m2', 'source_dist_m2',
        'source_cos_delta_alpha_m2', 'source_dca_m2',
        'source_dca_delta_m2'
    ]
    features.extend([float(row[col]) for col in source_m2_features])

    return torch.tensor(features, dtype=torch.float32)


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

        print(f"Original Number of Protons: {self.proton_metadata.num_rows}")
        print(f"Original Number of Gammas: {self.gamma_metadata.num_rows}")
        print(f"Calculated Number of Protons: {self.n_protons}")
        print(f"Calculated Number of Protons: {self.n_gammas}")

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
