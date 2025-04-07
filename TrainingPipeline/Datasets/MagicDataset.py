from typing import Any, Dict, Optional, Tuple
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
import numpy as np

NUM_OF_HEXAGONS = 1039


def replace_nan(value):
    """Replace missing Values with 0"""
    try:
        val = float(value)
        return 0.0 if pd.isna(val) else val
    except (ValueError, TypeError):
        return 0.0


def extract_features(row: pd.Series) -> torch.Tensor:
    features = []

    hillas_m1_features = [
        'hillas_length_m1', 'hillas_width_m1', 'hillas_delta_m1',
        'hillas_size_m1', 'hillas_cog_x_m1', 'hillas_cog_y_m1',
        'hillas_sin_delta_m1', 'hillas_cos_delta_m1'
    ]
    features.extend([replace_nan(row[col]) for col in hillas_m1_features])

    hillas_m2_features = [
        'hillas_length_m2', 'hillas_width_m2', 'hillas_delta_m2',
        'hillas_size_m2', 'hillas_cog_x_m2', 'hillas_cog_y_m2',
        'hillas_sin_delta_m2', 'hillas_cos_delta_m2'
    ]
    features.extend([replace_nan(row[col]) for col in hillas_m2_features])

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
    features.extend([replace_nan(row[col]) for col in stereo_features])

    assert len(features) == 37, "Total features count mismatch"

    return torch.tensor(features, dtype=torch.float32)


def read_parquet_limit(filename, max_rows):
    parquet_file_stream = pq.ParquetFile(filename).iter_batches(batch_size=max_rows)

    batch = next(parquet_file_stream)

    return batch.to_pandas()


class MagicDataset(Dataset):
    GAMMA_LABEL: str = 'gamma'
    PROTON_LABEL: str = 'proton'

    def __init__(self,
                 proton_filename: str,
                 gamma_filename: str,
                 max_samples: Optional[int] = None,
                 clean_image: bool = True,
                 rescale_image: bool = True,
                 debug_info: bool = True,
                 additional_features: list = [],
                 min_hillas_size: float = None,
                 min_true_energy: float = None,
                 ):
        self.debug_info = debug_info
        self.clean_image = clean_image
        self.rescale_image = rescale_image
        self.additional_features = additional_features

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

        if self.debug_info:
            print(f"Original Number of Protons: {self.proton_metadata.num_rows}")
            print(f"Original Number of Gammas: {self.gamma_metadata.num_rows}")
            print(f"Calculated Number of Protons: {self.n_protons}")
            print(f"Calculated Number of Protons: {self.n_gammas}")

        # Read the first num_rows rows
        self.proton_data = read_parquet_limit(proton_filename, self.n_protons)
        self.gamma_data = read_parquet_limit(gamma_filename, self.n_gammas)

        if min_hillas_size is not None:
            original_proton_count = len(self.proton_data)
            original_gamma_count = len(self.gamma_data)

            self.proton_data = self.proton_data[
                (self.proton_data['hillas_size_m1'] >= min_hillas_size) |
                (self.proton_data['hillas_size_m2'] >= min_hillas_size)
                ]
            self.gamma_data = self.gamma_data[
                (self.gamma_data['hillas_size_m1'] >= min_hillas_size) |
                (self.gamma_data['hillas_size_m2'] >= min_hillas_size)
                ]

            self.n_protons = len(self.proton_data)
            self.n_gammas = len(self.gamma_data)

            if self.debug_info:
                print(f"\nApplied Hillas size filter (hillas_size >= {min_hillas_size}):")
                print(
                    f"Protons: {original_proton_count} → {self.n_protons} ({self.n_protons / original_proton_count * 100:.1f}%)")
                print(
                    f"Gammas: {original_gamma_count} → {self.n_gammas} ({self.n_gammas / original_gamma_count * 100:.1f}%)")

        if min_true_energy is not None:
            original_proton_count = len(self.proton_data)
            original_gamma_count = len(self.gamma_data)

            self.proton_data = self.proton_data[
                self.proton_data['true_energy'] >= min_true_energy
                ]
            self.gamma_data = self.gamma_data[
                self.gamma_data['true_energy'] >= min_true_energy
                ]

            self.n_protons = len(self.proton_data)
            self.n_gammas = len(self.gamma_data)

            if self.debug_info:
                print(f"\nApplied true energy filter (true_energy >= {min_true_energy}):")
                print(
                    f"Protons: {original_proton_count} → {self.n_protons} ({self.n_protons / original_proton_count * 100:.1f}%)")
                print(
                    f"Gammas: {original_gamma_count} → {self.n_gammas} ({self.n_gammas / original_gamma_count * 100:.1f}%)")

        # self.proton_data = pd.read_parquet(proton_filename, engine='fastparquet', rows=self.n_protons)
        # self.gamma_data = pd.read_parquet(gamma_filename, engine='fastparquet', rows=self.n_gammas)
        self.length = self.n_protons + self.n_gammas
        self.labels = {self.PROTON_LABEL: 0, self.GAMMA_LABEL: 1}

        if self.debug_info:
            print(f"\nDataset initialized with {self.length} total samples:")
            print(f"Protons: {self.n_protons}")
            print(f"Gammas: {self.n_gammas}")
            print(f"Label mapping: {self.labels}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, list]:
        if idx < self.n_protons:
            row = self.proton_data.iloc[idx]
            label = self.PROTON_LABEL
        else:
            row = self.gamma_data.iloc[idx - self.n_protons]
            label = self.GAMMA_LABEL

        image_m1 = np.array(row['clean_image_m1' if self.clean_image else 'image_m1'])
        image_m2 = np.array(row['clean_image_m2' if self.clean_image else 'image_m2'])

        image_m1 = torch.tensor(
            self._convert_image(self._rescale_image(image_m1)),
            dtype=torch.float32
        )
        image_m2 = torch.tensor(
            self._convert_image(self._rescale_image(image_m2)),
            dtype=torch.float32
        )

        features = extract_features(row)

        return image_m1, image_m2, features, self.labels[label], [row[feat] for feat in self.additional_features]
    
    def _rescale_image(self, image):
        """Arrays are 1183 long, however the last 144 are always 0"""
        image = image[:NUM_OF_HEXAGONS]
        
        # Rescale image to values between 0 and 1.
        # Negative values are set to 0.
        if self.rescale_image:
            image[image < 0] = 0
            if image.max() > 0:
                image = (image - image.min()) / (image.max() - image.min())

        return image
    
    def _convert_image(self, image: np.ndarray) -> np.ndarray:
        return image

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
    
    def get_all_labels(self) -> np.ndarray:
        # First n_protons items are proton labels (Logic is mirrored magicDataset)
        labels = np.full(self.n_protons + self.n_gammas, self.labels[self.PROTON_LABEL])
        # Last n_gammas items are gamma labels
        labels[self.n_protons:] = self.labels[self.GAMMA_LABEL]
        
        return labels
