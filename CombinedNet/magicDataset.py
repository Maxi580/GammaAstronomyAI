from typing import Any, Dict, Optional, Tuple
import sys
import os
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from CNN.HexLayers.neighbor import find_center_pixel, get_neighbor_list_by_kernel

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

    true_features = [
        'true_energy', 'true_theta', 'true_phi',
        'true_telescope_theta', 'true_telescope_phi',
        'true_first_interaction_height',
        'true_impact_m1', 'true_impact_m2'
    ]
    features.extend([replace_nan(row[col]) for col in true_features])

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

    features.extend([
        replace_nan(row['pointing_zenith']),
        replace_nan(row['pointing_azimuth'])
    ])

    features.extend([
        replace_nan(row['time_gradient_m1']),
        replace_nan(row['time_gradient_m2'])
    ])

    source_m1_features = [
        'source_alpha_m1', 'source_dist_m1',
        'source_cos_delta_alpha_m1', 'source_dca_m1',
        'source_dca_delta_m1'
    ]
    features.extend([replace_nan(row[col]) for col in source_m1_features])

    source_m2_features = [
        'source_alpha_m2', 'source_dist_m2',
        'source_cos_delta_alpha_m2', 'source_dca_m2',
        'source_dca_delta_m2'
    ]
    features.extend([replace_nan(row[col]) for col in source_m2_features])

    assert len(features) == 59, "Total features count mismatch"

    return torch.tensor(features, dtype=torch.float32)


def resize_input(image):
    """Arrays are 1183 long, however the last 144 are always 0"""
    return image[:NUM_OF_HEXAGONS]


def create_neighbor_mask(cog: dict, neighbors_info) -> torch.Tensor:
    center_idx = find_center_pixel(cog['x'], cog['y'])

    mask = torch.ones(1039)
    mask[center_idx] = 0

    for neighbor in neighbors_info[center_idx]:
        if neighbor >= 0:
            mask[neighbor] = 0

    return mask


def read_parquet_limit(filename, max_rows):
    parquet_file_stream = pq.ParquetFile(filename).iter_batches(batch_size=max_rows)

    batch = next(parquet_file_stream)

    return batch.to_pandas()


class MagicDataset(Dataset):
    GAMMA_LABEL: str = 'gamma'
    PROTON_LABEL: str = 'proton'

    def __init__(self, proton_filename: str, gamma_filename: str, mask_rings: Optional[int] = None,
                 max_samples: Optional[int] = None, debug_info: bool = True):
        self.debug_info = debug_info
        self.mask_rings = mask_rings
        if mask_rings is not None:
            self.neighbors_info = get_neighbor_list_by_kernel(mask_rings, pooling=False, pooling_kernel_size=2,
                                                     num_pooling_layers=0)

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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if idx < self.n_protons:
            row = self.proton_data.iloc[idx]
            label = self.PROTON_LABEL
        else:
            row = self.gamma_data.iloc[idx - self.n_protons]
            label = self.GAMMA_LABEL

        noisy_m1 = resize_input(torch.tensor(row['image_m1'], dtype=torch.float32))
        noisy_m2 = resize_input(torch.tensor(row['image_m2'], dtype=torch.float32))

        if self.mask_rings is not None:
            m1_cog = {'x': row['hillas_cog_x_m1'], 'y': row['hillas_cog_y_m1']}
            m2_cog = {'x': row['hillas_cog_x_m2'], 'y': row['hillas_cog_y_m2']}

            mask_m1 = create_neighbor_mask(m1_cog, self.neighbors_info)
            mask_m2 = create_neighbor_mask(m2_cog, self.neighbors_info)

            noisy_m1 = noisy_m1 * mask_m1
            noisy_m2 = noisy_m2 * mask_m2

        features = extract_features(row)

        return noisy_m1, noisy_m2, features, self.labels[label]

    def analyze_noise(self):
        stats = {
            self.PROTON_LABEL: {
                'count': 0,
                'noisy_m1': {'negatives': 0, 'sum': 0, 'squared_sum': 0, 'min': float('inf'), 'max': float('-inf')},
                'clean_m1': {'negatives': 0, 'sum': 0, 'squared_sum': 0, 'min': float('inf'), 'max': float('-inf')},
                'noise_m1': {'negatives': 0, 'sum': 0, 'squared_sum': 0, 'min': float('inf'), 'max': float('-inf')},
                'noisy_m2': {'negatives': 0, 'sum': 0, 'squared_sum': 0, 'min': float('inf'), 'max': float('-inf')},
                'clean_m2': {'negatives': 0, 'sum': 0, 'squared_sum': 0, 'min': float('inf'), 'max': float('-inf')},
                'noise_m2': {'negatives': 0, 'sum': 0, 'squared_sum': 0, 'min': float('inf'), 'max': float('-inf')}
            },
            self.GAMMA_LABEL: {
                'count': 0,
                'noisy_m1': {'negatives': 0, 'sum': 0, 'squared_sum': 0, 'min': float('inf'), 'max': float('-inf')},
                'clean_m1': {'negatives': 0, 'sum': 0, 'squared_sum': 0, 'min': float('inf'), 'max': float('-inf')},
                'noise_m1': {'negatives': 0, 'sum': 0, 'squared_sum': 0, 'min': float('inf'), 'max': float('-inf')},
                'noisy_m2': {'negatives': 0, 'sum': 0, 'squared_sum': 0, 'min': float('inf'), 'max': float('-inf')},
                'clean_m2': {'negatives': 0, 'sum': 0, 'squared_sum': 0, 'min': float('inf'), 'max': float('-inf')},
                'noise_m2': {'negatives': 0, 'sum': 0, 'squared_sum': 0, 'min': float('inf'), 'max': float('-inf')}
            }
        }

        for idx in range(self.length):
            if idx < self.n_protons:
                row = self.proton_data.iloc[idx]
                label = self.PROTON_LABEL
            else:
                row = self.gamma_data.iloc[idx - self.n_protons]
                label = self.GAMMA_LABEL

            noisy_m1 = torch.tensor(row['image_m1'], dtype=torch.float32)
            clean_m1 = torch.tensor(row['clean_image_m1'], dtype=torch.float32)
            noisy_m2 = torch.tensor(row['image_m2'], dtype=torch.float32)
            clean_m2 = torch.tensor(row['clean_image_m2'], dtype=torch.float32)
            noise_m1 = noisy_m1 - clean_m1
            noise_m2 = noisy_m2 - clean_m2

            images = {
                'noisy_m1': noisy_m1,
                'clean_m1': clean_m1,
                'noise_m1': noise_m1,
                'noisy_m2': noisy_m2,
                'clean_m2': clean_m2,
                'noise_m2': noise_m2
            }

            stats[label]['count'] += 1

            for img_name, img in images.items():
                stats[label][img_name]['negatives'] += (img < 0).sum().item()
                stats[label][img_name]['sum'] += img.sum().item()
                stats[label][img_name]['squared_sum'] += (img ** 2).sum().item()
                stats[label][img_name]['min'] = min(stats[label][img_name]['min'], img.min().item())
                stats[label][img_name]['max'] = max(stats[label][img_name]['max'], img.max().item())

            if idx % 1000 == 0:
                print(f"Processed {idx}/{self.length} samples...")

        for label in [self.PROTON_LABEL, self.GAMMA_LABEL]:
            n = stats[label]['count']
            n_pixels = 1039
            total_pixels = n * n_pixels

            for img_type in ['noisy_m1', 'clean_m1', 'noise_m1', 'noisy_m2', 'clean_m2', 'noise_m2']:
                img_stats = stats[label][img_type]

                img_stats['mean'] = img_stats['sum'] / total_pixels
                img_stats['variance'] = (img_stats['squared_sum'] / total_pixels) - (img_stats['mean'] ** 2)
                img_stats['std'] = (img_stats['variance']) ** 0.5
                img_stats['negative_percentage'] = (img_stats['negatives'] / total_pixels) * 100

        print("\nAnalysis Results:")
        for label in [self.PROTON_LABEL, self.GAMMA_LABEL]:
            print(f"\n{label.upper()} ANALYSIS (Total samples: {stats[label]['count']})")
            for img_type in ['noisy_m1', 'clean_m1', 'noise_m1', 'noisy_m2', 'clean_m2', 'noise_m2']:
                print(f"\n  {img_type}:")
                img_stats = stats[label][img_type]
                print(f"    Negative values: {img_stats['negatives']} ({img_stats['negative_percentage']:.2f}%)")
                print(f"    Mean: {img_stats['mean']:.6f}")
                print(f"    Std: {img_stats['std']:.6f}")
                print(f"    Min: {img_stats['min']:.6f}")
                print(f"    Max: {img_stats['max']:.6f}")

        return stats

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

    def analyze_mask_coverage(self, radius_rings: int = 7):
        total_m1 = 0
        total_m2 = 0
        complete_coverage_m1 = 0
        complete_coverage_m2 = 0

        for idx in range(self.length):
            if idx < self.n_protons:
                row = self.proton_data.iloc[idx]
            else:
                row = self.gamma_data.iloc[idx - self.n_protons]

            clean_m1 = resize_input(torch.tensor(row['clean_image_m1'], dtype=torch.float32))
            clean_m2 = resize_input(torch.tensor(row['clean_image_m2'], dtype=torch.float32))

            m1_cog = {
                'x': row['hillas_cog_x_m1'],
                'y': row['hillas_cog_y_m1']
            }

            m2_cog = {
                'x': row['hillas_cog_x_m2'],
                'y': row['hillas_cog_y_m2']
            }

            neighbors_info = get_neighbor_list_by_kernel(self.mask_rings, pooling=False, pooling_kernel_size=2,
                                                         num_pooling_layers=0)

            if clean_m1.max() > 0:
                total_m1 += 1
                mask_m1 = create_neighbor_mask(m1_cog, neighbors_info)
                masked_m1 = clean_m1 * mask_m1

                if (masked_m1 > clean_m1.max() * 0.1).sum() == 0:
                    complete_coverage_m1 += 1

            if clean_m2.max() > 0:
                total_m2 += 1
                mask_m2 = create_neighbor_mask(m2_cog, neighbors_info)
                masked_m2 = clean_m2 * mask_m2

                if (masked_m2 > clean_m2.max() * 0.1).sum() == 0:
                    complete_coverage_m2 += 1

        print(f"total_m1: {total_m1}")
        print(f"total_m1: {total_m2}")
        print(f"complete_coverage_m1: {complete_coverage_m1}")
        print(f"complete_coverage_m2: {complete_coverage_m2}")


if __name__ == '__main__':
    proton_file = 'magic-protons.parquet'
    gamma_file = 'magic-gammas.parquet'

    md = MagicDataset(proton_file, gamma_file, mask_rings=7)
    md.analyze_mask_coverage()
