from typing import Any, Dict, Optional, Tuple
import os
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CNN.MagicConv.NeighborLogic import get_neighbor_list_by_kernel
from CNN.Architectures.StatsModel import get_batch_stats

NUM_OF_HEXAGONS = 1039


def replace_nan(value):
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

    assert len(features) == 51, "Total features count mismatch"

    return torch.tensor(features, dtype=torch.float32)


def resize_image(image):
    """Arrays are 1183 long, however the last 144 are always 0"""
    return image[:NUM_OF_HEXAGONS]


def precalculate_masks(neighbors_info) -> torch.Tensor:
    all_masks = torch.ones(1039, 1039)

    for center_idx in range(1039):
        all_masks[center_idx, center_idx] = 0
        for neighbor in neighbors_info[center_idx]:
            all_masks[center_idx, neighbor] = 0

    return all_masks


def read_parquet_limit(filename, max_rows):
    parquet_file_stream = pq.ParquetFile(filename).iter_batches(batch_size=max_rows)

    batch = next(parquet_file_stream)

    return batch.to_pandas()


def shuffle_tensor(tensor: torch.Tensor) -> torch.Tensor:
    indices = torch.randperm(len(tensor))
    return tensor[indices]


class MagicDataset(Dataset):
    GAMMA_LABEL: str = 'gamma'
    PROTON_LABEL: str = 'proton'

    def __init__(self, proton_filename: str, gamma_filename: str, mask_rings: Optional[int] = None,
                 shuffle: Optional[bool] = False, max_samples: Optional[int] = None, debug_info: bool = True,
                 min_hillas_size: float = None, min_true_energy: float = None):
        self.debug_info = debug_info
        self.shuffle = shuffle
        self.mask_rings = mask_rings
        if mask_rings is not None:
            neighbors_info = get_neighbor_list_by_kernel(mask_rings, pooling=False, pooling_kernel_size=2,
                                                         num_pooling_layers=0)
            self.all_masks = precalculate_masks(neighbors_info)

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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if idx < self.n_protons:
            row = self.proton_data.iloc[idx]
            label = self.PROTON_LABEL
        else:
            row = self.gamma_data.iloc[idx - self.n_protons]
            label = self.GAMMA_LABEL

        m1 = resize_image(torch.tensor(row['image_m1'], dtype=torch.float32))
        m2 = resize_image(torch.tensor(row['image_m2'], dtype=torch.float32))

        """if self.mask_rings is not None:
            # Masks are precalculated, but we need to know which to use
            clean_m1 = resize_image(torch.tensor(row['clean_image_m1'], dtype=torch.float32))
            clean_m2 = resize_image(torch.tensor(row['clean_image_m2'], dtype=torch.float32))
            m1_center_idx = torch.argmax(clean_m1).item()
            m2_center_idx = torch.argmax(clean_m2).item()

            mask_m1 = self.all_masks[m1_center_idx]
            mask_m2 = self.all_masks[m2_center_idx]

            noisy_m1 *= mask_m1
            noisy_m2 *= mask_m2

        if self.shuffle:
            noisy_m1 = shuffle_tensor(noisy_m1)
            noisy_m2 = shuffle_tensor(noisy_m2)"""

        features = extract_features(row)

        return m1, m2, features, self.labels[label]

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

    def analyze_mask_coverage(self) -> dict:
        if self.mask_rings is None:
            raise ValueError("Dataset must be initialized with mask_rings parameter")

        stats = {
            'total': {'m1': 0, 'm2': 0},
            'pixel_counts': {
                'm1': {'total': 0, 'masked': 0},
                'm2': {'total': 0, 'masked': 0}
            },
            'intensity_values': {
                'm1': {'total': 0.0, 'masked': 0.0},
                'm2': {'total': 0.0, 'masked': 0.0}
            }
        }

        for idx in range(self.length):
            if idx < self.n_protons:
                row = self.proton_data.iloc[idx]
            else:
                row = self.gamma_data.iloc[idx - self.n_protons]

            clean_m1 = torch.tensor(row['clean_image_m1'][:1039], dtype=torch.float32)
            m1_center_idx = torch.argmax(clean_m1).item()

            mask_m1 = self.all_masks[m1_center_idx]
            masked_m1 = clean_m1 * mask_m1

            stats['total']['m1'] += 1
            stats['pixel_counts']['m1']['total'] += (clean_m1 > 0).sum().item()
            stats['pixel_counts']['m1']['masked'] += (masked_m1 > 0).sum().item()
            stats['intensity_values']['m1']['total'] += clean_m1.sum().item()
            stats['intensity_values']['m1']['masked'] += masked_m1.sum().item()

            clean_m2 = torch.tensor(row['clean_image_m2'][:1039], dtype=torch.float32)
            m2_center_idx = torch.argmax(clean_m2).item()

            mask_m2 = self.all_masks[m2_center_idx]
            masked_m2 = clean_m2 * mask_m2

            stats['total']['m2'] += 1
            stats['pixel_counts']['m2']['total'] += (clean_m2 > 0).sum().item()
            stats['pixel_counts']['m2']['masked'] += (masked_m2 > 0).sum().item()
            stats['intensity_values']['m2']['total'] += clean_m2.sum().item()
            stats['intensity_values']['m2']['masked'] += masked_m2.sum().item()

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


def collect_statistics(dataset):
    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    m1_proton_stats = []
    m2_proton_stats = []
    m1_gamma_stats = []
    m2_gamma_stats = []

    total_processed = 0

    for m1, m2, _, labels in loader:
        proton_mask = labels == 0
        gamma_mask = labels == 1

        if proton_mask.any():
            batch_proton_stats_m1 = get_batch_stats(m1[proton_mask])
            batch_proton_stats_m2 = get_batch_stats(m2[proton_mask])
            m1_proton_stats.append(batch_proton_stats_m1)
            m2_proton_stats.append(batch_proton_stats_m2)

        if gamma_mask.any():
            batch_gamma_stats_m1 = get_batch_stats(m1[gamma_mask])
            batch_gamma_stats_m2 = get_batch_stats(m2[gamma_mask])
            m1_gamma_stats.append(batch_gamma_stats_m1)
            m2_gamma_stats.append(batch_gamma_stats_m2)

        total_processed += len(labels)
        if total_processed % 10000 == 0:
            print(f"Processed {total_processed} samples")

    m1_proton_stats = torch.cat(m1_proton_stats, dim=0)
    m2_proton_stats = torch.cat(m2_proton_stats, dim=0)
    m1_gamma_stats = torch.cat(m1_gamma_stats, dim=0)
    m2_gamma_stats = torch.cat(m2_gamma_stats, dim=0)

    stats_dict = {
        "m1_proton": m1_proton_stats.numpy(),
        "m2_proton": m2_proton_stats.numpy(),
        "m1_gamma": m1_gamma_stats.numpy(),
        "m2_gamma": m2_gamma_stats.numpy()
    }

    metric_names = [
        "mean", "std", "neg_ratio", "min", "max",
        "squared_mean", "nonzero_ratio", "q25", "q50", "q75"
    ]

    print("\nComplete Statistics Summary:")
    for telescope_type, stats in stats_dict.items():
        print(f"\nSummary for {telescope_type}:")
        for i, metric in enumerate(metric_names):
            values = stats[:, i]
            print(f"{metric:15} - Min: {values.min():10.6f}, Max: {values.max():10.6f}, "
                  f"Avg: {values.mean():10.6f}, Std: {values.std():10.6f}")

    return stats_dict


def print_summary_statistics(stats_dict):
    metrics = [
        "mean", "std", "neg_ratio", "min", "max",
        "squared_mean", "nonzero_ratio", "q25", "q50", "q75"
    ]

    for telescope_type, stats in stats_dict.items():
        print(f"\nSummary for {telescope_type}:")
        for i, metric in enumerate(metrics):
            values = stats[:, i]
            print(f"{metric:15} - Min: {values.min():10.6f}, Max: {values.max():10.6f}, "
                  f"Avg: {values.mean():10.6f}, Std: {values.std():10.6f}")


def plot_metric_distributions(stats_dict, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)

    metrics = [
        "mean", "std", "neg_ratio", "min", "max",
        "squared_mean", "nonzero_ratio", "q25", "q50", "q75"
    ]

    for metric_idx, metric_name in enumerate(metrics):
        for telescope_type, stats in stats_dict.items():
            plt.figure(figsize=(10, 6))
            values = stats[:, metric_idx]

            plt.plot(range(len(values)), values, linewidth=1)

            plt.title(f"{metric_name} Distribution - {telescope_type}")
            plt.xlabel("Sample Index")
            plt.ylabel(metric_name)
            plt.grid(True, alpha=0.3)

            mean_val = values.mean()
            std_val = values.std()
            plt.axhline(y=mean_val, color='r', linestyle='--', alpha=0.5,
                        label=f'Mean: {mean_val:.4f}')
            plt.axhline(y=mean_val + std_val, color='g', linestyle=':', alpha=0.5,
                        label=f'Mean ± Std')
            plt.axhline(y=mean_val - std_val, color='g', linestyle=':', alpha=0.5)

            plt.legend()
            plt.tight_layout()

            plt.savefig(os.path.join(output_dir, f"{metric_name}_{telescope_type}.png"),
                        dpi=300, bbox_inches='tight')
            plt.close()
