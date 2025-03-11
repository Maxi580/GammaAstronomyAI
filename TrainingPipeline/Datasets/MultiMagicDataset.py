from typing import Any, Dict, Optional, Tuple, List
import pandas as pd
import pyarrow.parquet as pq
import torch
from TrainingPipeline.Datasets.MagicDataset import (MagicDataset,
                                                    extract_features,
                                                    read_parquet_limit,
                                                    NUM_OF_HEXAGONS)
import numpy as np


class MultiMagicDataset(MagicDataset):

    def __init__(self, proton_filename: str, gamma_filename: str, max_samples: Optional[int] = None,
                 clean_image: bool = True, rescale_image: bool = True, debug_info: bool = True):
        self.debug_info = debug_info
        self.clean_image = clean_image
        self.rescale_image = rescale_image

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
            print(f"Calculated Number of Gammas: {self.n_gammas}")

        # Read the first num_rows rows
        self.proton_data = read_parquet_limit(proton_filename, self.n_protons)
        self.gamma_data = read_parquet_limit(gamma_filename, self.n_gammas)

        # Group the data by event_number and run_number.
        self.proton_groups = list(self.proton_data.groupby(['event_number', 'run_number']))
        self.gamma_groups = list(self.gamma_data.groupby(['event_number', 'run_number']))
        
        proton_img_counts = {}
        gamma_img_counts = {}
        
        for _, g in self.proton_groups:
            count = len(g)
            
            if count not in proton_img_counts:
                proton_img_counts[count] = 1
            else:
                proton_img_counts[count] += 1
                
        for _, g in self.gamma_groups:
            count = len(g)
            
            if count not in gamma_img_counts:
                gamma_img_counts[count] = 1
            else:
                gamma_img_counts[count] += 1
                
        print("PROTONS", proton_img_counts)
        print("GAMMAS", gamma_img_counts)


        # Total number of groups becomes the dataset length.
        self.n_protons = len(self.proton_groups)
        self.n_gammas = len(self.gamma_groups)
        self.length = self.n_protons + self.n_gammas
        self.labels = {self.PROTON_LABEL: 0, self.GAMMA_LABEL: 1}

        if self.debug_info:
            print(f"\nDataset initialized with {self.length} total groups:")
            print(f"Proton groups: {self.n_protons}")
            print(f"Gamma groups: {self.n_gammas}")
            print(f"Label mapping: {self.labels}")

    def __getitem__(self, idx: int) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]:
        # Determine whether the index corresponds to a proton or gamma group.
        if idx < self.n_protons:
            # proton_groups is a list of (group_key, group_dataframe)
            _, group = self.proton_groups[idx]
            label = self.PROTON_LABEL
        else:
            _, group = self.gamma_groups[idx - self.n_protons]
            label = self.GAMMA_LABEL

        images_m1 = []
        images_m2 = []
        features = []
        orig_len = 0
        # Process each row in the group
        for _, row in group.head(16).iterrows():
            image_m1 = torch.tensor(row['clean_image_m1' if self.clean_image else 'image_m1'], dtype=torch.float32)
            image_m2 = torch.tensor(row['clean_image_m2' if self.clean_image else 'image_m2'], dtype=torch.float32)

            images_m1.append(self._convert_image(self._rescale_image(image_m1)))
            images_m2.append(self._convert_image(self._rescale_image(image_m2)))
            features.append(torch.tensor(extract_features(row), dtype=torch.float32))
            orig_len += 1
            
        # Pad the lists so that each group has exactly 16 samples.
        pad_length = 16 - orig_len
        if pad_length > 0:
            # Here we assume each image vector is of length 1039.
            pad_m1 = torch.full((pad_length, 1039), -1.0, dtype=torch.float32)
            pad_m2 = torch.full((pad_length, 1039), -1.0, dtype=torch.float32)
            # For features, we assume a consistent feature length.
            feature_length = features[0].size(0) if features else 0
            pad_features = torch.full((pad_length, feature_length), -1.0, dtype=torch.float32)
            
            images_m1 = torch.cat([torch.stack(images_m1), pad_m1], dim=0)
            images_m2 = torch.cat([torch.stack(images_m2), pad_m2], dim=0)
            features = torch.cat([torch.stack(features), pad_features], dim=0)
        else:
            images_m1 = torch.stack(images_m1)
            images_m2 = torch.stack(images_m2)
            features = torch.stack(features)

        return images_m1, images_m2, features, orig_len, self.labels[label]

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

        return {
            'total_samples': total_samples,
            'distribution': distribution, 
        }
