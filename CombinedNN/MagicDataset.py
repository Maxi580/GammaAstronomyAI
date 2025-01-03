import torch
from torch.utils.data import Dataset
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import numpy as np
from typing import Tuple, List
from torch.utils.data import DataLoader, SubsetRandomSampler
from Labels import gamma_label, proton_label


class MAGICDataset(Dataset):
    TRAIN_TEST_SPLIT = 0.7
    CHUNK_SIZE = 1000

    def __init__(self, gamma_file: str, proton_file: str):
        # Open datasets without loading them entirely
        self.gamma_dataset = pq.ParquetFile(gamma_file)
        self.proton_dataset = pq.ParquetFile(proton_file)

        # Get metadata and row counts
        self.gamma_rows = self.gamma_dataset.metadata.num_rows
        self.proton_rows = self.proton_dataset.metadata.num_rows
        self.total_rows = self.gamma_rows + self.proton_rows

        # Get schema information
        schema = self.gamma_dataset.schema

        # Extract non_image/timing parameter columns
        self.param_columns = [
            field.name for field in schema
            if field.name != 'element' and field.name != '__index_level_0__'
        ]
        print(f"Param Columns: {self.param_columns}")

        print(f"Dataset initialized with:")
        print(f"- {self.gamma_rows} gamma events")
        print(f"- {self.proton_rows} proton events")
        print(f"- {len(self.param_columns)} parameters")

    def __len__(self) -> int:
        return self.total_rows

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """Only read the row that is requested"""
        # Determine the dataset and the corresponding idx of the global idx
        is_gamma = idx < self.gamma_rows
        local_idx = idx if is_gamma else idx - self.gamma_rows
        dataset = self.gamma_dataset if is_gamma else self.proton_dataset

        row_group_index = local_idx // dataset.metadata.row_group(0).num_rows
        offset_within_group = local_idx % dataset.metadata.row_group(0).num_rows

        # Read the specific row group
        table = dataset.read_row_group(
            row_group_index,
            columns=['clean_image_m1', 'clean_image_m2'] + self.param_columns
        )
        row = table.slice(offset_within_group, 1).to_pydict()

        # Extract M1 and M2 images
        m1_image = torch.tensor([row['clean_image_m1'][0]], dtype=torch.float32)
        m2_image = torch.tensor([row['clean_image_m2'][0]], dtype=torch.float32)

        # Extract parameters
        parameters = torch.tensor(
            [row[col][0] for col in self.param_columns],
            dtype=torch.float32
        )

        # Determine the label
        label = torch.tensor(gamma_label if is_gamma else proton_label, dtype=torch.long)

        return m1_image, m2_image, parameters, label

    def get_subset_indices(self, split: str = 'train') -> List[int]:
        """Get shuffled indices for train/val split with stratification"""
        # Create stratified indices
        gamma_indices = np.arange(self.gamma_rows)
        proton_indices = np.arange(self.gamma_rows, self.total_rows)

        # Shuffle indices
        rng = np.random.RandomState(42)
        rng.shuffle(gamma_indices)
        rng.shuffle(proton_indices)

        # Split each class according to ratio
        gamma_split = int(self.TRAIN_TEST_SPLIT * len(gamma_indices))
        proton_split = int(self.TRAIN_TEST_SPLIT * len(proton_indices))

        if split == 'train':
            indices = np.concatenate([
                gamma_indices[:gamma_split],
                proton_indices[:proton_split]
            ])
        else:  # validation
            indices = np.concatenate([
                gamma_indices[gamma_split:],
                proton_indices[proton_split:]
            ])

        rng.shuffle(indices)
        return indices.tolist()

    def get_data_loaders(self, batch_size: int = 32, num_workers: int = 4):
        """Get train and validation data loaders"""
        train_sampler = SubsetRandomSampler(self.get_subset_indices('train'))
        val_sampler = SubsetRandomSampler(self.get_subset_indices('val'))

        train_loader = DataLoader(
            self,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            self,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True
        )

        return train_loader, val_loader


if __name__ == "__main__":
    dataset = MAGICDataset(
        gamma_file='magic-gammas.parquet',
        proton_file='magic-protons.parquet'
    )

    # Test loading a single item
    print("\nTesting single item loading:")
    for i in range(0, 10):
        item = dataset[i]
        print(f"M1 image shape: {item[0].shape}")
        print(f"M2 image shape: {item[1].shape}")
        print(f"Parameters shape: {item[2].shape}")
        print(f"Label: {item[3]}")
