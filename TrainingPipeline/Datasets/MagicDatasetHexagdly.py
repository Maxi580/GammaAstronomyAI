from itertools import groupby

import numpy as np

from TrainingPipeline.Datasets.MagicDataset import MagicDataset
from CNN.HexCircleLayers.neighbors import get_axial_coords



class MagicDatasetHexagdly(MagicDataset):
    def _convert_image(self, image: np.ndarray) -> np.ndarray:
        coords = get_axial_coords(len(image))
        
        # Calculating size for x axis
        min_x = min(coords, key=lambda x: x[0])[0]
        max_x = max(coords, key=lambda x: x[0])[0]
        size_x = abs(min_x) + max_x + 1
        
        # Calculate size for y axis
        indexed_coords = list(enumerate(coords))
        indexed_coords.sort(key=lambda x: x[1][0])
        x_groups = [list(group[1]) for group in groupby(indexed_coords, key=lambda x: x[1][0])]
        size_y = max([len(g) for g in x_groups])
        
        grid = np.full((size_y, size_x), fill_value=-1, dtype=float)
        
        for group in x_groups:
            min_offset_y = min(group, key=lambda x: x[1][1])[1][1]
            offset_y = ((size_y - len(group)) // 2) - min_offset_y
            
            for idx, (x, y) in group:
                grid[y + offset_y][x + max_x] = image[idx]
            
        return grid