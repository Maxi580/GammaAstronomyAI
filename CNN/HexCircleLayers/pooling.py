from typing import List, Tuple, Dict
import math

import torch
import torch.nn

from CNN.HexCircleLayers.neighbor import get_axial_coords, get_neighbors_for_kernel, get_neighbors_coords

_CLUSTER_CACHE: Dict[Tuple[int, int, bool], List[List[int]]] = {}

def _get_clusters(orig_hex_count: int, pooling_size: int) -> List[List[int]]:
    # 1. Get axial coords and neighbors for original hex circle.
    orig_axial_coords = get_axial_coords(orig_hex_count)
    neighbors = get_neighbors_for_kernel(orig_hex_count, pooling_size)
    
    # Create a mapping from axial coordinate to its index
    axial_coords_index = {axial_coord: idx for idx, axial_coord in enumerate(orig_axial_coords)}
    
    # 2. Define directions for the center of the pooling clusters based on size.
    directions = [
        (-pooling_size-1, pooling_size*2+1),
        (pooling_size, pooling_size+1),
        (pooling_size*2+1, -pooling_size),
        (pooling_size+1, -2*pooling_size-1),
        (-pooling_size, -pooling_size-1),
        (-2*pooling_size-1, pooling_size),
    ]
    
    # Initiate lists of cluster centers and pixel indices to be clustered.
    clusters: List[List[int]] = []
    
    # Initiate axial coords in the very center of the circle.
    q, r = 0, 0
    
    idx = axial_coords_index[(q, r)]
    clusters.append(neighbors[idx])

    # Iterate over the original circle as long as the center coordinate of the next cluster exists.
    ring = 1
    while True:
        added_in_ring = False
        q += -pooling_size
        r += -pooling_size-1
        
        for direction in directions:
            for _ in range(ring):
                # If center axial coords of the kernel are not existent,
                # we calculate all coords that fit in the kernel and check
                # which ones match our original hex circle.
                if (q, r) not in axial_coords_index:
                    neighbs = get_neighbors_coords((q, r), pooling_size)
                    neighbs_filtered = []
                    
                    # Loop over all possible coords and check if they exists in the original circle
                    for neighbor_coords in neighbs:
                        if neighbor_coords in axial_coords_index:
                            neighbs_filtered.append(axial_coords_index[neighbor_coords])
                        else:
                            # -1 means the value is out of range of the hex circle
                            # required to keep the order of the neighbor indices
                            neighbs_filtered.append(-1)
                            
                    # Check if at least one pixel is inside the original circle.
                    if max(neighbs_filtered) > -1 and neighbs_filtered not in clusters:
                        clusters.append(neighbs_filtered)
                        added_in_ring = True

                # Center coordinate of kernel is in the original circle
                else:
                    idx = axial_coords_index[(q, r)]
                    neighbs = neighbors[idx]
                    
                    if neighbs not in clusters:
                        clusters.append(neighbs)
                        added_in_ring = True
                    
                # Step to the next hex in this direction
                q += direction[0]
                r += direction[1]
        
        # For each ring, check if at least one more cluster was added, else break the while-loop.
        if not added_in_ring:
            break
        ring += 1

    return clusters


def get_clusters(orig_hex_count: int, kernel_size: int) -> List[List[int]]:
    cache_key = (orig_hex_count, kernel_size)

    if cache_key not in _CLUSTER_CACHE:
        _CLUSTER_CACHE[cache_key] = _get_clusters(orig_hex_count, kernel_size)

    return _CLUSTER_CACHE[cache_key]