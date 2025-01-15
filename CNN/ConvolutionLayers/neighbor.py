from typing import NamedTuple
from ctapipe.instrument import CameraGeometry
from importlib.resources import files
import torch
import numpy as np


class NeighborInfo(NamedTuple):
    tensor: torch.Tensor
    max_neighbors: int


_NEIGHBOR_CACHE = {}


def _sort_by_angle(pixel_positions, center_idx: int, neighbor_indices: list[int]) -> list[int]:
    """Resorting indices geometrically, by calculating angle that is always the same."""
    center_pos = pixel_positions[center_idx]

    neighbor_angles = [
        np.arctan2(
            pixel_positions[n][1] - center_pos[1],  # y
            pixel_positions[n][0] - center_pos[0]  # x
        ) for n in neighbor_indices
    ]

    # Sort neighbors by angle (clockwise from top)
    return [n for _, n in sorted(zip(neighbor_angles, neighbor_indices))]


def _get_valid_indices_after_pooling(pooling_kernel_size: int, num_pooling_layers: int):
    stride = pooling_kernel_size ** num_pooling_layers
    valid_indices = torch.arange(0, 1039, stride)
    return valid_indices


def _get_neighbor_indices(pooled: bool, pooling_kernel_size: int, num_pooling_layers: int) -> list[list[int]]:
    """Returns a list of Geometrically sorted neighbor indices
       Geometric starts top left and goes clockwise"""
    f = str(files("ctapipe_io_magic").joinpath("resources/MAGICCam.camgeom.fits.gz"))
    geom = CameraGeometry.from_table(f)

    pixel_positions = np.column_stack([geom.pix_x, geom.pix_y])
    neighbors = geom.neighbors

    # Sort neighbors
    neighbors = [
        _sort_by_angle(pixel_positions, i, neighbor_list)
        for i, neighbor_list in enumerate(neighbors)
    ]

    # Reduce Spatial Size if data was pooled
    if pooled:
        pooled_neighbors = []
        valid_indices = _get_valid_indices_after_pooling(pooling_kernel_size, num_pooling_layers)

        for valid_index in valid_indices:
            valid_neighbor = neighbors[valid_index]
            new_neighbors = [n for n in valid_neighbor if n in valid_indices]
            pooled_neighbors.append(new_neighbors)
        neighbors = pooled_neighbors

    return neighbors


def _get_neighbor_list_by_kernel(kernel_size: int, pooled: bool, pooling_kernel_size: int, num_pooling_layers: int)\
        -> list[list[int]]:
    """
    Get list of neighbors up to specified kernel size rings away
    kernel_size=1 -> immediate neighbors only (6 neighbors)
    kernel_size=2 -> two rings (18 neighbors)
    kernel_size=3 -> three rings (36 neighbors)
    """
    # Get initial immediate neighbors for each hexagon
    base_neighbors = _get_neighbor_indices(pooled, pooling_kernel_size, num_pooling_layers)

    # For each hexagon, build expanded neighbor list
    expanded_neighbors = [[] for _ in range(len(base_neighbors))]

    # for each hexagon calculate all neighbour indices
    for hex_idx in range(len(base_neighbors)):
        current_ring = set(base_neighbors[hex_idx])  # First ring of neighbors

        # For each additional ring requested
        for ring in range(1, kernel_size):
            # Find next ring by getting neighbors of current ring
            new_ring = set()
            for n in current_ring:
                new_ring.update(base_neighbors[n])

            # Remove already processed hexagons and current hexagon
            new_ring -= current_ring

            # Update rings for next iteration
            current_ring.update(new_ring)

        expanded_neighbors[hex_idx] = list(current_ring)

    return expanded_neighbors


def get_neighbor_tensor(kernel_size: int, pooled: bool, pooling_kernel_size: int, num_pooling_layers: int) \
        -> NeighborInfo:
    """Move to gpu for efficiency"""
    if kernel_size not in _NEIGHBOR_CACHE:
        neighbors_list = _get_neighbor_list_by_kernel(kernel_size, pooled, pooling_kernel_size, num_pooling_layers)
        max_neighbors = max(len(neighbors) for neighbors in neighbors_list)

        padded_neighbors = [
            neighbors + [-1] * (max_neighbors - len(neighbors))
            for neighbors in neighbors_list
        ]
        tensor = torch.tensor(padded_neighbors, dtype=torch.long)
        _NEIGHBOR_CACHE[kernel_size] = NeighborInfo(tensor, max_neighbors)

    return _NEIGHBOR_CACHE[kernel_size]
