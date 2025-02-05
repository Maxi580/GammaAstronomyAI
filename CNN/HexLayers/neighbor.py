from typing import NamedTuple
from ctapipe.instrument import CameraGeometry
from importlib.resources import files
import torch
import numpy as np


class NeighborInfo(NamedTuple):
    tensor: torch.Tensor
    max_neighbors: int


_NEIGHBOR_CACHE = {}
_CAMERA_CACHE = None


def _get_camera_geometry():
    global _CAMERA_CACHE

    if _CAMERA_CACHE is None:
        f = str(files("ctapipe_io_magic").joinpath("resources/MAGICCam.camgeom.fits.gz"))
        geom = CameraGeometry.from_table(f)

        x_positions = geom.pix_x.to_value()
        y_positions = geom.pix_y.to_value()
        pixel_positions = np.column_stack([x_positions, y_positions])

        _CAMERA_CACHE = {
            'geometry': geom,
            'pixel_positions': pixel_positions,
            'neighbors': geom.neighbors
        }

    return _CAMERA_CACHE


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


def _get_valid_indices_after_pooling(pooling_kernel_size: int, num_pooling_layers: int) -> list[int]:
    stride = pooling_kernel_size ** num_pooling_layers
    pooled_size = 1039 // stride  # Size of Array after Pooling
    valid_indices = list(range(0, pooled_size * stride, stride))
    return valid_indices


def unpool_array(pooled_array: np.ndarray, pooling_kernel_size: int, num_pooling_layers: int) -> [np.ndarray, list[int]]:
    valid_indices = _get_valid_indices_after_pooling(pooling_kernel_size, num_pooling_layers)

    unpooled_array = np.zeros(1039)

    for i, idx in enumerate(valid_indices):
        if i < len(pooled_array):
            unpooled_array[idx] = pooled_array[i]

    return unpooled_array, valid_indices


def _get_pooled_neighbors(neighbors, pooling_kernel_size, num_pooling_layers, pixel_positions):
    # Get valid indices after pooling
    valid_indices = _get_valid_indices_after_pooling(pooling_kernel_size, num_pooling_layers)
    valid_indices_set = set(valid_indices)

    # Create mapping from original to pooled indices
    index_mapping = {idx: i for i, idx in enumerate(valid_indices)}

    pooled_neighbors = []
    for valid_index in valid_indices:
        current_neighbors = set()

        # Process the kernel region starting at this valid index
        # to get not only neighbors of the valid, but also of the indices that were pooled with it
        for i in range(valid_index, min(valid_index + pooling_kernel_size, len(neighbors))):
            # Get neighbors of current hex in kernel
            for neighbor in neighbors[i]:
                if neighbor in valid_indices_set:
                    # Add the not mapped index at first
                    current_neighbors.add(neighbor)

        # Center hex cant be a neighbor
        current_neighbors.discard(valid_index)
        # Sort the neighbors based on their original positions
        sorted_neighbors = _sort_by_angle(pixel_positions, valid_index, list(current_neighbors))
        # After the indices are sorted geometrically, we can map them to their new index
        pooled_sorted_neighbors = [index_mapping[n] for n in sorted_neighbors]
        pooled_neighbors.append(pooled_sorted_neighbors)

    return pooled_neighbors


def _get_neighbor_indices(pooling: bool, pooling_kernel_size: int, num_pooling_layers: int) -> list[list[int]]:
    """Returns a list of Geometrically sorted neighbor indices
       Geometric starts top left and goes clockwise"""
    camera_cache = _get_camera_geometry()
    pixel_positions = camera_cache['pixel_positions']
    neighbors = camera_cache['neighbors']

    if pooling:
        neighbors = _get_pooled_neighbors(neighbors, pooling_kernel_size, num_pooling_layers, pixel_positions)
    else:
        neighbors = [
            _sort_by_angle(pixel_positions, i, neighbor_list)
            for i, neighbor_list in enumerate(neighbors)
        ]

    return neighbors


def get_neighbor_list_by_kernel(kernel_size: int, pooling: bool, pooling_kernel_size: int, num_pooling_layers: int) \
        -> list[list[int]]:
    """
    Get list of neighbors up to specified kernel size rings away
    kernel_size=1 -> immediate neighbors only (6 neighbors)
    kernel_size=2 -> two rings (18 neighbors)
    kernel_size=3 -> three rings (36 neighbors)
    """
    # Get initial immediate neighbors for each hexagon
    base_neighbors = _get_neighbor_indices(pooling, pooling_kernel_size, num_pooling_layers)

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


def get_neighbor_tensor(kernel_size: int, pooling: bool, pooling_kernel_size: int, num_pooling_layers: int) \
        -> NeighborInfo:
    """Move to gpu for efficiency"""
    cache_key = (kernel_size, pooling, pooling_kernel_size, num_pooling_layers)

    if cache_key not in _NEIGHBOR_CACHE:
        neighbors_list = get_neighbor_list_by_kernel(kernel_size, pooling, pooling_kernel_size, num_pooling_layers)
        max_neighbors = max(len(neighbors) for neighbors in neighbors_list)

        padded_neighbors = [
            neighbors + [-1] * (max_neighbors - len(neighbors))
            for neighbors in neighbors_list
        ]
        tensor = torch.tensor(padded_neighbors, dtype=torch.long)
        _NEIGHBOR_CACHE[cache_key] = NeighborInfo(tensor, max_neighbors)

    return _NEIGHBOR_CACHE[cache_key]
