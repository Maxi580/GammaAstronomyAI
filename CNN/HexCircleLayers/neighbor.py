from math import sqrt
import torch
from typing import List, Tuple, Dict

_AXIAL_CACHE: Dict[str, List[Tuple[int, int]]] = {}
_NEIGHBOR_CACHE: Dict[Tuple[int, int], torch.Tensor] = {}


def point_in_circle(px: float, py: float, outer_radius: float) -> bool:
    """
    Return True if point (px,py) is within the circle (center, radius).
    Center coordinates are always equal to outer_radius.
    """
    dx = px - outer_radius
    dy = py - outer_radius
    return (dx * dx + dy * dy) <= outer_radius * outer_radius


def axial_to_pixel(q: int, r: int, center: float, hex_radius: float) -> Tuple[float, float]:
    """
    Convert axial coordinates (q, r) to pixel coordinates.
    (This is used only for checking if a hex falls inside the circle.)
    """
    x = center + (3 / 2) * hex_radius * q
    y = center + sqrt(3) * hex_radius * (r + q / 2)
    return x, y


def generate_spiral_hexagons(hex_radius: float, outer_radius: float) -> List[Tuple[int, int]]:
    """
    Generate hexagons in spiral order from the center outward.
    Returns a list of axial coordinates (q, r) for each hexagon that is inside the circle.
    """
    hexagons: List[Tuple[int, int]] = []

    # Axial directions used for spiral generation (order matters for the spiral order)
    directions = [
        (-1, 1),   # Southwest
        (0, 1),    # South
        (1, 0),    # Southeast
        (1, -1),   # Northeast
        (0, -1),   # North
        (-1, 0),   # Northwest
    ]

    # Start with the center hexagon (axial coordinate (0,0))
    q, r = 0, 0
    x, y = axial_to_pixel(q, r, outer_radius, hex_radius)
    if point_in_circle(x, y, outer_radius):
        hexagons.append((q, r))

    ring = 1
    while True:
        added_in_ring = False
        q, r = 0, -ring  # Starting point for the ring
        for direction in directions:
            for _ in range(ring):
                x, y = axial_to_pixel(q, r, outer_radius, hex_radius)
                if point_in_circle(x, y, outer_radius):
                    hexagons.append((q, r))
                    added_in_ring = True
                # Step to the next hex in this direction
                q += direction[0]
                r += direction[1]
        if not added_in_ring:
            break
        ring += 1

    return hexagons

def generate_neighbors_from_axial(hexagons: List[Tuple[int, int]], kernel_size: int) -> List[List[int]]:
    """
    Given a list of hexagons (as axial coordinates), return a list where
    each element is a list of indices corresponding to the neighbors of that hexagon.
    """
    # Create a mapping from axial coordinate to its index
    hex_index = {hex_coord: idx for idx, hex_coord in enumerate(hexagons)}

    neighbors: List[List[int]] = []
    for q, r in hexagons:
        n_list = []
        # Loop over all candidate offsets in the square defined by the kernel_size.
        for dq in range(-kernel_size, kernel_size + 1):
            for dr in range(-kernel_size, kernel_size + 1):
                # Check if the offset lies within a hexagon shape.
                # In axial coordinates, the distance can be defined as:
                #     distance = max(|dq|, |dr|, | -dq - dr| )
                if max(abs(dq), abs(dr), abs(-dq - dr)) <= kernel_size:
                    neighbor_coord = (q + dq, r + dr)
                    if neighbor_coord in hex_index:
                        n_list.append(hex_index[neighbor_coord])
                    else:
                        # -1 means the value is out of range of the hex circle
                        # required to keep the order of the neighbor indices
                        n_list.append(-1)
        neighbors.append(n_list)
    return neighbors


def get_axial_coords(hex_count_target: int) -> List[Tuple[int, int]]:
    # list of axial coordinates IS NOT cached, so we need to cache it first
    if str(hex_count_target) not in _AXIAL_CACHE:
        outer_radius = 1000

        # Binary search to find a hex_radius that produces approximately target_count hexagons.
        min_ratio = 0.01
        max_ratio = 0.2
        best_ratio = None
        best_count = 0

        while max_ratio - min_ratio > 0.0001:
            mid_ratio = (min_ratio + max_ratio) / 2
            hex_radius = outer_radius * mid_ratio
            hexagons = generate_spiral_hexagons(hex_radius, outer_radius)
            count = len(hexagons)

            if count == hex_count_target:
                best_ratio = mid_ratio
                best_count = count
                break
            elif count < hex_count_target:
                max_ratio = mid_ratio
            else:
                min_ratio = mid_ratio

            if best_ratio is None or abs(count - hex_count_target) < abs(best_count - hex_count_target):
                best_ratio = mid_ratio
                best_count = count

        final_hex_radius = outer_radius * best_ratio
        
        axial_coords = generate_spiral_hexagons(final_hex_radius, outer_radius)
        _AXIAL_CACHE[str(len(axial_coords))] = axial_coords
        return axial_coords

    # list of axial coordinates IS now cached 
    if _AXIAL_CACHE[str(hex_count_target)] == -1:
        raise ValueError(f"{hex_count_target} is an invalid hex_count.")
    
    axial_coords = _AXIAL_CACHE[str(hex_count_target)].copy()
    
    # Check if targeted count matches result, if not we safe the "invalid" hex count
    if len(axial_coords) != hex_count_target:
        _AXIAL_CACHE[str(hex_count_target)] = -1
        raise ValueError(f"{hex_count_target} is an invalid hex_count. Next possible count is {len(axial_coords)}")
    
    return axial_coords


def get_neighbors_for_kernel(hex_count_target: int, kernel_size: int) -> List[List[int]]:
    if kernel_size < 0:
        raise ValueError(f"kernel_size must be a positive integer or zero.")
    
    axial_coords = get_axial_coords(hex_count_target)
    
    return generate_neighbors_from_axial(axial_coords, kernel_size)


def get_neighbor_tensor(hex_count_target: int, kernel_size: int) -> torch.Tensor:
    if hex_count_target in _AXIAL_CACHE and _AXIAL_CACHE[str(hex_count_target)] == -1:
        raise ValueError(f"{hex_count_target} is an invalid hex_count.")

    cache_key = (hex_count_target, kernel_size)

    if cache_key not in _NEIGHBOR_CACHE:
        neighbors_list = get_neighbors_for_kernel(hex_count_target, kernel_size)
        _NEIGHBOR_CACHE[cache_key] = torch.tensor(neighbors_list, dtype=torch.long)

    return _NEIGHBOR_CACHE[cache_key]
