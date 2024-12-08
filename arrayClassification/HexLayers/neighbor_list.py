from ctapipe.core import Provenance
from ctapipe.instrument import CameraGeometry
from importlib.resources import files


def get_neighbor_indices() -> list[list[int]]:
    """Returns a list of 1039 lists, that has the index of neighbours of each idx"""
    f = str(files("ctapipe_io_magic").joinpath("resources/MAGICCam.camgeom.fits.gz"))
    Provenance().add_input_file(f, role="CameraGeometry")
    return CameraGeometry.from_table(f).neighbors


def get_neighbor_list_by_kernel(kernel_size: int) -> list[list[int]]:
    """
    Get list of neighbors up to specified kernel size rings away
    kernel_size=1 -> immediate neighbors only (6 neighbors)
    kernel_size=2 -> two rings (18 neighbors)
    kernel_size=3 -> three rings (36 neighbors)
    """
    # Get initial immediate neighbors for each hexagon
    base_neighbors = get_neighbor_indices()

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

print(get_neighbor_list_by_kernel(1))