import sys, os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Append the parent directory so we can import HexagonPlaneGenerator.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from SimulatedSampleGeneration.PlaneGenerators import HexagonPlaneGenerator
from CNN.HexCircleLayers.pooling import _get_clusters


def hex_circle_pooled(hex_count: int, kernel_size: int, number_pooled: bool, valid_padding: bool = False):
    # Get clusters from the pooling function.
    clusters = _get_clusters(hex_count, kernel_size)

    plane_generator = HexagonPlaneGenerator()
    # Generate the hexagon plane.
    # Note: Here we assume plane_info[2] contains the hexagon radius.
    background, hexagons, plane_info = plane_generator.generate_plane(2000, hex_count)
    hex_radius = plane_info[2]

    # Create a matplotlib figure and axes.
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, 2000)
    ax.set_ylim(0, 2000)
    ax.set_aspect("equal")
    ax.axis("off")  # Hide axes for a cleaner look

    # Determine the number of clusters and assign each one a unique color using the prism colormap.
    cluster_colors = plt.get_cmap("prism")(np.linspace(0, 1, 20))

    # Keep track of hexagons that are not part of any cluster.
    missing = list(range(hex_count))

    # Draw hexagons belonging to clusters.
    for c_idx, cluster in enumerate(clusters):
        if valid_padding and -1 in cluster:
            continue

        color = cluster_colors[c_idx % 20]
        all_points = {}
        for idx, neighbor_idx in enumerate(cluster):
            if neighbor_idx == -1:
                continue

            hex_center_x, hex_center_y, ring = hexagons[neighbor_idx]
            points = plane_generator.create_hexagon_points(hex_center_x, hex_center_y, hex_radius)

            # Save edge points of every hexagon to later use them to add an outline around every cluster
            for p in points:
                p = (round(p[0], 2), round(p[1], 2))
                if p in all_points:
                    all_points[p] += 1
                else:
                    all_points[p] = 1
            
            hexagon_patch = patches.Polygon(points, closed=True, facecolor=color, edgecolor="black", alpha=0.15)
            ax.add_patch(hexagon_patch)

            if number_pooled and idx == (len(cluster) // 2):
                ax.text(hex_center_x, hex_center_y, str(c_idx),
                        ha="center", va="center", fontsize=(16 * kernel_size), color=(50/255, 50/255, 50/255))
            elif not number_pooled:
                ax.text(hex_center_x, hex_center_y, str(neighbor_idx),
                        ha="center", va="center", fontsize=8, color=(50/255, 50/255, 50/255))

            if neighbor_idx in missing:
                missing.remove(neighbor_idx)

        # Find edge points and sort them to create an outline
        outside_points = [point for point, occurrence in all_points.items() if occurrence <= 2]
        if len(outside_points) == len(all_points.keys()):
            cx = np.mean([p[0] for p in outside_points])
            cy = np.mean([p[1] for p in outside_points])
            # Sort the points by angle relative to the centroid.
            sorted_outside_points = sorted(outside_points, key=lambda p: np.arctan2(p[1] - cy, p[0] - cx))
        else: 
            sorted_outside_points = []
            points_to_sort = outside_points

            while len(points_to_sort) > 0:
                if len(sorted_outside_points) == 0:
                    prev_point = (0,0)
                else:
                    prev_point = sorted_outside_points[-1]

                points_to_sort = sorted(points_to_sort, key=lambda p: math.dist(prev_point, p))
                sorted_outside_points.append(points_to_sort.pop(0))

        outline_patch = patches.Polygon(sorted_outside_points, closed=True, facecolor='none', edgecolor="black", linewidth=1)
        ax.add_patch(outline_patch)

    default_color = (50/255, 50/255, 50/255)
    for neighbor_idx in missing:
        hex_center_x, hex_center_y, *_ = hexagons[neighbor_idx]
        points = plane_generator.create_hexagon_points(hex_center_x, hex_center_y, hex_radius)
        hexagon_patch = patches.Polygon(points, closed=True, facecolor=default_color, edgecolor="black")
        ax.add_patch(hexagon_patch)
        text_to_show = "None" if number_pooled else str(neighbor_idx)
        ax.text(hex_center_x, hex_center_y, text_to_show,
                ha="center", va="center", fontsize=8, color="white")

    plt.savefig(f"../Visuals/hex_circle_pooled-{hex_count}-{kernel_size}.png", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    TARGET_HEX_COUNT = 1039
    KERNEL_SIZE = 1
    NUMBER_POOLED = True
    VALID_PADDING = False # Only for demonstration purposes
    hex_circle_pooled(TARGET_HEX_COUNT, KERNEL_SIZE, NUMBER_POOLED, VALID_PADDING)
