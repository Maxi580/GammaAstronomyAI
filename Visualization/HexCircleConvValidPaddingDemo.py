import sys, os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Append the parent directory so we can import HexagonPlaneGenerator.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from SimulatedSampleGeneration.PlaneGenerators import HexagonPlaneGenerator
from CNN.HexCircleLayers.neighbors import get_neighbors_for_kernel

"""
THIS FILE IS ONLY MEANT FOR DEMONSTRATION PURPOSES WHY VALID PADDING IS NOT DOABLE
"""

def calc_kernel_pixels(kernel_size: int):
    return 1 + sum(range(1, kernel_size+1)) * 6

def hex_circle_valid_padding(hex_count: int, kernel_size: int):
    # Get clusters from the pooling function.
    clusters = get_neighbors_for_kernel(hex_count, kernel_size)
    clusters = [[c2 for c2 in c if c2 >= 0] for c in clusters]

    plane_generator = HexagonPlaneGenerator()
    # Generate the hexagon plane.
    # Note: Here we assume plane_info[2] contains the hexagon radius.
    background, hexagons, plane_info = plane_generator.generate_plane(2000, hex_count)
    hex_radius = plane_info[2]

    # Create a matplotlib figure and axes.
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-1000, 1000)
    ax.set_ylim(-1000, 1000)
    ax.set_aspect("equal")
    ax.axis("off")  # Hide axes for a cleaner look


    # Keep track of hexagons that are not part of any cluster.
    missing = list(range(hex_count))

    kernel_pixels = calc_kernel_pixels(kernel_size)
    remaining_pixels = [c for c in clusters if len(c) == kernel_pixels]
    all_x_coords = []

    # Draw hexagons belonging to clusters.
    for c_idx, cluster in enumerate(remaining_pixels):

        existing_idx = cluster[(len(cluster) // 2)]
        hex_center_x, hex_center_y, ring = hexagons[existing_idx]
        points = plane_generator.create_hexagon_points(hex_center_x-1000, hex_center_y-1000, hex_radius)
        all_x_coords += [x for x,y in points]
        
        hexagon_patch = patches.Polygon(points, closed=True, facecolor="lightgray", edgecolor="black", alpha=0.15)
        ax.add_patch(hexagon_patch)

        if existing_idx in missing:
            missing.remove(existing_idx)

    

        # Find edge points and sort them to create an outline
        # outside_points = [point for point, occurrence in all_points.items() if occurrence <= 2]
        # if len(outside_points) == len(all_points.keys()):
        #     cx = np.mean([p[0] for p in outside_points])
        #     cy = np.mean([p[1] for p in outside_points])
        #     # Sort the points by angle relative to the centroid.
        #     sorted_outside_points = sorted(outside_points, key=lambda p: np.arctan2(p[1] - cy, p[0] - cx))
        # else: 
        #     sorted_outside_points = []
        #     points_to_sort = outside_points

        #     while len(points_to_sort) > 0:
        #         if len(sorted_outside_points) == 0:
        #             prev_point = (0,0)
        #         else:
        #             prev_point = sorted_outside_points[-1]

        #         points_to_sort = sorted(points_to_sort, key=lambda p: math.dist(prev_point, p))
        #         sorted_outside_points.append(points_to_sort.pop(0))

        # if c_idx in [766, 765, 764, 859, 858]:
        #     outline_patch = patches.Polygon(sorted_outside_points, closed=True, facecolor='none', edgecolor=color, linewidth=3)
        #     ax.add_patch(outline_patch)

    # Plot hexagons which would be removed because of the convolution
    default_color = (50/255, 50/255, 50/255)
    for neighbor_idx in missing:
        hex_center_x, hex_center_y, *_ = hexagons[neighbor_idx]
        points = plane_generator.create_hexagon_points(hex_center_x-1000, hex_center_y-1000, hex_radius)
        hexagon_patch = patches.Polygon(points, closed=True, facecolor=default_color, edgecolor="black")
        ax.add_patch(hexagon_patch)

    dist = round(((max(all_x_coords) * 2) * 50) / 45) -65 # -65 needs to be adapted manually to fit the outline correctly
    new_hexcount = len(remaining_pixels)
    print("Hexcount after convolution would be", new_hexcount)
    # Plot the outline of hex circle which would be correct for the number of remaining pixels after convolution
    background, hexagons, plane_info = plane_generator.generate_plane(dist, new_hexcount)
    hex_radius = plane_info[2]

    outline_points = {}
    for hex_center_x, hex_center_y, ring in hexagons:
        points = plane_generator.create_hexagon_points(hex_center_x-(dist//2), hex_center_y-(dist//2), hex_radius)

        # Save edge points of every hexagon to later use them to add an outline
        for p in points:
            p = (round(p[0], 2), round(p[1], 2))
            if p in outline_points:
                outline_points[p] += 1
            else:
                outline_points[p] = 1

    outside_points = [point for point, occurrence in outline_points.items() if occurrence <= 2]
    sorted_outside_points = []
    points_to_sort = outside_points

    while len(points_to_sort) > 0:
        if len(sorted_outside_points) == 0:
            prev_point = (0,0)
        else:
            prev_point = sorted_outside_points[-1]

        points_to_sort = sorted(points_to_sort, key=lambda p: math.dist(prev_point, p))
        sorted_outside_points.append(points_to_sort.pop(0))

    outline_patch = patches.Polygon(sorted_outside_points, closed=True, facecolor='none', edgecolor='red', linewidth=2)
    ax.add_patch(outline_patch)

    plt.show()
    # plt.savefig(f"./Visuals/hex_circle_pooled-{hex_count}-{kernel_size}.png")


if __name__ == "__main__":
    TARGET_HEX_COUNT = 1039
    KERNEL_SIZE = 1

    hex_circle_valid_padding(TARGET_HEX_COUNT, KERNEL_SIZE)
