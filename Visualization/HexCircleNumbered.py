import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.pyplot import cm

# Append the parent directory so we can import HexagonPlaneGenerator.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from SimulatedSampleGeneration.PlaneGenerators import HexagonPlaneGenerator


def visualize_hex_circle(target_count: int):
    plane_generator = HexagonPlaneGenerator()

    # Generate the hexagon plane; we only need the hexagon centers and circle_info.
    _, hexagons, circle_info = plane_generator.generate_plane(2000, target_count)
    center_x, center_y, hex_radius, outer_radius, total_rings = circle_info

    # Create a matplotlib figure and axes.
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, 2000)
    ax.set_ylim(0, 2000)
    ax.set_aspect("equal")
    ax.axis("off")  # Hide axes for a cleaner look

    colors = cm.prism(np.linspace(0, 1, total_rings))

    # Draw each hexagon using a color from the colormap based on its index.
    for idx, (hex_center_x, hex_center_y, ring) in enumerate(hexagons):
        # Get the points for the hexagon (using the same generator method)
        points = plane_generator.create_hexagon_points(hex_center_x, hex_center_y, hex_radius)
        # Create a polygon patch with the chosen fill color and a black outline.
        hexagon_patch = patches.Polygon(points, closed=True, facecolor=colors[ring], edgecolor="black", alpha=0.1)
        ax.add_patch(hexagon_patch)
        # Place the hexagon index as text in the center of the hexagon.
        ax.text(hex_center_x, hex_center_y, str(idx),
                ha="center", va="center", fontsize=8, color="black")

    plt.show()
    # plt.savefig("./hex_circle.png")

if __name__ == "__main__":
    TARGET_HEX_COUNT = 1039
    visualize_hex_circle(TARGET_HEX_COUNT)
