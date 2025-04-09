import sys, os
from itertools import groupby

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
from matplotlib import gridspec

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CNN.HexCircleLayers.neighbors import get_axial_coords


def convert_image(image: np.ndarray) -> np.ndarray:
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
    
    grid = np.full((size_y, size_x), fill_value=0, dtype=float)
    
    for group in x_groups:
        min_offset_y = min(group, key=lambda x: x[1][1])[1][1]
        offset_y = ((size_y - len(group)) // 2) - min_offset_y
        
        for idx, (x, y) in group:
            grid[y + offset_y][x + max_x] = image[idx]
        
    return grid

def plot_image_hexagdly(image: np.ndarray, as_hex: bool = False, cmap ='viridis'):
    converted = convert_image(image)

    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_aspect("equal")
    ax.axis("off")  # Hide axes for a cleaner look
    

    if as_hex:
        # Source: https://github.com/ai4iacts/hexagdly/blob/master/notebooks/hexagdly_tools.py
        ax.set_xlim([-1, len(converted)])
        ax.set_ylim([-len(converted), 1])

        npixel = 0
        intensities = []
        hexagons = []
        for x in range(len(converted[0])):
            for y in range(len(converted)):
                intensity = converted[y, x]
                hexagon = RegularPolygon(
                    (x * np.sqrt(3) / 2, -(y + np.mod(x, 2) * 0.5)),
                    6,
                    radius=0.577349,
                    orientation=np.pi / 6,
                )
                intensities.append(intensity)
                hexagons.append(hexagon)
                npixel += 1

        p = PatchCollection(
            np.array(hexagons), cmap=cmap, alpha=0.9, edgecolors="k", linewidth=1
        )
        p.set_array(np.array(np.array(intensities)))
        ax.add_collection(p)
    else:
        # Flipping upside-down because pcolormesh flips it too.
        ax.pcolormesh(np.flipud(converted), edgecolors='black', cmap=cmap, linewidth=0.5)

    plt.colorbar(converted, ax=ax, label='Intensity')
    plt.tight_layout()
    plt.savefig(f'../Visuals/hexagdly_{len(image)}px_{'hex' if as_hex else 'square'}.png', bbox_inches='tight')
    plt.show()



if __name__ == "__main__":
    IMAGE = np.full(1039, fill_value=1, dtype=float) # Or add a real image here
    AS_HEX = True
    plot_image_hexagdly(IMAGE, AS_HEX)