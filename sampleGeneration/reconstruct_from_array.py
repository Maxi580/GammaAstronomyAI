# File: reconstruct_from_array.py

import os
import json
from typing import Tuple, List

from PIL import Image, ImageDraw
from PlaneGenerators import HexagonPlaneGenerator
from Shapes import Ellipse, Square, Triangle
from shapely.geometry import Polygon
from math import cos, sin, pi

def create_hexagon_points(center_x, center_y, radius):
    """Get the 6 defining Points of hexagons of position and radius"""
    points = []
    for i in range(6):
        angle = i * pi / 3
        x = center_x + radius * cos(angle)
        y = center_y + radius * sin(angle)
        points.append((x, y))
    return points

def reconstruct_image(array_path: str, hexagons: List[Tuple[float, float]], hex_radius: float, outline_color=(40, 40, 40)):
    """
    Reconstructs an image from the array data.
    """
    # Initialize a transparent image
    # Assuming the size is known or stored; alternatively, store it as part of the array data
    size = 640  # Replace with actual size if different
    image = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    # Load the array
    with open(array_path, 'r') as f:
        pixel_array = json.load(f)

    # Iterate over hexagons and their corresponding array values
    for idx, (hex_center_x, hex_center_y) in enumerate(hexagons):
        overlap_ratio = pixel_array[idx]
        color_value = int(255 * overlap_ratio)
        points = create_hexagon_points(hex_center_x, hex_center_y, hex_radius)
        draw.polygon(points, fill=(color_value, color_value, color_value), outline=outline_color)

    return image

def load_hexagons(array_dir: str) -> Tuple[List[Tuple[float, float]], float]:
    """
    Loads hexagons and hex_radius from the first image generated.
    Assumes that all samples have the same hexagon layout.
    """
    # Load from the first array to get hexagon information
    first_array = os.listdir(array_dir)[0]
    first_array_path = os.path.join(array_dir, first_array)

    # To retrieve hexagons and hex_radius, you might need to store this information separately
    # For simplicity, assume we regenerate the hexagons
    hex_plane_gen = HexagonPlaneGenerator()
    _, hexagons, circle_info = hex_plane_gen.generate_plane(size=640, target_count=1039)
    hex_radius = circle_info[2]
    return hexagons, hex_radius

def main_reconstruct():
    array_dir = 'simulated_data/arrays'
    output_dir = 'reconstructed_images'
    os.makedirs(output_dir, exist_ok=True)

    hexagons, hex_radius = load_hexagons(array_dir)

    for array_file in os.listdir(array_dir):
        if not array_file.endswith('.json'):
            continue
        array_path = os.path.join(array_dir, array_file)
        image = reconstruct_image(array_path, hexagons, hex_radius)
        reconstructed_path = os.path.join(output_dir, array_file.replace('.json', '.png'))
        image.save(reconstructed_path, 'PNG')
        print(f"Reconstructed image saved to {reconstructed_path}")

if __name__ == '__main__':
    main_reconstruct()
