# File: reconstruct_from_array.py

import os
import json
from typing import Tuple, List

from PIL import Image, ImageDraw
from PlaneGenerators import HexagonPlaneGenerator
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
    Reconstructs an image from the combined array and noise data.
    """
    # Initialize a transparent image
    # Assuming the size is known or stored; alternatively, store it as part of the array data
    size = 640  # Replace with actual size if different
    image = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    # Load the combined array and noise data
    with open(array_path, 'r') as f:
        data = json.load(f)

    pixel_array = data.get("pixel_array", [])
    noise_array = data.get("noise", [])

    if not pixel_array or not noise_array:
        print(f"Missing data in {array_path}, skipping.")
        return

    # Iterate over hexagons and their corresponding array values
    for idx, (hex_center_x, hex_center_y) in enumerate(hexagons):
        if idx >= len(pixel_array) or idx >= len(noise_array):
            print(f"Index {idx} out of range for {array_path}, skipping hexagon.")
            continue

        overlap_ratio = pixel_array[idx]
        noise_value = noise_array[idx]

        # Combine overlap_ratio with noise_value
        # Example: average the two
        combined_value = (overlap_ratio + noise_value) / 2
        combined_value = min(1.0, combined_value)  # Ensure it doesn't exceed 1

        # Determine the color based on the combined value
        # You can customize how noise affects the color. Here, we'll blend red noise with grayscale.
        grayscale = int(255 * combined_value)
        noise_intensity = int(255 * noise_value)
        # Example blending: weighted average between grayscale and noise (red channel)
        blended_color = (
            int((grayscale * 0.7) + (noise_intensity * 0.3)),  # Red channel
            int((grayscale * 0.7)),                            # Green channel
            int((grayscale * 0.7)),                            # Blue channel
            255  # Alpha channel
        )

        points = create_hexagon_points(hex_center_x, hex_center_y, hex_radius)
        draw.polygon(points, fill=blended_color, outline=outline_color)

    return image

def load_hexagons(array_dir: str) -> Tuple[List[Tuple[float, float]], float]:
    """
    Loads hexagons and hex_radius from the first array file generated.
    Assumes that all samples have the same hexagon layout.
    """
    # Load from the first array to get hexagon information
    first_array_file = None
    for file in os.listdir(array_dir):
        if file.endswith('.json'):
            first_array_file = file
            break

    if not first_array_file:
        raise FileNotFoundError(f"No JSON files found in {array_dir}")

    first_array_path = os.path.join(array_dir, first_array_file)

    # To retrieve hexagons and hex_radius, we need to regenerate the hexagons
    # because the hexagon layout is consistent across all samples
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
        if image is None:
            continue
        reconstructed_path = os.path.join(output_dir, array_file.replace('.json', '.png'))
        image.save(reconstructed_path, 'PNG')
        print(f"Reconstructed image saved to {reconstructed_path}")

if __name__ == '__main__':
    main_reconstruct()
