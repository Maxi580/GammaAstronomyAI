import os
import random
from typing import List
from PIL import Image, ImageDraw
from .IPlaneGenerator import IPlaneGenerator
from .IShape import IShape
from .ShapeGenerator import ShapeGenerator
import json

from .PlaneGeneratorUtils import PlaneGeneratorUtils
from .PlaneGenerators import HexagonPlaneGenerator
from .INoiseGenerator import INoiseGenerator


class SampleGenerator:
    def __init__(self,
                 plane_generator: IPlaneGenerator,
                 shapes: List[IShape],
                 noise_generator: INoiseGenerator,  # New parameter
                 shape_probabilities: List[float] = None,
                 output_size: int = 640,
                 target_count: int = 1039,
                 image_dir: str = 'simulated_data/images',
                 annotation_dir: str = 'simulated_data/annotations',
                 array_dir: str = 'simulated_data/arrays'):
        self.plane_generator = plane_generator
        self.shapes = shapes
        self.noise_generator = noise_generator  # Store the noise generator
        self.shape_probabilities = shape_probabilities or [1.0 / len(shapes)] * len(shapes)
        self.output_size = output_size
        self.target_count = target_count
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.array_dir = array_dir

        self._create_directories()

    def _create_directories(self):
        for directory in [self.image_dir, self.annotation_dir, self.array_dir]:
            os.makedirs(directory, exist_ok=True)

    def generate_samples(self, num_samples: int):
        background, hexagons, plane_info = self.plane_generator.generate_plane(self.output_size, self.target_count)
        shape_generator = ShapeGenerator(self.shapes, plane_info)

        # Precompute hexagon polygons for intersection tests
        from shapely.geometry import Polygon
        hex_polygons = []
        for hex_center_x, hex_center_y in hexagons:
            hex_points = PlaneGeneratorUtils.create_hexagon_points_static(hex_center_x, hex_center_y, plane_info[2])
            hex_polygon = Polygon(hex_points)
            hex_polygons.append((hex_points, hex_polygon))

        for i in range(1, num_samples + 1):
            background_copy = background.copy()
            draw = ImageDraw.Draw(background_copy)
            shape = random.choices(self.shapes, weights=self.shape_probabilities, k=1)[0]

            # Select random position for the shape
            shape_center_x, shape_center_y = shape_generator.get_random_position()
            shape_polygon = shape.generate_polygon(shape_center_x, shape_center_y, plane_info[3])

            # Initialize the pixel array
            pixel_array = [0.0] * len(hexagons)

            # Compute overlap ratios and update pixel array
            for idx, (hex_points, hex_polygon) in enumerate(hex_polygons):
                if shape_polygon.intersects(hex_polygon):
                    intersection_area = shape_polygon.intersection(hex_polygon).area
                    hex_area = hex_polygon.area
                    overlap_ratio = intersection_area / hex_area
                    pixel_array[idx] = min(1.0, overlap_ratio)  # Ensure ratio doesn't exceed 1

                    # Update image based on overlap ratio
                    color_value = int(255 * overlap_ratio)
                    draw.polygon(hex_points, fill=(color_value, color_value, color_value), outline=HexagonPlaneGenerator.HEXAGON_OUTLINE_COLOR)

            # Generate unique noise for this sample
            noise_values = self.noise_generator.generate_noise(len(hexagons))

            # Apply noise to the image
            # We'll overlay semi-transparent noise based on the noise_values
            # Create a noise overlay
            noise_overlay = Image.new('RGBA', (self.output_size, self.output_size), (0, 0, 0, 0))
            noise_draw = ImageDraw.Draw(noise_overlay)

            for idx, (hex_center_x, hex_center_y) in enumerate(hexagons):
                noise_intensity = int(255 * noise_values[idx])
                # Define the noise color, e.g., red noise
                noise_color = (noise_intensity, noise_intensity, noise_intensity, noise_intensity)  # Semi-transparent red
                hex_points = PlaneGeneratorUtils.create_hexagon_points_static(hex_center_x, hex_center_y, plane_info[2])
                noise_draw.polygon(hex_points, fill=noise_color)

            # Composite the noise overlay onto the background image
            background_copy = Image.alpha_composite(background_copy.convert('RGBA'), noise_overlay)

            # Combine pixel_array and noise_values into a single data structure
            data = {
                "pixel_array": pixel_array,
                "noise": noise_values
            }

            # Save the image
            image_path = os.path.join(self.image_dir, f'image_{i:04d}.png')
            background_copy.save(image_path, 'PNG')

            # Save the annotation
            annotation_path = os.path.join(self.annotation_dir, f'image_{i:04d}.txt')
            with open(annotation_path, 'w') as f:
                f.write(shape.get_name())

            # Save the combined array and noise data
            array_path = os.path.join(self.array_dir, f'image_{i:04d}.json')
            with open(array_path, 'w') as f:
                json.dump(data, f)

            if i % 100 == 0 or i == num_samples:
                print(f"Generated {i} / {num_samples} samples")
