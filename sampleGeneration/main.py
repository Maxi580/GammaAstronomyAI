# File: main.py

from PlaneGenerators import HexagonPlaneGenerator
from Shapes import Ellipse, Square, Triangle
from SampleGenerator import SampleGenerator

def main():
    # Instantiate shapes
    shapes = [Ellipse(), Square(), Triangle(0.1, 0.2)]

    # Instantiate a hexagon plane generator
    hex_plane_gen = HexagonPlaneGenerator()

    # Create a sample generator with specified probabilities for each shape
    sample_gen = SampleGenerator(
        plane_generator=hex_plane_gen,
        shapes=shapes,
        shape_probabilities=[0.3, 0.3, 0.4],  # Adjust probabilities as needed
        output_size=640,
        target_count=1039,
        image_dir='simulated_data/images',
        annotation_dir='simulated_data/annotations',
        array_dir='simulated_data/arrays'  # Ensure array directory is specified
    )

    # Generate samples (e.g., 1000 samples)
    sample_gen.generate_samples(20)

if __name__ == '__main__':
    main()
