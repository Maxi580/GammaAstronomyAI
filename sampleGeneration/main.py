# File: main.py

from PlaneGenerators import HexagonPlaneGenerator
from Shapes import Ellipse, Square, Triangle
from SampleGenerator import SampleGenerator
from NoiseGenerators import SimpleNoiseGenerator, SpikyNoiseGenerator  # New import

def main():
    # Instantiate shapes
    shapes = [Ellipse(), Square(), Triangle(0.1, 0.2)]

    # Instantiate a hexagon plane generator
    hex_plane_gen = HexagonPlaneGenerator()

    # Instantiate a noise generator
    #noise_gen = SimpleNoiseGenerator(0.1, 0.7)  # You can pass a seed for reproducibility
    noise_gen = SpikyNoiseGenerator(0.1, 0.5)  # You can pass a seed for reproducibility

    # Create a sample generator with specified probabilities for each shape
    sample_gen = SampleGenerator(
        plane_generator=hex_plane_gen,
        shapes=shapes,
        noise_generator=noise_gen,  # Pass the noise generator
        shape_probabilities=[0.3, 0.3, 0.4],  # Adjust probabilities as needed
        output_size=640,
        target_count=1039,
        image_dir='simulated_data/images',
        annotation_dir='simulated_data/annotations',
        array_dir='simulated_data/arrays'  # Noise data is now within arrays
    )

    # Generate samples (e.g., 1000 samples)
    sample_gen.generate_samples(20)

if __name__ == '__main__':
    main()
