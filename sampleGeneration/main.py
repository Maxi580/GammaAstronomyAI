# File: main.py

from PlaneGenerators import HexagonPlaneGenerator
from Shapes import Ellipse, Square, Triangle
from SampleGenerator import SampleGenerator
from NoiseGenerators import SimpleNoiseGenerator, SpikyNoiseGenerator  # New import
from sampleGeneration.NoiseGenerators import NoNoiseGenerator


def main():
    # Instantiate shapes
    shapes = [Square(), Ellipse(0.2, 0.6)] # Triangle(0.2, 0.4)] # Ellipse(),

    # Instantiate a hexagon plane generator
    hex_plane_gen = HexagonPlaneGenerator()

    # Instantiate a noise generator
    #noise_gen = SimpleNoiseGenerator(0.1, 0.3)  # You can pass a seed for reproducibility
    noise_gen = SpikyNoiseGenerator(0.1, 0.5)  # You can pass a seed for reproducibility
    # noise_gen = NoNoiseGenerator()

    direc = "../simulated_data_15k_gn/"
    # Create a sample generator with specified probabilities for each shape
    sample_gen = SampleGenerator(
        plane_generator=hex_plane_gen,
        shapes=shapes,
        noise_generator=noise_gen,  # Pass the noise generator
        shape_probabilities=[0.5, 0.5],  # Adjust probabilities as needed 0.3,
        output_size=640,
        target_count=1039,
        image_dir= direc + 'images',
        annotation_dir= direc + 'annotations',
        array_dir= direc + 'arrays'  # Noise data is now within arrays
    )

    # Generate samples (e.g., 1000 samples)
    sample_gen.generate_samples(1000)

if __name__ == '__main__':
    main()
