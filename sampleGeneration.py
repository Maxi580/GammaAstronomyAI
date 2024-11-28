from sampleGeneration.PlaneGenerators import HexagonPlaneGenerator
from sampleGeneration.Shapes import Ellipse, Square, Triangle
from sampleGeneration.SampleGenerator import SampleGenerator
from sampleGeneration.NoiseGenerators import SimpleNoiseGenerator, SpikyNoiseGenerator

def main():
    """
    1.  Add the wanted shapes and their probability here.
    """
    shapes = [Ellipse(), Ellipse(centered=True)]
    probabilities = [0.5, 0.5]

    # Instantiate a hexagon plane generator
    hex_plane_gen = HexagonPlaneGenerator()

    """
    2.  Select Noise generator here.
        You can pass a seed for reproducibility.
    """
    # noise_gen = SimpleNoiseGenerator(0.1, 0.7)
    noise_gen = SpikyNoiseGenerator(0.1, 0.35)

    """
    3.  Adapt generator specifications here.
    """
    sample_gen = SampleGenerator(
        plane_generator=hex_plane_gen,
        shapes=shapes,
        noise_generator=noise_gen,
        shape_probabilities=probabilities,
        output_size=640,
        target_count=1039,
        image_dir='simulated_data/images',
        annotation_dir='simulated_data/annotations',
        array_dir='simulated_data/arrays'
    )

    """
    4.  Enter number of wanted samples here.
    """
    sample_gen.generate_samples(20)

if __name__ == '__main__':
    main()