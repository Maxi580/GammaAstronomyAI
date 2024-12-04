import argparse
import os
import sys

from sampleGeneration.NoiseGenerators import SimpleNoiseGenerator, SpikyNoiseGenerator
from sampleGeneration.PlaneGenerators import HexagonPlaneGenerator
from sampleGeneration.SampleGenerator import SampleGenerator
from sampleGeneration.Shapes import Ellipse, Square, Triangle


def main(count: int, name: str, shapes, probabilities):
    output_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "simulated_data", name
    )

    print("Starting Sample Generator with settings:")
    print(f"\t- Sample Count = {count}")
    print(f"\t- Output Dir = {output_dir}")
    print(f"\t- Shapes = {[s.get_name() for s in shapes]}")
    print(f"\t- Probabilities = {[round(p, 2) for p in probabilities]}")
    print("\n")

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
        image_dir=dir + "/images",
        annotation_dir=dir + "/annotations",
        array_dir=dir + "/arrays",
    )

    sample_gen.generate_samples(count)


def parse_shapes(input_str: str):
    shapes = []
    probabilities = []

    for shape in input_str.lower().split(","):
        s, p = shape.split(":")

        try:
            probabilities.append(int(p))
        except:
            raise ValueError(f"Invalid probability for Shape '{s}': '{p}'")

        match s:
            case "ellipse":
                shapes.append(Ellipse())
            case "ellipse-centered":
                shapes.append(Ellipse(centered=True))
            case "square":
                shapes.append(Square())
            case "triangle":
                shapes.append(Triangle())
            case _:
                raise ValueError(f"Invalid Shape: '{s}'")

    probs_sum = sum(probabilities)
    probabilities = [p / probs_sum for p in probabilities]

    return shapes, probabilities


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        metavar="sample_count",
        type=int,
        default="1000",
        help="Specified count of samples to generate.",
    )
    parser.add_argument(
        "--name",
        metavar="dataset_name",
        required=True,
        help="Specify the name of the generated dataset.",
    )
    parser.add_argument(
        "-s",
        "--shapes",
        metavar="shapes",
        default="ellipse:1,square:1",
        help="Specify what shapes to generate and their probabilities.",
    )
    args = parser.parse_args(sys.argv[1:])

    shapes, probabilities = parse_shapes(args.shapes)

    # Start generation
    main(args.n, args.name, shapes, probabilities)
