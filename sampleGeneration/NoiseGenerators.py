import random
from typing import List
from INoiseGenerator import INoiseGenerator


class SimpleNoiseGenerator(INoiseGenerator):
    def __init__(self, min_noise: float = 0.2, max_noise: float = 0.4, seed: int = None):
        """
        Initializes the SimpleNoiseGenerator.

        :param seed: Optional seed for reproducibility.
        """
        if seed is not None:
            random.seed(seed)
        super().__init__(min_noise, max_noise)

    def generate_noise(self, num_pixels: int) -> List[float]:
        """
        Generates random noise values between 0 and 1 for each pixel.

        :param num_pixels: The number of unique pixels to generate noise for.
        :return: A list of noise values.
        """
        return [random.uniform(self.min_noise, self.max_noise) for _ in range(num_pixels)]
