import random
from typing import List
from .INoiseGenerator import INoiseGenerator


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


class SpikyNoiseGenerator(INoiseGenerator):
    def __init__(self, min_noise: float = 0.2, max_noise: float = 0.4, seed: int = None):
        if seed is not None:
            random.seed(seed)
        super().__init__(min_noise, max_noise)

    def generate_noise(self, num_pixels: int) -> List[float]:
        """
        Generates random noise values between min_noise and max_noise for each pixel.
        But max_noise is way less likely to be chosen.

        :param num_pixels: The number of unique pixels to generate noise for.
        :return: A list of noise values.
        """

        noise_values = [1 for _ in range(num_pixels)]

        for x in range(num_pixels):
            rand = random.randint(0, 100)
            if rand < 90:
                noise_values[x] = random.uniform(self.min_noise, (self.max_noise+self.max_noise)/2)
            else:
                noise_values[x] = random.uniform((self.max_noise+self.max_noise)/2, self.max_noise*1.8)

        return noise_values
