from abc import ABC, abstractmethod
from typing import List

class INoiseGenerator(ABC):
    def __init__(self, min_noise: float = 0.2, max_noise: float = 0.4):
        if not (0 < min_noise <= max_noise):
            raise ValueError("min_size must be > 0 and <= max_size")
        self.min_noise = min_noise
        self.max_noise = max_noise

    @abstractmethod
    def generate_noise(self, num_pixels: int) -> List[float]:
        """
        Generates a list of noise values for the given number of pixels.

        :param num_pixels: The number of unique pixels to generate noise for.
        :return: A list of noise values between 0 and 1.
        """
        pass
