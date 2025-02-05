from abc import ABC, abstractmethod
from typing import List, Tuple
from PIL import Image

class IPlaneGenerator(ABC):
    @abstractmethod
    def generate_plane(self, size: int, target_count: int) -> Tuple[Image.Image, List[Tuple[float, float]], Tuple[int, int, float, float]]:
        """Generates the background plane and returns the image, hexagon centers, and plane info."""
        pass
