from abc import ABC, abstractmethod
from shapely.geometry import Polygon

class IShape(ABC):
    def __init__(self, min_size: float = 0.2, max_size: float = 0.4):
        """
        Initialize the shape with optional min_size and max_size.

        :param min_size: Minimum size factor relative to the outer_radius.
        :param max_size: Maximum size factor relative to the outer_radius.
        """
        if not (0 < min_size <= max_size):
            raise ValueError("min_size must be > 0 and <= max_size")
        self.min_size = min_size
        self.max_size = max_size

    @abstractmethod
    def generate_polygon(self, center_x: float, center_y: float, outer_radius: float) -> Polygon:
        """Generates the polygon representing the shape."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Returns the name of the shape."""
        pass