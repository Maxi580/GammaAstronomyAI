from typing import List, Tuple
from shapely.geometry import Polygon
from PIL import ImageDraw
import random
from math import pi, cos, sin
from PlaneGeneratorUtils import PlaneGeneratorUtils
from IShape import IShape

class ShapeGenerator:
    def __init__(self, shapes: List[IShape], plane_info: Tuple[int, int, float, float]):
        self.shapes = shapes
        self.plane_info = plane_info

    def select_shape(self) -> IShape:
        """Selects a shape based on predefined probabilities."""
        return random.choice(self.shapes)

    def get_random_position(self) -> Tuple[float, float]:
        """Selects a random position within the plane for the shape."""
        center_x, center_y, hex_radius, outer_radius = self.plane_info
        angle = random.uniform(0, 2 * pi)
        distance = random.uniform(0, outer_radius * 0.8)
        shape_center_x = center_x + cos(angle) * distance
        shape_center_y = center_y + sin(angle) * distance
        return shape_center_x, shape_center_y
