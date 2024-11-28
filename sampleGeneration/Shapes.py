import random
from math import pi, cos, sin
from shapely.geometry import Polygon
from .IShape import IShape


class Ellipse(IShape):
    def __init__(self, min_size: float = 0.2, max_size: float = 0.4):
        super().__init__(min_size, max_size)

    def generate_polygon(self, center_x: float, center_y: float, outer_radius: float) -> Polygon:
        # Width and height are scaled based on min_size and max_size
        width = random.uniform(outer_radius * self.min_size, outer_radius * self.max_size)
        height = random.uniform(outer_radius * self.min_size, outer_radius * self.max_size * random.uniform(0.1, 0.8))
        rotation = random.uniform(0, pi / 2)
        num_points = 64
        points = [
            (
                center_x + (width / 2 * cos(t) * cos(rotation) - height / 2 * sin(t) * sin(rotation)),
                center_y + (width / 2 * cos(t) * sin(rotation) + height / 2 * sin(t) * cos(rotation))
            )
            for t in [i * 2 * pi / num_points for i in range(num_points)]
        ]
        return Polygon(points)

    def get_name(self) -> str:
        return "ellipse"


class Square(IShape):
    def __init__(self, min_size: float = 0.2, max_size: float = 0.4):
        super().__init__(min_size, max_size)

    def generate_polygon(self, center_x: float, center_y: float, outer_radius: float) -> Polygon:
        size = random.uniform(outer_radius * self.min_size, outer_radius * self.max_size)
        rotation = random.uniform(0, pi / 2)
        half_size = size / 2
        points = [
            (
                center_x + (cos(rotation) * x - sin(rotation) * y),
                center_y + (sin(rotation) * x + cos(rotation) * y)
            )
            for x, y in
            [(-half_size, -half_size), (half_size, -half_size), (half_size, half_size), (-half_size, half_size)]
        ]
        return Polygon(points)

    def get_name(self) -> str:
        return "square"


class Triangle(IShape):
    def __init__(self, min_size: float = 0.2, max_size: float = 0.4):
        super().__init__(min_size, max_size)

    def generate_polygon(self, center_x: float, center_y: float, outer_radius: float) -> Polygon:
        # Generate random size within specified bounds
        size = random.uniform(outer_radius * self.min_size, outer_radius * self.max_size)
        rotation = random.uniform(0, pi / 2)

        # Calculate vertices for an equilateral triangle
        points = []
        for i in range(3):
            angle = (2 * pi * i / 3) + rotation
            x = center_x + size * cos(angle)
            y = center_y + size * sin(angle)
            points.append((x, y))

        return Polygon(points)

    def get_name(self) -> str:
        return "triangle"
