import random
from math import pi, cos, sin, atan2
from shapely.geometry import Polygon
from IShape import IShape  # Relative import


class Ellipse(IShape):
    def __init__(self, min_size: float = 0.2, max_size: float = 0.4, center_bias: float = 7.0):
        super().__init__(min_size, max_size)
        self.center_bias = center_bias

    def generate_polygon(self, center_x: float, center_y: float, outer_radius: float) -> Polygon:
        plane_center_x = center_x - (center_x - outer_radius)
        plane_center_y = center_y - (center_y - outer_radius)

        dx = plane_center_x - center_x
        dy = plane_center_y - center_y
        angle_to_center = atan2(dy, dx)

        base_rotation = angle_to_center + (pi * random.choice([0, 1]))

        if self.center_bias == 0:
            rotation = random.uniform(0, 2 * pi)
        else:
            max_deviation = pi / (2 * self.center_bias)  # Smaller when bias is higher
            rotation = base_rotation + random.uniform(-max_deviation, max_deviation)

        width = random.uniform(outer_radius * self.min_size, outer_radius * self.max_size)
        height = random.uniform(outer_radius * self.min_size, outer_radius * self.max_size * random.uniform(0.2, 0.6))

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
