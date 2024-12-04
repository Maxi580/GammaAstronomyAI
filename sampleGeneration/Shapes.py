import random
from math import atan2, cos, pi, sin
from typing import Tuple

from shapely.geometry import Polygon

from .IShape import IShape


class Ellipse(IShape):
    def __init__(
        self, min_size: float = 0.2, max_size: float = 0.4, centered: bool = False
    ):
        super().__init__(min_size, max_size)
        self._centered = centered

    def generate_polygon(
        self,
        center_x: float,
        center_y: float,
        plane_info: Tuple[int, int, float, float],
    ) -> Polygon:
        hex_center_x, hex_center_y, _, outer_radius = plane_info

        # Width and height are scaled based on min_size and max_size
        width = random.uniform(
            outer_radius * self.min_size, outer_radius * self.max_size
        )
        height = random.uniform(
            outer_radius * self.min_size * random.uniform(0.1, 0.8),
            outer_radius * self.max_size * random.uniform(0.1, 0.8),
        )

        dx = hex_center_x - center_x
        dy = hex_center_y - center_y
        rotation = atan2(dy, dx) if self._centered else random.uniform(0, pi / 2)

        num_points = 64
        points = [
            (
                center_x
                + (
                    width / 2 * cos(t) * cos(rotation)
                    - height / 2 * sin(t) * sin(rotation)
                ),
                center_y
                + (
                    width / 2 * cos(t) * sin(rotation)
                    + height / 2 * sin(t) * cos(rotation)
                ),
            )
            for t in [i * 2 * pi / num_points for i in range(num_points)]
        ]

        return Polygon(points)

    def get_name(self) -> str:
        if self._centered:
            return "ellipse-centered"
        return "ellipse"


class Square(IShape):
    def __init__(self, min_size: float = 0.2, max_size: float = 0.4):
        super().__init__(min_size, max_size)

    def generate_polygon(
        self,
        center_x: float,
        center_y: float,
        plane_info: Tuple[int, int, float, float],
    ) -> Polygon:
        *_, outer_radius = plane_info

        size = random.uniform(
            outer_radius * self.min_size, outer_radius * self.max_size
        )
        rotation = random.uniform(0, pi / 2)
        half_size = size / 2
        points = [
            (
                center_x + (cos(rotation) * x - sin(rotation) * y),
                center_y + (sin(rotation) * x + cos(rotation) * y),
            )
            for x, y in [
                (-half_size, -half_size),
                (half_size, -half_size),
                (half_size, half_size),
                (-half_size, half_size),
            ]
        ]
        return Polygon(points)

    def get_name(self) -> str:
        return "square"
