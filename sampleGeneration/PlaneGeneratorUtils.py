# File: PlaneGeneratorUtils.py

from math import pi, cos, sin

class PlaneGeneratorUtils:
    @staticmethod
    def create_hexagon_points_static(center_x, center_y, radius):
        points = []
        for i in range(6):
            angle = i * pi / 3
            x = center_x + radius * cos(angle)
            y = center_y + radius * sin(angle)
            points.append((x, y))
        return points
