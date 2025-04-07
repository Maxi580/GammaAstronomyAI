from PIL import Image, ImageDraw
from math import cos, sin, pi, sqrt
from typing import List, Tuple
from .IPlaneGenerator import IPlaneGenerator

class HexagonPlaneGenerator(IPlaneGenerator):
    HEXAGON_FILL_COLOR = (0, 0, 0)
    HEXAGON_OUTLINE_COLOR = (40, 40, 40)

    def create_hexagon_points(self, center_x, center_y, radius):
        points = []
        for i in range(6):
            angle = i * pi / 3
            x = center_x + radius * cos(angle)
            y = center_y + radius * sin(angle)
            points.append((x, y))
        return points

    def point_in_circle(self, px, py, center_x, center_y, radius):
        dx = px - center_x
        dy = py - center_y
        return (dx * dx + dy * dy) <= radius * radius

    def generate_spiral_hexagons(self, center_x, center_y, hex_radius, outer_radius) -> List[Tuple[float, float]]:
        """
        Generate hexagons in spiral order from the center outward.
        """
        hexagons = []
        count = 0

        # Define axial directions for hex grid
        directions = [
            # Anti-clockwise
            (-1, 1),    # Southwest
            (0, 1),     # South
            (1, 0),     # Southeast
            (1, -1),    # Northeast
            (0, -1),    # North
            (-1, 0),    # Northwest
            
            # OLD: clockwise directions
            # (1, 0),    # Southeast
            # (0, 1),    # South
            # (-1, 1),    # Southwest
            # (-1, 0),    # Northwest
            # (0, -1),    # North
            # (1, -1),    # Northeast
        ]

        # Mapping from axial coordinates to pixel positions
        def axial_to_pixel(q, r):
            x = center_x + (3/2) * hex_radius * q
            y = center_y + sqrt(3) * hex_radius * (r + q / 2)
            return x, y

        # Start with the center hexagon
        q, r = 0, 0
        x, y = axial_to_pixel(q, r)
        if self.point_in_circle(x, y, center_x, center_y, outer_radius):
            hexagons.append((x, y, 0))
            count += 1

        # Generate hexagons ring by ring
        ring = 1
        while True:
            added_in_ring = False
            q, r = 0, -ring
            for direction in directions:
                for _ in range(ring):
                    if self.point_in_circle(*axial_to_pixel(q, r), center_x, center_y, outer_radius):
                        hexagons.append((*axial_to_pixel(q, r), ring))
                        count += 1
                        added_in_ring = True
                    q += direction[0]
                    r += direction[1]
            if not added_in_ring:
                break
            ring += 1

        return hexagons, ring

    def generate_plane(self, size: int, target_count: int) -> Tuple[Image.Image, List[Tuple[float, float]], Tuple[int, int, float, float]]:
        image = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)

        center_x = size // 2
        center_y = size // 2
        outer_radius = size * 0.45

        # Binary search to find the best hex_radius to match target_count
        min_ratio = 0.01
        max_ratio = 0.2
        best_ratio = None
        best_count = 0

        while max_ratio - min_ratio > 0.0001:
            mid_ratio = (min_ratio + max_ratio) / 2
            hex_radius = outer_radius * mid_ratio
            hexagons, _ = self.generate_spiral_hexagons(center_x, center_y, hex_radius, outer_radius)
            count = len(hexagons)

            if count == target_count:
                best_ratio = mid_ratio
                best_count = count
                break
            elif count < target_count:
                max_ratio = mid_ratio
            else:
                min_ratio = mid_ratio

            if abs(count - target_count) < abs(best_count - target_count):
                best_ratio = mid_ratio
                best_count = count

        # Final hexagon generation with the best_ratio
        final_hex_radius = outer_radius * best_ratio
        final_hexagons, rings = self.generate_spiral_hexagons(center_x, center_y, final_hex_radius, outer_radius)

        # Draw hexagons
        for hex_center_x, hex_center_y, _ in final_hexagons:
            points = self.create_hexagon_points(hex_center_x, hex_center_y, final_hex_radius)
            draw.polygon(points, fill=self.HEXAGON_FILL_COLOR, outline=self.HEXAGON_OUTLINE_COLOR)

        circle_info = (center_x, center_y, final_hex_radius, outer_radius, rings)
        print(f"Created hexagon pattern with {len(final_hexagons)} hexagons")

        return image, final_hexagons, circle_info
