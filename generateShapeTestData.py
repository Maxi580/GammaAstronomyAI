from PIL import Image, ImageDraw
from math import cos, sin, pi, sqrt, tan
import random
from dataclasses import dataclass

HEXAGON_FILL_COLOR = (0, 0, 0)
HEXAGON_OUTLINE_COLOR = (40, 40, 40)
SHAPE_COLOR = (255, 255, 255)


def create_hexagon_points(center_x, center_y, radius):
    points = []
    for i in range(6):
        angle = i * pi / 3
        x = center_x + radius * cos(angle)
        y = center_y + radius * sin(angle)
        points.append((x, y))
    return points


def point_in_hexagon(px, py, center_x, center_y, radius):
    # Check if point is inside a regular hexagon
    # Convert to local coordinates
    x = px - center_x
    y = py - center_y

    # Check against hexagon boundaries
    x_abs = abs(x)
    y_abs = abs(y)

    # Hexagon boundary conditions
    return (x_abs <= radius * cos(pi / 6)) and (y_abs <= radius * sin(pi / 6) + (radius / 2 - x_abs * tan(pi / 6)))


def count_and_draw_hexagon_grid(draw, center_x, center_y, hex_radius, outer_radius, do_draw=True):
    hex_width = hex_radius * 2
    hex_height = hex_width * sqrt(3) / 2

    # Calculate grid size to cover the outer hexagon
    cols = int(outer_radius * 2 / (hex_width * 0.75)) + 2
    rows = int(outer_radius * 2 / hex_height) + 2

    hexagons = []

    count = 0
    for row in range(-rows // 2, rows // 2 + 1):
        for col in range(-cols // 2, cols // 2 + 1):
            x = center_x + col * hex_width * 0.75
            y = center_y + row * hex_height
            if col % 2:
                y += hex_height / 2

            # Check if center point is within the outer hexagon boundary
            if point_in_hexagon(x, y, center_x, center_y, outer_radius):
                count += 1
                if do_draw:
                    points = create_hexagon_points(x, y, hex_radius)
                    hexagons.append([x, y, hex_radius])
                    draw.polygon(points, fill=HEXAGON_FILL_COLOR, outline=HEXAGON_OUTLINE_COLOR)

    return count, hexagons


def calculate_background_ratios(size=512, target_count=1039):
    image = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    center_x = size // 2
    center_y = size // 2
    outer_radius = size * 0.45

    # Find The Right Hexagon Size (in percent of background hexagon) to get as close to target count as possible
    min_ratio = 0.01
    max_ratio = 0.2
    best_ratio = None
    best_count = 0

    # Break the search after the difference is getting irrelevant
    while max_ratio - min_ratio > 0.0001:
        mid_ratio = (min_ratio + max_ratio) / 2
        hex_radius = outer_radius * mid_ratio
        count, _ = count_and_draw_hexagon_grid(draw, center_x, center_y, hex_radius, outer_radius, do_draw=False)

        # If there are fewer hexagons than we want we decrease size else we increase it
        if count == target_count:
            best_ratio = mid_ratio
            break
        elif count < target_count:
            max_ratio = mid_ratio
        else:
            min_ratio = mid_ratio

        # Save the Best Ratio/ Count
        if abs(count - target_count) < abs(best_count - target_count):
            best_ratio = mid_ratio
            best_count = count

    final_hex_radius = outer_radius * best_ratio
    final_count, hexagons = count_and_draw_hexagon_grid(draw, center_x, center_y, final_hex_radius, outer_radius, do_draw=True)
    print(f"Created pattern with {final_count} hexagons")

    return image, hexagons


if __name__ == '__main__':
    background, hexagons = calculate_background_ratios(size=640, target_count=1039)
    draw = ImageDraw.Draw(background)

    num_white_hexagons = 5  # or however many you want
    for _ in range(num_white_hexagons):
        random_hexagon = random.choice(hexagons)
        x, y, radius = random_hexagon
        points = create_hexagon_points(x, y, radius)
        draw.polygon(points, fill=SHAPE_COLOR, outline=HEXAGON_OUTLINE_COLOR)

    background.save('hexagon_pattern_hexagonal.png', 'PNG')
    background.copy()
