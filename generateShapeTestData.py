from PIL import Image, ImageDraw
from math import cos, sin, pi, sqrt, tan
import random

HEXAGON_FILL_COLOR = (0, 0, 0)
HEXAGON_OUTLINE_COLOR = (40, 40, 40)
SHAPE_COLOR = (255, 255, 255)


def create_hexagon_points(center_x, center_y, radius):
    """Get the 6 defining Points of hexagons of position and radius"""
    points = []
    for i in range(6):
        angle = i * pi / 3
        x = center_x + radius * cos(angle)
        y = center_y + radius * sin(angle)
        points.append((x, y))
    return points


def point_in_circle(px, py, center_x, center_y, radius):
    # Check if point is inside a circle using distance formula
    dx = px - center_x
    dy = py - center_y
    return (dx * dx + dy * dy) <= radius * radius


def count_and_draw_hexagon_grid(draw, center_x, center_y, hex_radius, outer_radius, do_draw=True):
    hex_width = hex_radius * 2
    hex_height = hex_width * sqrt(3) / 2

    # Calculate grid size to cover the circle
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

            # Check if center point is within the circle boundary
            if point_in_circle(x, y, center_x, center_y, outer_radius):
                count += 1
                if do_draw:
                    points = create_hexagon_points(x, y, hex_radius)
                    hexagons.append([x, y])
                    draw.polygon(points, fill=HEXAGON_FILL_COLOR, outline=HEXAGON_OUTLINE_COLOR)

    return count, hexagons


def calculate_background_ratios(size=512, target_count=1039):
    image = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    center_x = size // 2
    center_y = size // 2
    outer_radius = size * 0.45

    # Find The Right Hexagon Size to get as close to target count as possible
    min_ratio = 0.01
    max_ratio = 0.2
    best_ratio = None
    best_count = 0

    # If there are fewer hexagons than we want we decrease max_size otherwise we increase min_size
    while max_ratio - min_ratio > 0.0001:
        mid_ratio = (min_ratio + max_ratio) / 2
        hex_radius = outer_radius * mid_ratio
        count, _ = count_and_draw_hexagon_grid(draw, center_x, center_y, hex_radius, outer_radius, do_draw=False)

        if count == target_count:
            best_ratio = mid_ratio
            break
        elif count < target_count:
            max_ratio = mid_ratio
        else:
            min_ratio = mid_ratio

        if abs(count - target_count) < abs(best_count - target_count):
            best_ratio = mid_ratio
            best_count = count

    # After we have found the best ratio we draw the image
    final_hex_radius = outer_radius * best_ratio
    final_count, hexagons = count_and_draw_hexagon_grid(draw, center_x, center_y, final_hex_radius, outer_radius,
                                                        do_draw=True)
    circle_info = (center_x, center_y, final_hex_radius, outer_radius)
    print(f"Created pattern with {final_count} hexagons")

    return image, hexagons, circle_info


def draw_random_square(draw, hexagons, circle_info):
    start_hexagon = random.choice(hexagons)
    start_x, start_y = start_hexagon
    center_x, center_y, hex_radius, outer_circle_radius = circle_info

    hex_width = hex_radius * 2
    hex_height = hex_width * sqrt(3) / 2

    top_factor = random.choice([-1, 1])
    diagonal_factor = random.choice([-1, 1])

    size = random.randint(1, 5)

    # Draw each vertical column
    for col in range(0, size + 1):
        col_x = start_x
        col_y = start_y + hex_height * top_factor * col

        if point_in_circle(col_x, col_y, center_x, center_y, outer_circle_radius):
            points = create_hexagon_points(col_x, col_y, hex_radius)
            draw.polygon(points, fill=SHAPE_COLOR, outline=HEXAGON_OUTLINE_COLOR)

        # Draw the diagonal row from this column
        for row in range(1, size + 1):
            # Offset calculation adjusted to maintain alignment
            diagonal_x = col_x + (hex_width * 0.75 * row)
            diagonal_y = col_y + (hex_height * 0.5 * row * diagonal_factor)

            if point_in_circle(diagonal_x, diagonal_y, center_x, center_y, outer_circle_radius):
                points = create_hexagon_points(diagonal_x, diagonal_y, hex_radius)
                draw.polygon(points, fill=SHAPE_COLOR, outline=HEXAGON_OUTLINE_COLOR)


if __name__ == '__main__':
    background, hexagons, circle_info = calculate_background_ratios(size=640, target_count=1039)
    draw = ImageDraw.Draw(background)

    draw_random_square(draw, hexagons, circle_info)

    background.save('hexagon_background.png', 'PNG')
