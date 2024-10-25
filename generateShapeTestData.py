from PIL import Image, ImageDraw
from math import cos, sin, pi, sqrt, tan
import random
from shapely.geometry import Polygon, box
import os

HEXAGON_FILL_COLOR = (0, 0, 0)
HEXAGON_OUTLINE_COLOR = (40, 40, 40)
SHAPE_COLOR = (255, 255, 255)

ELLIPSE = 'ellipse'
SQUARE = 'square'

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


def get_random_square_poly(center_x, center_y, outer_radius):
    angle = random.uniform(0, 2 * pi)
    distance = random.uniform(0, outer_radius * 0.3)
    shape_center_x = center_x + cos(angle) * distance
    shape_center_y = center_y + sin(angle) * distance

    width = height = random.uniform(outer_radius * 0.1, outer_radius * 0.4)

    rotation = random.uniform(0, pi / 2)

    shape_points = [
        (
            shape_center_x + (cos(rotation) * x - sin(rotation) * y),
            shape_center_y + (sin(rotation) * x + cos(rotation) * y)
        )
        for x, y in [
            (-width / 2, -height / 2),
            (width / 2, -height / 2),
            (width / 2, height / 2),
            (-width / 2, height / 2)
        ]
    ]

    shape_polygon = Polygon(shape_points)
    return shape_polygon


def get_random_ellipse_poly(center_x, center_y, outer_radius):
    angle = random.uniform(0, 2 * pi)
    distance = random.uniform(0, outer_radius * 0.3)
    ellipse_center_x = center_x + cos(angle) * distance
    ellipse_center_y = center_y + sin(angle) * distance

    width = random.uniform(outer_radius * 0.3, outer_radius * 0.4)
    height = random.uniform(outer_radius * 0.1, outer_radius * 0.2)

    rotation = random.uniform(0, pi / 2)

    points = []
    num_points = 64

    for i in range(num_points):
        t = i * 2 * pi / num_points

        x = width / 2 * cos(t)
        y = height / 2 * sin(t)

        rotated_x = x * cos(rotation) - y * sin(rotation)
        rotated_y = x * sin(rotation) + y * cos(rotation)

        final_x = ellipse_center_x + rotated_x
        final_y = ellipse_center_y + rotated_y

        points.append((final_x, final_y))

    return Polygon(points)


def draw_random_shape(draw, hexagons, circle_info, shape):
    center_x, center_y, hex_radius, outer_radius = circle_info

    if shape == ELLIPSE:
        shape_polygon = get_random_ellipse_poly(center_x, center_y, outer_radius)
    else:
        shape_polygon = get_random_square_poly(center_x, center_y, outer_radius)

    for hex_center_x, hex_center_y in hexagons:
        hex_points = create_hexagon_points(hex_center_x, hex_center_y, hex_radius)
        hex_polygon = Polygon(hex_points)

        if shape_polygon.intersects(hex_polygon):
            # Calculate overlap percentage
            intersection_area = shape_polygon.intersection(hex_polygon).area
            hex_area = hex_polygon.area
            overlap_ratio = intersection_area / hex_area

            # Color the hexagon based on overlap
            color_value = int(255 * overlap_ratio)
            draw.polygon(hex_points,
                         fill=(color_value, color_value, color_value),
                         outline=HEXAGON_OUTLINE_COLOR)

def main(num_pictures):
    background, hexagons, circle_info = calculate_background_ratios(size=640, target_count=1039)

    image_directory = 'simulated_data/images'
    annotation_directory = 'simulated_data/annotations'

    for directory in [image_directory, annotation_directory]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    for i in range(num_pictures):
        background_copy = background.copy()
        draw = ImageDraw.Draw(background_copy)
        shape = random.choice([ELLIPSE, SQUARE])

        draw_random_shape(draw, hexagons, circle_info, shape)

        image_path = os.path.join(image_directory, f'image_{i:04d}.png')
        background_copy.save(image_path, 'PNG')

        annotations_file = os.path.join(annotation_directory, f'image_{i:04d}.txt')
        with open(annotations_file, 'w') as f:
            f.write(shape)

if __name__ == '__main__':
   main(10)
