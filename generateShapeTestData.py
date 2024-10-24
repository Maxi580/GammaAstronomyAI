from PIL import Image, ImageDraw
import os
import random
from math import cos, sin, pi, atan2

MIN_SIZE_RATIO = 0.3
MAX_SIZE_RATIO = 0.4


def create_hexagon_points(center_x, center_y, radius):
    """Create points for a regular hexagon"""
    points = []
    for i in range(6):
        angle = i * pi / 3
        x = center_x + radius * cos(angle)
        y = center_y + radius * sin(angle)
        points.append((x, y))
    return points


def create_rotated_ellipse(draw, width, height, center_x, center_y, angle, color):
    num_points = 100
    points = []
    for i in range(num_points):
        t = 2 * pi * i / num_points
        # Generate ellipse points
        x = width / 2 * cos(t)
        y = height / 2 * sin(t)
        # Rotate points
        x_rot = x * cos(angle) - y * sin(angle)
        y_rot = x * sin(angle) + y * cos(angle)
        # Translate to center
        points.append((x_rot + center_x, y_rot + center_y))

    draw.polygon(points, fill=color, outline=color)


def create_training_image(width=512):
    """Create a single training image with either a square or ellipse inside a hexagon"""

    border_color = (50, 50, 50)
    background_color = (0, 0, 0)
    shape_color = (255, 0, 0)

    height = int(width * 0.866)  # Hexagon width != Height, but should fill the entire image
    center_x = width // 2
    center_y = height // 2

    hex_radius = width * 0.5

    # Calculate Size Range for shapes
    min_size = int(hex_radius * MIN_SIZE_RATIO)
    max_size = int(hex_radius * MAX_SIZE_RATIO)

    image = Image.new('RGB', (width, height), background_color)
    draw = ImageDraw.Draw(image)

    is_square = random.choice([True, False])
    if is_square:
        square_width = random.randint(min_size, max_size)

        left = random.randint(center_x - width // 5 - square_width, center_x + width // 5)
        top = random.randint(0, height - square_width)

        right = left + square_width
        bottom = top + square_width

        square_points = [
            (left, top),
            (right, top),
            (right, bottom),
            (left, bottom)
        ]
        draw.polygon(square_points, fill=shape_color, outline=shape_color, width=2)
        shape_type = 'square'
    else:
        ellipse_width = random.randint(min_size, max_size)
        ellipse_height = random.randint(min_size, max_size) * 0.5

        left = random.randint(center_x - width // 5 - ellipse_width, center_x + width // 5)
        top = random.randint(0, int(height - ellipse_height))

        right = (left + ellipse_width)
        bottom = (top + ellipse_height)

        shape_center_x = (left + right) / 2
        shape_center_y = (top + bottom) / 2

        angle = atan2(center_y - shape_center_y, center_x - shape_center_x)

        create_rotated_ellipse(draw, ellipse_width, ellipse_height, shape_center_x, shape_center_y, angle, shape_color)
        shape_type = 'ellipse'

    # Draw Outside of Hexagon, to overlap border shapes
    mask = Image.new('L', (width, height), 255)
    mask_draw = ImageDraw.Draw(mask)
    hex_points = create_hexagon_points(center_x, center_y, hex_radius)
    mask_draw.polygon(hex_points, fill=0)
    border = Image.new('RGB', (width, height), border_color)
    image.paste(border, mask=mask)

    return image, shape_type


def generate_dataset(num_images, output_dir='training_data'):
    """Generate a dataset of images with labels"""
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'labels.txt'), 'w') as f:
        for i in range(num_images):
            image, shape_type = create_training_image()

            filename = f'image_{i:04d}.png'
            image.save(os.path.join(output_dir, filename))

            f.write(f'{filename},{shape_type}\n')


if __name__ == "__main__":
    generate_dataset(100)
