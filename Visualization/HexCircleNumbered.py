from typing import Tuple

from PIL import Image, ImageDraw

from SimulatedSampleGeneration.PlaneGenerators import HexagonPlaneGenerator


def visualize_hex_circle(target_count: int):
    plane_generator = HexagonPlaneGenerator()

    background, hexagons, plane_info = plane_generator.generate_plane(2000, target_count)
    
    image = Image.new("RGBA", (2000, 2000), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    
    for idx, (hex_center_x, hex_center_y) in enumerate(hexagons):
        points = plane_generator.create_hexagon_points(
            hex_center_x, hex_center_y, plane_info[2]
        )
        draw.polygon(points, fill=(150,150,150), outline=(0, 0, 0))
        draw.text((hex_center_x, hex_center_y), str(idx), anchor="mm", align="center", font_size=20, stroke_width=1, fill=(255,255,255))
        
    image.show()
    
    # image.save("./hex_circle.png", "PNG")


if __name__ == "__main__":
    TARGET_HEX_COUNT = 1039
    visualize_hex_circle(TARGET_HEX_COUNT)