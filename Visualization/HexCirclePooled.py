from PIL import Image, ImageDraw
import numpy as np

from SimulatedSampleGeneration.PlaneGenerators import HexagonPlaneGenerator
from CNN.HexCircleLayers.pooling import _get_clusters
 

hex_count = 163

def hex_circle_pooled(hex_count: int, kernel_size: int, number_pooled: bool):
    clusters = _get_clusters(hex_count, kernel_size)


    plane_generator = HexagonPlaneGenerator()

    background, hexagons, plane_info = plane_generator.generate_plane(2000, hex_count)

    image = Image.new("RGBA", (2000, 2000), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    missing = list(range(hex_count))

    for c_idx, cluster in enumerate(clusters):
        color = tuple(np.random.choice(range(256), size=3))
        for idx in cluster:
            if idx == -1:
                continue
            
            hex_center_x, hex_center_y = hexagons[idx]
            points = plane_generator.create_hexagon_points(
                hex_center_x, hex_center_y, plane_info[2]
            )
            draw.polygon(points, fill=color, outline=(0, 0, 0))
            draw.text((hex_center_x, hex_center_y), str(c_idx if number_pooled else idx), anchor="mm", align="center", font_size=20, stroke_width=1)
            
            missing.remove(idx)

            
    for idx in missing:
        hex_center_x, hex_center_y = hexagons[idx]
        points = plane_generator.create_hexagon_points(
            hex_center_x, hex_center_y, plane_info[2]
        )
        draw.polygon(points, fill=(50,50,50), outline=(0, 0, 0))
        draw.text((hex_center_x, hex_center_y), str("None" if number_pooled else idx), anchor="mm", align="center", font_size=20, stroke_width=1)
        
    image.show()

    # image.save("./hex_circle_pooled.png", "PNG")
    

if __name__ == "__main__":
    TARGET_HEX_COUNT = 1039
    KERNEL_SIZE = 1
    NUMBER_POOLED = False
    hex_circle_pooled(TARGET_HEX_COUNT, KERNEL_SIZE, NUMBER_POOLED)