import os
import numpy as np
from PIL import Image
import glob
from image_classification.EnsembleML import EnsembleShapeClassifier


def load_data(image_dir, annotation_dir):
    images = []
    labels = []

    image_files = sorted(glob.glob(os.path.join(image_dir, '*.png')))

    for image_file in image_files:
        img = Image.open(image_file)
        img_array = np.array(img)

        base_name = os.path.basename(image_file)
        annotation_file = os.path.join(annotation_dir,
                                       base_name.replace('.png', '.txt'))

        with open(annotation_file, 'r') as f:
            label = f.read().strip()

        features = extract_features(img_array)

        images.append(features)
        labels.append(label)

    return np.array(images), np.array(labels)


def extract_features(image_array):
    if len(image_array.shape) == 3:
        image_array = np.mean(image_array, axis=2)

    # 1. Calculate the center of mass of white pixels
    white_pixels = np.where(image_array > 0)
    if len(white_pixels[0]) > 0:
        center_y = np.mean(white_pixels[0]) / image_array.shape[0]
        center_x = np.mean(white_pixels[1]) / image_array.shape[1]
    else:
        center_x, center_y = 0.5, 0.5

    # 2. Calculate the spread of white pixels
    if len(white_pixels[0]) > 0:
        spread_y = np.std(white_pixels[0]) / image_array.shape[0]
        spread_x = np.std(white_pixels[1]) / image_array.shape[1]
    else:
        spread_x, spread_y = 0, 0

    # 3. Calculate aspect ratio of the white region
    if len(white_pixels[0]) > 0:
        min_y, max_y = np.min(white_pixels[0]), np.max(white_pixels[0])
        min_x, max_x = np.min(white_pixels[1]), np.max(white_pixels[1])
        height = max_y - min_y + 1
        width = max_x - min_x + 1
        aspect_ratio = width / height if height != 0 else 1
    else:
        aspect_ratio = 1

    # 4. Calculate the total white area
    white_area = np.sum(image_array > 0) / (image_array.shape[0] * image_array.shape[1])

    # 5. Calculate intensity distribution features
    non_zero_intensities = image_array[image_array > 0]
    if len(non_zero_intensities) > 0:
        mean_intensity = np.mean(non_zero_intensities)
        std_intensity = np.std(non_zero_intensities)
    else:
        mean_intensity, std_intensity = 0, 0

    features = [
        center_x, center_y,  # Position features
        spread_x, spread_y,  # Spread features
        aspect_ratio,  # Shape feature
        white_area,  # Area feature
        mean_intensity,  # Intensity features
        std_intensity
    ]

    return features


if __name__ == "__main__":
    image_dir = "../simulated_data/images"
    annotation_dir = "../simulated_data/annotations"

    X, y = load_data(image_dir, annotation_dir)

    model = EnsembleShapeClassifier()
    model.train(X, y)