import os
import numpy as np
from PIL import Image
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import glob


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


def train_svm_classifier(X, y):
    """
    Train an SVM classifier
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svm = SVC(kernel='poly', degree=3,  C=2.0, random_state=42)
    svm.fit(X_train_scaled, y_train)

    train_score = svm.score(X_train_scaled, y_train)
    test_score = svm.score(X_test_scaled, y_test)
    print(f"Training accuracy: {train_score:.3f}")
    print(f"Testing accuracy: {test_score:.3f}")

    y_pred = svm.predict(X_test_scaled)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return svm, scaler


def predict_shape(svm, scaler, image_path):
    img = Image.open(image_path)
    img_array = np.array(img)
    features = extract_features(img_array)
    features_scaled = scaler.transform([features])
    prediction = svm.predict(features_scaled)[0]
    return prediction


if __name__ == "__main__":
    image_dir = "../simulated_data/images"
    annotation_dir = "../simulated_data/annotations"

    X, y = load_data(image_dir, annotation_dir)

    svm_model, scaler = train_svm_classifier(X, y)
