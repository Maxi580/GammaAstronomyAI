import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
import os
from PIL import Image
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

class EnsembleShapeClassifier:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

        self.svm = SVC(kernel='poly', degree=3, C=2.0, random_state=42, probability=True)
        self.rf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        self.xgb = xgb.XGBClassifier(
            n_estimators=100,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )

        self.classifiers = {
            'SVM': self.svm,
            'Random Forest': self.rf,
            'Gradient Boosting': self.gb,
            'Neural Network': self.mlp,
            'XGBoost': self.xgb
        }

    def train(self, X, y):
        y_encoded = self.label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.ensemble = VotingClassifier(
            estimators=[
                ('svm', self.svm),
                ('rf', self.rf),
                ('gb', self.gb),
                ('mlp', self.mlp),
                ('xgb', self.xgb)
            ],
            voting='soft'
        )

        print("Training individual classifiers:")
        print("-" * 50)

        for name, clf in self.classifiers.items():
            print(f"\nTraining {name}...")
            clf.fit(X_train_scaled, y_train)
            train_score = clf.score(X_train_scaled, y_train)
            test_score = clf.score(X_test_scaled, y_test)
            print(f"Training accuracy: {train_score:.3f}")
            print(f"Testing accuracy: {test_score:.3f}")

        print("\nTraining ensemble...")
        self.ensemble.fit(X_train_scaled, y_train)

        ensemble_train_score = self.ensemble.score(X_train_scaled, y_train)
        ensemble_test_score = self.ensemble.score(X_test_scaled, y_test)
        print("\nEnsemble:")
        print(f"Training accuracy: {ensemble_train_score:.3f}")
        print(f"Testing accuracy: {ensemble_test_score:.3f}")

        print("\nEnsemble Detailed Classification Report:")
        y_pred = self.ensemble.predict(X_test_scaled)
        y_test_original = self.label_encoder.inverse_transform(y_test)
        y_pred_original = self.label_encoder.inverse_transform(y_pred)
        print(classification_report(y_test_original, y_pred_original))

        return X_test_scaled, y_test_original

    def predict(self, features):
        features_scaled = self.scaler.transform([features])

        predictions = {}
        for name, clf in self.classifiers.items():
            pred = clf.predict(features_scaled)[0]
            prob = clf.predict_proba(features_scaled)[0]
            pred_original = self.label_encoder.inverse_transform([pred])[0]
            predictions[name] = {'prediction': pred_original, 'confidence': np.max(prob)}

        ensemble_pred = self.ensemble.predict(features_scaled)[0]
        ensemble_prob = self.ensemble.predict_proba(features_scaled)[0]
        ensemble_pred_original = self.label_encoder.inverse_transform([ensemble_pred])[0]
        predictions['Ensemble'] = {
            'prediction': ensemble_pred_original,
            'confidence': np.max(ensemble_prob)
        }

        return predictions


if __name__ == "__main__":
    image_dir = "../simulated_data/images"
    annotation_dir = "../simulated_data/annotations"

    X, y = load_data(image_dir, annotation_dir)

    model = EnsembleShapeClassifier()
    model.train(X, y)