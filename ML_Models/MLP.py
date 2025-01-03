import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier


class MLP:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def train(self, X, y):
        y_encoded = self.label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.mlp = MLPClassifier(
            hidden_layer_sizes=(128, 256, 128),
            alpha=0.01,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            max_iter=2000,
            random_state=42

        )
        self.mlp.fit(X_train_scaled, y_train)

        train_score = self.mlp.score(X_train_scaled, y_train)
        test_score = self.mlp.score(X_test_scaled, y_test)

        print("\nFinal Results:")
        print(f"Training accuracy: {train_score:.3f}")
        print(f"Testing accuracy: {test_score:.3f}")

        # Detailed classification report
        y_pred = self.mlp.predict(X_test_scaled)
        y_test_original = self.label_encoder.inverse_transform(y_test)
        y_pred_original = self.label_encoder.inverse_transform(y_pred)

        print("\nClassification Report:")
        print(classification_report(y_test_original, y_pred_original))

        return X_test_scaled, y_test_original

    def predict(self, features):
        features_scaled = self.scaler.transform([features])
        pred = self.mlp.predict(features_scaled)[0]
        prob = self.mlp.predict_proba(features_scaled)[0]
        pred_original = self.label_encoder.inverse_transform([pred])[0]

        return {
            'prediction': pred_original,
            'confidence': np.max(prob)
        }

    def plot_learning_curve(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.mlp.loss_curve_)
        plt.title('Learning Curve')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()
