from helper import load_and_preprocess_data, extract_features
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier
import xgboost as xgb


class EnsembleArrayClassifier:
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
            eval_metric='mlogloss'
        )

        self.classifiers = {
            'SVM': self.svm,
            'Random Forest': self.rf,
            'Gradient Boosting': self.gb,
            'Neural Network': self.mlp,
            'XGBoost': self.xgb
        }

    def train(self, arrays, labels):
        # Process arrays and extract features
        X, y = load_and_preprocess_data(arrays, labels)
        y_encoded = self.label_encoder.fit_transform(y)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )

        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Create and train ensemble
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

        # Train and evaluate individual classifiers
        for name, clf in self.classifiers.items():
            print(f"\nTraining {name}...")
            clf.fit(X_train_scaled, y_train)
            train_score = clf.score(X_train_scaled, y_train)
            test_score = clf.score(X_test_scaled, y_test)
            print(f"Training accuracy: {train_score:.3f}")
            print(f"Testing accuracy: {test_score:.3f}")

        # Train and evaluate ensemble
        print("\nTraining ensemble...")
        self.ensemble.fit(X_train_scaled, y_train)

        ensemble_train_score = self.ensemble.score(X_train_scaled, y_train)
        ensemble_test_score = self.ensemble.score(X_test_scaled, y_test)
        print("\nEnsemble:")
        print(f"Training accuracy: {ensemble_train_score:.3f}")
        print(f"Testing accuracy: {ensemble_test_score:.3f}")

        # Generate detailed classification report
        print("\nEnsemble Detailed Classification Report:")
        y_pred = self.ensemble.predict(X_test_scaled)
        y_test_original = self.label_encoder.inverse_transform(y_test)
        y_pred_original = self.label_encoder.inverse_transform(y_pred)
        print(classification_report(y_test_original, y_pred_original))

        return X_test_scaled, y_test_original

    def predict(self, array):
        features = extract_features(array)
        features_scaled = self.scaler.transform([features])

        predictions = {}

        # Get predictions from each classifier
        for name, clf in self.classifiers.items():
            pred = clf.predict(features_scaled)[0]
            prob = clf.predict_proba(features_scaled)[0]
            pred_original = self.label_encoder.inverse_transform([pred])[0]
            predictions[name] = {'prediction': pred_original, 'confidence': np.max(prob)}

        # Get ensemble prediction
        ensemble_pred = self.ensemble.predict(features_scaled)[0]
        ensemble_prob = self.ensemble.predict_proba(features_scaled)[0]
        ensemble_pred_original = self.label_encoder.inverse_transform([ensemble_pred])[0]
        predictions['Ensemble'] = {
            'prediction': ensemble_pred_original,
            'confidence': np.max(ensemble_prob)
        }

        return predictions


