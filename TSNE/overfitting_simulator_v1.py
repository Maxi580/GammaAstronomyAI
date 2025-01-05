import os
import json
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from tqdm import tqdm


class ImprovedTSNEShapeClassifier:
    def __init__(self, data_dir, ann_dir, n_components=3, perplexity=50):
        self.data_dir = data_dir
        self.ann_dir = ann_dir
        self.n_components = n_components
        self.perplexity = perplexity
        self.label_map = {"square": 0, "ellipse": 1}
        self.inv_label_map = {0: "square", 1: "ellipse"}
        self.scaler = RobustScaler()  # Changed to RobustScaler for better handling of outliers

        # Create multiple classifiers to compare
        self.classifiers = {
            'knn': KNeighborsClassifier(n_neighbors=7, weights='distance'),
            'rf': RandomForestClassifier(n_estimators=100, max_depth=10),
            'svm': SVC(kernel='rbf', probability=True)
        }
        self.best_classifier = None
        self.best_classifier_name = None

    def load_data(self):
        print("Loading data...")
        X = []
        y = []
        data_files = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.json')])

        for data_file in tqdm(data_files):
            with open(os.path.join(self.data_dir, data_file), 'r') as f:
                data_json = json.load(f)
                features = np.array(data_json["combined"])

                # Add statistical features
                additional_features = np.array([
                    np.mean(features),
                    np.std(features),
                    np.max(features),
                    np.min(features),
                    np.median(features),
                    np.percentile(features, 25),
                    np.percentile(features, 75)
                ])

                X.append(np.concatenate([features, additional_features]))

            ann_file = data_file.replace('.json', '.txt')
            with open(os.path.join(self.ann_dir, ann_file), 'r') as f:
                label_str = f.read().strip()
                y.append(self.label_map[label_str])

        return np.array(X), np.array(y)

    def create_tsne_features(self, X):
        print("Creating t-SNE features...")
        X_scaled = self.scaler.fit_transform(X)

        tsne = TSNE(
            n_components=self.n_components,
            perplexity=self.perplexity,
            n_iter=2000,  # Increased iterations
            random_state=42,
            learning_rate='auto',
            init='pca'  # Use PCA initialization
        )

        return tsne.fit_transform(X_scaled)

    def train(self):
        # Load and preprocess data
        X, y = self.load_data()

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # Create t-SNE features
        X_train_tsne = self.create_tsne_features(X_train)
        X_test_tsne = self.create_tsne_features(X_test)

        # Compare classifiers using cross-validation
        best_score = 0
        for name, clf in self.classifiers.items():
            scores = cross_val_score(clf, X_train_tsne, y_train, cv=5)
            mean_score = scores.mean()
            std_score = scores.std()
            print(f"{name.upper()} Cross-validation score: {mean_score:.4f} (+/- {std_score:.4f})")

            if mean_score > best_score:
                best_score = mean_score
                self.best_classifier = clf
                self.best_classifier_name = name

        # Train the best classifier on full training data
        print(f"\nTraining best classifier ({self.best_classifier_name})...")
        self.best_classifier.fit(X_train_tsne, y_train)

        # Evaluate
        train_accuracy = self.best_classifier.score(X_train_tsne, y_train)
        test_accuracy = self.best_classifier.score(X_test_tsne, y_test)

        print(f"\nBest classifier ({self.best_classifier_name}):")
        print(f"Training accuracy: {train_accuracy:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")

        return X_train_tsne, y_train, X_test_tsne, y_test

    def visualize(self, X_tsne, y, title="t-SNE Visualization of Shapes"):
        if X_tsne.shape[1] > 2:
            # If we have more than 2 components, create multiple 2D plots
            fig, axes = plt.subplots(1, 3, figsize=(20, 6))
            fig.suptitle(title)

            # Plot first two components
            scatter = axes[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
            axes[0].set_title("Components 1 vs 2")
            axes[0].set_xlabel("t-SNE component 1")
            axes[0].set_ylabel("t-SNE component 2")

            # Plot first and third components
            axes[1].scatter(X_tsne[:, 0], X_tsne[:, 2], c=y, cmap='viridis')
            axes[1].set_title("Components 1 vs 3")
            axes[1].set_xlabel("t-SNE component 1")
            axes[1].set_ylabel("t-SNE component 3")

            # Plot second and third components
            axes[2].scatter(X_tsne[:, 1], X_tsne[:, 2], c=y, cmap='viridis')
            axes[2].set_title("Components 2 vs 3")
            axes[2].set_xlabel("t-SNE component 2")
            axes[2].set_ylabel("t-SNE component 3")

            plt.colorbar(scatter, ax=axes.ravel().tolist())
        else:
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
            plt.colorbar(scatter)
            plt.title(title)
            plt.xlabel("t-SNE component 1")
            plt.ylabel("t-SNE component 2")

        plt.savefig(f"tsne_visualization_{title.lower().replace(' ', '_')}.png")
        plt.close()

    def report_misclassified(self, X_test_tsne, y_test, limit=300):
        predictions = self.best_classifier.predict(X_test_tsne)
        misclassified = np.where(predictions != y_test)[0]

        print(f"\nNumber of misclassified samples: {len(misclassified)}")
        print("\nSample of misclassified instances:")
        for idx in misclassified[:limit]:
            confidence = np.max(self.best_classifier.predict_proba(X_test_tsne[idx:idx + 1]))
            print(f"Sample {idx} - Actual: {self.inv_label_map[y_test[idx]]}, "
                  f"Predicted: {self.inv_label_map[predictions[idx]]}, "
                  f"Confidence: {confidence:.2f}")


def main():
    direc = "../simulated_data_10k_nn/"

    # Initialize and train the classifier
    classifier = ImprovedTSNEShapeClassifier(
        data_dir=direc + "arrays",
        ann_dir=direc + "annotations",
        n_components=3,  # Increased to 3 components
        perplexity=50  # Adjusted perplexity
    )

    # Train and get results
    X_train_tsne, y_train, X_test_tsne, y_test = classifier.train()

    # Visualize results
    classifier.visualize(X_train_tsne, y_train, "Training Data t-SNE Visualization")
    classifier.visualize(X_test_tsne, y_test, "Test Data t-SNE Visualization")

    # Report misclassified samples
    classifier.report_misclassified(X_test_tsne, y_test)


if __name__ == "__main__":
    main()