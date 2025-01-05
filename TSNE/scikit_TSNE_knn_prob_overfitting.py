import os
import json
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Function to load data and labels
def load_data(data_dir, ann_dir):
    data_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.json')])
    X = []
    y = []
    label_map = {"square": 0, "ellipse": 1}

    for data_file in data_files:
        with open(os.path.join(data_dir, data_file), 'r') as f:
            data_json = json.load(f)
        X.append(data_json["combined"])

        ann_file = data_file.replace('.json', '.txt')
        with open(os.path.join(ann_dir, ann_file), 'r') as f:
            label_str = f.read().strip()
        y.append(label_map[label_str])

    return np.array(X), np.array(y)

if __name__ == "__main__":
    # Load data and labels
    direc = "../simulated_data_10k_nn/"
    X, y = load_data(direc + "arrays", direc + "annotations")

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=2000) # Adjust parameters as needed
    X_tsne = tsne.fit_transform(X_scaled)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_tsne, y, test_size=0.5, random_state=42)

    # Train a k-NN classifier (you can experiment with other classifiers as well)
    knn = KNeighborsClassifier(n_neighbors=5)  # Adjust n_neighbors as needed
    knn.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = knn.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

    # Visualize the t-SNE results (optional)
    plt.figure(figsize=(10, 8))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', s=50, alpha=0.7)
    plt.title("t-SNE visualization of data")
    plt.colorbar()
    plt.show()

    # Optional: Visualize misclassified samples (similar to the Transformer example)
    misclassified_indices = np.where(y_pred != y_test)[0]
    if misclassified_indices.size > 0:
        print("\nMisclassified samples (index, actual, predicted):")
        for i in misclassified_indices[:min(10, len(misclassified_indices))]:  # Limit to 300 samples
            print(f"{i}, {y_test[i]}, {y_pred[i]}")
    else:
        print("\nNo misclassified samples found.")