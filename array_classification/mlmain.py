from helper import load_and_preprocess_data, load_data_from_json
from EnsembleML import EnsembleArrayClassifier
from MLP import MLP

ARRAYS_DIR = "../simulated_data/arrays"
ANNOTATIONS_DIR = "../simulated_data/annotations"


def main():
    arrays, labels = load_data_from_json(ARRAYS_DIR, ANNOTATIONS_DIR)

    print(f"Loaded {len(arrays)} samples")
    print(f"Unique labels found: {set(labels)}")

    classifier = EnsembleArrayClassifier()

    print("\nTraining model...")
    _, _ = classifier.train(arrays, labels)


if __name__ == "__main__":
    main()
