from helper import load_data_from_json

from arrayClassification.ML_Models.EnsembleML import EnsembleArrayClassifier

# TODO
ARRAYS_DIR = "../../datasets/Huge10kSet/arrays"
ANNOTATIONS_DIR = "../../datasets/Huge10kSet/annotations"


def main():
    arrays, labels = load_data_from_json(ARRAYS_DIR, ANNOTATIONS_DIR)

    print(f"Loaded {len(arrays)} samples")
    print(f"Unique labels found: {set(labels)}")

    classifier = EnsembleArrayClassifier()

    print("\nTraining model...")
    _, _ = classifier.train(arrays, labels)


if __name__ == "__main__":
    main()
