import torch
from torch_cnn_arrays import SimpleShapeCNN
import glob
import os
import json

LABELS = {'square': 0, 'ellipse': 1}


def predict(model_path, array_dir, annotation_dir):
    model = SimpleShapeCNN()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    correct_count = 0
    
    array_files = sorted(glob.glob(os.path.join(array_dir, '*.json')))
    
    for array_file in array_files:

        base_name = os.path.basename(array_file)
        annotation_file = os.path.join(annotation_dir,
                                       base_name.replace('.json', '.txt'))

        with open(annotation_file, 'r') as f:
            label = f.read().strip()
            
        with open(array_file, 'r') as f:
            array = json.loads(f.read().strip())

        # Convert the array to a tensor and add batch and channel dimensions
        input_tensor = torch.tensor(array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 1039]

        # Perform the prediction
        with torch.no_grad():
            result = model(input_tensor)

        # Get the predicted label
        predicted_label = torch.argmax(result, dim=1).item()

        # Check if the prediction is correct
        if predicted_label == LABELS[label]:
            correct_count += 1
            
    print("Correct Results:", f"{correct_count}/{len(array_files)}")
    
    
if __name__ == "__main__":
    MODEL_PATH = 'best_model_arrays.pth'
    ARRAYS_DIR = "../simulated_data1/arrays"
    ANNOTATIONS_DIR = "../simulated_data1/annotations"
    predict(MODEL_PATH, ARRAYS_DIR, ANNOTATIONS_DIR)