"""
MAGIC HexMagicNet Architecture Visualization using Visualkeras
This script allows you to visualize your PyTorch neural network architecture using Visualkeras.
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dropout, Dense, BatchNormalization, ReLU, Flatten, Concatenate
import visualkeras
from PIL import ImageFont
import os
from collections import defaultdict

# Define output directory for saving visualizations
OUTPUT_DIR = "architecture_visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_keras_model_from_pytorch_architecture():
    """
    Create a Keras model that mirrors your PyTorch architecture.
    """
    # Input shapes - based on your MagicDataset
    image_m1 = Input(shape=(1039, 1), name='Image_M1')
    image_m2 = Input(shape=(1039, 1), name='Image_M2')
    features = Input(shape=(51,), name='Features')

    # Define a function to create a ConvHex-like layer group
    def conv_hex_block(x, filters, kernel_size, pooling=False, dropout_rate=0.2):
        x = Conv1D(filters, kernel_size)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(dropout_rate)(x)
        if pooling:
            x = MaxPooling1D(2)(x)
        return x

    # Paste your CNN architecture here
    # CNN for telescope 1 image
    x1 = conv_hex_block(image_m1, 8, 1)
    x1 = conv_hex_block(x1, 16, 4)
    x1 = MaxPooling1D(2)(x1)
    x1 = conv_hex_block(x1, 32, 1, pooling=True)
    x1 = Flatten()(x1)

    # CNN for telescope 2 image
    x2 = conv_hex_block(image_m2, 8, 1)
    x2 = conv_hex_block(x2, 16, 4)
    x2 = MaxPooling1D(2)(x2)
    x2 = conv_hex_block(x2, 32, 1, pooling=True)
    x2 = Flatten()(x2)

    # Combine CNN features
    cnn_features = Concatenate()([x1, x2])

    # Classifier - based on your provided architecture
    x = Dense(768)(cnn_features)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.2)(x)

    x = Dense(384)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.2)(x)

    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.2)(x)

    # Output layer
    outputs = Dense(2, activation='softmax')(x)

    return Model(inputs=[image_m1, image_m2, features], outputs=outputs)


def visualize_architecture(model, prefix="hexmagicnet", include_graph=True):
    """
    Visualize the model architecture using visualkeras.

    Args:
        model: Keras model to visualize
        prefix: Prefix for output filenames
        include_graph: Whether to also generate a graph visualization
    """
    # Define custom colors by layer type as dictionary with fill property
    color_mapping = defaultdict(dict)
    color_mapping[Conv1D]['fill'] = '#FF6B6B'      # Red
    color_mapping[Dense]['fill'] = '#4ECDC4'       # Teal
    color_mapping[Dropout]['fill'] = '#556270'     # Dark Slate Gray
    color_mapping[BatchNormalization]['fill'] = '#C7F464'  # Lime
    color_mapping[MaxPooling1D]['fill'] = '#1A535C'  # Dark Teal
    color_mapping[ReLU]['fill'] = '#FFE66D'        # Yellow
    color_mapping[Flatten]['fill'] = '#FF9F1C'     # Orange
    color_mapping[Concatenate]['fill'] = '#6A0572'  # Purple
    color_mapping[Input]['fill'] = '#CCCCCC'       # Light Gray

    # Try to get a nice font
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = None

    # Generate the layered view visualization
    layered_path = os.path.join(OUTPUT_DIR, f"{prefix}_layered.png")
    visualkeras.layered_view(model,
                           to_file=layered_path,
                           background_fill='#F8F9FA',
                           draw_volume=True,
                           legend=True,
                           font=font,
                           scale_xy=2,
                           scale_z=0.8,
                           max_z=500,
                           spacing=40,
                           color_map=color_mapping)
    print(f"Layered visualization saved to {layered_path}")

    # Optionally generate a graph view
    if include_graph:
        graph_path = os.path.join(OUTPUT_DIR, f"{prefix}_graph.png")
        visualkeras.graph_view(model,
                             to_file=graph_path,
                             background_fill='#F8F9FA',
                             color_map=color_mapping)  # Removed legend parameter
        print(f"Graph visualization saved to {graph_path}")


def main():
    """
    Main function to create and visualize the HexMagicNet model.
    """
    print("Creating and visualizing HexMagicNet architecture...")
    cnn_model = create_keras_model_from_pytorch_architecture()
    visualize_architecture(cnn_model, prefix="hexmagicnet")

    print("\nSummary of the CNN model:")
    cnn_model.summary()

    print("\nVisualization complete! Check the output directory for the generated images.")


if __name__ == "__main__":
    main()