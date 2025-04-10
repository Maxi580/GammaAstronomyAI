import joblib
import matplotlib.pyplot as plt
from sklearn import tree
import os

# Load your pickled random forest model
rf_model_path = "rf_model.pkl"  # Update with your actual path
rf_model = joblib.load(rf_model_path)

# For RAPIDS cuML models, convert to CPU if needed
# Uncomment if you're using cuML:
# from cuml.ensemble import RandomForestClassifier as cuRF
# if isinstance(rf_model, cuRF):
#     rf_model = rf_model.to_sklearn()

# Define feature names based on your MagicDataset
feature_names = [
    'hillas_length_m1', 'hillas_width_m1', 'hillas_delta_m1',
    'hillas_size_m1', 'hillas_cog_x_m1', 'hillas_cog_y_m1',
    'hillas_sin_delta_m1', 'hillas_cos_delta_m1',
    'hillas_length_m2', 'hillas_width_m2', 'hillas_delta_m2',
    'hillas_size_m2', 'hillas_cog_x_m2', 'hillas_cog_y_m2',
    'hillas_sin_delta_m2', 'hillas_cos_delta_m2',
    'stereo_direction_x', 'stereo_direction_y', 'stereo_zenith',
    'stereo_azimuth', 'stereo_dec', 'stereo_ra', 'stereo_theta2',
    'stereo_core_x', 'stereo_core_y', 'stereo_impact_m1',
    'stereo_impact_m2', 'stereo_impact_azimuth_m1',
    'stereo_impact_azimuth_m2', 'stereo_shower_max_height',
    'stereo_xmax', 'stereo_cherenkov_radius',
    'stereo_cherenkov_density', 'stereo_baseline_phi_m1',
    'stereo_baseline_phi_m2', 'stereo_image_angle',
    'stereo_cos_between_shower',
    'pointing_zenith', 'pointing_azimuth',
    'time_gradient_m1', 'time_gradient_m2',
    'source_alpha_m1', 'source_dist_m1',
    'source_cos_delta_alpha_m1', 'source_dca_m1',
    'source_dca_delta_m1',
    'source_alpha_m2', 'source_dist_m2',
    'source_cos_delta_alpha_m2', 'source_dca_m2',
    'source_dca_delta_m2'
]

# Class names for your classification problem
class_names = ['proton', 'gamma']

# Visualize just the first tree in the forest with limited depth
if hasattr(rf_model, 'estimators_'):
    # Get the first tree
    first_tree = rf_model.estimators_[0]

    # Create a more reasonably sized figure
    plt.figure(figsize=(16, 10))

    # Plot the tree with a maximum depth of 4 (first few levels only)
    tree.plot_tree(
        first_tree,
        max_depth=2,  # Only show first 4 levels
        feature_names=feature_names,
        class_names=class_names,
        filled=True,  # Color nodes by class
        rounded=True,  # Round node corners
        proportion=True,  # Show proportion of samples
        precision=2,  # Precision of decimal places
        fontsize=10  # Larger font size
    )

    # Add a title
    plt.title('Decision Tree (First 2 Levels)', fontsize=16)

    # Save the visualization to a file
    plt.savefig('simplified_decision_tree3.png', dpi=300, bbox_inches='tight')
    print("Simplified decision tree visualization saved to 'simplified_decision_tree.png'")
else:
    print("This model doesn't have individual trees accessible via .estimators_")
    print("It may be using a different implementation (like cuML) that needs special handling.")