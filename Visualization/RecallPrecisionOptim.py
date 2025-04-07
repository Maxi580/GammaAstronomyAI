import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

# Define the confusion matrices for each model
standard_cm = np.array([
    [24735, 6817],
    [10796, 220558]
])

precision_cm = np.array([
    [29418, 2134],
    [29773, 201581]
])

recall_cm = np.array([
    [23332, 8220],
    [8322, 223032]
])

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(16, 6))

# Create a professional blue colormap
cmap = plt.cm.Blues


# Function to plot a confusion matrix
def plot_cm(ax, cm, title, reference_cm=None):
    # Normalize for color intensity
    norm_cm = cm / cm.max()

    # Plot with professional blue gradient
    im = ax.imshow(norm_cm, cmap=cmap, interpolation='nearest')

    # Calculate percentages
    total = cm.sum()
    percentages = cm / total * 100

    # Add text annotations
    for i in range(2):
        for j in range(2):
            # Determine cell type
            if (i == j):  # TN or TP (true predictions)
                cell_type = "TN" if i == 0 else "TP"
            else:  # FP or FN (false predictions)
                cell_type = "FP" if i == 0 else "FN"

            # Ensure text is visible regardless of background
            text_color = "black" if norm_cm[i, j] < 0.5 else "white"

            # Format the value
            value_text = f"{cm[i, j]:,}\n({percentages[i, j]:.1f}%)"

            # For non-standard models, calculate and show difference
            if reference_cm is not None:
                diff = cm[i, j] - reference_cm[i, j]
                sign = "+" if diff > 0 else ""

                # CORRECTED COLOR LOGIC:
                # For *all* cells, the color of the *number* matches the sign:
                # Positive changes (+) are green, negative changes (-) are red
                diff_color = "green" if diff > 0 else "red"
                diff_text = f"\n{sign}{diff:,}"

                # Add the difference information with appropriate color
                ax.text(j, i + 0.25, diff_text, ha="center", va="center",
                        color=diff_color, fontweight='bold', fontsize=10)

            # Add value text
            ax.text(j, i - 0.1, value_text, ha="center", va="center",
                    color=text_color, fontweight='bold')

            # Add cell type indicator (TN, FP, FN, TP) in top-left corner
            ax.text(j - 0.45, i - 0.45, cell_type, ha="left", va="top",
                    color=text_color, fontsize=9, alpha=0.8)

    # Add labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Proton', 'Gamma'])
    ax.set_yticklabels(['Proton', 'Gamma'])

    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)

    return im


# Plot each matrix
plot_cm(axes[0], precision_cm, "Precision Optimized", standard_cm)
plot_cm(axes[1], standard_cm, "Standard")
plot_cm(axes[2], recall_cm, "Recall Optimized", standard_cm)

# Define model metrics with changes
standard_metrics = {
    "Accuracy": 93.30,
    "Precision": 97.00,
    "Recall": 95.33,
    "F1": 96.16
}

precision_metrics = {
    "Accuracy": 87.86,
    "Precision": 98.95,
    "Recall": 87.13,
    "F1": 92.67
}

recall_metrics = {
    "Accuracy": 93.71,
    "Precision": 96.45,
    "Recall": 96.40,
    "F1": 96.42
}

# Calculate differences
precision_diffs = {k: precision_metrics[k] - standard_metrics[k] for k in standard_metrics}
recall_diffs = {k: recall_metrics[k] - standard_metrics[k] for k in standard_metrics}


# Function to create text for metrics
def get_metrics_text(metrics, diffs=None):
    text_lines = []

    for metric, value in metrics.items():
        line = f"{metric}: {value:.2f}%"

        if diffs is not None:
            diff = diffs[metric]
            sign = "+" if diff > 0 else ""
            color = "green" if diff > 0 else "red"  # Higher is always better for metrics
            line += f" ({sign}{diff:.2f}%)"

        text_lines.append(line)

    return "\n".join(text_lines)


# Position the metrics texts well below the confusion matrices
axes[0].text(0.5, -0.35, get_metrics_text(precision_metrics, precision_diffs),
             ha="center", va="center", transform=axes[0].transAxes)
axes[1].text(0.5, -0.35, get_metrics_text(standard_metrics),
             ha="center", va="center", transform=axes[1].transAxes)
axes[2].text(0.5, -0.35, get_metrics_text(recall_metrics, recall_diffs),
             ha="center", va="center", transform=axes[2].transAxes)

# Add a supertitle
plt.suptitle("Confusion Matrix Comparison", fontsize=16)

plt.tight_layout()
plt.subplots_adjust(top=0.9, bottom=0.28)
plt.savefig('confusion_matrix_comparison_corrected.png', dpi=300, bbox_inches='tight')
plt.show()