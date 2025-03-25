import matplotlib.pyplot as plt
import numpy as np
import os
import json


def _plot_config(filename, title, x_axis, y_axis, metrics):
    return {
        "filename": filename,
        "title": title,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "metrics": metrics
    }


class ResultsWriter:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def save_training_results(self, training_data: dict):
        self._save_json(training_data)
        self._create_all_plots(training_data)

    def _save_json(self, data: dict):
        with open(os.path.join(self.output_dir, "info.json"), "w") as file:
            json.dump(data, file, indent=4)

    def _create_all_plots(self, data: dict):
        epochs = np.arange(1, data["epochs"] + 1)
        train_metrics = data["training_metrics"]
        val_metrics = data["validation_metrics"]

        plots = [
            _plot_config("loss_diagram.png", "Loss Diagram",
                         ((1, data["epochs"]), None, "Epochs"),
                         ((0, 1), np.arange(0, 1.1, 0.1), "Loss Value"),
                         [("loss", train_metrics, "Training"),
                          ("loss", val_metrics, "Validation")]),

            _plot_config("loss_diagram_training.png", "Loss Diagram Training",
                         ((1, data["epochs"]), None, "Epochs"),
                         ((0, 1), np.arange(0, 1.1, 0.1), "Loss Value"),
                         [("proton_loss", train_metrics, "Protons"),
                          ("gamma_loss", train_metrics, "Gammas")]),

            _plot_config("loss_diagram_validation.png", "Loss Diagram Validation",
                         ((1, data["epochs"]), None, "Epochs"),
                         ((0, 1), np.arange(0, 1.1, 0.1), "Loss Value"),
                         [("proton_loss", val_metrics, "Protons"),
                          ("gamma_loss", val_metrics, "Gammas")]),

            _plot_config("training_metrics.png", "Training Metrics",
                         ((1, data["epochs"]), None, "Epochs"),
                         ((0, 100), np.arange(0, 101, 10), "Percentage"),
                         [("accuracy", train_metrics, "Accuracy"),
                          ("precision", train_metrics, "Precision"),
                          ("recall", train_metrics, "Recall"),
                          ("f1", train_metrics, "F1")]),

            _plot_config("validation_metrics.png", "Validation Metrics",
                         ((1, data["epochs"]), None, "Epochs"),
                         ((0, 100), np.arange(0, 101, 10), "Percentage"),
                         [("accuracy", val_metrics, "Accuracy"),
                          ("precision", val_metrics, "Precision"),
                          ("recall", val_metrics, "Recall"),
                          ("f1", val_metrics, "F1")]),

            _plot_config("accuracy_comparison.png", "Accuracy Comparison",
                         ((1, data["epochs"]), None, "Epochs"),
                         ((0, 100), np.arange(0, 101, 10), "Percentage"),
                         [("accuracy", train_metrics, "Training Accuracy"),
                          ("accuracy", val_metrics, "Validation Accuracy")])
        ]

        for plot in plots:
            self._create_plot(epochs, plot)

    def _create_plot(self, epochs, plot_config):
        ax = plt.subplot()

        for metric_name, metrics_data, label in plot_config["metrics"]:
            y_values = [m[metric_name] for m in metrics_data]
            ax.plot(epochs, y_values, label=label)

        xlim, xticks, xlabel = plot_config["x_axis"]
        ylim, yticks, ylabel = plot_config["y_axis"]

        # Add small padding if distance is too small (e.g. only one epoch)
        if xlim[0] == xlim[1]:
            padding = 0.5
            xlim = (xlim[0] - padding, xlim[1] + padding)

        ax.set(xlim=xlim, ylim=ylim)
        if xticks is not None: ax.set_xticks(xticks)
        if yticks is not None: ax.set_yticks(yticks)

        ax.set_title(plot_config["title"])
        ax.legend()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.figure.savefig(os.path.join(self.output_dir, plot_config["filename"]))
        plt.close(ax.figure)


def get_min_max_values(metrics_data, metric_names):
    """Get min and max values across all specified metrics"""
    min_val = float('inf')
    max_val = float('-inf')

    for metric in metric_names:
        for data in metrics_data:
            value = data[metric]
            min_val = min(min_val, value)
            max_val = max(max_val, value)

    return min_val, max_val


def create_plot_from_json(json_data, output_dir="./recovered_plots/"):
    os.makedirs(output_dir, exist_ok=True)

    data = json_data if isinstance(json_data, dict) else json.loads(json_data)
    epochs = np.arange(1, data["epochs"] + 1)
    train_metrics = data["training_metrics"]
    val_metrics = data["validation_metrics"]

    loss_min, loss_max = get_min_max_values([*train_metrics, *val_metrics], ["loss"])
    loss_range = ((1, data["epochs"]), None, "Epochs"), ((loss_min * 0.95, loss_max * 1.05), None, "Loss Value")

    acc_min, acc_max = get_min_max_values([*train_metrics, *val_metrics], ["accuracy"])
    acc_range = ((1, data["epochs"]), None, "Epochs"), ((acc_min * 0.95, acc_max * 1.05), None, "Accuracy (%)")

    metrics_min, metrics_max = get_min_max_values([*train_metrics, *val_metrics],
                                                  ["accuracy", "precision", "recall", "f1"])
    metrics_range = ((1, data["epochs"]), None, "Epochs"), (
    (metrics_min * 0.95, metrics_max * 1.05), None, "Percentage")

    plots = [
        ("loss_diagram.png", "Loss Diagram", loss_range,
         [("loss", train_metrics, "Training Loss"),
          ("loss", val_metrics, "Validation Loss")]),

        ("accuracy_comparison.png", "Accuracy Comparison", acc_range,
         [("accuracy", train_metrics, "Training Accuracy"),
          ("accuracy", val_metrics, "Validation Accuracy")]),

        ("training_metrics.png", "Training Metrics", metrics_range,
         [("accuracy", train_metrics, "Accuracy"),
          ("precision", train_metrics, "Precision"),
          ("recall", train_metrics, "Recall"),
          ("f1", train_metrics, "F1")]),

        ("validation_metrics.png", "Validation Metrics", metrics_range,
         [("accuracy", val_metrics, "Accuracy"),
          ("precision", val_metrics, "Precision"),
          ("recall", val_metrics, "Recall"),
          ("f1", val_metrics, "F1")])
    ]

    for filename, title, (x_axis, y_axis), metrics in plots:
        fig, ax = plt.subplots(figsize=(10, 6))

        for metric_name, metrics_data, label in metrics:
            y_values = [m[metric_name] for m in metrics_data]
            ax.plot(epochs, y_values, label=label, marker='o', markersize=4)

        xlim, xticks, xlabel = x_axis
        ylim, yticks, ylabel = y_axis

        ax.set(xlim=xlim, ylim=ylim)
        if xticks is not None: ax.set_xticks(xticks)
        if yticks is not None: ax.set_yticks(yticks)

        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_title(title)
        ax.legend()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{filename}", dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    with open("info.json", "r") as f:
        data = json.load(f)
    create_plot_from_json(data)