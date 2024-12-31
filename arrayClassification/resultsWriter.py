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

        ax.set(xlim=xlim, ylim=ylim)
        if xticks is not None: ax.set_xticks(xticks)
        if yticks is not None: ax.set_yticks(yticks)

        ax.set_title(plot_config["title"])
        ax.legend()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.figure.savefig(os.path.join(self.output_dir, plot_config["filename"]))
        plt.close(ax.figure)


