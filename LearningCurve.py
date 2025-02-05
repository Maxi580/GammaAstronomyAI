import glob
import json
import os
import time
import matplotlib.pyplot as plt
import numpy as np

from TrainingPipeline.TrainingSupervisor import TrainingSupervisor
from TrainingPipeline.magicDataset import MagicDataset

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'


def _plot_config(filename, title, x_axis, y_axis, metrics):
    return {
        'filename': filename,
        'title': title,
        'x_axis': x_axis,
        'y_axis': y_axis,
        'metrics': metrics
    }


def _create_plot(samples, plot_config, output_dir: str):
        ax = plt.subplot()

        for metric_name, metrics_data, label in plot_config['metrics']:
            y_values = [m[metric_name] for m in metrics_data]
            ax.plot(samples, y_values, label=label)

        xlim, xticks, xlabel = plot_config['x_axis']
        ylim, yticks, ylabel = plot_config['y_axis']

        # Add small padding if distance is too small (e.g. only one epoch)
        if xlim != None and xlim[0] == xlim[1]:
            padding = 0.5
            xlim = (xlim[0] - padding, xlim[1] + padding)

        ax.set(xlim=xlim, ylim=ylim)
        if xticks is not None: ax.set_xticks(xticks)
        if yticks is not None: ax.set_yticks(yticks)

        ax.set_title(plot_config['title'])
        ax.legend()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.figure.savefig(os.path.join(output_dir, plot_config['filename']))
        plt.close(ax.figure)


def write_result_plots(output_dir: str):
    info_files = glob.glob('./*/info.json', root_dir=output_dir)

    runs = []
    
    for filename in info_files:
        with open(os.path.join(output_dir, filename), 'r') as file:
            data = json.load(file)
            runs.append((
                data['dataset']['distribution']['total_samples'] / 1000,
                data['training_metrics'][-1],
                data['validation_metrics'][-1]
            ))
            
    runs.sort(key=lambda x: x[0])
    
    samples_counts = [x[0] for x in runs]
    train_metrics = [x[1] for x in runs]
    val_metrics = [x[2] for x in runs]
            
    min_samples = min(samples_counts)
    max_samples = max(samples_counts)
        
    plots = [
            _plot_config('learning_curve_accuracy.png', 'Learning Curve - Accuracy',
                         ((min_samples, max_samples), None, 'Samples (in thousands)'),
                         ((0, 100), None, 'Accuracy'),
                         [('accuracy', train_metrics, 'Training'),
                          ('accuracy', val_metrics, 'Validation')]),
            
            _plot_config('learning_curve_precision.png', 'Learning Curve - Precision',
                         ((min_samples, max_samples), None, 'Samples (in thousands)'),
                         ((0, 100), None, 'Precision'),
                         [('precision', train_metrics, 'Training'),
                          ('precision', val_metrics, 'Validation')]),
            
            _plot_config('learning_curve_recall.png', 'Learning Curve - Recall',
                         ((min_samples, max_samples), None, 'Samples (in thousands)'),
                         ((0, 100), None, 'Recall'),
                         [('recall', train_metrics, 'Training'),
                          ('recall', val_metrics, 'Validation')]),
            
            _plot_config('learning_curve_f1.png', 'Learning Curve - F1',
                         ((min_samples, max_samples), None, 'Samples (in thousands)'),
                         ((0, 100), None, 'F1'),
                         [('f1', train_metrics, 'Training'),
                          ('f1', val_metrics, 'Validation')]),
            
            _plot_config('learning_curve_loss.png', 'Learning Curve - Loss',
                         ((min_samples, max_samples), None, 'Samples (in thousands)'),
                         ((0, 1), np.arange(0, 1.1, 0.1), 'Loss Value'),
                         [('loss', train_metrics, 'Training'),
                          ('loss', val_metrics, 'Validation')]),
        ]
    
    for plot in plots:
        _create_plot(samples_counts, plot, output_dir)


def main(step_size: int, max_step: int, epochs: int, model_name: str):
    nametag = f'LearningCurve-{model_name}-{epochs}_{time.strftime('%Y-%m-%d_%H-%M-%S')}'
    proton_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'magic-protons.parquet')
    gamma_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'magic-gammas.parquet')
    parent_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'trained_models', nametag)

    print(f'Starting Training for Learning Curve with settings:')
    print(f'\t- Model = {model_name}')
    print(f'\t- Data = {proton_dir, gamma_dir}')
    print(f'\t- Epochs = {epochs}')
    print(f'\t- Steps = {step_size}')
    print(f'\t- Max Size = {max_step}')
    print(f'\t- Output = {parent_dir}\n')
    
    for sample_count in range(step_size, max_step+1, step_size):
        print(f'Training with {sample_count} samples:')
        
        dataset = MagicDataset('magic-protons.parquet', 'magic-gammas.parquet', sample_count, debug_info=False)
        
        output_dir = os.path.join(parent_dir, f'{sample_count}-samples')
        supervisor = TrainingSupervisor(model_name, dataset, output_dir, debug_info=False, save_model=False, save_debug_data=True)
        supervisor.train_model(epochs)
        
        with open(os.path.join(output_dir, 'info.json'), 'r') as file:
            data = json.load(file)
            print(f'\t- Training Accuracy: {data['training_metrics'][-1]['accuracy']:>6.2f}%')
            print(f'\t- Validation Accuracy: {data['validation_metrics'][-1]['accuracy']:>6.2f}%')
        
    print('Plotting result diagrams...')
    write_result_plots(parent_dir)


if __name__ == '__main__':
    step_size = 10_000
    max_step = 480_000
    epochs = 10
    model_name = 'TrainingPipeline'

    main(step_size, max_step, epochs, model_name)
    