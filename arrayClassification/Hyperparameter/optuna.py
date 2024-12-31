import optuna
from arrayClassification.trainingSupervisor import TrainingSupervisor
import torch.nn as nn
import torch.optim as optim
from arrayClassification.HexLayers.ConvHex import ConvHex


def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'dropout_rate1': trial.suggest_float('dropout_rate1', 0.025, 0.5),
        'dropout_rate2': trial.suggest_float('dropout_rate2', 0.025, 0.5),
        'dropout_rate3': trial.suggest_float('dropout_rate3', 0.025, 0.3),
        'kernel_size1': trial.suggest_int('kernel_size1', 1, 5),
        'kernel_size2': trial.suggest_int('kernel_size2', 1, 5),
        'kernel_size3': trial.suggest_int('kernel_size3', 1, 5),
        'channels1': trial.suggest_int('channels1', 16, 64, step=16),
        'channels2': trial.suggest_int('channels2', 32, 128, step=32),
        'channels3': trial.suggest_int('channels3', 64, 256, step=64),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True),
    }

    class TrialHexCNN(nn.Module):
        def __init__(self, params):
            super(TrialHexCNN, self).__init__()
            self.features = nn.Sequential(
                ConvHex(in_channels=1, out_channels=params['channels1'],
                        kernel_size=params['kernel_size1']),
                nn.BatchNorm1d(params['channels1']),
                nn.ReLU(),
                nn.Dropout1d(params['dropout_rate1']),

                ConvHex(in_channels=params['channels1'], out_channels=params['channels2'],
                        kernel_size=params['kernel_size2']),
                nn.BatchNorm1d(params['channels2']),
                nn.ReLU(),
                nn.Dropout1d(params['dropout_rate2']),

                ConvHex(in_channels=params['channels2'], out_channels=params['channels3'],
                        kernel_size=params['kernel_size3']),
                nn.BatchNorm1d(params['channels3']),
                nn.ReLU(),
                nn.Dropout1d(params['dropout_rate3']),
            )

            self.classifier = nn.Sequential(
                nn.Flatten(),

                nn.Linear(32 * 1039, 1024),
                nn.ReLU(),
                nn.Dropout(0.2),

                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Dropout(0.2),

                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(0.1),

                nn.Linear(64, 2)
            )

    supervisor = TrainingSupervisor(modelname="HexCNN", output_dir="optuna_trial")
    supervisor.load_training_data("your_dataset_path")

    optimizer = optim.Adam(model.parameters(),
                           lr=params['learning_rate'],
                           weight_decay=params['weight_decay'])

    validation_accuracy = supervisor.load_model(epochs=10, info_prints=False)

    return validation_accuracy


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

# Print best parameters and score
print("Best parameters:", study.best_params)
print("Best validation accuracy:", study.best_value)

# Plot optimization history
optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_parallel_coordinate(study)
