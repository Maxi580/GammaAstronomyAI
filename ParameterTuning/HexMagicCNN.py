import copy

import torch
import torch.nn as nn
import optuna

from CNN.MagicConv.MagicConv import MagicConv


def parameterize_hex_magicnet(trial: optuna.Trial):
    class HexMagicCNN(nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.cnn = nn.Sequential(*layers)

        def forward(self, x):
            return self.cnn(x)

    class HexMagicNet(nn.Module):
        def __init__(self):
            super().__init__()

            self.hyperparams = {
                'num_layers': trial.suggest_int('cnn_layers', 1, 3),
                'channels': [1],
                'kernel_sizes': [],
                'dropout_rates': [],
                'pooling_pattern': []
            }

            for i in range(1, self.hyperparams['num_layers'] + 1):
                self.hyperparams['channels'].append(
                    trial.suggest_int(f'cnn_channels{i}', 1, 32)
                )

            for i in range(1, self.hyperparams['num_layers'] + 1):
                self.hyperparams['kernel_sizes'].append(
                    trial.suggest_int(f'kernel_size{i}', 1, 5)
                )

            for i in range(1, self.hyperparams['num_layers'] + 1):
                self.hyperparams['dropout_rates'].append(
                    trial.suggest_float(f'dropout_cnn_{i}', 0.05, 0.6)
                )

            for i in range(self.hyperparams['num_layers']):
                self.hyperparams['pooling_pattern'].append(
                    trial.suggest_categorical(f'pooling_layer_{i + 1}', [True, False])
                )

            layers = []
            has_been_pooled = False
            pooling_ctr = 0

            for i in range(self.hyperparams['num_layers']):
                layers.extend([
                    MagicConv(
                        self.hyperparams['channels'][i],
                        self.hyperparams['channels'][i + 1],
                        kernel_size=self.hyperparams['kernel_sizes'][i],
                        pooling=has_been_pooled,
                        pooling_cnt=pooling_ctr,
                        pooling_kernel_size=2,
                    ),
                    nn.BatchNorm1d(self.hyperparams['channels'][i + 1]),
                    nn.ReLU(),
                ])

                if self.hyperparams['pooling_pattern'][i]:
                    layers.append(nn.MaxPool1d(kernel_size=2))
                    pooling_ctr += 1
                    has_been_pooled = True

                layers.append(nn.Dropout1d(
                    self.hyperparams['dropout_rates'][i]
                ))

            self.m1_cnn = HexMagicCNN(copy.deepcopy(layers))
            self.m2_cnn = HexMagicCNN(copy.deepcopy(layers))

            max_neuron = 4096
            final_channels = self.hyperparams['channels'][-1]
            input_size = final_channels * (1039 // (2 ** pooling_ctr)) * 2
            mlp_additional_layers = trial.suggest_int('mlp_additional_layers', 1, 4)

            sizes = [input_size]
            for i in range(mlp_additional_layers):
                max_size = min(sizes[-1], max_neuron)
                min_size = min(max(8, sizes[-1] // 8), max_size)

                next_size = trial.suggest_int(f'linear{i}_size', min_size, max_size)
                sizes.append(next_size)

                max_neuron = max_neuron // (2 ** i)

            layers = []
            for i in range(mlp_additional_layers):
                layers.extend([
                    nn.Linear(sizes[i], sizes[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(trial.suggest_float(f'dropout_{i}', 0.05, 0.6))
                ])

            self.classifier = nn.Sequential(
                *layers,
                nn.Linear(sizes[-1], 2)
            )

        def forward(self, m1_image, m2_image, measurement_features):
            m1_image = m1_image.unsqueeze(1)
            m2_image = m2_image.unsqueeze(1)
            m1_cnn_features = (self.m1_cnn(m1_image)).flatten(1)
            m2_cnn_features = (self.m2_cnn(m2_image)).flatten(1)

            combined = torch.cat([m1_cnn_features, m2_cnn_features], dim=1)

            return self.classifier(combined)

    return HexMagicNet()
