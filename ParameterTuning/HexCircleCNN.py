import copy
import math

import torch
import torch.nn as nn
import optuna

from CNN.HexCircleLayers.HexCircleConv import HexCircleConv
from CNN.HexCircleLayers.HexCirclePool import HexCirclePool
from CNN.HexCircleLayers.pooling import _get_clusters


def parameterize_HexCircleNet(trial: optuna.Trial):

    class HexCircleCNN(nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.cnn = nn.Sequential(*layers)

        def forward(self, x):
            return self.cnn(x)

    class HexCircleNet(nn.Module):
        def __init__(self):
            super().__init__()

            n_pixels = [1039]
            num_layers = trial.suggest_int('cnn_layers', 1, 3)

            channels = [1]
            channels.append(
                trial.suggest_int('cnn_channels1', 8, 32, step=8)
            )
            
            for i in range(2, num_layers+1):
                lower_bound = ((channels[-1] + 15) // 16) * 16
                upper_bound = min(channels[-1] * 4, 512)  # restrict growth factor to 512 channels
                channels.append(trial.suggest_int(f'cnn_channels{i}', lower_bound, upper_bound, step=16))
                
            
            pooling_pattern = [
                trial.suggest_categorical(f'pooling_layer_{i+1}', [True, False])
                for i in range(num_layers)
            ]

            layers = []
            for i in range(num_layers):
                layers.extend([
                    HexCircleConv(
                        channels[i],
                        channels[i + 1],
                        kernel_size=trial.suggest_int(f'conv_kernel_size{i + 1}', 1, 5),
                        padding_mode=trial.suggest_categorical(f'conv_padding_mode{i + 1}', ['zeros', 'mean']),
                    ),
                    nn.BatchNorm1d(channels[i+1]),
                    nn.ReLU(),
                ])
                
                if pooling_pattern[i]:
                    layers.append(HexCirclePool(
                        trial.suggest_int(f'pooling_layer{i+1}_kernel', 1, 3),
                        trial.suggest_categorical(f'pooling_layer{i+1}_mode', ['max', 'avg']),
                    ))
                    
                    n_pixels.append(len(_get_clusters(n_pixels[-1], trial.params[f'pooling_layer{i+1}_kernel'])))
                
                layers.append(nn.Dropout1d(
                    trial.suggest_float(f'dropout_cnn_{i + 1}', 0.05, 0.6, step=0.05)
                ))

            self.m1_cnn = HexCircleCNN(copy.deepcopy(layers))
            self.m2_cnn = HexCircleCNN(copy.deepcopy(layers))
            
            num_layers = trial.params['cnn_layers']

            final_channels = trial.params[f'cnn_channels{num_layers}']
            input_size = final_channels * n_pixels[-1] * 2
            
            num_layers = trial.suggest_int('linear_layers', 1, 4)

            sizes = [input_size]
            max_ub = 2048
            for i in range(1, num_layers+1):
                lb_candidate = max(2, sizes[-1] // 16)
                lb = max(16, math.ceil(lb_candidate / 16) * 16)
                ub = (sizes[-1] // 4) * 4
                ub = min(max_ub, ub)

                if lb > ub:
                    lb = ub // 16

                sizes.append(
                    trial.suggest_int(f'linear{i}_size', lb, ub, step=16)
                )
            
            layers = []
            for i in range(num_layers):
                layers.extend([
                    nn.Linear(sizes[i], sizes[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(trial.suggest_float(f'linear{i}_dropout', 0.05, 0.6, step=0.05))
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

    return HexCircleNet()
