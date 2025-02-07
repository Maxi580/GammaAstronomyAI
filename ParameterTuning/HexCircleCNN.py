import torch
import torch.nn as nn
import optuna

from CNN.HexCircleLayers.HexCircleConv import HexCircleConv
from CNN.HexCircleLayers.HexCirclePool import HexCirclePool
from CNN.HexCircleLayers.pooling import get_clusters


def parameterize_HexCircleNet(trial: optuna.Trial):
        
    n_pixels = [1039]

    class HexCircleCNN(nn.Module):
        def __init__(self):
            super().__init__()

            num_layers = trial.suggest_int('cnn_layers', 1, 4)

            channels = [1]
            for i in range(1, num_layers+1):
                channels.append(trial.suggest_int(f'cnn_channels{i}', channels[-1]+1, channels[-1]*8))
                
            
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
                        kernel_size=trial.suggest_int(f'kernel_size{i + 1}', 1, 5),
                        n_pixels=n_pixels[-1]
                    ),
                    nn.BatchNorm1d(channels[i+1]),
                    nn.ReLU(),
                ])
                
                if pooling_pattern[i]:
                    p_kernel = trial.suggest_int(f'pooling_layer{i+1}_kernel', 1, 3)
                    p_mode = trial.suggest_categorical(f'pooling_layer{i+1}_mode', ["max", "avg"])
                    
                    layers.append(HexCirclePool(
                        p_kernel,
                        n_pixels[-1],
                        p_mode,
                    ))
                    
                    n_pixels.append(len(get_clusters(n_pixels[-1], p_kernel)))
                
                layers.append(nn.Dropout1d(
                    trial.suggest_float(f'dropout_cnn_{i + 1}', 0.05, 0.6)
                ))


            self.cnn = nn.Sequential(*layers)

        def forward(self, x):
            return self.cnn(x)

    class HexCircleNet(nn.Module):
        def __init__(self):
            super().__init__()

            self.m1_cnn = HexCircleCNN()
            self.m2_cnn = HexCircleCNN()

            channels3 = trial.params['cnn_channels3']
            input_size = channels3 * n_pixels[-1] * 2
            
            num_layers = trial.suggest_int('linear_layers', 1, 4)

            sizes = [input_size]
            for i in range(1, num_layers+1):
                sizes.append(trial.suggest_int(f'linear{i}_size', max(2, sizes[-1]//8), sizes[-1]))
            
            layers = []
            for i in range(num_layers):
                layers.extend([
                    nn.Linear(sizes[i], sizes[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(trial.suggest_float(f'linear{i}_dropout', 0.05, 0.6))
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
