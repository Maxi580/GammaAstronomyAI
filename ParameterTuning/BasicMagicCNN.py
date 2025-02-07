import torch
import torch.nn as nn
import optuna

from CNN.MagicConv.MagicConv import MagicConv

def parameterize_BasicMagicNet(trial: optuna.Trial):

    def suggest_group_norm_params(trial, num_channels, name_prefix):
        divisors = [i for i in range(1, num_channels + 1) if num_channels % i == 0]

        divisor_ratio = trial.suggest_int(f'{name_prefix}_group_ratio', 1, 10)
        suggested_number_of_groups = num_channels // divisor_ratio

        return min(divisors, key=lambda x: abs(x - suggested_number_of_groups))

    class BasicMagicCNN(nn.Module):
        def __init__(self, prefix):
            super().__init__()
            self.prefix = prefix

            pooling_pattern = [
                trial.suggest_categorical(f'pooling_layer_{i}', [True, False])
                for i in range(3)
            ]

            channels = [
                1,
                trial.suggest_int(f'cnn_channels1', 2, 16),
                trial.suggest_int(f'cnn_channels2', 4, 32),
                trial.suggest_int(f'cnn_channels3', 8, 48)
            ]

            layers = []
            pooling_count = 0
            has_previous_pooling = False

            for i in range(3):
                needs_pooling = has_previous_pooling or pooling_pattern[i]

                layers.append(
                    MagicConv(
                        channels[i],
                        channels[i + 1],
                        kernel_size=trial.suggest_int(f'kernel_size{i + 1}', 1, 5),
                        pooling=needs_pooling,
                        pooling_cnt=pooling_count,
                        pooling_kernel_size=2
                    )
                )

                layers.extend([
                    nn.GroupNorm(suggest_group_norm_params(trial, channels[i+1], f'cnn_{i + 1}'),
                                 channels[i + 1]),
                    nn.ReLU(),
                ])

                if pooling_pattern[i]:
                    layers.append(nn.MaxPool1d(kernel_size=2))
                    pooling_count += 1
                    has_previous_pooling = True

                layers.append(nn.Dropout1d(
                    trial.suggest_float(f'dropout_cnn_{i + 1}', 0.05, 0.6)
                ))

            self.cnn = nn.Sequential(*layers)

        def forward(self, x):
            return self.cnn(x)

    class BasicMagicNet(nn.Module):
        def __init__(self):
            super().__init__()

            self.m1_cnn = BasicMagicCNN("m1_")
            self.m2_cnn = BasicMagicCNN("m2_")

            channels3 = trial.params['cnn_channels3']
            num_pooling = sum(1 for i in range(3) if trial.params[f'pooling_layer_{i}'])
            input_size = channels3 * (1039 // (2 ** num_pooling)) * 2

            linear1_size = trial.suggest_int('linear1_size', 512, 2048, step=256)
            linear2_size = trial.suggest_int('linear2_size', 128, 512, step=64)
            linear3_size = trial.suggest_int('linear3_size', 64, 256, step=32)
            dropout_linear_1 = trial.suggest_float('dropout_linear_1', 0.05, 0.6)
            dropout_linear_2 = trial.suggest_float('dropout_linear_2', 0.05, 0.6)
            dropout_linear_3 = trial.suggest_float('dropout_linear_3', 0.05, 0.6)

            self.classifier = nn.Sequential(
                nn.Linear(input_size, linear1_size),
                nn.GroupNorm(suggest_group_norm_params(trial, linear1_size, f'mlp_1'), linear1_size),
                nn.ReLU(),
                nn.Dropout(dropout_linear_1),

                nn.Linear(linear1_size, linear2_size),
                nn.GroupNorm(suggest_group_norm_params(trial, linear2_size, f'mlp_2'), linear2_size),
                nn.ReLU(),
                nn.Dropout(dropout_linear_2),

                nn.Linear(linear2_size, linear3_size),
                nn.GroupNorm(suggest_group_norm_params(trial, linear3_size, f'mlp_3'), linear3_size),
                nn.ReLU(),
                nn.Dropout(dropout_linear_3),

                nn.Linear(linear3_size, 2)
            )

        def forward(self, m1_image, m2_image, measurement_features):
            m1_image = m1_image.unsqueeze(1)
            m2_image = m2_image.unsqueeze(1)
            m1_cnn_features = (self.m1_cnn(m1_image)).flatten(1)
            m2_cnn_features = (self.m2_cnn(m2_image)).flatten(1)

            combined = torch.cat([m1_cnn_features, m2_cnn_features], dim=1)

            return self.classifier(combined)

    return BasicMagicNet()
