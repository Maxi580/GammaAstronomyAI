import copy

import torch
import torch.nn as nn
import hexagdly
import optuna



def parameterize_HexagdlyNet(trial: optuna.Trial):

    class HexagdlyCNN(nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.cnn = nn.Sequential(*layers)

        def forward(self, x):
            return self.cnn(x)

    class HexagdlyNet(nn.Module):
        def __init__(self):
            super().__init__()

            num_layers = trial.suggest_int('cnn_layers', 1, 3)

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
                    hexagdly.Conv2d(
                        channels[i],
                        channels[i + 1],
                        kernel_size=trial.suggest_int(f'cnn_kernel_size{i + 1}', 1, 5)
                    ),
                    nn.BatchNorm2d(channels[i+1]),
                    nn.ReLU(),
                ])
                
                if pooling_pattern[i]:
                    layers.append(hexagdly.MaxPool2d(
                        trial.suggest_int(f'pooling_layer{i+1}_kernel', 1, 3),
                        trial.suggest_int(f'pooling_layer{i+1}_stride', 1, 3),
                    ))
                
                layers.append(nn.Dropout2d(
                    trial.suggest_float(f'dropout_cnn_{i + 1}', 0.05, 0.6)
                ))

            self.m1_cnn = HexagdlyCNN(copy.deepcopy(layers))
            self.m2_cnn = HexagdlyCNN(copy.deepcopy(layers))
            
            num_layers = trial.params['cnn_layers']
            
            # Calculate output dimensions of the CNN based on pooling layers
            out_dims = (34, 39) # Magic Images converted to hexagdly format have size 34x39 (height x width)
            for i in range(num_layers):
                if pooling_pattern[i]:
                    kernel = trial.params[f'pooling_layer{i+1}_kernel']
                    stride = trial.params[f'pooling_layer{i+1}_stride']
                    out_dims = hexagdly_maxpool2d_output_shape(out_dims, kernel, stride)

            final_channels = trial.params[f'cnn_channels{num_layers}']
            input_size = final_channels * out_dims[0] * out_dims[1] * 2
            
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

    return HexagdlyNet()


def hexagdly_maxpool2d_output_shape(input_shape, kernel_size, stride):
    H, W = input_shape

    odd_left = 0
    odd_right = max(0, 0 - ((W - 1) % (2 * stride)))
    odd_top = kernel_size

    T = stride // 2
    constraint = (H - 1) - ((H - 1 - T) // stride) * stride
    odd_bottom = max(0, kernel_size - constraint)

    odd_padded_H = H + odd_top + odd_bottom
    odd_padded_W = W + odd_left + odd_right

    kernel_h = 1 + 2 * kernel_size
    kernel_w = 1

    out_h_odd = (odd_padded_H - kernel_h) // stride + 1
    out_w_odd = (odd_padded_W - kernel_w) // (2 * stride) + 1

    effective_W_even = max(W - stride, 0)
    even_right = max(0, 0 - ((W - 1 - stride) % (2 * stride)))
    out_w_even = (effective_W_even - kernel_w + even_right) // (2 * stride) + 1

    out_h = out_h_odd
    if out_w_odd == out_w_even:
        out_w = 2 * out_w_odd
    else:
        out_w = 2 * min(out_w_odd, out_w_even) + 1

    return out_h, out_w