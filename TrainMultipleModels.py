from TrainModel import train_model

def train_multiple(models: list[str], proton_file: str, gamma_file: str, epochs: int, **kwargs):
    for model in models:
        train_model(model, proton_file, gamma_file, epochs, **kwargs)


if __name__ == '__main__':
    # Comment out the model which you dont want to train
    MODELS = [
        'HexMagicNet',
        'HexCircleNet',
        'HexagdlyNet',
        'Simple1DNet',
        # ...
    ]

    PROTON_FILE = "magic-protons.parquet"
    GAMMA_FILE = "magic-gammas.parquet"
    EPOCHS = 20

    train_multiple(MODELS, PROTON_FILE, GAMMA_FILE, EPOCHS,
                   early_stopping=False, use_custom_params=False,
                   clean_image=False, rescale_image=False, max_samples=None
                )
