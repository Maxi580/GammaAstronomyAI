import os
import time
from TrainingPipeline.TrainingSupervisor import TrainingSupervisor
from TrainingPipeline.Datasets import *

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'


def train_model(model_name: str, proton_file: str, gamma_file: str, epochs: int,
         early_stopping: bool = False, use_custom_params: bool = False,
         clean_image: bool = False, rescale_image: bool = False,
         max_samples: bool | None = None
         ):
    nametag = f"{model_name}_{time.strftime('%d-%m-%Y_%H-%M-%S')}"
    proton_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), proton_file)
    gamma_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), gamma_file)
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "trained_models", nametag)

    print(f"Starting Training with settings:")
    print(f"\t- Model = {model_name}")
    print(f"\t- Data = {proton_dir, gamma_dir}")
    print(f"\t- Epochs = {epochs}")
    print(f"\t- Output = {output_dir}\n")

    if model_name.lower() == 'hexagdlynet':
        dataset = MagicDatasetHexagdly(proton_file, gamma_file, max_samples=max_samples,
                           clean_image=clean_image, rescale_image=rescale_image
                           )
    else:
        dataset = MagicDataset(proton_file, gamma_file, max_samples=max_samples,
                            clean_image=clean_image, rescale_image=rescale_image
                            )

    supervisor = TrainingSupervisor(model_name, dataset, output_dir, debug_info=True,
                                    save_model=True, save_debug_data=True,
                                    early_stopping=early_stopping, use_custom_params=use_custom_params
                                    )

    print(f"Model has {supervisor._count_trainable_weights()} trainable weights.")
    supervisor.train_model(epochs)


if __name__ == "__main__":
    train_model("hexmagicnet", "magic-protons.parquet", "magic-gammas-new.parquet", 20)
