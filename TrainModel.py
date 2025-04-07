import os
import time
from TrainingPipeline.TrainingSupervisor import TrainingSupervisor
from TrainingPipeline.Datasets import *

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'


def main(model_name: str, proton_file: str, gamma_file: str, epochs: int):
    nametag = f"{model_name}_{time.strftime('%d-%m-%Y_%H-%M-%S')}"
    proton_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), proton_file)
    gamma_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), gamma_file)
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "trained_models", nametag)

    print(f"Starting Training with settings:")
    print(f"\t- Model = {model_name}")
    print(f"\t- Data = {proton_dir, gamma_dir}")
    print(f"\t- Epochs = {epochs}")
    print(f"\t- Output = {output_dir}\n")

    dataset = MagicDataset(proton_file, gamma_file)
    supervisor = TrainingSupervisor(model_name, dataset, output_dir, debug_info=True, save_model=True,
                                    save_debug_data=True, early_stopping=False)

    print(f"Model has {supervisor._count_trainable_weights()} trainable weights.")
    supervisor.train_model(epochs)


if __name__ == "__main__":
    """parser = argparse.ArgumentParser()
    parser.add_argument( 
        "-e",
        "--epochs",
        metavar="epochs",
        type=int,
        default=30,
        help="Specify number of epochs for training the model.",
    )
    parser.add_argument(
        "-m",
        "--model",
        metavar="modelname",
        required=True,
        help="Specify the model you want to train.",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        metavar="dataset_name",
        required=True,
        help="Specify what dataset to use. (Must be in ./datasets)",
    )
    args = parser.parse_args(sys.argv[1:])"""

    main("HybridNet", "magic-protons.parquet", "magic-gammas-new.parquet", 20)
