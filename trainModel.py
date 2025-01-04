import argparse
import os
import sys
import time
from SimulatedData.trainingSupervisor import TrainingSupervisor

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'


def main(model_name: str, dataset: str, epochs: int):
    nametag = f"{model_name}_{dataset}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    dataset_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "datasets", dataset)
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "trained_models", nametag)

    print(f"Starting Training with settings:")
    print(f"\t- Model = {model_name}")
    print(f"\t- Data = {dataset_dir}")
    print(f"\t- Epochs = {epochs}")
    print(f"\t- Output = {output_dir}\n")

    supervisor = TrainingSupervisor(model_name, dataset_dir, output_dir)
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

    main("HexCNN", "DebugSet", 1)
