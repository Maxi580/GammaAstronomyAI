import argparse
import os
import sys
import time

from arrayClassification.trainingSupervisor import TrainingSupervisor


def main(modelname: str, dataset: str, epochs: int, info_prints: bool):
    nametag = f"{modelname}_{time.strftime("%Y-%m-%d_%H.%M.%S")}"
    dataset_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "datasets", dataset
    )
    output_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "trained_models", nametag
    )

    print("Starting Training with settings:")
    print(f"\t- Modelname = {modelname}")
    print(f"\t- Training Data = {dataset_dir}")
    print(f"\t- Number Epochs = {epochs}")
    print(f"\t- Print Info = {info_prints}")
    print(f"Outputs will be saved under: {output_dir}\n")

    supervisor = TrainingSupervisor(modelname, output_dir)

    supervisor.load_training_data(dataset_dir)

    supervisor.start_training(epochs, info_prints)

    supervisor.write_results()


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        "-i",
        "--info",
        action="store_true",
        help="Print out more info during the training process, e.g. metrics for every epoch.",
    )
    args = parser.parse_args(sys.argv[1:])

    # Start generation
    main(args.model, args.dataset, args.epochs, args.info)
