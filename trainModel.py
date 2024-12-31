import argparse
import os
import sys
import time

from arrayClassification.trainingSupervisor import TrainingSupervisor


def main(modelname: str, dataset: str, epochs: int):
    nametag = f"{modelname}_{dataset}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    dataset_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "datasets", dataset)
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "trained_models", nametag)

    print(f"Starting Training with settings:")
    print(f"\t- Model = {modelname}")
    print(f"\t- Data = {dataset_dir}")
    print(f"\t- Epochs = {epochs}")
    print(f"\t- Output = {output_dir}\n")

    supervisor = TrainingSupervisor(modelname, dataset_dir, output_dir)
    supervisor.train_model(epochs)


if __name__ == "__main__":
    main("HexCNN", "DebugSet", 5)
