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
    
    
    # # Params for Simple1DNet
    # supervisor.LEARNING_RATE = 0.0024372693219380376
    # supervisor.WEIGHT_DECAY = 0.00046404836789216026
    # supervisor.GRAD_CLIP_NORM = 3.1
    # supervisor.SCHEDULER_CYCLE_MOMENTUM = False
    # supervisor.SCHEDULER_STEP_SIZE = 4
    # supervisor.SCHEDULER_BASE_LR = 0.0001089415103064346
    # supervisor.SCHEDULER_MAX_LR = 0.004629201448534882
    
    # # Params for HexCircleNet
    # supervisor.LEARNING_RATE = 0.0006250708520118225
    # supervisor.WEIGHT_DECAY = 0.00012551308112717833
    # supervisor.GRAD_CLIP_NORM = 1.0
    # supervisor.SCHEDULER_CYCLE_MOMENTUM = True
    # supervisor.SCHEDULER_STEP_SIZE = 6
    # supervisor.SCHEDULER_BASE_LR = 1.9730551292086e-05
    # supervisor.SCHEDULER_MAX_LR = 0.0008631873109885323
    
    # # Params for HexagdlyNet
    # supervisor.LEARNING_RATE = 0.000817861258020137
    # supervisor.WEIGHT_DECAY = 0.000721390625278987
    # supervisor.GRAD_CLIP_NORM = 2.0
    # supervisor.SCHEDULER_CYCLE_MOMENTUM = False
    # supervisor.SCHEDULER_STEP_SIZE = 4
    # supervisor.SCHEDULER_BASE_LR = 0.0001108823441121981
    # supervisor.SCHEDULER_MAX_LR = 0.004212829321789141

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
