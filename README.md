# Gamma Astronomy AI

**This repository contains testdata generation and ML models for our coursework about classifying raw data from the MAGIC-Telescopes.**

## Sample Generation

You can execute the script `genSamples.py` with the following arguments:

```sh
usage: genSamples.py [-h] [-n sample_count] --name dataset_name [-s shapes]

options:
  -h, --help            show this help message and exit
  -n sample_count       Specified count of samples to generate.
  --name dataset_name   Specify the name of the generated dataset.
  -s shapes, --shapes shapes
                        Specify what shapes to generate and their probabilities.
```

Additionally, you can adjust the noise settings inside the script itself.

Valid Shapes are:
- `ellipse` for Ellipses
- `ellipse-centered` for Ellipses that point to the center
- `square` for Squares

Shapes must be provided in the following format: `<shape1>:<probability1>,<shape2>:<probability2>,...`.
Example: `ellipse:1,square:3`. This will produce a dataset containing roughly 25% ellipses and 75% squares.

All generated Datasets will be saved in `./datasets`.


## Model Training

You can execute the script `trainModel.py` with the following arguments to train any model:

```sh
usage: trainModel.py [-h] [-e epochs] -m modelname -d dataset_name [-i]

options:
  -h, --help            show this help message and exit
  -e epochs, --epochs epochs
                        Specify number of epochs for training the model.
  -m modelname, --model modelname
                        Specify the model you want to train.
  -d dataset_name, --dataset dataset_name
                        Specify what dataset to use. (Must be in ./datasets)
  -i, --info            Print out more info during the training process, e.g. metrics for every epoch.
```

The Dataset used for training must be saved in `./datasets`.

Valid Models are:
- `HexCNN`
- `SimpleShapeCNN`

The trained models and other information about the training process, like metrics will be saved in `./trained_models`.
