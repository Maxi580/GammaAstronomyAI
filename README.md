# Gamma Astronomy AI

**This repository contains testdata generation and ML models for our coursework about classifying raw data from the MAGIC-Telescopes.**

## Sample Generation

You can execute the script `sampleGeneration.py` with the following arguments:

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
All generated Datasets will be saved in `./simulated_data`.

