# Munis

## Install

First install the dependencies by running:

`pip install -r requirements.txt`

Once all the dependencies have been installed successfully, simply run:

`pip install .`

Installation, including dependencies, should only take a few minutes.

> Note: the software was only tested on the versions specified in the requirements.txt file under python 3.10, though we expect it *should* run fine on other python versions, and same major releases of the dependencies.

## Running predictions

To run predictions, simply run:

`python predict.py`

> Note: we recommend running predictions using a GPU for speed, but CPU will work.

The following options are available:

- `--peptides`: path a CSV file with columns: `pep`, `mhc`, `left`, `right`
- `--fasta`: alternative input, as a path to a fasta file containing full proteins sequences
- `--mhc`: list of MHC's (comma separated) to run predictions for (only used for fasta input)
- `--output`: path to output csv containing predictions with columns: `mhc`, `pep`, `left`, `right`, `score`
- `--min_len`: the minimum peptide length (default: 8), supports >=8
- `--max_len`: the maximum peptide length (default: 15), supports <= 15
- `--device`: which device to use for prediction (default to `cuda`)

> Note: only one of `--fasta` or `--peptides` should be used at the same time.

To run an example you may run the following command to run predictions on an example protein:

`python predict.py --fasta example.fasta --mhc HLA-A*02:01 --outdir ./ --cache`

Running this example should take under a few seconds with a GPU and under 5 minutes on CPU.

## Training a new model

First download the data from:
https://data.mendeley.com/preview/5w2zg5jn27?a=e53ed26b-1a4d-4d77-aa76-a0b0d489ac2f

You will only need the `el_train.csv` file. Point the `scripts/config.yaml` to this files's path on your system.

Then, run `python train.py config.yaml` to execute training. It may be useful to check the relevant options under `python train.py --help`.
