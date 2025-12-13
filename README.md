# γ-Quant
## Inertial Experiments

This folder contains all relevant files to reproduce experiments mentioned within Section 4 of the main paper (i.e. all experiments using inertial data as input).

## Installation
Please follow instructions mentioned in the [INSTALL.md](/inertial/INSTALL.md) file within this folder.

## Download
Datasets used for conducting experiments can be downloaded [here](https://uni-siegen.sciebo.de/s/gq2S7nomosqfNPa). Place the downloaded folder into this subfolder (`inertial`) of the main repository. If not already the case, rename the folder to be called `data`. You can also specify a different data directory within the `configs` files by adjusting the `sens_folder` and `anno_json` accordingly.

## Reproduce Experiments
Rerun experiments by running the `main.py` script. For example:

````
python main.py --config ./configs/deepconvlstm/hangtime_loso.yaml --seed 1
````

Each config file represents one type of experiment. The folder `job_scripts` contains collections of commands of all experiments mentioned in the paper.

## Logging using Neptune.ai
In order to log experiments to [Neptune.ai](https://neptune.ai) please provide `project` and `api_token` information in your local deployment (see `main.py`)

## Contact
Marius Bock (marius.bock(at)uni-siegen.de)
