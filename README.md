# About

This repository provides the code and experiments from the paper:

<p align="center"><strong>Closed-Loop Neural Operator-Based Observer of Traffic Density</strong></p>
<p align="center">Alice Harting<sup>$\dagger$</sup>, Karl Henrik Johansson, and Matthieu Barreau</p>
<p align="center">KTH Royal Institute of Technology</p>

The paper is published here: [https://arxiv.org/abs/2504.04873](https://arxiv.org/abs/2504.04873)

<sup>$\dagger$</sup> Corresponding author: aharting@kth.se
# Set-up
### Install required packages
`pip install -r requirements.txt`

Python 3.11.5
### Download the data and the models
The datasets and models used in the paper are stored on Google drive.

To download the datasets, run the following from the command line

    DATA_ZIP=./data.zip
    gdown --id "1NNKMMrOo04uLgewvvPhLwhToJm8AT8RD" -O "$DATA_ZIP"
    unzip -o "$DATA_ZIP" -d "./"
    rm $DATA_ZIP

The trained models can similarly be downloaded by

    OUTPUT_DIR=./models
    mkdir -p $OUTPUT_DIR
    gdown --id 1XkXyZArWbJCqsCOWhpXf-NNX4P8NU9eX -O $OUTPUT_DIR/openloop.pt
    gdown --id 1JIkY98QQYI9-ULlAhuxGc911ryCYdOal -O $OUTPUT_DIR/closedloop.pt


This is summarized in `download.sh`.
# Reproducing the experiments
To **reproduce the plots from the paper**, make sure the models and the datasets from the previous step are downloaded according to the instructions, then run:

`python scripts/report.py --config configs/report.yaml`

This includes an evaluation of the observers on all SUMO scenarios in the test set.

The **prediction operator** was trained by running:

`python scripts/train_prediction_operator.py --config configs/openloop.yaml`.

You can inspect its performance on individual SUMO scenarios by running:

`python scripts/evaluation_openloop.py --config configs/openloop.yaml`. 

The **correction operator** was trained by running:

`python scripts/train_correction_operator.py --config configs/closedloop.yaml`

The resulting closed-loop observer can be evaluated on individual SUMO scenarios by running:

`python scripts/evaluation_closedloop.py --config configs/closedloop.yaml`

# Cite
Please consider citing our work if you find our paper and/or this code useful.

    @article{harting2025closed,
        title={Closed-Loop Neural Operator-Based Observer of Traffic Density},
        author={Harting, Alice and Johansson, Karl Henrik and Barreau, Matthieu},
        journal={arXiv preprint arXiv:2504.04873},
        year={2025}
    }
# References
The files under `imported/` contain code from repositories

[https://github.com/ziqi-ma/neuraloperator](https://github.com/ziqi-ma/neuraloperator)

[https://github.com/Plasma-FNO/FNO_Isothermal_Blob](https://github.com/Plasma-FNO/FNO_Isothermal_Blob)

as indicated.

# Acknowledgements
This work is supported by the Wallenberg AI, Autonomous Systems and Software Program (WASP) funded by the Knut and Alice Wallenberg Foundation.

A. Harting, K.H. Johansson, and M. Barreau are with the Division of Decision and Control Systems, Digital Futures, KTH Royal Institute of Technology, SE-100 44 Stockholm, Sweden
