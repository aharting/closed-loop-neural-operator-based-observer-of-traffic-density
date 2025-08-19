#!/bin/bash

OUTPUT_DIR=./models
mkdir -p $OUTPUT_DIR
gdown --id 1XkXyZArWbJCqsCOWhpXf-NNX4P8NU9eX -O $OUTPUT_DIR/openloop.pt

DATA_ZIP=./data.zip
gdown --id "1NNKMMrOo04uLgewvvPhLwhToJm8AT8RD" -O "$DATA_ZIP"
unzip -o "$DATA_ZIP" -d "./"
