#!/bin/bash

OUTPUT_DIR=./models
mkdir -p $OUTPUT_DIR
gdown --id 1XkXyZArWbJCqsCOWhpXf-NNX4P8NU9eX -O $OUTPUT_DIR/openloop.pt
gdown --id 1JIkY98QQYI9-ULlAhuxGc911ryCYdOal -O $OUTPUT_DIR/closedloop.pt

DATA_ZIP=./data.zip
gdown --id "1NNKMMrOo04uLgewvvPhLwhToJm8AT8RD" -O "$DATA_ZIP"
unzip -o "$DATA_ZIP" -d "./"
rm $DATA_ZIP