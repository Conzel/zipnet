#!/bin/bash
activation=$1
log_folder=$2
input_image=$3

mkdir -p ./results
python run.py $activation $log_folder $input_image
