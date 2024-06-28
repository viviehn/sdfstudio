#!/bin/bash

input_dir=$1
output_dir=$2

ns-process-data images --data $input_dir --output-dir $output_dir --num-downscales 0 --gpu
