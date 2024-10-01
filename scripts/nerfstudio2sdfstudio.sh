#!/bin/bash

hostname

DATA_ID=$1
echo $DATA_ID

python /n/fs/lines/scripts/process_nerfstudio_to_sdfstudio.py \
    --data /n/fs/3d-indoor/data/$DATA_ID/dslr \
    --data-type colmap --scene-type indoor --scene-scale-mult 1.5 \
    --crop-rgb --crop-mult 3\
    --output-dir /n/fs/3d-indoor/data/$DATA_ID/dslr/sdfstudio
