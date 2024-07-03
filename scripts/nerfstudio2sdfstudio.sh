#!/bin/bash

hostname

DATA_ID=$1
echo $DATA_ID

python /n/fs/lines/sdfstudio/scripts/datasets/process_nerfstudio_to_sdfstudio.py \
    --data /n/fs/3d-indoor/data/$DATA_ID/dslr/nerfstudio \
    --data-type colmap --scene-type indoor --scene-scale-mult 1.5 \
    --output-dir /n/fs/3d-indoor/data/$DATA_ID/dslr/sdfstudio
