#!/bin/bash

hostname


TMP_STR=$(date +%Y%m%d_%H%M%S)_$RANDOM
local_outdir=/scratch/vivienn/outputs/$TMP_STR/
experiment=$1
data_path=/n/fs/lines/macro_bug/$experiment.json
method=neus-facto-angelo
resolution=2048
mkdir -p $local_outdir

OMP_NUM_THREADS=4 ns-train $method \
    --output-dir $local_outdir \
    --trainer.max-num-iterations 20001  --trainer.steps_per_save 10000\
    --trainer.steps-per-eval-batch 10000 --trainer.steps-per-eval-image 10000 \
    --pipeline.model.sdf-field.inside-outside True     \
    --pipeline.model.sdf-field.num-layers 2     \
    --pipeline.model.sdf-field.hidden-dim 64     \
    --pipeline.model.sdf-field.geo-feat-dim 64     \
    --pipeline.model.sdf-field.num-layers-color 2     \
    --pipeline.model.sdf-field.log2-hashmap-size 22\
    --pipeline.model.sdf-field.hash-features-per-level 4\
    --pipeline.model.sdf-field.base-res 16\
    --pipeline.model.sdf-field.max-res 2048\
    --pipeline.model.enable-progressive-hash-encoding False\
    --pipeline.model.sdf-field.use-appearance-embedding True\
    --pipeline.model.sdf-field.use-position-encoding True\
    --pipeline.model.sdf-field.vanilla-ngp True\
    --pipeline.model.sdf-field.geometric-init False\
    --pipeline.model.sdf-field.bias 0.8\
    --pipeline.model.sdf-field.fix-geonet False \
    --pipeline.model.background-model none\
    --pipeline.datamanager.train_num_images_to_sample_from -1\
    --pipeline.datamanager.train_num_times_to_repeat_images -1\
    --pipeline.datamanager.eval_num_images_to_sample_from 8 --vis tensorboard\
    --timestamp $TMP_STR \
    --experiment-name macro-bug-1-$experiment     sdfstudio-data \
    --data $data_path

full_output_path=/scratch/vivienn/outputs/$TMP_STR/macro-bug-1-$experiment/$method/$TMP_STR
ns-extract-mesh --load-config $full_output_path/config.yml \
    --resolution $resolution \
    --output-path $full_output_path/$resolution_mesh.ply \
    --bounding-box-min -.3 -.3 -.3 \
    --bounding-box-max .3 .3 .3 \
    --use-point-color True \
    --create-visibility-mask True

# path is /scratch/vivienn/outputs/$TMP_STR/macro-bug-1-$experiment/$method/$TMP_STR/
mkdir -p /n/fs/lines/sdfstudio_outputs/macro-bug-1-$experiment/$method
mv /scratch/vivienn/outputs/$TMP_STR/macro-bug-1-$experiment/$method/$TMP_STR /n/fs/lines/sdfstudio_outputs/macro-bug-1-$experiment/$method
