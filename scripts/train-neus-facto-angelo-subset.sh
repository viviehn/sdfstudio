#!/bin/bash

hostname
echo $CONDA_PREFIX
git --git-dir=/n/fs/lines/sdfstudio-indoors/sdfstudio/.git branch --show-current
echo $0


TMP_STR=$(date +%Y%m%d_%H%M%S)_$RANDOM
LOCAL_OUTDIR=/scratch/vivienn/outputs/$TMP_STR/

DATA_ID=$1
MODEL_NAME=neus-facto-angelo
EXP_CATEGORY=nfa-subsets
EXP_NAME=$DATA_ID/subset-$2

mkdir -p $LOCAL_OUTDIR


OMP_NUM_THREADS=4 ns-train $MODEL_NAME \
    --viewer.quit-on-train-completion True \
    --trainer.max-num-iterations 200001  --trainer.steps_per_save 10000\
    --trainer.steps-per-eval-image 10000\
    --trainer.steps_per_eval_batch 1000 \
    --output-dir $LOCAL_OUTDIR \
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
    --pipeline.datamanager.train_num_images_to_sample_from $2\
    --pipeline.datamanager.train_num_times_to_repeat_images -1\
    --pipeline.datamanager.eval_num_images_to_sample_from 8 --vis tensorboard\
    --timestamp $TMP_STR \
    --experiment-name $EXP_NAME     sdfstudio-data \
    --data /n/fs/3d-indoor/data/$DATA_ID/dslr/sdfstudio \

FULL_OUTPUT_PATH=$LOCAL_OUTDIR/$EXP_NAME/$MODEL_NAME/$TMP_STR
RESOLUTION=1024
ns-extract-mesh --load-config $FULL_OUTPUT_PATH/config.yml \
    --resolution $RESOLUTION\
    --output-path $FULL_OUTPUT_PATH/$RESOLUTION-mesh.ply \
    --use-point-color True \

FINAL_PATH=/n/fs/3d-indoor/sdfstudio_outputs/3d-indoor/$EXP_CATEGORY/$EXP_NAME/$MODEL_NAME
mkdir -p $FINAL_PATH
mv $LOCAL_OUTDIR/$EXP_NAME/$MODEL_NAME/$TMP_STR $FINAL_PATH
