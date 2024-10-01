#!/bin/bash

hostname
echo $CUDA_VISIBLE_DEVICES
nvidia-smi

DATA_ID=$1
EXP_NAME=on-40b56bf310-cl-hm-fr-gn
TMP_STR=$(date +%Y%m%d_%H%M%S)_$RANDOM
LOCAL_OUTDIR=/scratch/vivienn/outputs/$TMP_STR/
MODEL_NAME=neus-facto-angelo
mkdir -p $LOCAL_OUTDIR


ns-train $MODEL_NAME \
    --trainer.max-num-iterations 200001  --trainer.steps_per_save 10000\
    --trainer.steps-per-eval-image 1000\
    --trainer.steps-per-eval-batch 1000\
    --output-dir $LOCAL_OUTDIR \
    --viewer.websocket-port 12700 \
    --pipeline.model.sdf-field.inside-outside True     \
    --pipeline.model.sdf-field.num-layers 2     \
    --pipeline.model.sdf-field.hidden-dim 256     \
    --pipeline.model.sdf-field.geo-feat-dim 256     \
    --pipeline.model.sdf-field.num-layers-color 2     \
    --pipeline.model.sdf-field.log2-hashmap-size 22\
    --pipeline.model.sdf-field.hash-features-per-level 4\
    --pipeline.model.sdf-field.base-res 16\
    --pipeline.model.sdf-field.max-res 2048\
    --pipeline.model.enable-progressive-hash-encoding False\
    --pipeline.model.sdf-field.use-appearance-embedding True\
    --pipeline.model.sdf-field.appearance-embedding-dim 128\
    --pipeline.model.sdf-field.use-position-encoding True\
    --pipeline.model.sdf-field.vanilla-ngp True\
    --pipeline.model.sdf-field.geometric-init False\
    --pipeline.model.sdf-field.bias 0.8\
    --pipeline.model.sdf-field.fix-geonet True \
    --pipeline.model.background-model none\
    --pipeline.datamanager.train_num_images_to_sample_from -1\
    --pipeline.datamanager.train_num_times_to_repeat_images -1\
    --pipeline.datamanager.eval_num_images_to_sample_from 8 --vis tensorboard\
    --pipeline.pop-sdf-pretrain True \
    --timestamp $TMP_STR \
    --trainer.load-dir /n/fs/3d-indoor/sdfstudio_outputs/3d-indoor/ngp-sdf-baseline/40b56bf310 \
    --experiment-name $DATA_ID     sdfstudio-data \
    --data /n/fs/3d-indoor/data/$DATA_ID/dslr/sdfstudio

FULL_OUTPUT_PATH=$LOCAL_OUTDIR/$DATA_ID/$MODEL_NAME/$TMP_STR
RESOLUTION=1024

FINAL_PATH=/n/fs/3d-indoor/sdfstudio_outputs/3d-indoor/nfa-w-pretraining/$EXP_NAME/$DATA_ID/$MODEL_NAME
mkdir -p $FINAL_PATH
mv /scratch/vivienn/outputs/$TMP_STR/$DATA_ID/neus-facto-angelo/$TMP_STR $FINAL_PATH
mv $FULL_OUTPUT_PATH $FINAL_PATH

