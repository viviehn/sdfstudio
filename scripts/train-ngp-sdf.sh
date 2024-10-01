#!/bin/bash

hostname
echo $CUDA_VISIBLE_DEVICES
nvidia-smi

DATA_ID=$1
EXP_NAME=ngp-sdf-baseline
TMP_STR=$(date +%Y%m%d_%H%M%S)_$RANDOM
LOCAL_OUTDIR=/scratch/vivienn/outputs/$TMP_STR/
MODEL_NAME=neus-facto-angelo
mkdir -p /scratch/vivienn/outputs/$TMP_STR

ns-train $MODEL_NAME \
    --output-dir $LOCAL_OUTDIR\
    --trainer.max-num-iterations 6101  --trainer.steps_per_save 1000\
    --trainer.steps-per-eval-image 2000\
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
    --pipeline.model.sdf-field.fix-geonet False\
    --pipeline.model.sdf-field.use-numerical-gradients False\
    --pipeline.model.sdf-field.fields-geometry.optimizer.lr .0001 \
    --pipeline.model.sdf-field.fields-geometry.optimizer.betas 0.9 0.99 \
    --pipeline.model.sdf-field.fields-geometry.scheduler.warm-up-end 0 \
    --pipeline.model.sdf-field.fields-geometry.scheduler.milestones 3660 \
    --pipeline.model.background-model none\
    --pipeline.model.sparse_points_sdf_loss_mult 1.0\
    --pipeline.model.curvature-loss-warmup-steps 2000\
    --pipeline.model.curvature-loss-multi 0.0\
    --pipeline.model.eikonal-loss-mult 0.0\
    --pipeline.datamanager.train_num_rays_per_batch 2\
    --pipeline.datamanager.train_num_images_to_sample_from -1\
    --pipeline.datamanager.train_num_times_to_repeat_images -1\
    --pipeline.datamanager.eval_num_images_to_sample_from 1 --vis tensorboard\
    --timestamp $TMP_STR \
    --experiment-name $DATA_ID     sdfstudio-data \
    --data /n/fs/3d-indoor/data/$DATA_ID/dslr/sdfstudio \
    --include_sdf_samples True\

FULL_OUTPUT_PATH=$LOCAL_OUTDIR/$DATA_ID/$method/$TMP_STR
RESOLUTION=1024
ns-extract-mesh --load-config $FULL_OUTPUT_PATH/config.yml \
    --resolution $RESOLUTION\
    --output-path $FULL_OUTPUT_PATH/$RESOLUTION-mesh.ply \
    --use-point-color True \

FINAL_PATH=/n/fs/3d-indoor/sdfstudio_outputs/3d-indoor/$EXP_NAME/$DATA_ID/$MODEL_NAME
mkdir -p $FINAL_PATH
mv /scratch/vivienn/outputs/$TMP_STR/$DATA_ID/neus-facto-angelo/$TMP_STR $FINAL_PATH
