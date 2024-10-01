#!/bin/bash

hostname
echo $CUDA_VISIBLE_DEVICES
nvidia-smi

TMP_STR=$(date +%Y%m%d_%H%M%S)_$RANDOM
MODEL_NAME=neus-facto-angelo-multi
LOCAL_OUTDIR=/scratch/vivienn/outputs/$TMP_STR/
EXP_NAME=5-scenes
mkdir -p $LOCAL_OUTDIR

#config=/n/fs/3d-indoor/vivien_data/data/scenes.txt
config=/n/fs/3d-indoor/data/002_scenes.txt
readarray -t DATA_IDS < $config

LIST_OF_SCENES=""

for data_id in "${DATA_IDS[@]}";
do
    echo $data_id
    LIST_OF_SCENES+=" /n/fs/3d-indoor/data/$data_id/dslr/sdfstudio"
done

echo $LIST_OF_SCENES

ns-train $MODEL_NAME \
    --output-dir $LOCAL_OUTDIR\
    --trainer.max-num-iterations 10001  --trainer.steps_per_save 2000\
    --trainer.steps-per-eval-image 1000\
    --trainer.steps-per-eval-batch 1000\
    --trainer.steps-per-eval-all-images 100000\
    --logging.steps-per-log 100\
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
    --pipeline.model.sdf-field.num-scenes 2\
    --optimizers.fields-geometry.optimizer.lr .0001 \
    --optimizers.fields-geometry.optimizer.betas 0.9 0.99\
    --optimizers.fields-geometry.scheduler.warm-up-end 0 \
    --optimizers.fields-geometry.scheduler.milestones 3660 \
    --pipeline.model.background-model none\
    --pipeline.model.sparse_points_sdf_loss_mult 1.0\
    --pipeline.model.curvature-loss-warmup-steps 2000\
    --pipeline.model.curvature-loss-multi 0.0\
    --pipeline.model.eikonal-loss-mult 0.0\
    --pipeline.datamanager.train_num_rays_per_batch 2\
    --pipeline.datamanager.train_num_images_to_sample_from -1\
    --pipeline.datamanager.train_num_times_to_repeat_images -1\
    --pipeline.datamanager.eval_num_images_to_sample_from 1 --vis tensorboard\
    --experiment-name $EXP_NAME\
    --timestamp $TMP_STR \
    --pipeline.datamanager.dataparser.multiscene-data $LIST_OF_SCENES \
    --pipeline.datamanager.dataparser.multiscene True \
    --pipeline.datamanager.dataparser.include-sdf-samples True \

FULL_OUTPUT_PATH=$LOCAL_OUTDIR/$EXP_NAME/$MODEL_NAME/$TMP_STR
RESOLUTION=1024
ns-extract-mesh --load-config $FULL_OUTPUT_PATH/config.yml \
    --resolution $RESOLUTION\
    --output-path $FULL_OUTPUT_PATH/$RESOLUTION-mesh.ply \
    --use-point-color True \

FINAL_PATH=/n/fs/3d-indoor/sdfstudio_outputs/3d-indoor/multiscene/$EXP_NAME/$MODEL_NAME
mkdir -p $FINAL_PATH
mv $FULL_OUTPUT_PATH $FINAL_PATH
