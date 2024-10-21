#!/bin/bash

hostname
echo $CONDA_PREFIX
git --git-dir=/n/fs/lines/sdfstudio-indoors/sdfstudio/.git branch --show-current
echo $0


TMP_STR=$(date +%Y%m%d_%H%M%S)_$RANDOM
LOCAL_OUTDIR=/scratch/vivienn/outputs/$TMP_STR/

MODEL_NAME=nfa-multi
EXP_CATEGORY=multiscene
EXP_NAME=$5
BASE_OUTDIR=/n/fs/3d-indoor/sdfstudio_outputs/3d_indoor

mkdir -p $LOCAL_OUTDIR
#config=/n/fs/3d-indoor/vivien_data/data/scenes.txt
#config=/n/fs/3d-indoor/data/002_scenes.txt
#readarray -t DATA_IDS < $config

#DATA_IDS=("3f1e1610de" "8b5caf3398" "210f741378" "785e7504b9" "bfd3fd54d2")
#DATA_IDS=("3f1e1610de" "8b5caf3398" "210f741378" "785e7504b9")
DATA_IDS=($1 $2 $3 $4)
#DATA_IDS=("8b5caf3398" "3f1e1610de")
#DATA_IDS=("3f1e1610de")
#DATA_IDS=("785e7504b9")

LIST_OF_SCENES=""

for data_id in "${DATA_IDS[@]}";
do
    echo $data_id
    LIST_OF_SCENES+=" /n/fs/3d-indoor/data/$data_id/dslr/sdfstudio"
done

echo $LIST_OF_SCENES

ns-train $MODEL_NAME \
    --viewer.quit-on-train-completion True \
    --output-dir $LOCAL_OUTDIR\
    --trainer.max-num-iterations 10001  --trainer.steps_per_save 2000\
    --trainer.steps-per-eval-image 500\
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
    --pipeline.model.sdf-field.fix-geonet False \
    --pipeline.model.sdf-field.use-numerical-gradients False\
    --pipeline.model.background-model none\
    --optimizers.fields-geometry.optimizer.lr .0001 \
    --optimizers.fields-geometry.optimizer.betas 0.9 0.99 \
    --optimizers.fields-geometry.scheduler.warm-up-end 0 \
    --optimizers.fields-geometry.scheduler.milestones 3660 \
    --pipeline.model.sdf_sample_training True \
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
    --pipeline.datamanager.dataparser.include-sdf-samples True \
    --pipeline.datamanager.dataparser.use_point_color True \
    --pipeline.datamanager.dataparser.multiscene-data $LIST_OF_SCENES \
    #sdfstudio-data --data /n/fs/3d-indoor/data/785e7504b9/dslr/sdfstudio \
    #--include_sdf_samples True \
    #--use_point_color True \


FULL_OUTPUT_PATH=$LOCAL_OUTDIR/$EXP_NAME/$MODEL_NAME/$TMP_STR
RESOLUTION=1024
ns-extract-mesh --load-config $FULL_OUTPUT_PATH/config.yml \
    --resolution $RESOLUTION\
    --output-path $FULL_OUTPUT_PATH/$RESOLUTION-mesh.ply \
    --use-point-color True \
    --all_scenes True \

FINAL_PATH=$BASE_OUTDIR/$EXP_CATEGORY/$EXP_NAME/$MODEL_NAME
mkdir -p $FINAL_PATH
mv $LOCAL_OUTDIR/$EXP_NAME/$MODEL_NAME/$TMP_STR $FINAL_PATH
