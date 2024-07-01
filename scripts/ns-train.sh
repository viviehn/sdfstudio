#!/bin/bash

hostname


TMP_STR=$(date +%Y%m%d_%H%M%S)_$RANDOM
local_outdir=/scratch/vivienn/outputs/$TMP_STR/
experiment=bunny_2_zoom_22_baseline
#data_path=/n/fs/lines/macro_bug/$experiment.json
data_path=/n/fs/lines/printed_miniatures_dataset/bunny_2_zoom_22/sdfstudio/meta_data.json
#method=neus-facto-angelo
method=neus-facto-angelo
mkdir -p $local_outdir

all_str="all"

if [[ $experiment =~ $all_str ]]; then
    num_images=750
    num_times=11
else
    num_images=-1
    num_times=-1
fi
    


OMP_NUM_THREADS=4 ns-train $method \
    --output-dir $local_outdir \
    --trainer.max-num-iterations 60001  --trainer.steps_per_save 10000\
    --trainer.steps-per-eval-batch 10000 --trainer.steps-per-eval-image 100000 \
    --pipeline.model.sdf-field.inside-outside False     \
    --pipeline.model.sdf-field.num-layers 2     \
    --pipeline.model.sdf-field.hidden-dim 64     \
    --pipeline.model.sdf-field.geo-feat-dim 64     \
    --pipeline.model.sdf-field.num-layers-color 2     \
    --pipeline.model.sdf-field.log2-hashmap-size 22\
    --pipeline.model.sdf-field.hash-features-per-level 4\
    --pipeline.model.sdf-field.base-res 16\
    --pipeline.model.sdf-field.max-res 4096\
    --pipeline.model.enable-progressive-hash-encoding True\
    --pipeline.model.sdf-field.use-appearance-embedding True\
    --pipeline.model.sdf-field.use-position-encoding True\
    --pipeline.model.sdf-field.vanilla-ngp True\
    --pipeline.model.sdf-field.geometric-init False\
    --pipeline.model.sdf-field.bias 0.8\
    --pipeline.model.sdf-field.fix-geonet False \
    --pipeline.model.background-model none\
    --optimizers.fields-color.optimizer.lr .0001 \
    --optimizers.fields-geometry.optimizer.lr .0001 \
    --pipeline.datamanager.train_num_images_to_sample_from $num_images\
    --pipeline.datamanager.train_num_times_to_repeat_images $num_times\
    --pipeline.datamanager.eval_num_images_to_sample_from 1 --vis tensorboard\
    --timestamp $TMP_STR \
    --experiment-name $experiment     sdfstudio-data \
    --data $data_path \

full_output_path=/scratch/vivienn/outputs/$TMP_STR/macro-bug-1-$experiment/$method/$TMP_STR
resolution=2048
ns-extract-mesh --load-config $full_output_path/config.yml \
    --resolution $resolution \
    --output-path $full_output_path/$resolution-mesh.ply \
    --use-point-color True \
    --create-visibility-mask True


# path is /scratch/vivienn/outputs/$TMP_STR/macro-bug-1-$experiment/$method/$TMP_STR/
mkdir -p /n/fs/lines/sdfstudio_outputs/miniatures/$experiment/$method
mv /scratch/vivienn/outputs/$TMP_STR/$experiment/$method/$TMP_STR /n/fs/lines/sdfstudio_outputs/miniatures/$experiment/$method
