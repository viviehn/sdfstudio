#!/bin/bash
hostname
echo $CUDA_VISIBLE_DEVICES
nvidia-smi



DATA_ID=785e7504b9
TMP_STR=$(date +%Y%m%d_%H%M%S)_$RANDOM
mkdir -p /scratch/vivienn/outputs/$TMP_STR/$DATA_ID

local_data_path=/scratch/vivienn/local_input/sdfstudio_data/$DATA_ID
mkdir -p $local_data_path/dslr/sdfstudio
mkdir -p $local_data_path/scans
cp -r \
    /n/fs/lines/scannetpp/data/$DATA_ID/dslr/sdfstudio \
    $local_data_path/dslr/
cp -r \
    /n/fs/lines/scannetpp/data/$DATA_ID/scans \
    $local_data_path/

OMP_NUM_THREADS=4 ns-train neus-facto  \
    --output-dir /scratch/vivienn/outputs/$TMP_STR/ \
    --trainer.max_num_iterations 200000 \
    --pipeline.model.sdf-field.geometric-init True \
    --pipeline.model.sdf-field.bias 0.8  \
    --pipeline.model.sdf-field.inside-outside True \
    --pipeline.model.mono-depth-loss-mult 0.1 \
    --pipeline.model.mono-normal-loss-mult 0.05 \
    --pipeline.datamanager.train_num_rays_per_batch 2048 \
    --pipeline.datamanager.eval_num_rays_per_batch 1024 \
    --pipeline.datamanager.train_num_images_to_sample_from 128 \
    --pipeline.datamanager.train_num_times_to_repeat_images 100 \
    --pipeline.datamanager.eval_num_images_to_sample_from 16 \
    --vis tensorboard --experiment-name neus-facto-mono256_100-$s \
    sdfstudio-data \
    --data $local_data_path/dslr/sdfstudio \
    --include_mono_prior True
    #--trainer.load-dir outputs/neus-facto-mono256_100/neus-facto/2023-12-06_034653/sdfstudio_models\

mv /scratch/vivienn/outputs/$TMP_STR/ /n/fs/lines/sdfstudio_outputs/
