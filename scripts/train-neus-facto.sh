#!/bin/bash
hostname
echo $CUDA_VISIBLE_DEVICES
nvidia-smi



DATA_ID=785e7504b9
TMP_STR=$(date +%Y%m%d_%H%M%S)_$RANDOM
#mkdir -p /home/vivienn/outputs/$TMP_STR/$DATA_ID
#mkdir -p /scratch/vivienn/local_input
mkdir -p /scratch/vivienn/outputs/$TMP_STR/$DATA_ID
ns-train neus-facto  --trainer.max_num_iterations 200000 \
    --output-dir /scratch/vivienn/outputs/$TMP_STR/ \
    --pipeline.model.sdf-field.geometric-init True \
    --pipeline.model.sdf-field.bias 0.8  --pipeline.model.sdf-field.inside-outside True \
    --pipeline.datamanager.train_num_rays_per_batch 2048 --vis tensorboard \
    --experiment-name neus-facto-$DATA_ID sdfstudio-data \
    --data /n/fs/lines/scannetpp/data/$DATA_ID/dslr/sdfstudio
    #--viewer.websocket-port 16008 

mv /scratch/vivienn/outputs/$TMP_STR/ /n/fs/lines/sdfstudio_outputs/
