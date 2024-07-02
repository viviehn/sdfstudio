#!/bin/bash
#!/bin/bash

hostname
echo $CUDA_VISIBLE_DEVICES
nvidia-smi

DATA_ID1=785e7504b9
DATA_ID2=0e75f3c4d9
DATA_ID3=c49a8c6cff
DATA_ID4=0a7cc12c0e
TMP_STR=$(date +%Y%m%d_%H%M%S)_$RANDOM
#mkdir -p /home/vivienn/outputs/$TMP_STR/$DATA_ID
#mkdir -p /scratch/vivienn/local_input
mkdir -p /scratch/vivienn/outputs/$TMP_STR


OMP_NUM_THREADS=4 ns-train neus-facto-angelo-multi \
    --output-dir /scratch/vivienn/outputs/$TMP_STR \
    --trainer.max-num-iterations 20101  --trainer.steps_per_save 5000\
    --trainer.steps-per-eval-image 1000\
    --trainer.steps-per-eval-batch 1000\
    --trainer.steps-per-eval-all-images 100000\
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
    --pipeline.model.sdf-field.num_scenes 4\
    --pipeline.model.background-model none\
    --pipeline.model.sparse_points_sdf_loss_mult 1.0\
    --pipeline.model.curvature-loss-warmup-steps 2000\
    --pipeline.model.curvature-loss-multi 0.0\
    --pipeline.model.eikonal-loss-mult 0.0\
    --pipeline.datamanager.train_num_rays_per_batch 2\
    --pipeline.datamanager.train_num_images_to_sample_from -1\
    --pipeline.datamanager.train_num_times_to_repeat_images -1\
    --pipeline.datamanager.eval_num_images_to_sample_from 1 --vis tensorboard\
    --experiment-name multiscene \
    --timestamp $TMP_STR \
    --logging.local-writer.enable False \
    --pipeline.datamanager.dataparser.multiscene-data /n/fs/3d-indoor/vivien_data/data/$DATA_ID1/dslr/sdfstudio /n/fs/3d-indoor/vivien_data/data/$DATA_ID2/dslr/sdfstudio /n/fs/3d-indoor/vivien_data/data/$DATA_ID3/dslr/sdfstudio /n/fs/3d-indoor/vivien_data/data/$DATA_ID4/dslr/sdfstudio \
    --pipeline.datamanager.dataparser.multiscene True \
    --pipeline.datamanager.dataparser.include-sdf-samples True
    #--pipeline.model.curvature-loss-multi 0.0\
    #--pipeline.model.eikonal-loss-mult 0.0\
    #--trainer.load-dir outputs/ngp-best/neus-facto-angelo/2024-02-26_105859/sdfstudio_models \
    #--pipeline.datamanager.dataparser.multiscene-data /n/fs/3d-indoor/vivien_data/data/$DATA_ID1/dslr/sdfstudio /n/fs/3d-indoor/vivien_data/data/$DATA_ID2/dslr/sdfstudio /n/fs/3d-indoor/vivien_data/data/$DATA_ID3/dslr/sdfstudio \
    #--pipeline.datamanager.dataparser.multiscene-data /n/fs/3d-indoor/vivien_data/data/$DATA_ID1/dslr/sdfstudio \

mkdir -p /n/fs/3d-indoor/vivien_sdfstudio_outputs/ngp-sdf-multiscene/$DATA_ID/neus-facto-angelo
#mv /scratch/vivienn/outputs/$TMP_STR/multiscene/neus-facto-angelo-multi/$TMP_STR /n/fs/3d-indoor/vivien_sdfstudio_outputs/ngp-sdf-multiscene/multiscene/neus-facto-angelo-multi
