#!/bin/bash
#!/bin/bash

hostname
echo $CUDA_VISIBLE_DEVICES
nvidia-smi

DATA_ID=785e7504b9
TMP_STR=$(date +%Y%m%d_%H%M%S)_$RANDOM
#mkdir -p /home/vivienn/outputs/$TMP_STR/$DATA_ID
#mkdir -p /scratch/vivienn/local_input
mkdir -p /scratch/vivienn/outputs/$TMP_STR


OMP_NUM_THREADS=4 ns-train neus-facto-angelo \
    --output-dir /scratch/vivienn/outputs/$TMP_STR \
    --trainer.max-num-iterations 6101  --trainer.steps_per_save 1000\
    --trainer.steps-per-eval-image 2000\
    --pipeline.model.sdf-field.inside-outside True     \
    --pipeline.model.sdf-field.num-layers 2     \
    --pipeline.model.sdf-field.hidden-dim 64     \
    --pipeline.model.sdf-field.geo-feat-dim 64     \
    --pipeline.model.sdf-field.num-layers-color 2     \
    --pipeline.model.sdf-field.log2-hashmap-size 22\
    --pipeline.model.sdf-field.hash-features-per-level 2\
    --pipeline.model.sdf-field.base-res 16\
    --pipeline.model.sdf-field.max-res 2048\
    --pipeline.model.sdf-field.use-appearance-embedding True\
    --pipeline.model.sdf-field.use-position-encoding True\
    --pipeline.model.sdf-field.vanilla-ngp True\
    --pipeline.model.sdf-field.geometric-init True\
    --pipeline.model.sdf-field.bias 0.8\
    --pipeline.model.sdf-field.fix-geonet False\
    --pipeline.model.sdf-field.use-numerical-gradients False\
    --pipeline.model.background-model none\
    --pipeline.model.sparse_points_sdf_loss_mult 1.0\
    --pipeline.datamanager.train_num_rays_per_batch 2\
    --pipeline.datamanager.train_num_images_to_sample_from 1\
    --pipeline.datamanager.train_num_times_to_repeat_images -1\
    --pipeline.datamanager.eval_num_images_to_sample_from 1 --vis tensorboard\
    --experiment-name ngp-best-t     sdfstudio-data \
    --data /n/fs/3d-indoor/vivien_data/data/$DATA_ID/dslr/sdfstudio \
    --include_sdf_samples True
    #--pipeline.model.curvature-loss-multi 0.0\
    #--pipeline.model.eikonal-loss-mult 0.0\
    #--trainer.load-dir outputs/ngp-best/neus-facto-angelo/2024-02-26_105859/sdfstudio_models \

mv /scratch/vivienn/outputs/$TMP_STR/ /n/fs/3d-indoor/vivien_sdfstudio_outputs/
