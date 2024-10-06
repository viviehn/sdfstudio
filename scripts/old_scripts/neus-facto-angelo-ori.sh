CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 ns-train neus-facto-angelo \
    --trainer.max-num-iterations 200001  --trainer.steps_per_save 1000\
    --pipeline.model.sdf-field.inside-outside True     \
    --pipeline.model.sdf-field.num-layers 1     \
    --pipeline.model.sdf-field.hidden-dim 256     \
    --pipeline.model.sdf-field.geo-feat-dim 256     \
    --pipeline.model.sdf-field.num-layers-color 2     \
    --pipeline.model.sdf-field.log2-hashmap-size 18\
    --pipeline.model.sdf-field.hash-features-per-level 4\
    --pipeline.model.sdf-field.base-res 16\
    --pipeline.model.sdf-field.max-res 2048\
    --pipeline.model.steps-per-level 800\
    --pipeline.model.sdf-field.use-appearance-embedding True\
    --pipeline.model.sdf-field.use-position-encoding True\
    --pipeline.model.sdf-field.vanilla-ngp False\
    --pipeline.model.sdf-field.geometric-init True\
    --pipeline.model.sdf-field.bias 0.8\
    --pipeline.model.background-model none\
    --pipeline.datamanager.train_num_images_to_sample_from 1\
    --pipeline.datamanager.train_num_times_to_repeat_images -1\
    --pipeline.datamanager.eval_num_images_to_sample_from 8 --vis tensorboard\
    --experiment-name neus-facto-angelo-ft     sdfstudio-data \
    --data /home/fangyin/data/scannetpp/785e7504b9/dslr/sdfstudio\
    #--include_sdf_samples True
    #--pipeline.model.sparse_points_sdf_loss_mult 1.0\
    #--trainer.load-dir outputs/ngp_models\
    #--trainer.load-dir outputs/neus-facto-angelo-improve/neus-facto-angelo/2023-12-06_075728/sdfstudio_models\
    #--pipeline.model.sdf-field.log2-hashmap-size 18\
    #--pipeline.model.sdf-field.hash-features-per-level 4\
    #--pipeline.model.sdf-field.use-position-encoding True\
    #--trainer.load-dir outputs/ngp_models/785e7504b9 \
